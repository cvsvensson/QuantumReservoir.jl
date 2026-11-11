using FermionicHilbertSpaces, LinearAlgebra
using LinearAlgebra
using Plots
using ExponentialUtilities
using MLJLinearModels
using OrdinaryDiffEqHighOrderRK, OrdinaryDiffEqTsit5
using Statistics
using Random
includet("opensystems.jl")
includet("system.jl")
includet("misc.jl")

Random.seed!(1234)
training_parameters = generate_training_parameters(1000);
validation_parameters = generate_training_parameters(1000);
#includet("gpu.jl")
##
N = 1
labels = vec(Base.product(0:N, 1:2) |> collect)
@fermions f
space = hilbert_space(f, labels, NumberConservation())
bases = (; f, space)
hopping_labels = [(labels[k1], labels[k2]) for k1 in 1:length(labels), k2 in 1:length(labels) if k1 > k2 && is_nearest_neighbours(labels[k1], labels[k2])]
Ilabels = filter(x -> first(x) <= 0, labels)
Rlabels = filter(x -> first(x) > 0, labels)
Ihalflabels = filter(x -> isone(x[2]), Ilabels)
Iconnections = filter(k -> first(k[1]) + first(k[2]) == 0, hopping_labels)
Rconnections = filter(k -> first(k[1]) + first(k[2]) > 1, hopping_labels)
IRconnections = filter(k -> abs(first(k[1]) - first(k[2])) == 1, hopping_labels)
##
J = Dict((k1, k2) => 2(rand() - 0.5) for (k1, k2) in hopping_labels) # random_hoppings(hopping_labels)
# J = Dict((k1, k2) => rand() * exp(2pi * 1im * rand()) for (k1, k2) in hopping_labels) # random_hoppings(hopping_labels)
V = Dict((k1, k2) => rand() for (k1, k2) in hopping_labels) # random_hoppings(hopping_labels, 10)
ε = Dict(l => rand() - 0.5 for l in labels)
Γ = 0.5ones(2, 2)#.5 *1e0 * (rand(2, 2))
##
HR = representation(hopping_hamiltonian(f, J; labels=Rconnections), space)
HI = representation(hopping_hamiltonian(f, J; labels=Iconnections), space)
HIR = representation(hopping_hamiltonian(f, J; labels=IRconnections), space)
HV = representation(coulomb_hamiltonian(f, V; labels=hopping_labels), space)
Hqd = representation(qd_level_hamiltonian(f, ε), space)
##
H0 = HR + HI + 5HV + 0Hqd
H = H0 + HIR
eigvals(Hermitian(Matrix(H))) |> sort
eigvals(Hermitian(Matrix(H - 5HV))) |> sort
eigvals(Hermitian(Matrix(H - 5.1HV))) |> sort |> diff |> minimum
eigvals(Hermitian(Matrix(H))) |> sort |> diff |> minimum
##
μmin = -1e5
μs = [10, 0]#[μmin, μmin]#rand(2)
T = 10norm(Γ)
leads = Tuple(CombinedLead((representation(f[N, k]' * Γ[k, k] + f[N, mod1(k + 1, 2)]' * Γ[k, mod1(k + 1, 2)], space),); T, μ=μs[k]) for k in 1:1)
input_dissipator = CombinedLead(Tuple(representation(f[i]', space) for i in Ilabels); T, μ=μmin)
leads0 = (input_dissipator, leads...)

# ls = LindbladSystem(H, leads0)
# vals, vecs = eigen(Matrix(ls.total))
reservoir = QuantumReservoir(H0, H, leads0, leads, bases, Ilabels, Rlabels);
rhoI0 = partial_trace_to(reservoir.rho0, bases, Ilabels)
rhoR0 = partial_trace_to(reservoir.rho0, bases, Rlabels)

##
M = 400
training_ensemble = InitialEnsemble(training_parameters[1:M], reservoir)
test_ensemble = InitialEnsemble(validation_parameters[1:M], reservoir)

##
tspan = (0, 40 / (norm(Γ)^1))#*log(norm(Γ)))
t_obs = range(tspan[end] / 100, tspan[end] / 2, 20)
t_nl = [1]
proc = CPU();
op = representation(f[N, 1]' * f[N, 1], space)
@time training_sols = time_evolve_exp(reservoir, training_ensemble, t_obs; proc);
@time training_sols2 = time_evolve_exp(reservoir, training_ensemble, t_obs, t_nl; op, proc);
@time test_sols = time_evolve_exp(reservoir, test_ensemble, t_obs; proc);
@time timesols = time_evolve_ode(reservoir, training_ensemble[1:2], tspan, t_obs; proc, alg=Tsit5());
@time timesols = time_evolve_ode(reservoir, training_ensemble[1:2], tspan, t_obs, t_nl; op=representation(f[N, 1] * f[N, 1]', space), proc, alg=Tsit5());
superpos_states = [training_ensemble.rho0s[1], training_ensemble.rho0s[2], (training_ensemble.rho0s[1] + training_ensemble.rho0s[2]) / 2]
superpos_ensemble = InitialEnsemble(superpos_states, training_data(superpos_states, reservoir.bases, reservoir.I_occ_ops, Ilabels))
@time timesols = time_evolve_ode(reservoir, superpos_ensemble, tspan, t_obs, t_nl; op=representation(f[N, 1] * f[N, 1]', space), proc, alg=Tsit5());
##
p = plot();
map((sol, ls) -> plot!(p, sol.ts, sol.currents; ls, lw=2, c=[:red :blue], label="Lead" .* string.(eachindex(reservoir.leads))), timesols, [:solid, :dash, :dashdot]);
vline!(t_obs, label="observations");
p

## Training
X = training_sols2.data
# X .+= 0randn(size(X)) * 1e-3 * mean(abs, X)
y = training_ensemble.data
ridge = RidgeRegression(1e-6; fit_intercept=true)
W1 = reduce(hcat, map(data -> fit(ridge, X', data), eachrow(y))) |> permutedims
W2 = y * X' * inv(X * X' + 1e-8 * I)
W3 = y * pinv(X)
##
titles = ["entropy of one input dot", "purity of inputs", "ρ11", "ρ22", "ρ33", "ρ44", "real(ρ23)", "imag(ρ23)", "n1", "n2"]
let is = 1:10, perm, W = W1, X = test_sols.data, y = test_ensemble.data, b
    p = plot(; size=1.2 .* (600, 400))
    colors = cgrad(:seaborn_dark, size(y, 1))
    # colors2 = cgrad(:seaborn_dark, size(y, 2))
    colors2 = cgrad(:seaborn_bright, size(y, 1))
    for i in is
        perm = sortperm(y[i, :])
        Wi, b = size(W, 2) > size(X, 1) ? (W[i, 1:end-1], W[i, end] * ones(M)) : (W[i, :], zeros(M))
        plot!(p, (Wi' * X[:, perm])' .+ b; label=titles[i] * " pred", lw=3, c=colors[i], frame=:box)
        plot!(y[i, perm]; label=titles[i] * " truth", lw=3, ls=:dash, c=colors2[i])
    end
    display(p)
end

##
function loss_function(W, X, y)
    loss = 0.0
    M = size(y, 2)
    for i in axes(W, 1)
        Wi, b = size(W, 2) > size(X, 1) ? (W[i, 1:end-1], W[i, end] * ones(M)) : (W[i, :], zeros(M))
        loss += sum(abs2, (Wi' * X)' .- y[i, :] .+ b)
    end
    return sqrt(loss) #/ size(X, 2)
end
loss_function(W1, training_sols.data, training_ensemble.data)
loss_function(W1, test_sols.data, test_ensemble.data)
##

function get_loss_function(bases, J, V, ε, Γ, M, training_parameters, validation_parameters)
    (; f, space) = bases
    HR = representation(hopping_hamiltonian(f, J; labels=Rconnections), space)
    HI = representation(hopping_hamiltonian(f, J; labels=Iconnections), space)
    HIR = representation(hopping_hamiltonian(f, J; labels=IRconnections), space)
    HV = representation(coulomb_hamiltonian(f, V; labels=hopping_labels), space)
    Hqd = representation(qd_level_hamiltonian(f, ε), space)
    function _loss_function(v, ϵ, μs, γ, tend)
        H0 = HR + HI + v * HV + ϵ * Hqd
        H = H0 + HIR
        Γ = γ * Γ
        T = 10norm(Γ)
        input_dissipator = CombinedLead(Tuple(representation(f[i]', space) for i in Ilabels); T, μ=-1e5)
        leads = Tuple(CombinedLead((representation(f[N, k]' * Γ[k, k] + f[N, mod1(k + 1, 2)]' * Γ[k, mod1(k + 1, 2)], space),); T, μ=μs[k]) for k in 1:1)
        leads0 = (input_dissipator, leads...)
        reservoir = QuantumReservoir(H0, H, leads0, leads, bases, Ilabels, Rlabels)
        training_ensemble = InitialEnsemble(training_parameters[1:M], reservoir)
        test_ensemble = InitialEnsemble(validation_parameters[1:M], reservoir)
        t_obs = range(tend / 50, tend, 20)
        training_sols = time_evolve_exp(reservoir, training_ensemble, t_obs)
        test_sols = time_evolve_exp(reservoir, test_ensemble, t_obs)
        X = training_sols.data
        y = training_ensemble.data
        ridge = RidgeRegression(1e-6; fit_intercept=true)
        W1 = reduce(hcat, map(data -> fit(ridge, X', data), eachrow(y))) |> permutedims
        return loss_function(W1, test_sols.data, test_ensemble.data)
    end
end
##
fl = get_loss_function(bases, J, V, ε, Γ, 100, training_parameters, validation_parameters)
@time fl(5, 0, 10, 1, 20)
plot([fl(v, 0.0, 0.0, 1.0, 10.0) for v in range(0, 100, 10)])
plot([fl(6.0, v, 100.0, 1.0, 10.0) for v in range(-1, 1, 10)])
plot([fl(6.0, 0.0, v, 1.0, 10.0) for v in range(0, 10, 10)])
plot([fl(6.0, 0.0, 100.0, 1.0, v) for v in range(0.5, 10, 10)])
##
using Optimization, OptimizationOptimJL
# (v, ϵ, μs, γ, tend)
prob = OptimizationProblem((u, p) -> fl(u...), [1.0, 1.0, 1.0, 1.0, 10.0]; lb=Float64[0, -10, -10, 0.5, 1], ub=Float64[10, 10, 100, 2, 40])
sol = solve(prob, ParticleSwarm(); maxiters=1000, maxtime=20, show_trace=true)
##
prob = OptimizationProblem((u, p) -> fl(u..., 1, 40), [1.0, 1.0, 1.0]; lb=Float64[0, -10, -10], ub=Float64[10, 10, 100])
sol1 = solve(prob, ParticleSwarm(); maxiters=1000, maxtime=20, show_trace=true)
# sol2 = solve(prob, BBO_probabilistic_descent(); maxiters=1000, maxtime=20, show_trace=true)
# sol3 = solve(prob, BBO_adaptive_de_rand_1_bin_radiuslimited(); maxiters=1000, maxtime=20, show_trace=true)

##
sol = solve(OptimizationProblem((x, p) -> norm(x .^ 2), [1.0, 2.0, 3.0]), ParticleSwarm(); maxiters=100, maxtime=1, show_trace=true)
## Analyze initial state
ls0 = reservoir.ls0
rho0 = reservoir.rho0
rhointernal0 = reservoir.rhointernal0
particle_number = representation(number_operator(f, labels), space)
rhod0 = diag(rho0)
@assert isapprox(tr(rho0), 1; atol=1e-3)
currents_ops = collect(values(current_operators(ls0, particle_number)))
currents = [tr(tomatrix(diss * rhointernal0, ls0) * particle_number) for diss in values(ls0.dissipators)]
currents2 = [rhointernal0' * op for op in currents_ops]
@assert norm(real.(currents) - real.(currents2)) < 1e-2

##
rhoI0 = partial_trace_to(rho0, bases, Ilabels)
rhoR0 = partial_trace_to(rho0, bases, Rlabels)
@assert isapprox(rhoI0[1], 1; atol=1e-3)
@assert isapprox(tr(rhoI0), 1; atol=1e-3)
rhoI0
rhoR0

##
ls = reservoir.ls
current_ops = collect(values(current_operators(ls, particle_number)))
R_occ_ops = map(k -> vecrep(representation(f[k]' * f[k], space), ls), Rlabels)
I_occ_ops = map(k -> representation(f[k]' * f[k], space), Ilabels)

## Entanglement non-linearity
perm = sortperm(training_ensemble.data[1, :])
entropies = training_ensemble.data[1, perm];
rhos = training_ensemble.rho0s[perm];
ent1 = [input_entanglement((r1 + r2) / 2, Ihalflabels, bases) for (n1, r1) in enumerate(rhos), (n2, r2) in enumerate(rhos)]
ent2 = [(entropies[n1] + entropies[n2]) / 2 for (n1, r1) in enumerate(rhos), (n2, r2) in enumerate(rhos)]

heatmap(ent1)
heatmap(ent2)
