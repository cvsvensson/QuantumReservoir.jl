using QuantumDots, QuantumDots.BlockDiagonals, LinearAlgebra
using LinearAlgebra
using Plots
using LinearSolve
using ExponentialUtilities
using MLJLinearModels
using OrdinaryDiffEq
using Statistics
using Folds
using Random
includet("misc.jl")

Random.seed!(1234)
training_parameters = generate_training_parameters(1000);
validation_parameters = generate_training_parameters(1000);
#includet("gpu.jl")
##
N = 1
# labels = push!(vec(Base.product(0:N, 1:2) |> collect), (0, 0))
labels = vec(Base.product(0:N, 1:2) |> collect)
qn = QuantumDots.fermionnumber
c = FermionBasis(labels; qn)
hopping_labels = [(labels[k1], labels[k2]) for k1 in 1:length(labels), k2 in 1:length(labels) if k1 > k2 && is_nearest_neighbours(labels[k1], labels[k2])]
Ilabels = filter(x -> first(x) <= 0, labels)
Rlabels = filter(x -> first(x) > 0, labels)
Ihalflabels = filter(x -> isone(x[2]), Ilabels)
Iconnections = filter(k -> first(k[1]) + first(k[2]) == 0, hopping_labels)
Rconnections = filter(k -> first(k[1]) + first(k[2]) > 1, hopping_labels)
IRconnections = filter(k -> abs(first(k[1]) - first(k[2])) == 1, hopping_labels)
##
J = random_hoppings(hopping_labels)
V = random_hoppings(hopping_labels, 10)
##
HR = hopping_hamiltonian(c, J; labels=Rconnections)
HI = hopping_hamiltonian(c, J; labels=Iconnections)
HIR = hopping_hamiltonian(c, J; labels=IRconnections)
HV = coulomb_hamiltonian(c, V; labels=hopping_labels)
H0 = HR + HI + HV
H = H0 + HIR
##
Γ = 1e0 * (rand(2, 2))
μmin = -1e5
μs = [1, 0]#[μmin, μmin]#rand(2)
T = 10norm(Γ)
# leads = Tuple(CombinedLead((c[N, k]' * Γ[k, k], c[N, mod1(k + 1, 2)]' * Γ[k, mod1(k + 1, 2)]); T, μ=μs[k]) for k in 1:1)
leads = Tuple(CombinedLead((c[N, k]' * Γ[k, k] + c[N, mod1(k + 1, 2)]' * Γ[k, mod1(k + 1, 2)],); T, μ=μs[k]) for k in 1:1)
input_dissipator = CombinedLead(Tuple(c[i]' for i in Ilabels); T, μ=μmin)
# input_dissipator2 = CombinedLead((c[0, 2]',); T, μ=μmin)
leads0 = (input_dissipator, leads...)

ls = LindbladSystem(H, leads0)
vals, vecs = eigen(Matrix(ls.total))
reservoir = QuantumReservoir(H0, H, leads0, leads, c, Ilabels, Rlabels);
##
M = 200
training_ensemble = InitialEnsemble(training_parameters[1:M], reservoir)
test_ensemble = InitialEnsemble(training_parameters[M+1:2M], reservoir)

##
tspan = (0, 100 / (norm(Γ)^1))#*log(norm(Γ)))
t_obs = range(tspan[end] / 100, tspan[end] / 2, 20)
proc = CPU();
@time timesols = time_evolve(reservoir, training_ensemble[1:2], tspan, t_obs; proc, alg=ROCK4());
@time training_sols = time_evolve(reservoir, training_ensemble, t_obs; proc);
@time test_sols = time_evolve(reservoir, test_ensemble, t_obs; proc);
##
p = plot();
map((sol, ls) -> plot!(p, sol.ts, sol.currents; ls, lw=2, c=[:red :blue], label="Lead" .* string.(eachindex(reservoir.leads))), timesols, [:solid, :dash, :dashdot]);
vline!(t_obs, label="observations");
p

## Training
X = training_sols.data
# X .+= 0randn(size(X)) * 1e-3 * mean(abs, X)
y = training_ensemble.data
ridge = RidgeRegression(1e-6; fit_intercept=true)
W1 = reduce(hcat, map(data -> fit(ridge, X', data), eachrow(y))) |> permutedims
W2 = y * X' * inv(X * X' + 1e-8 * I)
W3 = y * pinv(X)
##
titles = ["entropy of one input dot", "purity of inputs", "ρ11", "ρ22", "ρ33", "ρ44", "real(ρ23)", "imag(ρ23)", "n1", "n2"]
let is = 1:4, perm, W = W1, X = test_sols.data, y = test_ensemble.data, b
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
    return sqrt(loss) / size(X, 2)
end
loss_function(W1, training_sols.data, training_ensemble.data)
loss_function(W1, test_sols.data, test_ensemble.data)
##

##


## Analyze initial state
rhod0 = diag(rho0)
@assert isapprox(tr(rho0), 1; atol=1e-3)
internal_N = QuantumDots.internal_rep(particle_number, ls0)
currents_ops = map(diss -> diss' * internal_N, ls0.dissipators)
currents = map(diss -> tr(QuantumDots.tomatrix(diss * rhointernal0, ls0) * particle_number), ls0.dissipators)
currents2 = map(op -> rhointernal0' * op, currents_ops)
@assert norm(map(-, currents, currents2)) / norm(currents) < 1e-2

##
cR = FermionBasis(Rlabels; qn)
cI = FermionBasis(Ilabels; qn)
rhoI0 = partial_trace(rho0, Ilabels, c, cI.symmetry)
rhoR0 = partial_trace(rho0, Rlabels, c, cR.symmetry)
@assert isapprox(rhoI0[1], 1; atol=1e-3)
@assert isapprox(tr(rhoI0), 1; atol=1e-3)
pretty_print(rhoI0, cI)
pretty_print(rhoR0, cR)

##
# ls = QuantumDots.LazyLindbladSystem(H, leads)
ls = QuantumDots.LindbladSystem(H, leads)
internal_N = QuantumDots.internal_rep(particle_number, ls)
current_ops = map(diss -> diss' * internal_N, ls.dissipators)
R_occ_ops = map(k -> QuantumDots.internal_rep(c[k]' * c[k], ls), Rlabels)
I_occ_ops = map(k -> c[k]' * c[k], Ilabels)