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
labels = vec(Base.product(0:N, 1:2) |> collect)
qn = QuantumDots.fermionnumber
c = FermionBasis(labels; qn)
hopping_labels = [(labels[k1], labels[k2]) for k1 in 1:length(labels), k2 in 1:length(labels) if k1 > k2 && is_nearest_neighbours(labels[k1], labels[k2])]
Ilabels = filter(iszero ∘ first, labels)
Rlabels = filter(!iszero ∘ first, labels)
Ihalflabels = filter(x -> isone(x[2]), Ilabels)
Iconnections = filter(k -> first(k[1]) + first(k[2]) == 0, hopping_labels)
Rconnections = filter(k -> first(k[1]) + first(k[2]) > 1, hopping_labels)
IRconnections = filter(k -> abs(first(k[1]) - first(k[2])) == 1, hopping_labels)
##
J = random_hoppings(hopping_labels)
V = random_hoppings(hopping_labels)
##
HR = hopping_hamiltonian(c, J; labels=Rconnections)
HI = hopping_hamiltonian(c, J; labels=Iconnections)
HIR = hopping_hamiltonian(c, J; labels=IRconnections)
HV = coulomb_hamiltonian(c, V; labels=IRconnections)

##
Γ = 1e1 * (rand(2, 2))
μmin = -10000
μs = [μmin, μmin]#rand(2)
T = 10norm(Γ)
# leads = Tuple(CombinedLead((c[N, k]' * Γ[k, k], c[N, mod1(k + 1, 2)]' * Γ[k, mod1(k + 1, 2)]); T, μ=μs[k]) for k in 1:1)
leads = Tuple(CombinedLead((c[N, k]' * Γ[k, k] + c[N, mod1(k + 1, 2)]' * Γ[k, mod1(k + 1, 2)], ); T, μ=μs[k]) for k in 1:1)
input_dissipator = CombinedLead((c[0, 1]', c[0, 2]'); T, μ=μmin)
leads0 = (input_dissipator, leads...)

##
particle_number = blockdiagonal(numberoperator(c), c)
H0 = HR + HI + HV
ls0 = LindbladSystem(H0, leads0)
lazyls0 = LazyLindbladSystem(H0, leads0)

##
prob0 = StationaryStateProblem(ls0)
lazyprob0 = StationaryStateProblem(lazyls0)
@time rhointernal0 = solve(prob0, LinearSolve.KrylovJL_LSMR(); abstol=1e-6);
@time lazyrhointernal0 = solve(lazyprob0, LinearSolve.KrylovJL_LSMR(); abstol=1e-6);
rho0 = QuantumDots.tomatrix(rhointernal0, ls0)
lazyrho0 = reshape(lazyrhointernal0, size(H0))
@assert norm(rho0 - lazyrho0) < 1e-4

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
H = H0 + HIR
# ls = QuantumDots.LazyLindbladSystem(H, leads)
ls = QuantumDots.LindbladSystem(H, leads)
internal_N = QuantumDots.internal_rep(particle_number, ls)
current_ops = map(diss -> diss' * internal_N, ls.dissipators)
R_occ_ops = map(k -> QuantumDots.internal_rep(c[k]' * c[k], ls), Rlabels)
I_occ_ops = map(k -> c[k]' * c[k], Ilabels)
##
M = 100
training_rho0s = generate_initial_states(training_parameters, rho0; M)
validation_rho0s = generate_initial_states(validation_parameters, rho0; M)
train_data = training_data(training_rho0s, c; occ_ops=I_occ_ops, Ilabels)
val_data = training_data(validation_rho0s, c; occ_ops=I_occ_ops, Ilabels)

##
tspan = (0, 100 / (norm(Γ)^1))#*log(norm(Γ)))
t_obs = range(0.1 / norm(Γ), tspan[end] / 2, 10)
proc = CPU();
timesols = map(rho0 -> time_evolve(proc, rho0, ls, tspan, t_obs; current_ops, occ_ops=R_occ_ops), training_rho0s[1:2]);
@time sols = map(rho0 -> time_evolve(proc, rho0, ls, t_obs; current_ops, occ_ops=R_occ_ops), training_rho0s);
observed_data = reduce(hcat, sols) |> permutedims;

val_sols = map(rho0 -> time_evolve(proc, rho0, ls, t_obs; current_ops, occ_ops=R_occ_ops), validation_rho0s);
val_observed_data = reduce(hcat, val_sols) |> permutedims;
##
p = plot();
map((sol, ls) -> plot!(p, sol.ts, sol.currents; ls, lw=2, c=[:red :blue]), timesols, [:solid, :dash, :dashdot]);
vline!(t_obs);
p
##
p = plot()
map((sol, ls) -> plot!(p, sol.ts, sol.observations; ls, lw=2, c=[:red :blue]), timesols, [:solid, :dash, :dashdot])
vline!(t_obs)
p

## Training
X = observed_data + randn(size(observed_data)) * 1e-3 * mean(abs, observed_data)
y = train_data
ridge = RidgeRegression(1e-8; fit_intercept=true)
W1 = reduce(hcat, map(data -> fit(ridge, X, data), eachcol(y)))
W2 = inv(X' * X + 1e-8 * I) * X' * y
W3 = pinv(X) * y
##
titles = ["entropy of one input dot", "purity of inputs", "ρ11", "ρ22", "ρ33", "ρ44", "real(ρ23)", "imag(ρ23)", "n1", "n2"]
let is = 1:10, perm, W = W1, X = val_observed_data, y = val_data, b
    p = plot(; size=1.2 .* (600, 400))
    colors = cgrad(:seaborn_dark, size(y, 2))
    # colors2 = cgrad(:seaborn_dark, size(y, 2))
    colors2 = cgrad(:seaborn_bright, size(y, 2))
    for i in is
        perm = sortperm(y[:, i])
        Wi, b = size(W, 1) > size(X, 2) ? (W[1:end-1, i], ones(M) * W[end, i]') : (W[:, i], zeros(M))
        plot!(p, X[perm, :] * Wi .+ b; label=titles[i] * " pred", lw=3, c=colors[i], frame=:box)
        plot!(y[perm, i]; label=titles[i] * " truth", lw=3, ls=:dash, c=colors2[i])
    end
    display(p)
end
