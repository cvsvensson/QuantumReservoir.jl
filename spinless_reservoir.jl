using QuantumDots
using QuantumDots.BlockDiagonals
using LinearAlgebra
using Plots
using LinearSolve
using ExponentialUtilities
using MLJLinearModels
using OrdinaryDiffEq
##
N = 1
labels = Base.product(0:N, 1:2) |> collect
spatial_labels = labels
qn = QuantumDots.fermionnumber
c = FermionBasis(labels; qn)
connection_labels = filter((ks) -> ks[1] != ks[2], Base.product(spatial_labels, spatial_labels) |> collect)
Ilabels = filter(iszero ∘ first, spatial_labels)
Rlabels = filter(k -> first(k) > 0, spatial_labels)
fullIlabels = filter(iszero ∘ first, labels)
fullRlabels = filter(k -> first(k) > 0, labels)
fullIhalflabels = filter(x -> isone(x[2]), fullIlabels)
IRconnections = filter(k -> abs(first(k[1]) - first(k[2])) == 1, connection_labels)
Rconnections = filter(k -> first(k[1]) + first(k[2]) > 1, connection_labels)
Iconnections = filter(k -> first(k[1]) + first(k[2]) == 0, connection_labels)
##
function hamiltonian(c, J; Jkeys=keys(J))
    H = deepcopy(first(c))
    for (k1, k2) in Jkeys
        #Hoppings
        H += J[(k1, k2)] * c[k1]'c[k2] + hc
    end
    return blockdiagonal(H, c)
end

## Generate couplings
is_nearest_neighbours(k1, k2) = k1 != k2 && abs(first(k1) - first(k2)) ∈ (0, 1)
function random_nearest_hoppings(spatial_labels, s=1)
    couplings = [(k1, k2) => s * rand() * is_nearest_neighbours(k1, k2) for k1 in spatial_labels, k2 in spatial_labels]
    Dict(couplings)
end
##

J = random_nearest_hoppings(spatial_labels)
HR = hamiltonian(c, J; Jkeys=Rconnections)
HI = hamiltonian(c, J; Jkeys=Iconnections)
HIR = hamiltonian(c, J; Jkeys=IRconnections)

##
Γ = rand(2, 2)
μL, μR = rand(2)
T = 0.1
leftlead = CombinedLead((c[N, 1]' * Γ[1, 1], c[N, 2]' * Γ[1, 2]); T, μ=μL)
rightlead = CombinedLead((c[N, 1,]' * Γ[2, 1], c[N, 2]' * Γ[2, 2]); T, μ=μR)
input_dissipator = CombinedLead((c[0, 1]', c[0, 2]'); T, μ=-10)
leads0 = (; input=input_dissipator, left=leftlead, right=rightlead)
leads = (; left=leftlead, right=rightlead)

##
function modify_initial_state((r, θ), c=c; kwargs...)
    # A = α * c[0, 1]' * c[0, 2]' - hc
    # b -> expv(one(eltype(A)), A, b; kwargs...)

    # e = exp(Matrix(A))
    # e = rand() < 0.5 ? e : c[0, 1]'e
    si, co = sincos(θ)
    e = r * (c[0, 1]' * si + co * c[0, 2]') + I
    b -> e * b * e'
end
I_n_ops = map(k -> c[k]' * c[k], fullIlabels)
normalize_rho!(rho) = rdiv!(rho, tr(rho))
function generate_training_data(M, rho)
    phases = 2pi * rand(M)
    r = 10 * rand(M)
    αs = r .* exp.(1im * phases)
    ops = modify_initial_state.(zip(r, phases))
    rhonew = [op(rho) for op in ops]
    normalize_rho!.(rhonew)
    true_data = reduce(hcat, get_target_data.(rhonew)) |> permutedims
    return (; αs, rhos=rhonew, true_data)
end
function get_target_data(rho)
    occupations = [real(tr(rho * op)) for op in I_n_ops]
    entropy = input_entanglement(rho)
    return [entropy, occupations...]
end

function input_entanglement(rho, c=c)
    rhosub = partial_trace(rho, fullIhalflabels, c)
    real(-tr(rhosub * log(rhosub)))
end
##
particle_number = blockdiagonal(numberoperator(c), c)
H0 = HR + HI
ls0 = LindbladSystem(H0, leads0)
lazyls0 = LazyLindbladSystem(H0, leads0)

##
prob0 = StationaryStateProblem(ls0)
lazyprob0 = StationaryStateProblem(lazyls0)
@time rhointernal0 = solve(prob0, LinearSolve.KrylovJL_LSMR(); abstol=1e-6);
@time lazyrhointernal0 = solve(lazyprob0, LinearSolve.KrylovJL_LSMR(); abstol=1e-6);
rho0 = QuantumDots.tomatrix(rhointernal0, ls0)
lazyrho0 = reshape(lazyrhointernal0, size(H0))
@assert norm(rho0 - lazyrho0) < 1e-6

## Analyze initial state
rhod0 = diag(rho0)
@assert tr(rho0) ≈ 1
internal_N = QuantumDots.internal_rep(particle_number, ls0)
currents_ops = map(diss -> diss' * internal_N, ls0.dissipators)
currents = map(diss -> tr(QuantumDots.tomatrix(diss * rhointernal0, ls0) * particle_number), ls0.dissipators)
currents2 = map(op -> rhointernal0' * op, currents_ops)
@assert norm(map(-, currents, currents2)) / norm(currents) < 1e-3

cR = FermionBasis(fullRlabels; qn)
cI = FermionBasis(fullIlabels; qn)
rhoI0 = partial_trace(rho0, fullIlabels, c, cI.symmetry)
rhoR0 = partial_trace(rho0, fullRlabels, c, cR.symmetry)
@assert isapprox(rhoI0[1], 1; atol=1e-3)
@assert tr(rhoI0) ≈ 1
pretty_print(rhoI0, cI)
pretty_print(rhoR0, cR)

##
H = H0 + HIR
ls = QuantumDots.LindbladSystem(H, leads)

##
function get_obs_data(sol, t_obs, current_ops)
    reduce(vcat, [[real(sol(t)' * op) for op in current_ops] for t in t_obs])
end
function time_evolve(rho, ls, current_ops, t_obs=[0.5, 1, 2])
    rhointernal = QuantumDots.internal_rep(rho, ls)
    drho!(out, rho, p, t) = mul!(out, ls, rho)
    prob = ODEProblem(drho!, rhointernal, (0, 20))
    sol = solve(prob, Tsit5())
    currents = reduce(hcat, [[real(sol(t)' * op) for op in current_ops] for t in sol.t]) |> permutedims
    observations = get_obs_data(sol, t_obs, current_ops)
    trs = [tr(QuantumDots.tomatrix(sol(t), ls)) |> real for t in sol.t]
    subrhos = [partial_trace(QuantumDots.tomatrix(sol(t), ls), fullIlabels, c) for t in sol.t]
    tr2s = [tr(QuantumDots.tomatrix(sol(t), ls)^2) |> real for t in sol.t]
    return (; sol, currents, trs, tr2s, observations, subrhos)
end
internal_N = QuantumDots.internal_rep(particle_number, ls)
current_ops = map(diss -> diss' * internal_N, ls.dissipators)
##
M = 100
train_data = generate_training_data(M, rho0)
sols = map(rho0 -> time_evolve(deepcopy(rho0), ls, current_ops), train_data.rhos);
observed_data = reduce(hcat, map(sol -> sol.observations, sols)) |> permutedims
##
p = plot()
map(sol -> plot!(p, sol.sol.t, sol.trs, ylims=(0, 1.1)), sols[1:3])
map(sol -> plot!(p, sol.sol.t, sol.tr2s, ylims=(0, 1.1)), sols[1:3])
p
##
p = plot()
map((sol, ls) -> plot!(p, sol.sol.t, sol.currents; ls, lw=2, c=[:red :blue]), sols, [:solid, :dash, :dashdot])
p

## Training
X = observed_data
y = train_data.true_data

ridge = RidgeRegression(; fit_intercept=false)
W = reduce(hcat, map(data -> fit(ridge, X, data), eachcol(y)))

R = X' * X
P = X' * y
W2 = inv(R + 1e-6 * I) * P

W3 = pinv(X) * y

map((x, yr) -> x' * W[1:end-1, :] .+ W[end, :] .- yr |> norm, eachrow(X), eachrow(y)) |> norm
norm.(eachcol(X * W - y))
norm.(eachcol(X * W2 - y))
norm.(eachcol(X * W3 - y))
y_pred = X * W

norm.(eachcol(y - y_pred))
##
plot((y_pred[:, 1] - y[:, 1]) ./ (y[:, 1] + y_pred[:, 1]))
plot((y_pred[:, 2] - y[:, 2]) ./ (y[:, 2] + y_pred[:, 2]))
plot((y_pred[:, 3] - y[:, 3]) ./ (y[:, 3] + y_pred[:, 3]))