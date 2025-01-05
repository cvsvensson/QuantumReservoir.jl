using QuantumDots
using QuantumDots.BlockDiagonals
using LinearAlgebra
using Plots
using LinearSolve
using ExponentialUtilities

N = 1
# qn=QuantumDots.fermionnumber
labels = Base.product(0:N, 1:2, (:↑, :↓)) |> collect
spatial_labels = Base.product(0:N, 1:2) |> collect
uplabels = map(l -> (l..., :↑), spatial_labels)
downlabels = map(l -> (l..., :↓), spatial_labels) |> vec
qn = f -> (QuantumDots.fermionnumber(uplabels, vec(labels))(f), QuantumDots.fermionnumber(downlabels, vec(labels))(f))
c = FermionBasis(labels; qn)
connection_labels = filter((ks) -> ks[1] != ks[2], Base.product(spatial_labels, spatial_labels) |> collect)
Ilabels = filter(iszero ∘ first, spatial_labels)
Rlabels = filter(k -> first(k) > 0, spatial_labels)
fullIlabels = filter(iszero ∘ first, labels)
fullRlabels = filter(k -> first(k) > 0, labels)
IRconnections = filter(k -> first(k[1]) + first(k[2]) != 1, connection_labels)
Rconnections = filter(k -> first(k[1]) + first(k[2]) > 1, connection_labels)
Iconnections = filter(k -> first(k[1]) + first(k[2]) == 0, connection_labels)
##
function hamiltonian(c, J, U; Jkeys=keys(J), Ukeys=keys(U))
    H = deepcopy(first(c))
    for (k1, k2) in Jkeys
        #Hoppings
        foreach(σ -> H += (J[(k1, k2)] * c[k1..., σ]'c[k2..., σ] + hc), (:↑, :↓))
    end
    for k in Ukeys
        #Coulomb term
        H += U[k] * c[k..., :↑]'c[k..., :↑] * c[k..., :↓]'c[k..., :↓]
    end
    return blockdiagonal(H, c)
end

## Generate couplings
is_nearest_neighbours(k1, k2) = k1 != k2 && abs(first(k1) - first(k2)) ∈ (0, 1)
function random_nearest_hoppings(spatial_labels, s=1)
    couplings = [(k1, k2) => s * rand() * is_nearest_neighbours(k1, k2) for k1 in spatial_labels, k2 in spatial_labels]
    Dict(couplings)
end
function random_charging_energies(spatial_labels, s=1)
    U = [k => s * rand() for k in spatial_labels]
    Dict(U)
end
##

J = random_nearest_hoppings(spatial_labels)
U = random_charging_energies(spatial_labels)
HR = hamiltonian(c, J, U; Jkeys=Rconnections, Ukeys=Ilabels)
HI = hamiltonian(c, J, U; Jkeys=Iconnections, Ukeys=Rlabels)
HIR = hamiltonian(c, J, U; Jkeys=IRconnections, Ukeys=[])

##
Γ = rand(2, 2)
μL, μR = rand(2)
T = 0.1
leftlead = CombinedLead((c[N, 1, :↑]' * Γ[1, 1], c[N, 2, :↑]' * Γ[1, 2], c[N, 1, :↓]' * Γ[1, 1], c[N, 2, :↓]' * Γ[1, 2]); T, μ=μL)
rightlead = CombinedLead((c[N, 1, :↑]' * Γ[2, 1], c[N, 2, :↑]' * Γ[2, 2], c[N, 1, :↓]' * Γ[2, 1], c[N, 2, :↓]' * Γ[2, 2]); T, μ=μR)
input_dissipator = CombinedLead((c[0, 1, :↑]', c[0, 2, :↑]', c[0, 1, :↓]', c[0, 2, :↓]'); T, μ=-10)
leads0 = (; input=input_dissipator, left=leftlead, right=rightlead)
leads = (; left=leftlead, right=rightlead)

##
killer_op = c[0, 1, :↑] * c[0, 2, :↑] * c[0, 1, :↓] * c[0, 2, :↓];
initial_state_ops = [c[0, 1, :↑]', c[0, 2, :↓]', c[0, 1, :↑]' + c[0, 2, :↓]'];
normalize_rho!(rho) = rdiv!(rho, tr(rho))
function initial_state(op, rho)
    rho = killer_op * rho * killer_op'
    rho = op * rho * op'
    normalize_rho!(rho)
    rho
end

##
function squeeze_operator(α, c=c; kwargs...)
    A = α * c[0, 1, :↑]' * c[0, 2, :↑]' - hc
    # b -> expv(one(eltype(A)), A, b; kwargs...)
    e = exp(A)
    b -> e * b
end
function generate_training_data(M, rho)
    phases = 2pi * rand(M)
    r = rand(M)
    αs = r .* exp.(1im * phases)
    rho0 = killer_op * rho * killer_op'
    # rho0intern = internal_rep(rho0intern, sys)
    ops = squeeze_operators.(α)
    rhonew = [op * rho0 * op' for op in ops]
    rhonew = rhonew ./ tr.(rhonew)
    purity = tr.(rhonew .^ 2)
    occupations = [tr(rhonew * c[k..., :↑]' * c[k..., :↑]) for k in spatial_labels]
    return (; αs, rhonew, purity, occupations)
end

##
particle_number = blockdiagonal(numberoperator(c), c)
H0 = HR + HI
ls0 = LindbladSystem(H0, leads0)
lazyls0 = LazyLindbladSystem(H0, leads0)

##
prob0 = StationaryStateProblem(ls0)
lazyprob0 = StationaryStateProblem(lazyls0)
@time rhointernal0 = solve(prob0, LinearSolve.KrylovJL_LSMR(); abstol=1e-6)
@time lazyrhointernal0 = solve(lazyprob0, LinearSolve.KrylovJL_LSMR(); abstol=1e-6)
rho0 = QuantumDots.tomatrix(rhointernal0, ls0)
lazyrho0 = reshape(lazyρinternal0, size(H0))
@assert norm(rho0 - lazyrho0) < 1e-6
# TODO: Fix types in Lazylindblad so that mul! does not hit generic_matmul
# TODO: Fix performance in khatri_rao_dissipator!

## Analyze initial state
rhod0 = diag(rho0)
@assert tr(rho0) ≈ 1
internal_N = QuantumDots.internal_rep(particle_number, ls0)
currents_ops = map(diss -> diss' * internal_N, ls0.dissipators)
currents = map(diss -> tr(QuantumDots.tomatrix(diss * ρinternal0, ls0) * particle_number), ls0.dissipators)
currents2 = map(op -> ρinternal0' * op, currents_ops)
@assert norm(map(-, currents, currents2)) / norm(currents) < 1e-6

cR = FermionBasis(fullRlabels; qn)
cI = FermionBasis(fullIlabels; qn)
rhoI0 = partial_trace(rho0, fullIlabels, c, cI.symmetry)
rhoR0 = partial_trace(rho0, fullRlabels, c, cR.symmetry)
@assert rhoI0[1] ≈ 1
@assert tr(rhoI0) ≈ 1
pretty_print(rhoI0, cI)
pretty_print(rhoR0, cR)

##
H = H0 + HIR
ls = QuantumDots.LindbladSystem(H, leads)

##
function time_evolve(op, rho0, ls)
    rho = initial_state(op, rho0)
    # rho = rho0
    drho!(out, rho, p, t) = mul!(out, ls, rho)
    prob = ODEProblem(drho!, rho, (0, 5))
    # drho(rho, p, t) = ls * rho
    # prob = ODEProblem(drho, rho, (0, 5))
    sol = solve(prob, Tsit5())
    current_ops = map(diss -> diss' * diagonalsystem.transformed_measurements[1], ls.dissipators)
    currents = reduce(hcat, [[real(tr(sol(t) * op)) for op in current_ops] for t in sol.t]) |> transpose
    trs = [tr(sol(t)) |> real for t in sol.t]
    tr2s = [tr(sol(t)^2) |> real for t in sol.t]
    return (; sol, currents, trs, tr2s)
end

sols = map(op -> time_evolve(op, deepcopy(rho0), ls), initial_state_ops);

##
p = plot()
map(sol -> plot!(p, sol.sol.t, sol.trs, ylims=(0, 1.1)), sols)
map(sol -> plot!(p, sol.sol.t, sol.tr2s, ylims=(0, 1.1)), sols)
p
##
p = plot()
map((sol, ls) -> plot!(p, sol.sol.t, sol.currents; ls, lw=2), sols, [:solid, :dash, :dashdot])
p
