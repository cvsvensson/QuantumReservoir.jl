using QuantumDots
using QuantumDots.BlockDiagonals
using LinearAlgebra
using Plots

N = 1
c = FermionBasis(0:N, 1:2, (:↑, :↓); qn=QuantumDots.fermionnumber)
spatial_labels = Base.product(0:N, 1:2) |> collect
connection_labels = filter((ks) -> ks[1] != ks[2], Base.product(spatial_labels, spatial_labels) |> collect)
Ilabels = filter(iszero ∘ first, spatial_labels)
Rlabels = filter(k -> first(k) > 0, spatial_labels)
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
leftlead = QuantumDots.CombinedLead((c[N, 1, :↑]' * Γ[1, 1], c[N, 2, :↑]' * Γ[1, 2], c[N, 1, :↓]' * Γ[1, 1], c[N, 2, :↓]' * Γ[1, 2]); T, μ=μL)
rightlead = QuantumDots.CombinedLead((c[N, 1, :↑]' * Γ[2, 1], c[N, 2, :↑]' * Γ[2, 2], c[N, 1, :↓]' * Γ[2, 1], c[N, 2, :↓]' * Γ[2, 2]); T, μ=μR)
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
function generate_training_data(M)
    m = rand(ComplexF64,2^4,2^4)
    θ = rand()
    [c[0, 1, :↑]', c[0, 2, :↓]', c[0, 1, :↑]' + c[0, 2, :↓]']
    rho = m'*m
    rho = rho / tr(rho)
    purity = tr(rho)^2
    occupations = [tr(rho * c[k..., :↑]' * c[k..., :↑]) for k in spatial_labels]
end

##
particle_number = blockdiagonal(numberoperator(c), c)
H0 = HR + HI
ls0 = LazyLindbladSystem(H0, leads)

##
prob0 = StationaryStateProblem(ls0)
ρinternal0 = solve(prob0, LinearSolve.KrylovJL_LSMR(); abstol=1e-12)

# TODO: Fix types in Lazylindblad so that mul! does not hit generic_matmul
# TODO: Fix performance in khatri_rao_dissipator!
# TODO: add one(::BlockDiagonal) 
rho0 = reshape(ρinternal0, size(H0)...)
rhod0 = diag(rho0)
tr(rho0) ≈ 1
currents0 = map(diss -> tr(diss' * diagonalsystem.transformed_measurements[1]), ls0.dissipators)

##
H = H0 + HIR
system = QuantumDots.OpenSystem(H, leads, measurements);
diagonalsystem = QuantumDots.diagonalize(system)
ls = QuantumDots.LazyLindbladSystem(diagonalsystem)

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
