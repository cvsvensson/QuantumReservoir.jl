using FermionicHilbertSpaces
using LinearAlgebra
using Plots
using ExponentialUtilities
using OrdinaryDiffEqTsit5
includet("opensystems.jl")

N = 1
labels = Base.product(0:N, 1:2, (:↑, :↓)) |> collect
spatial_labels = Base.product(0:N, 1:2) |> collect
@fermions f
space = hilbert_space(f, vec(labels), NumberConservation())
bases = (; f, space)
upnumber = representation(sum(f[l..., :↑]' * f[l..., :↑] for l in spatial_labels), space)
downnumber = representation(sum(f[l..., :↓]' * f[l..., :↓] for l in spatial_labels), space)
qn = collect(zip(round.(Int, real(diag(Matrix(upnumber)))), round.(Int, real(diag(Matrix(downnumber))))))
connection_labels = filter((ks) -> ks[1] != ks[2], Base.product(spatial_labels, spatial_labels) |> collect)
Ilabels = filter(iszero ∘ first, spatial_labels)
Rlabels = filter(k -> first(k) > 0, spatial_labels)
fullIlabels = filter(iszero ∘ first, labels)
fullRlabels = filter(k -> first(k) > 0, labels)
IRconnections = filter(k -> first(k[1]) + first(k[2]) != 1, connection_labels)
Rconnections = filter(k -> first(k[1]) + first(k[2]) > 1, connection_labels)
Iconnections = filter(k -> first(k[1]) + first(k[2]) == 0, connection_labels)
##
function hamiltonian(bases, J, U; Jkeys=keys(J), Ukeys=keys(U))
    (; f, space) = bases
    H = 0
    for (k1, k2) in Jkeys
        #Hoppings
        foreach(σ -> H += (J[(k1, k2)] * f[k1..., σ]'f[k2..., σ] + hc), (:↑, :↓))
    end
    for k in Ukeys
        #Coulomb term
        H += U[k] * f[k..., :↑]'f[k..., :↑] * f[k..., :↓]'f[k..., :↓]
    end
    return representation(H, space)
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
HR = hamiltonian(bases, J, U; Jkeys=Rconnections, Ukeys=Ilabels)
HI = hamiltonian(bases, J, U; Jkeys=Iconnections, Ukeys=Rlabels)
HIR = hamiltonian(bases, J, U; Jkeys=IRconnections, Ukeys=[])

##
Γ = rand(2, 2)
μL, μR = rand(2)
T = 0.1
leftlead = CombinedLead((representation(f[N, 1, :↑]' * Γ[1, 1], space), representation(f[N, 2, :↑]' * Γ[1, 2], space), representation(f[N, 1, :↓]' * Γ[1, 1], space), representation(f[N, 2, :↓]' * Γ[1, 2], space)); T, μ=μL)
rightlead = CombinedLead((representation(f[N, 1, :↑]' * Γ[2, 1], space), representation(f[N, 2, :↑]' * Γ[2, 2], space), representation(f[N, 1, :↓]' * Γ[2, 1], space), representation(f[N, 2, :↓]' * Γ[2, 2], space)); T, μ=μR)
input_dissipator = CombinedLead((representation(f[0, 1, :↑]', space), representation(f[0, 2, :↑]', space), representation(f[0, 1, :↓]', space), representation(f[0, 2, :↓]', space)); T, μ=-10)
leads0 = (; input=input_dissipator, left=leftlead, right=rightlead)
leads = (; left=leftlead, right=rightlead)

##
killer_op = representation(f[0, 1, :↑] * f[0, 2, :↑] * f[0, 1, :↓] * f[0, 2, :↓], space)
initial_state_ops = [representation(f[0, 1, :↑]', space), representation(f[0, 2, :↓]', space), representation(f[0, 1, :↑]' + f[0, 2, :↓]', space)];
normalize_rho!(rho) = rdiv!(rho, tr(rho))
function initial_state(op, rho)
    rho = killer_op * rho * killer_op'
    rho = op * rho * op'
    normalize_rho!(rho)
    rho
end

##
function squeeze_operator(α, f=f, space=space)
    A = representation(α * f[0, 1, :↑]' * f[0, 2, :↑]' - hc, space)
    exp(Matrix(A))
end
function generate_training_data(M, rho)
    phases = 2pi * rand(M)
    r = rand(M)
    αs = r .* exp.(1im * phases)
    rho0 = killer_op * rho * killer_op'
    ops = squeeze_operator.(αs)
    rhonew = [op * rho0 * op' for op in ops]
    rhonew = rhonew ./ tr.(rhonew)
    purity = tr.(rhonew .^ 2)
    occupations = [tr(rhonew * representation(f[k..., :↑]' * f[k..., :↑], space)) for k in spatial_labels]
    return (; αs, rhonew, purity, occupations)
end

##
particle_number = representation(sum(f[l]' * f[l] for l in vec(labels)), space)
H0 = HR + HI
ls0 = LindbladSystem(H0, leads0; qn)

##
@time rhointernal0 = stationary_state(ls0)
rho0 = tomatrix(rhointernal0, ls0)

## Analyze initial state
rhod0 = diag(rho0)
@assert tr(rho0) ≈ 1
currents_ops = collect(values(current_operators(ls0, particle_number)))
currents = [tr(tomatrix(diss * rhointernal0, ls0) * particle_number) for diss in values(ls0.dissipators)]
currents2 = [rhointernal0' * op for op in currents_ops]
@assert norm(real.(currents) - real.(currents2)) < 1e-6

rhoI0 = partial_trace(rho0, space => hilbert_space(f, fullIlabels, NumberConservation()))
rhoR0 = partial_trace(rho0, space => hilbert_space(f, fullRlabels, NumberConservation()))
@assert rhoI0[1] ≈ 1
@assert tr(rhoI0) ≈ 1
rhoI0
rhoR0

##
H = H0 + HIR
ls = LindbladSystem(H, leads; qn)

##
function time_evolve(op, rho0, ls)
    rho = initial_state(op, rho0)
    drho!(out, v, p, t) = mul!(out, ls, v)
    prob = ODEProblem(drho!, vecrep(rho, ls), (0, 5))
    sol = solve(prob, Tsit5())
    current_ops = collect(values(current_operators(ls, particle_number)))
    currents = reduce(hcat, [[real(dot(op, sol(t))) for op in current_ops] for t in sol.t]) |> transpose
    trs = [real(tr(tomatrix(sol(t), ls))) for t in sol.t]
    tr2s = [real(tr(tomatrix(sol(t), ls)^2)) for t in sol.t]
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
