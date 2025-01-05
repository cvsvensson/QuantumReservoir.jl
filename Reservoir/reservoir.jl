using QuantumDots
using QuantumDots.BlockDiagonals
using Random
using OrdinaryDiffEqTsit5
using LinearSolve
Random.seed!(1234)
include("..\\system.jl")
##
N = 2
labels = 1:N
qn = FermionConservation()
c = FermionBasis(labels; qn)
hopping_labels = [(labels[k1], labels[k2]) for k1 in 1:length(labels), k2 in 1:length(labels) if k1 > k2] #&& is_nearest_neighbours(labels[k1], labels[k2])]
##
J = Dict((k1, k2) => 2(rand() - 0.5) for (k1, k2) in hopping_labels) # random_hoppings(hopping_labels)
# J = Dict((k1, k2) => rand() * exp(2pi * 1im * rand()) for (k1, k2) in hopping_labels) # random_hoppings(hopping_labels)
V = Dict((k1, k2) => rand() for (k1, k2) in hopping_labels) # random_hoppings(hopping_labels, 10)
ε = Dict(l => rand() - 0.5 for l in labels)
Γ = rand(N)
##
Ht = hopping_hamiltonian(c, J; labels=hopping_labels)
HV = coulomb_hamiltonian(c, V; labels=hopping_labels)
Hqd = qd_level_hamiltonian(c, ε)
##
H = Ht + HV + Hqd
# H = blockdiagonal(Matrix.(blocks(H)), c)
##
μmin = -1e5
μs0 = zeros(4)#[μmin, μmin]#rand(2)
T = 10norm(Γ)
leads = NamedTuple(Symbol(:l, l) => NormalLead(c[l]' * Γ[l]; T, μ=μs0[l]) for l in labels)
@time ls = LindbladSystem(H, leads; usecache=true);

##

struct ContinuousInput{I}
    input::I
end
# struct DiscreteInput{I,T}
#     input::I
#     dt::T
# end
# struct ReservoirTask1{I,T}
#     input::I
#     target::T
# end
# ReservoirTask = ReservoirTask1
# struct DelayTask{I,D}
#     input::I
#     delay::D
# end
##

function voltage_input(input, labels)
    NamedTuple(map((l, v) -> Symbol(:l, l) => (; μ=v), labels, input))
end
struct VoltageWrapper1{I,L}
    input::I
    labels::L
end
VoltageWrapper = VoltageWrapper1
(v::VoltageWrapper)(t) = voltage_input(v.input(t), v.labels)
(c::ContinuousInput)(t) = c.input(t)

function odeproblem(_ls::LindbladSystem, input::ContinuousInput, tspan)
    ls = deepcopy(_ls)
    ss_prob = StationaryStateProblem(ls)
    rho0 = solve(ss_prob; abstol=1e-12)
    p = (ls, input, ls.cache)
    ODEProblem(f_ode!, rho0, tspan, p)
end
function f_ode!(du, u, (ls, input, cache), t)
    QuantumDots.update_lindblad_system!(ls, input(t), cache)
    mul!(du, ls.total, u)
end
##
ls_kron = LindbladSystem(H, leads, QuantumDots.KronVectorizer(size(H,1)); usecache=true);
ls = LindbladSystem(H, leads; usecache=true);
input = ContinuousInput(VoltageWrapper(t -> sin(t) .* (1:N), 1:N));
@time sol = solve(odeproblem(ls, input, (0, 10_000)), Tsit5());#, abstol=1e-4, reltol=1e-4);
@time sol_kron = solve(odeproblem(ls_kron, input, (0, 10_000)), Tsit5());#, abstol=1e-4, reltol=1e-4);
@profview solve(odeproblem(ls, input, (0, 10000)), Tsit5());