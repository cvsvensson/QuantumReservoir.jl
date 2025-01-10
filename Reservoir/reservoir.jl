using QuantumDots
using QuantumDots.BlockDiagonals
using Random
using OrdinaryDiffEqTsit5
using LinearSolve
Random.seed!(1234)
include("..\\system.jl")
##
N = 4
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
number_operator = sum([c[l]'c[l] for l in labels])
Ht = hopping_hamiltonian(c, J; labels=hopping_labels)
HV = coulomb_hamiltonian(c, V; labels=hopping_labels)
Hqd = qd_level_hamiltonian(c, ε)
##
H = Ht + HV + Hqd
# H = blockdiagonal(Matrix.(blocks(H)), c)
##
μmin = -1e5
μs0 = zeros(N)#[μmin, μmin]#rand(2)
T = 10norm(Γ)
leads = Dict(l => NormalLead(c[l]' * Γ[l]; T, μ=μs0[l]) for l in labels)
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
    # OrderedDict(map((l,v) zip(labels, input))
    Dict(map((l, v) -> l => (; μ=v), labels, input))
end
struct VoltageWrapper1{I,L}
    input::I
    labels::L
end
VoltageWrapper = VoltageWrapper1
(v::VoltageWrapper)(t) = voltage_input(v.input(t), v.labels)
(c::ContinuousInput)(t) = c.input(t)

function stationary_state(ls::LazyLindbladSystem; kwargs...)
    ss_prob = StationaryStateProblem(ls)
    reshape(solve(ss_prob; abstol=get(kwargs, :abstol, 1e-12)), size(ls.hamiltonian))
end
function stationary_state(ls::LindbladSystem; kwargs...)
    ss_prob = StationaryStateProblem(ls)
    solve(ss_prob; abstol=get(kwargs, :abstol, 1e-12))
end
function odeproblem(_ls, input::ContinuousInput, tspan, initial_state=collect(stationary_state(_ls)))
    ls = deepcopy(_ls)
    p = (ls, input)
    ODEProblem(f_ode!, initial_state, tspan, p)
end
function f_ode!(du, u, (ls, input), t)
    QuantumDots.update_coefficients!(ls, input(t))
    mul!(du, ls, u)
end
##
ls_kron = LindbladSystem(H, leads, QuantumDots.KronVectorizer(size(H, 1)); usecache=true);
ls = LindbladSystem(H, leads; usecache=true);
ls_lazy = LazyLindbladSystem(H, leads);
input = ContinuousInput(VoltageWrapper(t -> sin(t) .* (1:N), 1:N));
T = 100
ode_kwargs = (; abstol=1e-6, reltol=1e-6)
@time sol = solve(odeproblem(ls, input, (0, T)), Tsit5(); ode_kwargs...);
@time sol_lazy = solve(odeproblem(ls_lazy, input, (0, T)), Tsit5(); ode_kwargs...);
#@time sol_kron = solve(odeproblem(ls_kron, input, (0, 100)), Tsit5());#, abstol=1e-4, reltol=1e-4);
@profview solve(odeproblem(ls, input, (0, T)), Tsit5());
@profview solve(odeproblem(ls_lazy, input, (0, 10T)), Tsit5());

##measure(rho, op, ls::AbstractOpenSystem) 
mutable struct Res4{L,I,T}
    ls0::L
    ls::L
    input::I
    tspan::T
    initial_state
    sol
end
Res = Res4
_Res(ls, input, tspan) = Res(ls, deepcopy(ls), input, tspan, nothing, nothing)
_Res(ls, input, tspan, initial_state) = Res(ls, deepcopy(ls), input, tspan, initial_state, nothing)
function stationary_state!(res::Res; kwargs...)
    res.initial_state = collect(stationary_state(res.ls0; kwargs...))
end
default_initial_state(res::Res) = default_initial_state(res.ls)#stationary_state!(res)
function default_initial_state(ls::LazyLindbladSystem)
    Matrix{eltype(ls)}(I, size(ls.hamiltonian))
end
function default_initial_state(ls::LindbladSystem)
    ls.vectorizer.idvec
end
function SciMLBase.solve!(res::Res, initial_state=res.initial_state; kwargs...)
    if isnothing(res.initial_state)
        res.initial_state = default_initial_state(res)
    end
    res.sol = solve(odeproblem(res.ls, res.input, res.tspan, res.initial_state), Tsit5(); kwargs...)
end
get_currents(res::Res, t, op=number_operator) = get_currents(res.sol, res.ls, res.input, t, op)

function get_currents(sol, ls, input, t, op=number_operator)
    rho = sol(t)
    QuantumDots.update_coefficients!(ls, input(t))
    get_currents(rho, ls, op)
end
function get_currents(rho, ls, op=number_operator)
    real(QuantumDots.measure(rho, op, ls))
end
##
res = _Res(ls_lazy, input, (0, T));
sol = solve!(res; callback);
using BenchmarkTools
@btime solve!($res);
solve!(res)
cu = get_currents(res, 0.0)
cu2 = get_currents(res, 0.0)
@btime get_currents($res, $(0.0));
@profview foreach(t -> get_currents(res, t), range(0, T, 10000));
@profview_allocs foreach(t -> get_currents(res, t), range(0, T, 10000));
@time [get_currents(res, t) for t in range(0, T, 1000)];
##
@profview_allocs foreach(j -> solve!(res), 1:10);

##
