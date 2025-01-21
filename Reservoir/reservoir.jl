using QuantumDots, QuantumDots.BlockDiagonals
using LinearAlgebra
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
temperature = 2norm(Γ)
leads = Dict(l => NormalLead(c[l]' * Γ[l]; T=temperature, μ=μs0[l]) for l in labels)
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
struct MaskedInput{M,S}
    mask::M
    signal::S
end
(m::MaskedInput)(t) = Dict(l => m.signal(t) * v for (l, v) in pairs(m.mask))
function voltage_input(signal)
    Dict(l => (; μ=v) for (l, v) in pairs(signal))
    # Dict(map((l, v) -> l => (; μ=v), labels, input))
end
struct VoltageWrapper2{S}
    signal::S
end
VoltageWrapper = VoltageWrapper2
(v::VoltageWrapper)(t) = voltage_input(v.signal(t))
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
mask = Dict(l => l for l in keys(leads))
input = ContinuousInput(VoltageWrapper(MaskedInput(mask, sin)));
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
    complex(ls.vectorizer.idvec)
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
function get_spectrum(ls, input, t)
    QuantumDots.update_coefficients!(ls, input(t))
    eigvals(Matrix(ls))
end
function get_currents(rho, ls, op=number_operator)
    real(QuantumDots.measure(rho, op, ls))
end
##
res = _Res(ls_lazy, input, (0, T));
res_dense = _Res(ls, input, (0, T));
sol = solve!(res);
solve!(res_dense);
using BenchmarkTools
@btime solve!($res);
@btime solve!($res_dense);
solve!(res)
cu = get_currents(res, 0.0)
cu2 = get_currents(res, 0.0)
@btime get_currents($res, $(0.0));
@btime get_currents($res_dense, $(0.0));
@profview foreach(t -> get_currents(res, t), range(0, T, 10000));
@profview_allocs foreach(t -> get_currents(res, t), range(0, T, 10000));
@time [get_currents(res, t) for t in range(0, T, 1000)];
##
@profview_allocs foreach(j -> solve!(res), 1:10);

##
J = Dict((k1, k2) => 2(rand() - 0.5) for (k1, k2) in hopping_labels)
V = Dict((k1, k2) => rand() for (k1, k2) in hopping_labels)
ε = Dict(l => rand() - 0.5 for l in labels)
Γ = rand(N)
Ht = hopping_hamiltonian(c, J; labels=hopping_labels)
HV = coulomb_hamiltonian(c, V; labels=hopping_labels)
Hqd = qd_level_hamiltonian(c, ε)
H = Ht + HV + Hqd
temperature = 10norm(Γ)
leads = Dict(l => NormalLead(c[l]' * Γ[l]; T=temperature, μ=μs0[l]) for l in labels)
result = let H = H, leads = leads, signal = sin, T = 50, t_measures = range(0, T, 100)
    tspan = (0, T)
    ls = LindbladSystem(H, leads; usecache=true)
    lazyls = LazyLindbladSystem(H, leads)
    mask = Dict(l => l^2 * 10 * rand() for l in keys(leads))
    input = ContinuousInput(VoltageWrapper(MaskedInput(mask, sin)))
    ode_kwargs = (; abstol=1e-6, reltol=1e-6)
    # sol = solve(odeproblem(ls, input, (0, T)), Tsit5(); ode_kwargs...)
    sol_lazy = solve(odeproblem(ls_lazy, input, tspan), Tsit5(); ode_kwargs...)
    t_measures = range(tspan..., 100)

    currents = [get_currents(sol_lazy, ls_lazy, input, t, number_operator) for t in t_measures]
    spectrum = [get_spectrum(ls, input, t) for t in t_measures]
    (; spectrum, currents, sol=sol_lazy, ts=t_measures, T, input, H)
end;
## Training a linear layer to predict the target given the current
using MLJLinearModels
target = sin
n_train = 10:80
n_test = 81:length(result.ts)
Xtrain = stack(result.currents[n_train])
ytrain = transpose(target.(result.ts[n_train]))
Xtest = stack(result.currents[n_test])
ytest = transpose(target.(result.ts[n_test]))
ridge = RidgeRegression(1e-6; fit_intercept=true)
W = reduce(hcat, map(data -> fit(ridge, Xtrain', data), eachrow(ytrain))) |> permutedims
ztrain = W[:, 1:end-1] * Xtrain .+ W[:, end]
ztest = W[:, 1:end-1] * Xtest .+ W[:, end]
##
xlims = maximum(abs ∘ real, first(result.spectrum)) .* (-1.01, 0.01)
ylims = maximum(abs ∘ imag, first(result.spectrum)) .* (-1.01, 1.01)
ediffs = QuantumDots.commutator(Diagonal(eigvals(Matrix(H)))).diag
leadlabels = transpose(collect(keys(result.input(0))))
signal = stack([collect(values(result.input.input.signal(t))) for t in result.ts])'
pcurrent = plot(result.ts, stack(result.currents)', label=leadlabels, legendtitle="Lead", xlabel="t", ylabel="current")
ptarget = plot(result.ts, target.(result.ts), label="target", xlabel="t", ylabel="target")
plot!(ptarget, result.ts[n_train], ztrain', label="train")
plot!(ptarget, result.ts[n_test], ztest', label="test")
anim = @animate for (s, t) in zip(result.spectrum, result.ts)
    p1 = scatter(real(s), imag(s); xlims, ylims, size=(800, 800))
    input = result.input
    boltz = stack([QuantumDots.fermidirac.(ediffs, leads[l].T, input(t)[l].μ) |> sort for l in keys(leads)])
    psignal = plot(result.ts, signal, labels=leadlabels, xlabel="t", ylabel="voltage", legendtitle="Lead")
    vline!(psignal, [t], color=:red, label="t")
    pboltz = plot(boltz, marker=true, ylims=(0, 1), labels=leadlabels)
    plot(p1, psignal, pboltz, pcurrent, ptarget, layout=(3, 2))
end
gif(anim, "anim_fps15.gif", fps=15)
