struct ContinuousInput{I}
    input::I
end
struct DiscreteInput{I,T}
    input::I
    dt::T
end
struct MaskedInput{M,S}
    mask::M
    signal::S
end
(m::MaskedInput)(t) = Dict(l => m.signal(t) * v for (l, v) in pairs(m.mask))
Base.getindex(m::MaskedInput, i) = Dict(l => getindex(m.signal, i) * v for (l, v) in pairs(m.mask))
function voltage_input(signal)
    Dict(l => (; μ=v) for (l, v) in pairs(signal))
end
struct VoltageWrapper{S}
    signal::S
end
(v::VoltageWrapper)(t) = voltage_input(v.signal(t))
Base.getindex(v::VoltageWrapper, l) = voltage_input(getindex(v.signal, l))
(c::ContinuousInput)(t) = c.input(t)
(d::DiscreteInput)(t) = d.input[1+Int(div(t + 100eps(), d.dt))]

function stationary_state(ls::LazyLindbladSystem; kwargs...)
    ss_prob = StationaryStateProblem(ls)
    reshape(solve(ss_prob; abstol=get(kwargs, :abstol, 1e-12)), size(ls.hamiltonian))
end
function stationary_state(ls; kwargs...)
    ss_prob = StationaryStateProblem(ls)
    solve(ss_prob; abstol=get(kwargs, :abstol, 1e-12))
end

struct PiecewiseTimeSteppingMethod{A}
    alg::A
end
struct EXP_KRYLOV end
struct EXP_SCIML end
struct ODE{A}
    alg::A
end
default_lindblad(H, leads, ::ContinuousInput) = LazyLindbladSystem(H, leads)
default_lindblad(H, leads, ::DiscreteInput) = LindbladSystem(H, leads, usecache=true)
default_alg(::LindbladSystem, ::ContinuousInput) = ODE(Tsit5())
default_alg(::LazyLindbladSystem, ::ContinuousInput) = ODE(Tsit5())
default_alg(::LindbladSystem, ::DiscreteInput) = PiecewiseTimeSteppingMethod(EXP_SCIML())
default_alg(::LazyLindbladSystem, ::DiscreteInput) = PiecewiseTimeSteppingMethod(EXP_KRYLOV())
default_alg(::PauliSystem, ::DiscreteInput) = PiecewiseTimeSteppingMethod(EXP_SCIML())

struct StationaryState end
struct Reservoir{C,H,L,LS,I,IS,IIS,M}
    c::C
    H::H
    leads::L
    input::I
    ls0::LS
    ls::LS
    initial_state::IS
    internal_initial_state::IIS
    measure::M
end
struct CurrentMeasurements{O}
    op::O
end
(measure::CurrentMeasurements)(rho, ls::Union{LindbladSystem,LazyLindbladSystem}) = real(QuantumDots.measure(rho, measure.op, ls))
(measure::CurrentMeasurements)(rho, p::PauliSystem) = QuantumDots.get_currents(rho, p)

# function get_currents(rho, ls::, op)
#     real(QuantumDots.measure(rho, op, ls))
# end
# function get_currents(sol, ls, input::ContinuousInput, t, op)
#     rho = sol(t)
#     QuantumDots.update_coefficients!(ls, input(t))
#     get_currents(rho, ls, op)
# end
# function get_currents(sol, ls, input::DiscreteInput, t, op)
#     n = 1 + Int(div(t + 100eps(), input.dt))
#     rho = sol[n]
#     QuantumDots.update_coefficients!(ls, input(t))
#     get_currents(rho, ls, op)
# end
function reservoir(c::FermionBasis, H, leads, input, initial_state=StationaryState(), measure=CurrentMeasurements(numberoperator(c)))
    ls = default_lindblad(H, leads, input)
    Reservoir(c, H, leads, input, ls, deepcopy(ls), initial_state, nothing, measure)
end
function SciMLBase.init(res::Reservoir{<:Any,<:Any,<:Any,<:Any,<:Any,StationaryState}, t0)
    QuantumDots.update_coefficients!(res.ls, res.input(t0))
    internal_initial_state = collect(stationary_state(res.ls))
    Reservoir(res.c, res.H, res.leads, res.input, res.ls, res.ls, res.initial_state, internal_initial_state, res.measure)
end
function SciMLBase.init(res::Reservoir, t0)
    QuantumDots.update_coefficients!(res.ls, res.input(t0))
    internal_initial_state = QuantumDots.internal_rep(res.ls, res.initial_state)
    Reservoir(res.c, res.H, res.leads, res.input, res.ls, res.ls, res.initial_state, internal_initial_state, res.measure)
end
function run_reservoir(res::Reservoir, tspan, alg=default_alg(res.ls, res.input), measure=CurrentMeasurements(numberoperator(res.c)); kwargs...)
    resinit = init(res, tspan[1])
    run_reservoir!(resinit.ls, resinit.input, tspan, resinit.internal_initial_state, alg, measure; kwargs...)
end

function run_reservoir!(ls, input::DiscreteInput, tspan, initial_state, alg::PiecewiseTimeSteppingMethod, measure; time_multiplexing=1, kwargs...)
    # ls = copy ? deepcopy(_ls) : _ls
    QuantumDots.update_coefficients!(ls, input(tspan[1]))
    dt = input.dt
    ts = range(tspan..., step=dt)
    rho0 = initial_state
    measurements = []
    for t in ts[1:end-1]
        QuantumDots.update_coefficients!(ls, input(t))
        rhos = [rho0]
        for n in 1:time_multiplexing
            rho = expstep(alg.alg, ls, dt / time_multiplexing, rhos[end])
            push!(rhos, rho)
        end
        rho0 = rhos[end]
        push!(measurements, reduce(vcat, map(rho -> measure(rho, ls), Iterators.take(rhos, time_multiplexing))))
    end
    return measurements
end
expstep(method::EXP_KRYLOV, ls, dt, rho; kwargs...) = exponentiate(ls, dt, rho; kwargs...)[1]
expstep(method::EXP_SCIML, ls::LindbladSystem, dt, rho; kwargs...) = expv(dt, ls.total, rho; kwargs...)
expstep(method::EXP_SCIML, p::PauliSystem, dt, rho; kwargs...) = expv(dt, p.total_master_matrix, rho; kwargs...)
using DiffEqCallbacks
function run_reservoir!(ls, input::DiscreteInput, tspan, initial_state, alg::ODE, measure; time_multiplexing=1, kwargs...)
    p = (ls, input)
    ts = range(tspan..., step=input.dt)[1:end-1]
    small_dt = input.dt / time_multiplexing
    # t_measure = reduce(vcat, [[t + n * small_dt for n in 0:time_multiplexing-1] for t in ts])
    t_measure = range(tspan..., step=input.dt / time_multiplexing)[1:end-1]
    tspan = (tspan[1], t_measure[end])
    affect!(integrator) = QuantumDots.update_coefficients!(integrator.p[1], input(integrator.t))
    cb = PresetTimeCallback(ts, affect!)
    # savingls = deepcopy(ls)
    # save_func(u, t, integrator) = (QuantumDots.update_coefficients!(savingls, input(t)); measure(u, savingls))
    save_func(u, t, integrator) = measure(u, integrator.p[1])
    MT = typeof(collect(measure(initial_state, ls)))
    saved_values = SavedValues(Float64, MT)
    savingcb = SavingCallback(save_func, saved_values; saveat=t_measure)
    callback = CallbackSet(cb, savingcb)
    solve(ODEProblem(f_ode!, initial_state, tspan, p; callback, kwargs...), alg.alg; kwargs...)
    return map(data -> reduce(vcat, data), Iterators.partition(saved_values.saveval, time_multiplexing))
end
function run_reservoir!(ls, input::ContinuousInput, tspan, initial_state, measure, alg::ODE; kwargs...)
    p = (ls, input)
    solve(ODEProblem(f_ode!, initial_state, tspan, p; kwargs...))
end
function f_ode!(du, u, (ls, input), t)
    QuantumDots.update_coefficients!(ls, input(t))
    mul!(du, ls, u)
end
##

function get_spectrum(ls, input, t)
    QuantumDots.update_coefficients!(ls, input(t))
    eigvals(Matrix(ls))
end
function reservoir_properties(res, measurements, tspan)
    ls = res.ls
    input = res.input
    QuantumDots.update_coefficients!(ls, input(tspan[1]))
    dt = input.dt
    ts = range(tspan..., step=dt)
    evals = eigvals(Matrix(res.H))
    gapratios = map(x -> x > 1 ? inv(x) : x, diff(evals)[1:end-1] ./ diff(evals)[2:end])
    average_gapratio = mean(gapratios)
    ediffs = QuantumDots.commutator(Diagonal(evals)).diag
    spectrum = []
    for t in ts[1:end-1]
        QuantumDots.update_coefficients!(ls, input(t))
        push!(spectrum, eigvals(Matrix(ls)))
    end
    # use pca on meauserments from MultivariateStats.jl
    pca = MultivariateStats.fit(PCA, stack(measurements))
    smallest_decay_rate = mean(spec -> abs(partialsort(spec, 2, by=abs)), spectrum)

    return (; evals, spectrum, gapratios, average_gapratio, ediffs, smallest_decay_rate, pca)
end
