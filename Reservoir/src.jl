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
(m::MaskedInput)(t) = ArrayDictionary(keys(m.mask), m.signal(t) * v for (l, v) in pairs(m.mask))
Base.getindex(m::MaskedInput, i) = ArrayDictionary(keys(m.mask), getindex(m.signal, i) * v for (l, v) in pairs(m.mask))
function voltage_input(signal)
    ArrayDictionary(keys(signal), (; μ=v) for (l, v) in pairs(signal))
    # (; l = (; μ=v) for (l, v) in pairs(signal))
end
struct VoltageWrapper{S}
    signal::S
end
(v::VoltageWrapper)(t) = voltage_input(v.signal(t))
Base.getindex(v::DiscreteInput, l) = getindex(v.input, l)
Base.getindex(v::VoltageWrapper, l) = voltage_input(getindex(v.signal, l))
(c::ContinuousInput)(t) = c.input(t)
(d::DiscreteInput)(t) = d.input[1+Int(div(t + d.dt / 1000, d.dt))]

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
struct Generators end
default_lindblad(H, leads, ::ContinuousInput) = LazyLindbladSystem(H, leads)
default_lindblad(H, leads, ::DiscreteInput) = LindbladSystem(H, leads, usecache=true)
default_alg(::LindbladSystem, ::ContinuousInput) = ODE(Tsit5())
default_alg(::LazyLindbladSystem, ::ContinuousInput) = ODE(Tsit5())
default_alg(::LindbladSystem, ::DiscreteInput) = PiecewiseTimeSteppingMethod(EXP_SCIML())
default_alg(::LazyLindbladSystem, ::DiscreteInput) = PiecewiseTimeSteppingMethod(EXP_KRYLOV())
default_alg(::PauliSystem, ::DiscreteInput) = PiecewiseTimeSteppingMethod(EXP_SCIML())

struct Pauli end
struct Lindblad end
struct LazyLindblad end

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
# (measure::CurrentMeasurements)(rho, ls::Union{LindbladSystem,LazyLindbladSystem}) = real(QuantumDots.measure(rho, measure.op, ls))
# (measure::CurrentMeasurements)(rho, ls::Union{LindbladSystem,LazyLindbladSystem}) = real(_measure(rho, measure.op, ls))
function specialize(measure::CurrentMeasurements, ls::Union{LindbladSystem,LazyLindbladSystem})
    op = QuantumDots.internal_rep(measure.op, ls)
    ops = QuantumDots.AxisKeys.KeyedArray([d' * op for (k, d) in pairs(ls.dissipators)], collect(keys(ls.dissipators)))
    cache = deepcopy(complex(ls.vectorizer.idvec))
    # _measure(rho, diss) = real(dot(op, diss * QuantumDots.internal_rep(rho, ls)))
    # _measure(rho, diss) = real(dot(op, mul!(cache, diss, QuantumDots.internal_rep(rho, ls))))
    _measure(rho, op) = real(dot(QuantumDots.internal_rep(rho, ls), op))
    function f(rho)
        # QuantumDots.AxisKeys.KeyedArray([_measure(rho, d) for (k, d) in pairs(ls.dissipators)], collect(keys(ls.dissipators)))
        QuantumDots.AxisKeys.KeyedArray([_measure(rho, d) for (k, d) in pairs(ops)], collect(keys(ls.dissipators)))
    end
end
function specialize(::CurrentMeasurements, p::PauliSystem)
    real ∘ Base.Fix2(QuantumDots.get_currents, p)
end

# _measure(rho, op, ls::QuantumDots.AbstractOpenSystem) = QuantumDots.AxisKeys.KeyedArray([_measure(rho, op, d, ls) for (k, d) in pairs(ls.dissipators)], collect(keys(ls.dissipators)))

# _measure(rho, op, dissipator::QuantumDots.AbstractDissipator, ls::QuantumDots.AbstractOpenSystem) = dot(op, dissipator * QuantumDots.internal_rep(rho, ls))

# (measure::CurrentMeasurements)(rho, p::PauliSystem) = QuantumDots.get_currents(rho, p)

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
# function reservoir(c::FermionBasis, H, leads, input, initial_state=StationaryState(), measure=CurrentMeasurements(numberoperator(c)))
#     ls = default_lindblad(H, leads, input)
#     Reservoir(c, H, leads, input, ls, deepcopy(ls), initial_state, nothing, measure)
# end
# function SciMLBase.init(res::Reservoir{<:Any,<:Any,<:Any,<:Any,<:Any,StationaryState}, t0)
#     QuantumDots.update_coefficients!(res.ls, res.input(t0))
#     internal_initial_state = collect(stationary_state(res.ls))
#     Reservoir(res.c, res.H, res.leads, res.input, res.ls, res.ls, res.initial_state, internal_initial_state, res.measure)
# end
function get_internal_initial_state(system, input)
    t0 = input.tspan[1]
    QuantumDots.update_coefficients!(system, input.voltage_input(t0))
    QuantumDots.internal_rep(collect(stationary_state(system)), system)
end
# function SciMLBase.init(res::Reservoir, t0)
#     QuantumDots.update_coefficients!(res.ls, res.input(t0))
#     internal_initial_state = QuantumDots.internal_rep(res.ls, res.initial_state)
#     Reservoir(res.c, res.H, res.leads, res.input, res.ls, res.ls, res.initial_state, internal_initial_state, res.measure)
# end
# function run_reservoir(res::Reservoir, tspan, alg=default_alg(res.ls, res.input), measure=CurrentMeasurements(numberoperator(res.c)); kwargs...)
#     resinit = init(res, tspan[1])
#     run_reservoir!(resinit.ls, resinit.input, tspan, resinit.internal_initial_state, alg, measure; kwargs...)
# end
get_ham(ls::LindbladSystem) = ls.matrixhamiltonian
get_ham(ls::LazyLindbladSystem) = ls.hamiltonian
get_ham(p::PauliSystem) = first(values(p.dissipators)).H
function run_reservoir!(ls, input::DiscreteInput, ts, initial_state, alg::PiecewiseTimeSteppingMethod, measure; time_multiplexing=1, save_spectrum=false, kwargs...)
    # ls = copy ? deepcopy(_ls) : _ls
    QuantumDots.update_coefficients!(ls, input(first(ts)))
    dt = input.dt
    rho0 = initial_state
    measurements = typeof(measure(rho0, ls))[]
    spectrum = save_spectrum ? [] : nothing

    evals = sort(get_ham(ls).values)
    gapratios = map(x -> x > 1 ? inv(x) : x, diff(evals)[1:end-1] ./ diff(evals)[2:end])
    average_gapratio = mean(gapratios)
    ediffs = QuantumDots.commutator(Diagonal(evals)).diag

    cache = get_cache(alg.alg, ls)
    for t in ts[1:end-1]
        QuantumDots.update_coefficients!(ls, input(t))
        rhos = [rho0]
        save_spectrum && push!(spectrum, eigvals(Matrix(ls)))
        for _ in 1:time_multiplexing
            rho = expstep(alg.alg, ls, dt / time_multiplexing, rhos[end], cache)
            push!(rhos, rho)
        end
        rho0 = rhos[end]
        # push!(measurements, reduce(vcat, map(rho -> measure(rho, ls), Iterators.take(rhos, time_multiplexing))))
        push!(measurements, reduce(vcat, map(rho -> measure(rho, ls), Iterators.drop(rhos, 1))))
    end

    smallest_decay_rate = save_spectrum ? mean(spec -> abs(partialsort(spec, 2, by=abs)), spectrum) : nothing

    return (; measurements, evals, spectrum, gapratios, average_gapratio, ediffs, smallest_decay_rate)
end
expstep(method::EXP_KRYLOV, ls, dt, rho, cache=nothing; kwargs...) = exponentiate(ls, dt, rho; kwargs...)[1]
expstep(method::EXP_KRYLOV, p::PauliSystem, dt, rho, cache=nothing; kwargs...) = exponentiate(p.total_master_matrix, dt, rho; kwargs...)[1]
expstep(method::EXP_SCIML, ls::LindbladSystem, dt, rho, cache=nothing; kwargs...) = expv(dt, ls.total, rho; cache, kwargs...)
expstep(method::EXP_SCIML, p::PauliSystem, dt, rho, cache=nothing; kwargs...) = expv(dt, p.total_master_matrix, rho; cache, kwargs...)
get_cache(::EXP_SCIML, ls) = ExpvCache{eltype(ls)}(30)
get_cache(::EXP_KRYLOV, ls) = nothing
# ksA = KrylovSubspace{complex(eltype(A))}(n, maxiter)
# arnoldi!(ksA, A, vrho0; tol=abstol, m)

using DiffEqCallbacks
function run_reservoir!(ls, input::DiscreteInput, ts, initial_state, alg::ODE, measure; time_multiplexing=1, kwargs...)
    p = (ls, input)
    dt = input.dt
    ts = input.ts
    # ts = range(tspan..., step=input.dt)[1:end-1]
    small_dt = dt / time_multiplexing
    # t_measure = reduce(vcat, [[t + n * small_dt for n in 0:time_multiplexing-1] for t in ts])
    t_measure = sort(vcat(ts, ts .+ small_dt))#range(tspan..., step=small_dt)[1:end-1]
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
function run_reservoir!(ls, input::ContinuousInput, ts, initial_state, measure, alg::ODE; kwargs...)
    p = (ls, input)
    solve(ODEProblem(f_ode!, initial_state, ts, p; kwargs...))
end
function f_ode!(du, u, (ls, input), t)
    QuantumDots.update_coefficients!(ls, input(t))
    mul!(du, ls, u)
end
##

# function get_spectrum(ls, input, t)
#     QuantumDots.update_coefficients!(ls, input(t))
#     eigvals(Matrix(ls))
# end
# function reservoir_properties(res, input)
#     tspan = input.tspan
#     ls = res.ls
#     input = res.input
#     QuantumDots.update_coefficients!(ls, input(tspan[1]))
#     dt = input.dt
#     ts = range(tspan..., step=dt)
#     evals = eigvals(Matrix(res.H))
#     gapratios = map(x -> x > 1 ? inv(x) : x, diff(evals)[1:end-1] ./ diff(evals)[2:end])
#     average_gapratio = mean(gapratios)
#     ediffs = QuantumDots.commutator(Diagonal(evals)).diag
#     spectrum = []
#     for t in ts[1:end-1]
#         QuantumDots.update_coefficients!(ls, input(t))
#         push!(spectrum, eigvals(Matrix(ls)))
#     end
#     smallest_decay_rate = mean(spec -> abs(partialsort(spec, 2, by=abs)), spectrum)

#     return (; evals, spectrum, gapratios, average_gapratio, ediffs, smallest_decay_rate)
# end

# function reservoir_properties(reservoir, lead, input, opensystem)
# H = hamiltonian(reservoir.params)
# leads = Dict(l => NormalLead(c[l]' * lead.Γ[l]; T=lead.temperature, μ=0.0) for l in lead.labels)
# # system = opensystem(H, leads)
# reservoir_properties(H, input)
# end
# function reservoir_properties(H)
#     # @unpack voltage_input, ts, tspan, dt = input
#     # H = hamiltonian(reservoir.params)

#     # QuantumDots.update_coefficients!(system, voltage_input(tspan[1]))
#     evals = eigvals(Matrix(H))
#     gapratios = map(x -> x > 1 ? inv(x) : x, diff(evals)[1:end-1] ./ diff(evals)[2:end])
#     average_gapratio = mean(gapratios)
#     ediffs = QuantumDots.commutator(Diagonal(evals)).diag
#     # spectrum = get_spectrum(system, input)
#     # smallest_decay_rate = mean(spec -> abs(partialsort(spec, 2, by=abs)), spectrum)
#     # return (; evals, spectrum, gapratios, average_gapratio, ediffs, smallest_decay_rate)
#     return (; evals, gapratios, average_gapratio, ediffs)
# end
# function get_spectrum(reservoir, lead, input, opensystem)
#     H = hamiltonian(reservoir.params)
#     leads = Dict(l => NormalLead(c[l]' * lead.Γ[l]; T=lead.temperature, μ=0.0) for l in lead.labels)
#     system = opensystem(H, leads)
#     get_spectrum(system, input)
# end
# function get_spectrum(system, input)
#     @unpack voltage_input, ts = input
#     T = Vector{complex(eltype(eigvals(Matrix(system))))}
#     spectrum = T[]
#     for t in ts
#         QuantumDots.update_coefficients!(system, voltage_input(t))
#         push!(spectrum, eigvals(Matrix(system)))
#     end
#     return spectrum
# end

function fullanalysis(reservoir, lead, input, opensystem, measurement, target, training; kwargs...)
    c = reservoir.c
    H = hamiltonian(c, reservoir.params)
    leads = lead_dict(lead, c)#ArrayDictionary([l => NormalLead(c[l]' * lead.Γ[l]; T=lead.temperature, μ=0.0) for l in lead.labels])
    system = opensystem(H, leads)
    fullanalysis(system, H, input, measurement, target, training; kwargs...)
end
function fullanalysis(system::QuantumDots.AbstractOpenSystem, H, input, measurement, targets, training; alg=default_alg(system, input.voltage_input), kwargs...)
    @unpack voltage_input, ts, tspan, dt = input
    @unpack measure, time_multiplexing = measurement
    rho0 = get_internal_initial_state(system, input)
    simulation = run_reservoir!(system, voltage_input, ts, rho0, alg, measure; save_spectrum=true, time_multiplexing, kwargs...)
    @unpack warmup, train = training
    task_props = fit(simulation.measurements, targets; warmup=0.2, train=0.5)
    return (; simulation, fit=task_props)
end

(::Pauli)(H, leads) = PauliSystem(H, leads)
(::Lindblad)(H, leads) = LindbladSystem(H, leads, usecache=true)
(::LazyLindblad)(H, leads) = LazyLindbladSystem(H, leads)
function run_reservoir(reservoir, lead, input, measurement, opensystem, c, alg=PiecewiseTimeSteppingMethod(EXP_SCIML()); kwargs...)
    H = hamiltonian(c, reservoir.params)
    leads = lead_dict(lead, c)#ArrayDictionary([l => NormalLead(c[l]' * lead.Γ[l]; T=lead.temperature, μ=0.0) for l in lead.labels])
    system = opensystem(H, leads)
    @unpack ts, voltage_input = input
    @unpack measure, time_multiplexing = measurement
    rho0 = get_internal_initial_state(system, input)
    run_reservoir!(system, voltage_input, ts, rho0, alg, measure; time_multiplexing, kwargs...)
end

function get_input_values(input::Union{DiscreteInput,VoltageWrapper,MaskedInput})
    unique_input_indices = get_unique_input_indices(input)
    [input[ind] for ind in unique_input_indices]
end
get_unique_input_indices(input::DiscreteInput) = get_unique_input_indices(input.input)
get_unique_input_indices(input::VoltageWrapper) = get_unique_input_indices(input.signal)
get_unique_input_indices(input::MaskedInput) = get_unique_input_indices(input.signal)
get_unique_input_indices(input::Vector) = [findfirst(x -> v == x, input) for v in unique(input)]

function generate_propagators(reservoir, lead, input, opensystem, c; kwargs...)
    H = hamiltonian(c, reservoir.params)
    leads = lead_dict(lead, c)#ArrayDictionary([l => NormalLead(c[l]' * lead.Γ[l]; T=lead.temperature, μ=0.0) for l in lead.labels])
    system = opensystem(H, leads)
    generate_propagators(system, get_input_values(input.voltage_input), input.dt)
end
function generate_propagators(system, input_values, dt)
    method = ExpMethodHigham2005()
    cache = ExponentialUtilities.alloc_mem(Matrix(system), method)
    systems = typeof(system)[]
    propagators = map(input_values) do input
        QuantumDots.update_coefficients!(system, input)
        push!(systems, deepcopy(system))
        exponential!(dt * Matrix(system), method, cache)
    end
    return propagators, systems
end
function generate_master_matrices(reservoir, lead, input, opensystem, c; kwargs...)
    H = hamiltonian(c, reservoir.params)
    leads = lead_dict(lead, c)#ArrayDictionary([l => NormalLead(c[l]' * lead.Γ[l]; T=lead.temperature, μ=0.0) for l in lead.labels])
    system = opensystem(H, leads)
    generate_master_matrices(system, get_input_values(input.voltage_input))
end
function generate_master_matrices(system, input_values)
    # method = ExpMethodHigham2005()
    # cache = ExponentialUtilities.alloc_mem(Matrix(system), method)
    map(input_values) do input
        QuantumDots.update_coefficients!(system, input)
        Matrix(Matrix(system))
    end
end

using Combinatorics
function generate_algebra(propagators::AbstractVector, depth::Int)
    elements = [propagators]
    for _ in 1:depth-1
        push!(elements, increase_depth(elements))
    end
    return elements#reduce(vcat, elements)
end

function increase_depth(elements::Vector{<:AbstractVector{E}}) where {E}
    new_depth = length(elements) + 1
    combinations = partitions(new_depth, 2)
    new_elements = E[]
    for (n1, n2) in combinations
        for (k1, e1) in enumerate(elements[n1])
            for (k2, e2) in enumerate(elements[n2])
                if n1 == n2 && k1 >= k2
                    continue
                end
                push!(new_elements, e1 * e2 - e2 * e1)
            end
        end
    end
    return new_elements
end

function algebra_svdvals(elements)
    svdvals(mapreduce(vec, hcat, reduce(vcat, elements)))
end

function moving_average(A::AbstractArray, m::Int)
    out = similar(A)
    R = CartesianIndices(A)
    Ifirst, Ilast = first(R), last(R)
    I1 = m ÷ 2 * oneunit(Ifirst)
    for I in R
        n, s = 0, zero(eltype(out))
        for J in max(Ifirst, I - I1):min(Ilast, I + I1)
            s += A[J]
            n += 1
        end
        out[I] = s / n
    end
    return out
end

transition_matrix(propagators, signal, input_values) = prod(propagators[findfirst(v -> v == s, input_values)] for s in signal)


struct PropagatorMethod end
function lead_dict(lead, c)
    # ArrayDictionary(lead.labels, NormalLead(c[l]' * lead.Γ[l]; T=lead.temperature, μ=0.0) for l in lead.labels)
    Dict(zip(lead.labels, NormalLead(c[l]' * lead.Γ[l]; T=lead.temperature, μ=0.0) for l in lead.labels))
end

function run_reservoir(reservoir, lead, input, measurement, opensystem, c, alg::PropagatorMethod; kwargs...)
    H = hamiltonian(c, reservoir.params)
    leads = lead_dict(lead, c)
    system = opensystem(H, leads)
    @unpack ts, voltage_input, dt = input
    @unpack measure, time_multiplexing = measurement
    # propagators = generate_propagators(system, get_input_values(voltage_input), dt)
    propagators, systems = generate_propagators(system, get_input_values(voltage_input), dt / time_multiplexing)
    rho0 = get_internal_initial_state(system, input)

    evals = sort(get_ham(system).values)
    gapratios = map(x -> x > 1 ? inv(x) : x, diff(evals)[1:end-1] ./ diff(evals)[2:end])
    average_gapratio = mean(gapratios)
    ediffs = QuantumDots.commutator(Diagonal(evals)).diag
    input_values = get_input_values(voltage_input)
    # cache = get_cache(alg.alg, ls)

    specialized_measure = specialize(measure, system)

    measurement_type = typeof(reduce(vcat, map(specialized_measure, [rho0])))
    # measurement_type = typeof(reduce(vcat, map(rho -> measure(rho, system), [rho0])))
    measurements = measurement_type[]
    spectrum = []
    rhos = [deepcopy(rho0) for k in 1:time_multiplexing+1]
    for t in ts[1:end-1]
        voltage = voltage_input(t)
        ind = findfirst(v -> v == voltage, input_values)
        # QuantumDots.update_coefficients!(system, voltage_input(t))
        system = systems[ind]
        propagator = propagators[ind]
        # rhos = [rho0]
        save_spectrum && push!(spectrum, eigvals(Matrix(system)))
        for k in 1:time_multiplexing
            # rho = expstep(alg.alg, ls, dt / time_multiplexing, rhos[end], cache)
            # rho = propagator * rhos[end]
            mul!(rhos[k+1], propagator, rhos[k])
            # push!(rhos, rho)
        end
        # rho0 = rhos[end]
        rhos[1] .= rhos[end]
        # push!(measurements, reduce(vcat, map(rho -> measure(rho, system), Iterators.drop(rhos, 1))))
        push!(measurements, reduce(vcat, map(specialized_measure, Iterators.drop(rhos, 1))))
    end

    smallest_decay_rate = save_spectrum ? mean(spec -> abs(partialsort(spec, 2, by=abs)), spectrum) : nothing

    return (; measurements, evals, spectrum, gapratios, average_gapratio, ediffs, smallest_decay_rate)
end
