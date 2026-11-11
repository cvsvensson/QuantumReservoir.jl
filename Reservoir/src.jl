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

struct PiecewiseTimeSteppingMethod{A}
    alg::A
end
struct EXP_KRYLOV end
struct EXP_SCIML end
struct ODE{A}
    alg::A
end
struct Generators end
default_alg(::LindbladSystem, ::ContinuousInput) = ODE(Tsit5())
default_alg(::LindbladSystem, ::DiscreteInput) = PiecewiseTimeSteppingMethod(EXP_SCIML())
default_alg(::PauliSystem, ::DiscreteInput) = PiecewiseTimeSteppingMethod(EXP_SCIML())

struct Pauli end
struct Lindblad end

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
function specialize(measure::CurrentMeasurements, sys::Union{LindbladSystem,PauliSystem})
    ops = current_operators(sys, measure.op)
    labels = sort(collect(keys(ops)))
    rho -> [real(dot(ops[k], rho)) for k in labels]
end
(measure::CurrentMeasurements)(rho, sys) = specialize(measure, sys)(rho)

function get_internal_initial_state(system, input)
    t0 = input.tspan[1]
    update_coefficients!(system, input.voltage_input(t0))
    vecrep(collect(stationary_state(system)), system)
end
function run_reservoir!(ls, input::DiscreteInput, ts, initial_state, alg::PiecewiseTimeSteppingMethod, measure; time_multiplexing=1, save_spectrum=false, kwargs...)
    # ls = copy ? deepcopy(_ls) : _ls
    update_coefficients!(ls, input(first(ts)))
    dt = input.dt
    rho0 = initial_state
    measurements = typeof(measure(rho0, ls))[]
    spectrum = save_spectrum ? [] : nothing

    evals = sort(ls.eigen.values)
    gapratios = map(x -> x > 1 ? inv(x) : x, diff(evals)[1:end-1] ./ diff(evals)[2:end])
    average_gapratio = mean(gapratios)
    ediffs = vec(evals .- evals')

    cache = get_cache(alg.alg, ls)
    for t in ts[1:end-1]
        update_coefficients!(ls, input(t))
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
expstep(method::EXP_KRYLOV, sys, dt, rho, cache=nothing; kwargs...) = exponentiate(sys.total, dt, rho; kwargs...)[1]
expstep(method::EXP_SCIML, sys, dt, rho, cache=nothing; kwargs...) = expv(dt, sys.total, rho; cache, kwargs...)
get_cache(::EXP_SCIML, sys) = ExpvCache{eltype(sys)}(30)
get_cache(::EXP_KRYLOV, sys) = nothing
# ksA = KrylovSubspace{complex(eltype(A))}(n, maxiter)
# arnoldi!(ksA, A, vrho0; tol=abstol, m)

using DiffEqCallbacks
function run_reservoir!(ls, input::DiscreteInput, ts, initial_state, alg::ODE, measure; time_multiplexing=1, kwargs...)
    small_dt = input.dt / time_multiplexing
    t_measure = [t + n * small_dt for t in ts[1:end-1] for n in 1:time_multiplexing]
    update_coefficients!(ls, input(first(ts)))
    saved_values = SavedValues(Float64, typeof(measure(initial_state, ls)))
    # Saving comes first so that a measurement at a switching time uses the voltage of the step that just ended.
    callback = CallbackSet(SavingCallback((u, t, integrator) -> measure(u, ls), saved_values; saveat=t_measure),
        PresetTimeCallback(ts[2:end-1], integrator -> update_coefficients!(ls, input(integrator.t))))
    rhs!(du, u, ls, t) = mul!(du, ls, u)
    solve(ODEProblem(rhs!, initial_state, (first(ts), last(t_measure)), ls), alg.alg; callback, kwargs...)
    measurements = map(data -> reduce(vcat, data), Iterators.partition(saved_values.saveval, time_multiplexing))
    return (; measurements)
end
function run_reservoir!(ls, input::ContinuousInput, ts, initial_state, measure, alg::ODE; kwargs...)
    p = (ls, input)
    solve(ODEProblem(f_ode!, initial_state, ts, p; kwargs...))
end
function f_ode!(du, u, (ls, input), t)
    update_coefficients!(ls, input(t))
    mul!(du, ls, u)
end
##

function fullanalysis(reservoir, lead, input, opensystem, f, space, measurement, target, training; kwargs...)
    system = build_system(opensystem, reservoir, lead, f, space)
    fullanalysis(system, input, measurement, target, training; kwargs...)
end
function fullanalysis(system::Union{LindbladSystem,PauliSystem}, input, measurement, targets, training; alg=default_alg(system, input.voltage_input), kwargs...)
    (; voltage_input, ts, tspan, dt) = input
    (; measure, time_multiplexing) = measurement
    rho0 = get_internal_initial_state(system, input)
    simulation = run_reservoir!(system, voltage_input, ts, rho0, alg, measure; save_spectrum=true, time_multiplexing, kwargs...)
    (; warmup, train) = training
    task_props = fit(simulation.measurements, targets; warmup=0.2, train=0.5)
    return (; simulation, fit=task_props)
end

(::Pauli)(H, leads; qn) = PauliSystem(H, leads; qn)
(::Lindblad)(H, leads; qn) = LindbladSystem(H, leads; qn)
function build_system(opensystem, reservoir, lead, f, space)
    H = representation(hamiltonian(f, reservoir.params), space)
    leads = lead_dict(lead, f, space)
    opensystem(H, leads; qn=particle_numbers(space))
end
function run_reservoir(reservoir, lead, input, measurement, opensystem, f, space, alg=PiecewiseTimeSteppingMethod(EXP_SCIML()); kwargs...)
    system = build_system(opensystem, reservoir, lead, f, space)
    (; ts, voltage_input) = input
    (; measure, time_multiplexing) = measurement
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

function generate_propagators(reservoir, lead, input, opensystem, f, space; kwargs...)
    system = build_system(opensystem, reservoir, lead, f, space)
    generate_propagators(system, get_input_values(input.voltage_input), input.dt)
end
function generate_propagators(system, input_values, dt)
    method = ExpMethodHigham2005()
    cache = ExponentialUtilities.alloc_mem(Matrix(system), method)
    systems = typeof(system)[]
    propagators = map(input_values) do input
        update_coefficients!(system, input)
        push!(systems, deepcopy(system))
        exponential!(dt * Matrix(system), method, cache)
    end
    return propagators, systems
end
function generate_master_matrices(reservoir, lead, input, opensystem, f, space; kwargs...)
    system = build_system(opensystem, reservoir, lead, f, space)
    generate_master_matrices(system, get_input_values(input.voltage_input))
end
function generate_master_matrices(system, input_values)
    # method = ExpMethodHigham2005()
    # cache = ExponentialUtilities.alloc_mem(Matrix(system), method)
    map(input_values) do input
        update_coefficients!(system, input)
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
function lead_dict(lead, f, space)
    Dict(l => NormalLead(lead.Γ[l] * representation(f[l]', space); T=lead.temperature, μ=0.0) for l in lead.labels)
end

function run_reservoir(reservoir, lead, input, measurement, opensystem, f, space, alg::PropagatorMethod; save_spectrum=false)
    system = build_system(opensystem, reservoir, lead, f, space)
    (; ts, voltage_input, dt) = input
    (; measure, time_multiplexing) = measurement
    input_values = get_input_values(voltage_input)
    propagators, systems = generate_propagators(system, input_values, dt / time_multiplexing)
    measures = map(s -> specialize(measure, s), systems)
    rho0 = get_internal_initial_state(system, input)

    evals = sort(system.eigen.values)
    gapratios = map(x -> x > 1 ? inv(x) : x, diff(evals)[1:end-1] ./ diff(evals)[2:end])
    average_gapratio = mean(gapratios)
    ediffs = vec(evals .- evals')

    measurements = typeof(first(measures)(rho0))[]
    spectrum = []
    rhos = [deepcopy(rho0) for k in 1:time_multiplexing+1]
    for t in ts[1:end-1]
        ind = findfirst(==(voltage_input(t)), input_values)
        save_spectrum && push!(spectrum, eigvals(Matrix(systems[ind])))
        for k in 1:time_multiplexing
            mul!(rhos[k+1], propagators[ind], rhos[k])
        end
        rhos[1] .= rhos[end]
        push!(measurements, reduce(vcat, map(measures[ind], Iterators.drop(rhos, 1))))
    end

    smallest_decay_rate = save_spectrum ? mean(spec -> abs(partialsort(spec, 2, by=abs)), spectrum) : nothing

    return (; measurements, evals, spectrum, gapratios, average_gapratio, ediffs, smallest_decay_rate)
end
