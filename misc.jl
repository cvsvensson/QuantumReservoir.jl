abstract type AbstractPU end
struct CPU <: AbstractPU end

function hopping_hamiltonian(c, J; labels=keys(J))
    H = deepcopy(1.0 * first(c))
    for (k1, k2) in labels
        H .+= J[(k1, k2)] * c[k1]'c[k2] + hc
    end
    return blockdiagonal(H, c)
end
function coulomb_hamiltonian(c, V; labels=keys(V))
    H = deepcopy(1.0 * first(c))
    for (k1, k2) in labels
        H .+= V[(k1, k2)] * c[k1]'c[k1] * c[k2]'c[k2]
    end
    return blockdiagonal(H, c)
end

is_nearest_neighbours(k1, k2) = k1 != k2 && abs(first(k1) - first(k2)) ∈ (0, 1)
# function random_nearest_hoppings(spatial_labels, s=1)
#     couplings = [(k1, k2) => s * rand() * is_nearest_neighbours(k1, k2) for k1 in spatial_labels, k2 in spatial_labels]
#     Dict(couplings)
# end

function random_hoppings(labels, s=1)
    couplings = [(k1, k2) => s * rand() for (k1, k2) in labels]
    Dict(couplings)
end
function random_nearest_hoppings(labels, s=1)
    couplings = [(k1, k2) => s * rand() for (k1, k2) in labels if is_nearest_neighbours(k1, k2)]
    Dict(couplings)
end

function modify_initial_state((rs, θ, ϕ), rho, c=c; kwargs...)
    si, co = sincos(θ)
    channels = (I, c[0, 1]' * si + exp(1im * ϕ) * co * c[0, 2]', c[0, 1]' * c[0, 2]')
    rhonew = sum(r * L * rho * L' for (L, r) in zip(channels, rs))
    normalize_rho!(rhonew)
    return rhonew
end

normalize_rho!(rho) = rdiv!(rho, tr(rho))

function generate_training_parameters(M)
    θ = 2pi * rand(M)
    ϕ = 2pi * rand(M)
    r = 10 * rand(M, 3)
    return collect(zip(eachrow(r), θ, ϕ))
end
function generate_initial_states(ps, rho)
    # itr = Iterators.take(zip(eachrow(r), θ, ϕ), M)
    [modify_initial_state(p, rho) for p in ps]
end
function training_data(rhos, c; occ_ops, Ilabels)
    y = reduce(hcat, map(rho -> get_target_data(rho, occ_ops, c, Ilabels), rhos))
    return y
end
function get_target_data(rho, I_n_ops, c, Ilabels)
    occupations = [real(tr(rho * op)) for op in I_n_ops]
    entropy = input_entanglement(rho)
    rhoI = partial_trace(rho, Ilabels, c)
    purity = real(tr(rhoI^2))
    rv = get_rho_vec(rhoI)
    return [entropy, purity, rv..., occupations...]
end

function get_rho_vec(rho)
    @assert size(rho) == (4, 4)
    [real(diag(rho))..., real(rho[2, 3]), imag(rho[2, 3])]
end

function input_entanglement(rho, c=c)
    rhosub = partial_trace(rho, Ihalflabels, c)
    real(-tr(rhosub * log(rhosub)))
end

function get_obs_data(rhointernal, current_ops, occ_ops)
    cur = [real(tr(rhointernal' * op)) for op in current_ops]
    # occ = [real(tr(rhointernal' * op)) for op in occ_ops]
    # vcat(cur, occ)
    cur
    # occ
end

_time_evolve(rho, A::SciMLBase.MatrixOperator, t_obs; current_ops, occ_ops, kwargs...) = _time_evolve(rho, A.A, t_obs; current_ops, occ_ops, kwargs...)
function _time_evolve(rho, A, t_obs; current_ops, occ_ops, kwargs...)
    rhos = eachcol(expv_timestep(collect(t_obs), A, rho; kwargs...))
    reduce(vcat, [get_obs_data(rho, current_ops, occ_ops) for rho in rhos])
end
using KrylovKit
function _time_evolve(rho, ls::LazyLindbladSystem, t_obs; current_ops, occ_ops, kwargs...)
    rhos = [exponentiate(ls, t, rho; kwargs...)[1] for t in t_obs]
    reduce(vcat, [get_obs_data(rho, current_ops, occ_ops) for rho in rhos])
end
time_evolve(proc::CPU, args...; kwargs...) = time_evolve(args...; kwargs...)
function time_evolve(rho, ls::LindbladSystem, args...; kwargs...)
    A = QuantumDots.LinearOperator(ls)
    _time_evolve(vecrep(rho, ls), A, args...; kwargs...)
end
function time_evolve(rho, ls::LazyLindbladSystem, args...; kwargs...)
    _time_evolve(rho, ls, args...; kwargs...)
end
function _time_evolve(rho, A, tspan::Tuple, t_obs; current_ops, occ_ops, kwargs...)
    # drho!(out, rho, p, t) = mul!(out, A, rho)
    prob = ODEProblem(A, rho, tspan)
    sol = solve(prob, Tsit5(); abstol=1e-3, kwargs...)
    ts = range(tspan..., 200)
    currents = [real(tr(sol(t)' * op)) for op in current_ops, t in ts] |> permutedims
    # observations = reduce(hcat, [get_obs_data(sol(t), current_ops, occ_ops) for t in ts]) |> permutedims
    observations = reduce(vcat, [get_obs_data(sol(t), current_ops, occ_ops) for t in t_obs])
    return (; ts, sol, currents, observations)
end

QuantumDots.internal_rep(rho, ls::LazyLindbladSystem) = reshape(rho, size(ls.hamiltonian))
# LinearOperatorRep(rho, ls::LindbladSystem) = QuantumDots.internal_rep(rho, ls)
# LinearOperatorRep(rho, ::LazyLindbladSystem) = vec(rho)

vecrep(rho, ls::LazyLindbladSystem) = vec(rho)
vecrep(rho, ls::LindbladSystem) = QuantumDots.internal_rep(rho, ls)


struct QuantumReservoir{H,L0,L,LS0,LS,P,Pi,B,C,R,I,IL,RL}
    H0::H
    H::H
    leads0::L0
    leads::L
    ls0::LS0
    ls::LS
    rho0::P
    rhointernal0::Pi
    c::B
    current_ops::C
    R_occ_ops::R
    I_occ_ops::I
    Ilabels::IL
    Rlabels::RL
end
function QuantumReservoir(H0::h, H::h, leads0::L0, leads::L, c::B, Ilabels::IL, Rlabels::RL) where {h,L0,L,B,IL,RL}
    particle_number = blockdiagonal(numberoperator(c), c)
    ls0 = LindbladSystem(H0, leads0)
    prob0 = StationaryStateProblem(ls0)
    rhointernal0 = solve(prob0, LinearSolve.KrylovJL_LSMR(); abstol=1e-6)
    rho0 = QuantumDots.tomatrix(rhointernal0, ls0)
    @assert isapprox(tr(rho0), 1; atol=1e-3)
    ls = LindbladSystem(H, leads)
    internal_N = QuantumDots.internal_rep(particle_number, ls)
    current_ops = map(diss -> diss' * internal_N, ls.dissipators)
    R_occ_ops = map(k -> QuantumDots.internal_rep(c[k]' * c[k], ls), Rlabels)
    I_occ_ops = map(k -> c[k]' * c[k], Ilabels)
    QuantumReservoir{h,L0,L,typeof(ls0),typeof(ls),typeof(rho0),typeof(rhointernal0),B,typeof(current_ops),typeof(R_occ_ops),typeof(I_occ_ops),IL,RL}(H0, H, leads0, leads, ls0, ls, rho0, rhointernal0, c, current_ops, R_occ_ops, I_occ_ops, Ilabels, Rlabels)
end

struct InitialEnsemble{P,D}
    rho0s::P
    data::D
end
function InitialEnsemble(parameters, res::QuantumReservoir)
    rho0s = generate_initial_states(parameters, res.rho0)
    data = training_data(rho0s, res.c; occ_ops=res.I_occ_ops, Ilabels=res.Ilabels)
    InitialEnsemble{typeof(rho0s),typeof(data)}(rho0s, data)
end

function time_evolve(res, rho0, tspan, t_obs, proc=CPU())
    time_evolve(proc, rho0, res.ls, tspan, t_obs; current_ops=res.current_ops, occ_ops=res.R_occ_ops)
end
function time_evolve(res, ens::InitialEnsemble, tspan, t_obs; proc=CPU())
    map(rho0 -> time_evolve(proc, rho0, res.ls, tspan, t_obs; current_ops=res.current_ops, occ_ops=res.R_occ_ops), ens.rho0s)
end
function time_evolve(res, ens::InitialEnsemble, t_obs; proc=CPU())
    sols = map(rho0 -> time_evolve(proc, rho0, res.ls, t_obs; current_ops=res.current_ops, occ_ops=res.R_occ_ops), ens.rho0s)
    data = reduce(hcat, sols)
    return (; sols, data)
end

Base.getindex(ens::InitialEnsemble, I) = InitialEnsemble(ens.rho0s[I], ens.data[I, :])