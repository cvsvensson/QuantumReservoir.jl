abstract type AbstractPU end
struct CPU <: AbstractPU end

hopping_hamiltonian(c::FermionBasis{0}, J; labels=keys(J)) = 0
function hopping_hamiltonian(c, J; labels=keys(J))
    T = valtype(J)
    H = deepcopy(zero(T) * first(c))
    length(J) == 0 && return H
    for (k1, k2) in labels
        H .+= J[(k1, k2)] * c[k1]'c[k2] + hc
    end
    return blockdiagonal(H, c)
end
coulomb_hamiltonian(c::FermionBasis{0}, V; labels=keys(V)) = 0
function coulomb_hamiltonian(c, V; labels=keys(V))
    T = valtype(V)
    H = deepcopy(zero(T) * first(c))
    length(V) == 0 && return H
    for (k1, k2) in labels
        H .+= V[(k1, k2)] * c[k1]'c[k1] * c[k2]'c[k2]
    end
    return blockdiagonal(H, c)
end
qd_level_hamiltonian(c::FermionBasis{0}, ε; labels=keys(ε)) = 0
function qd_level_hamiltonian(c, ε; labels=keys(ε))
    T = valtype(ε)
    H = deepcopy(zero(T) * first(c))
    length(ε) == 0 && return H
    for l in labels
        H .+= ε[l] * c[l]'c[l]
    end
    return blockdiagonal(H, c)
end

is_nearest_neighbours(k1, k2) = k1 != k2 && all(map((l1, l2) -> abs(l1 - l2) ∈ (0, 1), k1, k2))
# function random_nearest_hoppings(spatial_labels, s=1)
#     couplings = [(k1, k2) => s * rand() * is_nearest_neighbours(k1, k2) for k1 in spatial_labels, k2 in spatial_labels]
#     Dict(couplings)
# end

function one_hoppings(labels, s=1)
    couplings = [(k1, k2) => s * 1.0 for (k1, k2) in labels]
    Dict(couplings)
end
function random_hoppings(labels, s=1)
    couplings = [(k1, k2) => s * rand() for (k1, k2) in labels]
    Dict(couplings)
end
function random_nearest_hoppings(labels, s=1)
    couplings = [(k1, k2) => s * rand() for (k1, k2) in labels if is_nearest_neighbours(k1, k2)]
    Dict(couplings)
end

function modify_initial_state((rs, θ, ϕ), rho, c)
    si, co = sincos(θ)
    # channels = (I, c[0, 1]' * si + exp(1im * ϕ) * co * c[0, 2]', c[0, 1]' * c[0, 2]')
    channels = (c[0, 1]' * si + exp(1im * ϕ) * co * c[0, 2]',)
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
function generate_initial_states(ps, rho, c)
    # itr = Iterators.take(zip(eachrow(r), θ, ϕ), M)
    [modify_initial_state(p, rho, c) for p in ps]
end
function training_data(rhos, c, occ_ops, Ilabels)
    y = reduce(hcat, map(rho -> get_target_data(rho, occ_ops, c, Ihalflabels, Ilabels), rhos))
    return y
end
function training_data(rhos, c, Ihalflabels, Ilabels)
    y = reduce(hcat, map(rho -> get_target_data(rho, c, Ihalflabels, Ilabels), rhos))
    return y
end
function get_target_data(rho, c, Ihalflabels, Ilabels)
    entropy = input_entanglement(rho, Ihalflabels, c)
    rhoI = partial_trace(rho, Ilabels, c)
    purity = real(tr(rhoI^2))
    rv = get_rho_vec(rhoI)
    return [entropy, purity, rv...]
end
function get_target_data(rho, I_n_ops, c, Ihalflabels, Ilabels)
    occupations = [real(tr(rho * op)) for op in I_n_ops]
    entropy = input_entanglement(rho, Ihalflabels, c)
    rhoI = partial_trace(rho, Ilabels, c)
    purity = real(tr(rhoI^2))
    rv = get_rho_vec(rhoI)
    return [entropy, purity, rv..., occupations...]
end

function get_rho_vec(rho)
    if size(rho) == (4, 4)
        return [real(diag(rho))..., real(rho[2, 3]), imag(rho[2, 3])]
    end
    if size(rho) == (8, 8)
        utri = triu!(trues(4, 4), 1)
        rho1 = rho[2:5, 2:5][utri]
        utri2 = triu!(trues(2, 2), 1)
        rho2 = rho[6:7, 6:7][utri2]
        return [real(diag(rho))..., real(rho1)..., imag(rho1)..., real(rho2)..., imag(rho2)...]
    end
end

function get_rho_mat(rho)
    if length(rho) == 6
        rho1 = [rho[1];;]
        rho2 = Diagonal(rho[2:3]) + [0 1; 1 0] * rho[5] + [0 1im; -1im 0] * rho[6]
        rho3 = [rho[4];;]
        return BlockDiagonal([rho1, rho2, rho3])
    else
        throw(ArgumentError("Don't know what to do with vectors of length(rho) == $(length(rho))"))
    end
end


function input_entanglement(rho, Ihalflabels, c)
    rhosub = partial_trace(rho, Ihalflabels, c)
    real(-tr(rhosub * log(rhosub)))
end

get_obs_data(rhointernal, current_ops) = [real(tr(rhointernal' * op)) for op in current_ops]
function get_obs_data(rhointernal, current_ops, occ_ops)
    cur = [real(tr(rhointernal' * op)) for op in current_ops]
    occ = [real(tr(rhointernal' * op)) for op in occ_ops]
    vcat(cur, occ)
end

time_evolve_exp(rho, A::SciMLBase.MatrixOperator, t_obs; current_ops, occ_ops, kwargs...) = time_evolve_exp(rho, A.A, t_obs; current_ops, occ_ops, kwargs...)
function time_evolve_exp(rho, A, t_obs; current_ops, occ_ops, kwargs...)
    rhos = eachcol(expv_timestep(collect(t_obs), A, rho; kwargs...))
    # rhos = [expv(t, A, rho; kwargs...) for t in t_obs]
    reduce(vcat, [get_obs_data(rho, current_ops, occ_ops) for rho in rhos])
end
using KrylovKit
function _time_evolve(rho, ls::LazyLindbladSystem, t_obs; current_ops, occ_ops, kwargs...)
    rhos = [exponentiate(ls, t, rho; kwargs...)[1] for t in t_obs]
    reduce(vcat, [get_obs_data(rho, current_ops, occ_ops) for rho in rhos])
end
time_evolve_ode(proc::CPU, args...; kwargs...) = _time_evolve_ode(args...; kwargs...)
time_evolve_exp(proc::CPU, args...; kwargs...) = time_evolve_exp(args...; kwargs...)
time_evolve_nonlinear(proc::CPU, args...; kwargs...) = time_evolve_nonlinear(args...; kwargs...)
function time_evolve_ode(rho::AbstractArray, ls::LindbladSystem, args...; kwargs...)
    A = QuantumDots.LinearOperator(ls)
    time_evolve_ode(vecrep(rho, ls), A, args...; kwargs...)
end
function time_evolve_exp(rho::AbstractArray, ls::LindbladSystem, args...; kwargs...)
    A = QuantumDots.LinearOperator(ls)
    time_evolve_exp(vecrep(rho, ls), A, args...; kwargs...)
end
# function time_evolve(rho, ls::LazyLindbladSystem, args...; kwargs...)
#     _time_evolve(rho, ls, args...; kwargs...)
# end
function _time_evolve_ode(rho::AbstractArray, A, tspan, t_obs; current_ops, occ_ops, alg=ROCK4(), kwargs...)
    # drho!(out, rho, p, t) = mul!(out, A, rho)
    prob = ODEProblem(A, rho, tspan)
    sol = solve(prob, alg; abstol=1e-3, kwargs...)
    ts = range(tspan..., 200)
    currents = [real(tr(sol(t)' * op)) for op in current_ops, t in ts] |> permutedims
    # observations = reduce(hcat, [get_obs_data(sol(t), current_ops, occ_ops) for t in ts]) |> permutedims
    observations = reduce(vcat, [get_obs_data(sol(t), current_ops, occ_ops) for t in t_obs])
    return (; ts, sol, currents, observations)
end

function time_evolve_nonlinear(rho, ls::LindbladSystem, t::Number, op; kwargs...)
    A = ls.unitary
    rho_measured = op * tomatrix(expv(t, A, vecrep(rho, ls); kwargs...), ls) * op'
    vecrep(normalize_rho!(rho_measured), ls)
end
function time_evolve_nonlinear(rho, ls, ts, op; kwargs...)
    t0 = first(ts)
    dt = diff(ts)
    newts = [t0, dt...]
    for t in newts
        rho = time_evolve_nonlinear(rho, ls, t, op; kwargs...)
    end
    return rho
end

QuantumDots.internal_rep(rho, ls::LazyLindbladSystem) = reshape(rho, size(ls.hamiltonian))
# LinearOperatorRep(rho, ls::LindbladSystem) = QuantumDots.internal_rep(rho, ls)
# LinearOperatorRep(rho, ::LazyLindbladSystem) = vec(rho)

vecrep(rho, ::LazyLindbladSystem) = vec(rho)
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
    normalize_rho!(rho0)
    rhointernal0 = vecrep(rho0, ls0)
    @assert isapprox(tr(rho0), 1; atol=1e-3) "tr(ρ) = $(tr(rho0)) != 1"
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

function time_evolve_ode(res::QuantumReservoir, rho0::AbstractArray, tspan, t_obs; proc=CPU(), kwargs...)
    time_evolve_ode(proc, rho0, res.ls, tspan, t_obs; current_ops=res.current_ops, occ_ops=res.R_occ_ops, kwargs...)
end
function time_evolve_ode(res::QuantumReservoir, ens::InitialEnsemble, tspan, t_obs, t_nl=Float64[]; proc=CPU(), op=nothing, kwargs...)
    rhos = length(t_nl) > 0 ? map(rho0 -> time_evolve_nonlinear(proc, rho0, res.ls, t_nl, op), ens.rho0s) : ens.rho0s
    map(rho0 -> time_evolve_ode(proc, rho0, res.ls, tspan, t_obs; current_ops=res.current_ops, occ_ops=res.R_occ_ops, kwargs...), rhos)
end
function time_evolve_exp(res::QuantumReservoir, ens::InitialEnsemble, t_obs, t_nl=Float64[]; proc=CPU(), op=nothing, kwargs...)
    rhos = length(t_nl) > 0 ? map(rho0 -> time_evolve_nonlinear(proc, rho0, res.ls, t_nl, op; kwargs...), ens.rho0s) : ens.rho0s
    sols = map(rho0 -> time_evolve_exp(proc, rho0, res.ls, t_obs; current_ops=res.current_ops, occ_ops=res.R_occ_ops, kwargs...), rhos)
    data = reduce(hcat, sols)
    return (; sols, data)
end

Base.getindex(ens::InitialEnsemble, I) = InitialEnsemble(ens.rho0s[I], ens.data[I, :])