function hopping_hamiltonian(c, J, V; labels=keys(J))
    H = deepcopy(1.0 * first(c))
    for (k1, k2) in labels
        H .+= J[(k1, k2)] * c[k1]'c[k2] + hc
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

function generate_training_data(M, rho, c; occ_ops, Ilabels)
    θ = 2pi * rand(M)
    ϕ = 2pi * rand(M)
    r = 10 * rand(M, 3)
    rhonew = [modify_initial_state(p, rho) for p in zip(eachrow(r), θ, ϕ)]
    true_data = reduce(hcat, map(rho -> get_target_data(rho, occ_ops, c, Ilabels), rhonew)) |> permutedims
    return (; θ, ϕ, r, rhos=rhonew, true_data)
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
    cur = [real(rhointernal' * op) for op in current_ops]
    occ = [real(rhointernal' * op) for op in occ_ops]
    # vcat(cur, occ)
    cur
    # occ
end

_time_evolve(rho, A::SciMLBase.MatrixOperator, t_obs; current_ops, occ_ops, kwargs...) = _time_evolve(rho, A.A, t_obs; current_ops, occ_ops, kwargs...)
function _time_evolve(rho, A, t_obs; current_ops, occ_ops, kwargs...)
    rhos = eachcol(expv_timestep(collect(t_obs), A, rho; kwargs...))
    # rhos = [exponentiate(A, t, rho; kwargs...)[1] for t in t_obs]
    reduce(vcat, [get_obs_data(rho, current_ops, occ_ops) for rho in rhos])
end
function _time_evolve(rho, ls::LazyLindbladSystem, t_obs; current_ops, occ_ops, kwargs...)
    rhos = [exponentiate(ls, t, rho; kwargs...)[1] for t in t_obs]
    reduce(vcat, [get_obs_data(rho, current_ops, occ_ops) for rho in rhos])
end
function time_evolve(rho, ls::LindbladSystem, args...; kwargs...)
    A = QuantumDots.LinearOperator(ls)
    time_evolve(LinearOperatorRep(rho, ls), A, args...; kwargs...)
end
function time_evolve(rho, A, args...; gpu=false, occ_ops, current_ops, kwargs...)
    if gpu
        _time_evolve(cu(rho), MatrixOperator(cu(A.A)), map(arg -> Float32.(arg), args)...; current_ops=cu.(current_ops), occ_ops=cu.(occ_ops), kwargs...)
    else
        _time_evolve(rho, A, args...; current_ops, occ_ops, kwargs...)
    end
end
function time_evolve(rho, ls::LazyLindbladSystem, args...; gpu=false, kwargs...)
    if gpu
        @warn "GPU not implemented for LazyLindbladSystem"
    end
    _time_evolve(rho, ls, args...; kwargs...)
end
function _time_evolve(rho, A, tspan::Tuple, t_obs; current_ops, occ_ops, kwargs...)
    # drho!(out, rho, p, t) = mul!(out, A, rho)
    prob = ODEProblem(A, rho, tspan)
    sol = solve(prob, Tsit5(); abstol=1e-3, kwargs...)
    ts = range(tspan..., 200)
    currents = [real(tr(sol(t)' * op)) for op in current_ops, t in ts] |> permutedims
    observations = reduce(hcat, [get_obs_data(sol(t), current_ops, occ_ops) for t in ts]) |> permutedims
    # observations = reduce(vcat, [get_obs_data(sol(t), current_ops) for t in t_obs])
    return (; ts, sol, currents, observations)
end

QuantumDots.internal_rep(rho, ls::LazyLindbladSystem) = reshape(rho, size(ls.hamiltonian))
LinearOperatorRep(rho, ls::LindbladSystem) = QuantumDots.internal_rep(rho, ls)
LinearOperatorRep(rho, ::LazyLindbladSystem) = vec(rho)
