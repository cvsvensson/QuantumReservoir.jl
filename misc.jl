function hopping_hamiltonian(c, J; Jkeys=keys(J))
    H = deepcopy(1.0 * first(c))
    for (k1, k2) in Jkeys
        H .+= J[(k1, k2)] * c[k1]'c[k2] + hc
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

function modify_initial_state((rs, θ), rho, c=c; kwargs...)
    si, co = sincos(θ)
    channels = (I, c[0, 1]' * si + co * c[0, 2]', c[0, 1]' * c[0, 2]')
    rhonew = sum(r * L * rho * L' for (L, r) in zip(channels, rs))
    normalize_rho!(rhonew)
    return rhonew
end

normalize_rho!(rho) = rdiv!(rho, tr(rho))

function generate_training_data(M, rho, c; occ_ops)
    θ = 2pi * rand(M)
    rs = 10 * rand(M, 3)
    rhonew = [modify_initial_state(p, rho) for p in zip(eachrow(rs), θ)]
    true_data = reduce(hcat, map(rho -> get_target_data(rho, occ_ops, c), rhonew)) |> permutedims
    return (; θ, rs, rhos=rhonew, true_data)
end
function get_target_data(rho, I_n_ops, c)
    occupations = [real(tr(rho * op)) for op in I_n_ops]
    entropy = input_entanglement(rho)
    purity = real(tr(partial_trace(rho, Ilabels, c)^2))
    return [entropy, purity, occupations...]
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

function time_evolve(rho, ls, t_obs; current_ops, occ_ops, kwargs...)
    L = Matrix(ls)
    rhointernal = QuantumDots.internal_rep(rho, ls)
    # f = t -> expv(t, L, rhointernal; kwargs...)
    # rhos = f.(t_obs)
    rhos = eachcol(expv_timestep(collect(t_obs), L, rhointernal; kwargs...))
    reduce(vcat, [get_obs_data(rho, current_ops, occ_ops) for rho in rhos])
end
function time_evolve(rho, ls, tspan::Tuple, t_obs; current_ops, occ_ops, kwargs...)
    rhointernal = QuantumDots.internal_rep(rho, ls)
    drho!(out, rho, p, t) = mul!(out, ls, rho)
    prob = ODEProblem(drho!, rhointernal, tspan)
    sol = solve(prob, Tsit5(); abstol=1e-3, kwargs...)
    ts = range(tspan..., 200)
    currents = [real(sol(t)' * op) for op in current_ops, t in ts] |> permutedims
    observations = reduce(hcat, [get_obs_data(sol(t), current_ops, occ_ops) for t in ts]) |> permutedims
    # observations = reduce(vcat, [get_obs_data(sol(t), current_ops) for t in t_obs])
    return (; ts, sol, currents, observations)
end