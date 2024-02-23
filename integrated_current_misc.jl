struct DenseLindblad end
struct LazyLindblad end
struct ReservoirConnections{L,C,Cl,B}
    labels::L
    Ilabels::L
    Ihalflabels::L
    Rlabels::L
    hopping_labels::C
    Iconnections::C
    Rconnections::C
    IRconnections::C
    lead_connections::Cl
    bases::B
end
function ReservoirConnections(N, M=1; qn=QuantumDots.fermionnumber)
    labels = vec(Base.product(0:N, 1:2) |> collect)
    hopping_labels = [(labels[k1], labels[k2]) for k1 in 1:length(labels), k2 in 1:length(labels) if k1 > k2 && is_nearest_neighbours(labels[k1], labels[k2])]
    Ilabels = filter(x -> first(x) <= 0, labels)
    Rlabels = filter(x -> first(x) > 0, labels)
    Ihalflabels = filter(x -> isone(x[2]), Ilabels)
    Iconnections = filter(k -> first(k[1]) + first(k[2]) == 0, hopping_labels)
    Rconnections = filter(k -> first(k[1]) + first(k[2]) > 1, hopping_labels)
    IRconnections = filter(k -> abs(first(k[1]) - first(k[2])) == 1, hopping_labels)
    lead_connections = [(m, [(N, k) for k in 1:2]) for m in 1:M]

    cI = FermionBasis(Ilabels; qn)
    cR = FermionBasis(Rlabels; qn)
    cIR = wedge(cI, cR)

    return ReservoirConnections(labels, Ilabels, Ihalflabels, Rlabels, hopping_labels, Iconnections, Rconnections, IRconnections, lead_connections, (; cI, cR, cIR))
end


struct Sampled{G}
    grid::G
end
struct Exponentiation{K}
    alg::K
end
struct EXP_krylovkit end
struct EXP_sciml end
struct EXP_sciml_full end
struct IntegratedODE{A}
    alg::A
end
struct ODE{A}
    alg::A
end
ODE() = ODE(DP8())
IntegratedODE() = IntegratedODE(DP8())
Exponentiation() = Exponentiation(EXP_sciml())

struct Reservoir{RC,H,LS,I,CO}
    rc::RC
    H::H
    ls::LS
    initials::I
    current_ops::CO
end
struct ReservoirParameters{S,E}
    JVΓ::S
    εs::E
end

function random_parameters(rc::ReservoirConnections, Nmeas)
    J = Dict((k1, k2) => 2(rand() - 0.5) for (k1, k2) in rc.hopping_labels)
    V = Dict((k1, k2) => rand() for (k1, k2) in rc.hopping_labels)
    Γ = Dict(reduce(vcat, [(m, l...) => 2(rand() - 0.5) for l in ls] for (m, ls) in rc.lead_connections))
    εs = [Dict(l => 2(rand() - 0.5) for l in rc.labels) for i in 1:Nmeas]
    return ReservoirParameters((J, V, Γ), εs)
end
scale_dict(d, s) = Dict(map(p -> p[1] => s * p[2], collect(d)))
scale_dicts(ds, ss) = map(scale_dict, ds, ss)


function get_hamiltonian(c, J, V, ε; Jlabels=keys(J), Vlabels=keys(V), εlabels=keys(ε))
    hopping_hamiltonian(c, J; labels=Jlabels) + coulomb_hamiltonian(c, V; labels=Vlabels) + qd_level_hamiltonian(c, ε; labels=εlabels)
end

function get_hamiltonians(rc, J, V, ε)
    cI = rc.bases.cI
    cR = rc.bases.cR
    cIR = rc.bases.cIR
    HI = get_hamiltonian(cI, J, V, ε; Jlabels=rc.Iconnections, Vlabels=rc.Iconnections, εlabels=rc.Ilabels)
    HR = get_hamiltonian(cR, J, V, ε; Jlabels=rc.Rconnections, Vlabels=rc.Rconnections, εlabels=rc.Rlabels)
    HIR = get_hamiltonian(cIR, J, V, ε; Jlabels=rc.hopping_labels, Vlabels=rc.hopping_labels, εlabels=rc.labels)
    HIR0 = get_hamiltonian(cIR, J, V, ε; Jlabels=rc.Iconnections, Vlabels=rc.Iconnections, εlabels=rc.Ilabels) + get_hamiltonian(cIR, J, V, ε; Jlabels=rc.Rconnections, Vlabels=rc.Rconnections, εlabels=rc.Rlabels)
    return (; I0=HI, R0=HR, IR0=HIR0, IR=HIR)
end

function get_leads(rc, Γ, (T, μ), μmin)
    cR = rc.bases.cR
    cI = rc.bases.cI
    c = rc.bases.cIR
    leadsR0 = length(cR) > 0 ? Tuple(NormalLead(sum(cR[N, k]' * Γ[(m, N, k)] for (N, k) in ls); T, μ=μ[m]) for (m, ls) in rc.lead_connections) : tuple()
    leadsIR = Tuple(NormalLead(sum(c[N, k]' * Γ[(m, N, k)] for (N, k) in ls); T, μ=μ[m]) for (m, ls) in rc.lead_connections)
    leadsI0 = tuple(CombinedLead(Tuple(cI[i]' for i in rc.Ilabels); T, μ=μmin))
    leadsIR0 = length(cR) > 0 ? (CombinedLead(Tuple(c[i]' for i in rc.Ilabels); T, μ=μmin), leadsIR...) : (CombinedLead(Tuple(c[i]' for i in rc.Ilabels); T, μ=μmin),)
    return (; I0=leadsI0, R0=leadsR0, IR0=leadsIR0, IR=leadsIR)
end

get_lindblad(H, leads, ::DenseLindblad) = LindbladSystem(H, leads)
get_lindblad(H, leads, ::LazyLindblad) = LazyLindbladSystem(H, leads)

function get_initial_state(rc::ReservoirConnections, (lsI, lsR, lsIR0, lsIR); abstol, kwargs...)
    # probI = StationaryStateProblem(lsI)
    # rhointernalI = solve(probI, LinearSolve.KrylovJL_LSMR(); kwargs...)
    # rhoI = QuantumDots.tomatrix(rhointernalI, lsI)
    # normalize_rho!(rhoI)
    # rhoIvec = vecrep(rhoI, lsI)

    if length(QuantumDots.LinearOperator(lsR)) > 1
        probR = StationaryStateProblem(lsR)
        rhointernalR = solve(probR, LinearSolve.KrylovJL_LSMR(); abstol, kwargs...)
        rhoR = QuantumDots.tomatrix(rhointernalR, lsR)
        normalize_rho!(rhoR)
        rhoRvec = vecrep(rhoR, lsR)
    else
        rhoRvec = [1]
        rhoR = [1;;]
        lsR = nothing
    end
    # rhoIR = wedge(rhoI, res.cI, rhoR, res.cR, res.c)
    # rhoIRvec = vecrep(rhoIR, lsIR)
    T = promote_type(eltype(lsI), eltype(lsIR))
    nout = size(QuantumDots.LinearOperator(lsIR), 2)
    nin = size(QuantumDots.LinearOperator(lsI), 2)
    lm = LinearMap{T}(rhoIvec -> vecrep(wedge(QuantumDots.tomatrix(rhoIvec, lsI), rc.bases.cI, rhoR, rc.bases.cR, rc.bases.cIR), lsIR), nout, nin)

    # probIR2 = StationaryStateProblem(lsIR2)
    # rhointernalIR2 = solve(probIR2, LinearSolve.KrylovJL_LSMR(); kwargs...)
    # rhoIR2 = QuantumDots.tomatrix(rhointernalIR2, lsIR2)
    # normalize_rho!(rhoIR2)

    # abstol = kwargs[:abstol]
    # if norm(rhoIR - rhoIR2) > 100 * (abstol)
    #     @warn "Inconsistent initial states" norm(rhoIR - rhoIR2) abstol
    # end

    # return (; I=(vec=rhointernalI, mat=rhoI), R=(mat=rhoR, vec=rhoRvec), IR=(vec=rhoIRvec, vec2=vecrep(rhoIR2, lsIR), mat=rhoIR), wedgemap=Matrix(lm))
    return (; mat=rhoR, vec=rhoRvec, wedgemap=Matrix(lm))
end
function prepare_reservoir(rc::ReservoirConnections, params::ReservoirParameters, (T, μ); kwargs...)
    εs = params.εs
    [prepare_reservoir(rc, (params.JVΓ..., ε), (T, μ); kwargs...) for ε in εs]
end
function prepare_reservoir(rc::ReservoirConnections, (J, V, Γ, ε), (T, μ); μmin=-100000, lindbladian=DenseLindblad(), kwargs...)
    Hs = get_hamiltonians(rc, J, V, ε)
    leads = get_leads(rc, Γ, (T, μ), μmin)
    ls = map((H, l) -> get_lindblad(H, l, lindbladian), Hs, leads)
    initials = get_initial_state(rc, ls; kwargs...)

    c = rc.bases.cIR
    lsIR = ls.IR
    particle_number = blockdiagonal(numberoperator(c), c)
    internal_N = QuantumDots.internal_rep(particle_number, lsIR)
    current_ops = map(diss -> vecrep((diss' * internal_N), lsIR), lsIR.dissipators)
    return Reservoir(rc, Hs, ls, initials, current_ops)
end

function get_initial_state(res::Reservoir, rho0I::AbstractMatrix)
    rho0Ivec = vecrep(rho0I, res.ls.I0)
    rho0mat = wedge(rho0I, res.rc.bases.cI, res.initials.mat, res.rc.bases.cR, res.rc.bases.cIR)
    rho0vec = vecrep(rho0mat, res.ls.IR)
    @assert rho0vec ≈ res.initials.wedgemap * rho0Ivec
    return rho0mat, rho0vec
end
function calculate_currents(rho, current_ops)
    [real(dot(op, rho)) for op in current_ops]
end

function run_reservoir_trajectories(reservoirs::Vector{<:Reservoir}, rho0Is, tmax, alg; kwargs...)
    trajs = [run_reservoir_trajectories(res, rho0Is, tmax, alg; kwargs...) for res in reservoirs]
    integrated = mapreduce(x -> x.integrated, vcat, trajs)

    currents = ismissing(first(trajs).currents) ? missing : [reduce(vcat, map(t -> t.currents[n], trajs)) for n in eachindex(rho0Is)]
    return (; integrated, currents)
end
function run_reservoir_trajectories(res::Reservoir, rho0Is, tmax, alg; time_trace=false, kwargs...)
    rhovecs = [get_initial_state(res, rho0I)[2] for rho0I in rho0Is]
    current_ops = res.current_ops
    ls = res.ls.IR
    A = QuantumDots.LinearOperator(ls)
    get_current = Base.Fix2(calculate_currents, current_ops)
    results = integrated_current(A, rhovecs, tmax, get_current, alg; kwargs...)
    currents = time_trace ? get_current_time_trace(A, rhovecs, tmax, get_current; kwargs...) : missing
    return (; integrated=results, currents)
end


function integrated_current(ls, rho0s, tmax, get_current, solver::Sampled; int_alg, kwargs...)
    grid = solver.grid
    @assert tmax ≈ grid[end]
    function solve_int(vrho0)
        vals = current_integrand(grid, A, vrho0, get_current; kwargs...)
        prob = SampledIntegralProblem(vals, grid; kwargs...)
        solve(prob, int_alg; kwargs...)
    end
    mapreduce(solve_int, hcat, rho0s)
end
(m::SciMLOperators.AbstractSciMLOperator)(v) = m(v, SciMLBase.NullParameters(), 0.0)
function integrated_current(A, rho0s, tmax, get_current, alg::Exponentiation{EXP_krylovkit}; kwargs...)
    u0 = complex(zero(first(rho0s)))
    mapreduce(rho0 -> get_current(krylovkit_exponentiation(A, tmax, u0, rho0; kwargs...)), hcat, rho0s)
end
function integrated_current(A, rho0s, tmax, get_current, alg::Exponentiation{EXP_sciml}; maxiter=100, kwargs...)
    n = size(A, 1)
    wa = zeros(ComplexF64, n, 2)
    ksA = KrylovSubspace{complex(eltype(A))}(n, maxiter)
    mapreduce(rho0 -> get_current(sciml_exponentiation(A, tmax, rho0, (wa, ksA); kwargs...)), hcat, rho0s)
end

function sciml_exponentiation(A, tmax, vrho0, (wa, ksA), m=100; abstol, kwargs...)
    count = 0
    arnoldi!(ksA, A, vrho0; tol=abstol, m)
    sol, errest = phiv!(wa, tmax, ksA, 1; correct=true, errest=true)
    while errest > abstol && count < 2
        @warn "phiv! not converged. Increasing krylovdim and maxiter" count m errest
        count += 1
        m = 2 * m
        resize!(ksA, 2 * ksA.m)
        arnoldi!(ksA, A, vrho0; tol=abstol, m)
        sol, errest = phiv!(wa, tmax, ksA, 1; correct=true, errest=true)
    end
    tmax * sol[:, 2]
end
function krylovkit_exponentiation(A, tmax, u0, vrho0; abstol, krylovdim=50, maxiter=100, kwargs...)
    sol, info = expintegrator(A, tmax, u0, vrho0; tol=abstol, krylovdim, maxiter)
    count = 0
    while !(info.converged > 0) && count < 2
        @warn info krylovdim maxiter count
        count += 1
        krylovdim = 2 * krylovdim
        maxiter = 2 * maxiter
        sol, info = expintegrator(A, tmax, u0, vrho0; tol=abstol, krylovdim, maxiter)
    end
    @debug info
    sol
end

function integrated_current(A, rho0s, tmax, get_current, _alg::ODE; int_alg, ensemblealg=EnsembleThreads(), kwargs...)
    tspan = (zero(tmax), tmax)
    alg = _alg.alg
    u0 = first(rho0s)
    prob = ODEProblem(A, u0, tspan)
    function prob_func(prob, i, repeat)
        prob.u0 .= rho0s[i]
        prob
    end
    eprob = EnsembleProblem(prob;
        output_func=(sol, i) -> (integrate_current(sol, tmax, get_current; int_alg, kwargs...), false),
        prob_func, u_init=Matrix{Float64}(undef, length(get_current(u0)), 0),
        reduction=(u, data, I) -> (hcat(u, reduce(hcat, data)), false))
    solve(eprob, alg, ensemblealg; trajectories=length(rho0s), kwargs...).u

    # currents = Vector{Float64}[]
    # for rho0 in rho0s
    #     prob = ODEProblem(A, rho0, tspan)
    #     sol = solve(prob, alg; kwargs...)
    #     solInt = integrate_current(sol, tmax, get_current; int_alg, kwargs...)
    #     push!(currents, solInt.u)
    # end
    # reduce(hcat, currents)
end
using SciMLOperators
function integrated_current(A, rho0s, tmax, get_current, _alg::IntegratedODE; ensemblealg=EnsembleThreads(), int_alg=nothing, kwargs...)
    domain = (zero(tmax), tmax)
    alg = _alg.alg
    u0 = zero(complex(first(rho0s)))
    f = (v, u, p, t) -> (mul!(v, A, u); v .+= p)
    prob = ODEProblem(f, u0, domain, first(rho0s); kwargs...)
    function prob_func(prob, i, repeat)
        prob = remake(prob, p=rho0s[i])
        prob
    end
    eprob = EnsembleProblem(prob;
        output_func=(sol, i) -> (get_current(sol(tmax)), false),
        prob_func, u_init=Matrix{Float64}(undef, length(get_current(u0)), 0),
        reduction=(u, data, I) -> (hcat(u, reduce(hcat, data)), false))
    solve(eprob, alg, ensemblealg; trajectories=length(rho0s), kwargs...).u
end
function current_integrand(ts, A, vrho0, get_current; abstol, kwargs...)
    rhos = expv_timestep(ts, A, vrho0; tol=abstol, adaptive=true)
    stack([get_current(rho) for rho in eachcol(rhos)])
end
function get_current_time_trace(A, rho0s, tmax, get_current; alg=ROCK4(), ensemblealg=EnsembleThreads(), kwargs...)
    tspan = (zero(tmax), tmax)
    ts = range(first(tspan), last(tspan), 200)
    currents = Matrix{Float64}[]
    for rho0 in rho0s
        u0 = rho0
        prob = ODEProblem(A, u0, tspan)
        sol = solve(prob, alg; kwargs...)
        outsol = sol(first(tspan))
        push!(currents, stack([get_current(sol(outsol, t)) for t in ts]))
    end
    currents
end

function integrate_current(sol, tmax, get_currents; int_alg, kwargs...)
    outsol = sol(0.0)
    function f(t, outsol)#::Vector{Float64}
        get_currents(sol(outsol, t))
    end
    IntegralFunction(f)
    domain = (zero(tmax), tmax)
    prob = IntegralProblem(f, domain, outsol)
    sol = solve(prob, int_alg; kwargs...)
end

function measurement_matrix(reservoirs::Vector{<:Reservoir}, tmax, alg; kwargs...)
    sols = [_measurement_matrix(res, tmax, alg; kwargs...) for res in reservoirs]
    reduce(hcat, sols)
end
function _measurement_matrix(res::Reservoir, tmax, alg; kwargs...)
    current_ops = res.current_ops
    ls = res.ls[end]
    A = QuantumDots.LinearOperator(ls)'
    _mat = __measurement_matrix(A, tmax, current_ops, alg; kwargs...)
    Matrix(res.initials.wedgemap)' * _mat
end
function __measurement_matrix(A, tmax, current_ops, alg::Exponentiation{EXP_krylovkit}; kwargs...)
    mapreduce(op -> krylovkit_exponentiation(A, tmax, zero(op), op; kwargs...), hcat, current_ops)
end
function __measurement_matrix(A, tmax, current_ops, alg::Exponentiation{EXP_sciml}; m=50, kwargs...)
    n = size(A, 1)
    wa = zeros(ComplexF64, n, 2)
    ksA = KrylovSubspace{complex(eltype(A))}(n, m)
    mapreduce(op -> sciml_exponentiation(A, tmax, op, (wa, ksA); kwargs...), hcat, current_ops)
end
function __measurement_matrix(A, tmax, current_ops, alg::Exponentiation{EXP_sciml_full}; kwargs...)
    Abar = [Matrix(A) I; 0I 0I]
    Ufull = exponential!(Abar * tmax)
    n = size(A, 1)
    U = Ufull[1:n, n+1:end]
    mapreduce(Base.Fix1(*, U), hcat, current_ops)
end


function __measurement_matrix(A, tmax, current_ops, alg::IntegratedODE; ensemblealg=EnsembleThreads(), abstol, kwargs...)
    domain = (zero(tmax), tmax)
    u0 = zero(complex(first(current_ops)))
    f = (v, u, p, t) -> (mul!(v, A, u); v .+= p)
    prob = ODEProblem(f, u0, domain, first(current_ops); kwargs...)
    function prob_func(prob, i, repeat)
        prob = remake(prob, p=current_ops[i])
        prob
    end

    eprob = EnsembleProblem(prob;
        output_func=(sol, i) -> (sol(tmax), false),
        prob_func, u_init=Matrix{Float64}(undef, length(u0), 0),
        reduction=(u, data, I) -> (hcat(u, reduce(hcat, data)), false))
    solve(eprob, alg.alg, ensemblealg; abstol, trajectories=length(current_ops), kwargs...).u
end
function __measurement_matrix(A, tmax, current_ops, alg::ODE; int_alg, ensemblealg=EnsembleThreads(), abstol, kwargs...)
    domain = (zero(tmax), tmax)
    u0 = zero(complex(first(current_ops)))
    prob = ODEProblem(A, u0, domain; kwargs...)
    function prob_func(prob, i, repeat)
        prob.u0 .= current_ops[i]
        prob
    end
    eprob = EnsembleProblem(prob;
        output_func=(sol, i) -> (integrate_current(sol, tmax, identity; int_alg, kwargs...), false),
        prob_func, u_init=Matrix{Float64}(undef, length(u0), 0),
        reduction=(u, data, I) -> (hcat(u, reduce(hcat, data)), false))
    solve(eprob, alg.alg, ensemblealg; abstol, trajectories=length(current_ops), kwargs...).u
end

using RandomMatrices
struct DensityMatrixDistribution{D}
    dist::D
    DensityMatrixDistribution(dist::D) where {D} = new{D}(dist)
end
default_distribution(N) = DensityMatrixDistribution(Ginibre(2, N))
function Base.rand(d::DensityMatrixDistribution{<:Ginibre})
    X = rand(d.dist)
    _to_density_matrix(X)
end
function _to_density_matrix(X)
    rho = X' * X
    rho ./= tr(rho)
end

function random_inputs(rc::ReservoirConnections, M)
    cI = rc.bases.cI
    bd = blockdiagonal(I + 0first(cI), cI)
    sizes = size.(bd.blocks, 1)
    dists = [Ginibre(2, N) for N in sizes]
    [_to_density_matrix(BlockDiagonal(rand.(dists))) for n in 1:M]
end
function vecrep(rho, res::Reservoir)
    ls = res.ls.I0
    vecrep(rho, ls)
end
function vecrep(rho, res::Vector{<:Reservoir})
    vecrep(rho, first(res))
end

##
function get_training_data(rho)
    entropy = real(-tr(rho * log(Matrix(rho))))
    purity = real(tr(rho^2))
    rv = get_rho_vec(rho)
    return [entropy, purity, rv...]
end