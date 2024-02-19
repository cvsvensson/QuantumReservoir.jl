using QuantumDots, QuantumDots.BlockDiagonals, LinearAlgebra
using LinearAlgebra
using Plots
using LinearSolve
using ExponentialUtilities
using MLJLinearModels
using OrdinaryDiffEq
using Statistics
using Folds
using Random
using LinearMaps
# using DiffEqCallbacks
using Integrals, FastGaussQuadrature
includet("misc.jl")

Random.seed!(1234)
training_parameters = generate_training_parameters(1000);
validation_parameters = generate_training_parameters(1000);
#includet("gpu.jl")
##
struct DenseLindblad end
struct LazyLindblad end
struct ReservoirConnections{L,C,Cl}
    labels::L
    Ilabels::L
    Ihalflabels::L
    Rlabels::L
    hopping_labels::C
    Iconnections::C
    Rconnections::C
    IRconnections::C
    lead_connections::Cl
end
function ReservoirConnections(N, M=1)
    labels = vec(Base.product(0:N, 1:2) |> collect)
    hopping_labels = [(labels[k1], labels[k2]) for k1 in 1:length(labels), k2 in 1:length(labels) if k1 > k2 && is_nearest_neighbours(labels[k1], labels[k2])]
    Ilabels = filter(x -> first(x) <= 0, labels)
    Rlabels = filter(x -> first(x) > 0, labels)
    Ihalflabels = filter(x -> isone(x[2]), Ilabels)
    Iconnections = filter(k -> first(k[1]) + first(k[2]) == 0, hopping_labels)
    Rconnections = filter(k -> first(k[1]) + first(k[2]) > 1, hopping_labels)
    IRconnections = filter(k -> abs(first(k[1]) - first(k[2])) == 1, hopping_labels)
    lead_connections = [(m, [(N, k) for k in 1:2]) for m in 1:M]

    return ReservoirConnections(labels, Ilabels, Ihalflabels, Rlabels, hopping_labels, Iconnections, Rconnections, IRconnections, lead_connections)
end

function random_static_parameters(rc::ReservoirConnections)
    J = Dict((k1, k2) => 2(rand() - 0.5) for (k1, k2) in rc.hopping_labels)
    V = Dict((k1, k2) => rand() for (k1, k2) in rc.hopping_labels)
    Γ = Dict(reduce(vcat, [(m, l...) => 2(rand() - 0.5) for l in ls] for (m, ls) in rc.lead_connections))
    return J, V, Γ
end
scale_dict(d, s) = Dict(map(p -> p[1] => s * p[2], collect(d)))
scale_dicts(ds, ss) = map(scale_dict, ds, ss)

struct IntegratedQuantumReservoir{C,CI,CR,RC,Hf,L}
    c::C
    cI::CI
    cR::CR
    rc::RC
    Hfunc::Hf
    leads::L
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
Exponentiation() = Exponentiation(EXP_krylovkit())
# function get_leads(c, lead_connections, Ilabels, Γ, T, μ, μmin=-1000)
#     leads = Tuple(NormalLead(sum(c[N, k]' * Γ[(m, N, k)] for (N, k) in ls); T, μ=μ[m]) for (m, ls) in lead_connections)
#     input_dissipator = CombinedLead(Tuple(c[i]' for i in Ilabels); T, μ=μmin)
#     leads0 = (input_dissipator, leads...)
#     return leads0, leads
# end
function static_hamiltonian(c, rc::ReservoirConnections, (J, V), (sJ, sV))
    HRJ = sJ * hopping_hamiltonian(c, J; labels=rc.Rconnections)
    HIJ = sJ * hopping_hamiltonian(c, J; labels=rc.Iconnections)
    HIR = sJ * hopping_hamiltonian(c, J; labels=rc.IRconnections)
    HIV = sV * coulomb_hamiltonian(c, V; labels=rc.Iconnections)
    HRV = sV * coulomb_hamiltonian(c, V; labels=rc.Rconnections)
    HIV = sV * coulomb_hamiltonian(c, V; labels=rc.IRconnections)

    HI0 = HIJ + HIV
    HR0 = HRJ + HRV
    H = HR0 + HI0 + HIR
    return HI0, HR0, H
end
function initialize_reservoir(rc, (J, V, Γ), (sJ, sV, sΓ), (T, μ), μmin)
    qn = QuantumDots.fermionnumber
    cI = FermionBasis(rc.Ilabels; qn)
    cR = FermionBasis(rc.Rlabels; qn)
    c = wedge(cI, cR)#FermionBasis(rc.labels; qn)

    static_HI = sJ * hopping_hamiltonian(cI, J; labels=rc.Iconnections) + sV * coulomb_hamiltonian(cI, V; labels=rc.Iconnections)
    static_HR = sJ * hopping_hamiltonian(cR, J; labels=rc.Rconnections) + sV * coulomb_hamiltonian(cR, V; labels=rc.Rconnections)
    static_HIR = sJ * hopping_hamiltonian(c, J; labels=rc.hopping_labels) + sV * coulomb_hamiltonian(c, V; labels=rc.hopping_labels)
    static_HIR0 = sJ * hopping_hamiltonian(c, J; labels=rc.Iconnections) + sV * coulomb_hamiltonian(c, V; labels=rc.Iconnections) + sJ * hopping_hamiltonian(c, J; labels=rc.Rconnections) + sV * coulomb_hamiltonian(c, V; labels=rc.Rconnections)
    function get_hamiltonian(ϵ)
        HI = static_HI + qd_level_hamiltonian(cI, ϵ; labels=rc.Ilabels)
        HR = static_HR + qd_level_hamiltonian(cR, ϵ; labels=rc.Rlabels)
        Hϵ = qd_level_hamiltonian(c, ϵ)
        HIR = static_HIR + Hϵ
        HIR0 = static_HIR0 + Hϵ
        return HI, HR, HIR, HIR0
    end
    leadsR0 = length(cR) > 0 ? Tuple(NormalLead(sum(cR[N, k]' * Γ[(m, N, k)] for (N, k) in ls); T, μ=μ[m]) for (m, ls) in rc.lead_connections) : tuple()
    leads = Tuple(NormalLead(sum(c[N, k]' * Γ[(m, N, k)] for (N, k) in ls); T, μ=μ[m]) for (m, ls) in rc.lead_connections)
    leadsI0 = tuple(CombinedLead(Tuple(cI[i]' for i in rc.Ilabels); T, μ=μmin))
    leadsIR0 = length(cR) > 0 ? (CombinedLead(Tuple(c[i]' for i in rc.Ilabels); T, μ=μmin), leadsIR...) : (CombinedLead(Tuple(c[i]' for i in rc.Ilabels); T, μ=μmin),)
    # leads0, leads = get_leads(c, rc.lead_connections, rc.Ilabels, scale_dict(Γ, sΓ), T, μ, μmin)
    # _get_leads() = (leads0, leads)
    return IntegratedQuantumReservoir(c, cI, cR, rc, get_hamiltonian, (; leadsR0, leadsI0, leadsIR0, leads))
end

get_lindblad(H0, leads0, lindbladian::DenseLindblad) = LindbladSystem(H0, leads0)
get_lindblad(H0, leads0, lindbladian::LazyLindblad) = LazyLindbladSystem(H0, leads0)

function get_initial_state(res, ε, lindbladian; kwargs...)
    HI, HR, HIR, HIR0 = res.Hfunc(ε)
    (; leadsI0, leadsR0, leadsIR0, leads) = res.leads

    lsI = get_lindblad(HI, leadsI0, lindbladian)
    probI = StationaryStateProblem(lsI)
    rhointernalI = solve(probI, LinearSolve.KrylovJL_LSMR(); kwargs...)
    rhoI = QuantumDots.tomatrix(rhointernalI, lsI)
    normalize_rho!(rhoI)
    rhoIvec = vecrep(rhoI, lsI)

    if length(res.cR) > 0
        lsR = get_lindblad(HR, leadsR0, lindbladian)
        probR = StationaryStateProblem(lsR)
        rhointernalR = solve(probR, LinearSolve.KrylovJL_LSMR(); kwargs...)
        rhoR = QuantumDots.tomatrix(rhointernalR, lsR)
        normalize_rho!(rhoR)
        rhoRvec = vecrep(rhoR, lsR)
    else
        rhoRvec = [1]
        rhoR = [1;;]
        lsR = nothing
    end

    lsIR = get_lindblad(HIR, leads, lindbladian)
    rhoIR = wedge(rhoI, res.cI, rhoR, res.cR, res.c)
    rhoIRvec = vecrep(rhoIR, lsIR)
    lm = LinearMap{eltype(rhoIR)}(rhoIvec -> vecrep(wedge(QuantumDots.tomatrix(rhoIvec, lsI), res.cI, rhoR, res.cR, res.c), lsIR), length(rhoIRvec), length(rhoIvec))

    lsIR2 = get_lindblad(HIR0, leadsIR0, lindbladian)
    probIR2 = StationaryStateProblem(lsIR2)
    rhointernalIR2 = solve(probIR2, LinearSolve.KrylovJL_LSMR(); abstol=abstol)
    rhoIR2 = QuantumDots.tomatrix(rhointernalIR2, lsIR2)
    normalize_rho!(rhoIR2)
    if norm(rhoIR - rhoIR2) > 100 * (abstol)
        @warn "Inconsistent initial states" norm(rhoIR - rhoIR2) abstol
    end

    return (; I=(ls=lsI, rhovec=rhointernalI, rho=rhoI), R=(ls=lsR, rho=rhoR, rhovec=rhoRvec), IR=(ls=lsIR, rhovec=rhoIRvec, rhovec2=vecrep(rhoIR2, lsIR), rho=rhoIR), lmap=Matrix(lm), vecrep=Base.Fix2(vecrep, lsIR))
end


function run_reservoir(res::IntegratedQuantumReservoir, ε, initial_state_parameters, tmax, alg; time_trace=false, lindbladian=DenseLindblad(), kwargs...)
    # H0, H = res.Hfunc(ε)
    # leads0, leadsR,leads = res.leadsfunc()
    c = res.c
    particle_number = blockdiagonal(numberoperator(c), c)

    # ls0 = get_lindblad(H0, leads0, lindbladian)
    # # ls0 = LindbladSystem(H0, leads0)
    # prob0 = StationaryStateProblem(ls0)
    # rhointernal0 = solve(prob0, LinearSolve.KrylovJL_LSMR(); abstol=ss_abstol)
    # rho0 = QuantumDots.tomatrix(rhointernal0, ls0)
    # normalize_rho!(rho0)
    # # rhointernal0 = vecrep(rho0, ls0)
    # rhointernal0 = QuantumDots.internal_rep(rho0, ls0)
    # @assert isapprox(tr(rho0), 1; atol=1e-3) "tr(ρ) = $(tr(rho0)) != 1"
    # # ls = LindbladSystem(H, leads)
    # ls = get_lindblad(H, leads, lindbladian)

    initials = get_initial_state(res, ε, lindbladian; kwargs...)
    ls = initials.IR.ls
    internal_N = QuantumDots.internal_rep(particle_number, ls)
    current_ops = map(diss -> vecrep((diss' * internal_N), ls), ls.dissipators)
    rho0vec = initials.IR.rhovec
    rho0mat = QuantumDots.tomatrix(rho0vec, ls)
    #let's compare with measurement evolution
    # A = QuantumDots.LinearOperator(ls)'
    # mm = mapreduce(op -> solve_int(A, tmax, op; abstol), hcat, current_ops)

    rho0s = generate_initial_states(initial_state_parameters, rho0mat, c)
    rhoI0s = generate_initial_states(initial_state_parameters, initials.I.rho, res.cI)
    data = training_data(rho0s, res.c, res.rc.Ihalflabels, res.rc.Ilabels)
    dataI = training_data(rhoI0s, res.cI, res.rc.Ihalflabels, res.rc.Ilabels)
    ens = InitialEnsemble{typeof(rho0s),typeof(data)}(rho0s, data)
    ensI = InitialEnsemble{typeof(rhoI0s),typeof(dataI)}(rhoI0s, dataI)
    vrhoI0s = map(rho -> vecrep(rho, initials.I.ls), rhoI0s)
    vecensI = InitialEnsemble{typeof(vrhoI0s),typeof(dataI)}(vrhoI0s, dataI)
    results = integrated_current(ls, ens, tmax, current_ops, alg; kwargs...)
    current = time_trace ? get_current_time_trace(ls, ens, tmax, current_ops; kwargs...) : missing

    # integrated2 = mapreduce(rho -> mm' * vecrep(rho, ls), hcat, rho0s)
    return (; integrated=results, ensemble=ens, current, ls, ensembleI=ensI, vecensembleI=vecensI, initials)
end


function integrated_current(ls, ens::InitialEnsemble, tmax, current_ops, solver::Sampled; int_alg, kwargs...)
    grid = solver.grid
    @assert tmax ≈ grid[end]
    function solve_int(vrho0)
        vals = current_integrand(grid, ls, vrho0, current_ops; kwargs...)
        prob = SampledIntegralProblem(vals, grid; kwargs...)
        solve(prob, int_alg; kwargs...)
    end
    mapreduce(solve_int ∘ Base.Fix2(vecrep, ls), hcat, ens.rho0s)
end
(m::SciMLOperators.AbstractSciMLOperator)(v) = m(v, SciMLBase.NullParameters(), 0.0)
function integrated_current(ls, ens::InitialEnsemble, tmax, current_ops, alg::Exponentiation{EXP_krylovkit}; kwargs...)
    A = QuantumDots.LinearOperator(ls; normalizer=false)
    u0 = complex(zero(vecrep(first(ens.rho0s), ls)))
    mapreduce(rho0 -> [real(krylovkit_exponentiation(A, tmax, u0, vecrep(rho0, ls); abstol)' * op) for op in current_ops], hcat, ens.rho0s)
end
function integrated_current(ls, ens::InitialEnsemble, tmax, current_ops, alg::Exponentiation{EXP_sciml}; kwargs...)
    A = QuantumDots.LinearOperator(ls; normalizer=false)
    n = size(A, 1)
    wa = zeros(ComplexF64, n, 2)
    maxiter = 100
    ksA = KrylovSubspace{complex(eltype(A))}(n, maxiter)
    mapreduce(rho0 -> [real(sciml_exponentiation(A, tmax, vecrep(rho0, ls), (wa, ksA); abstol)' * op) for op in current_ops], hcat, ens.rho0s)
end

function sciml_exponentiation(A, tmax, vrho0, (wa, ksA), m=50; abstol)
    count = 0
    arnoldi!(ksA, A, vrho0; tol=abstol, m)
    sol, errest = phiv!(wa, tmax, ksA, 1; errest=true)
    out = tmax * sol[:, 2]
    while errest > abstol && count < 10
        count += 1
        m = 2 * m
        arnoldi!(ksA, A, vrho0; tol=abstol, m)
        sol, errest = phiv!(wa, tmax, ksA, 1; errest=true)
        out = tmax * sol[:, 2]
        @warn count m errest
    end
    out
end
function krylovkit_exponentiation(A, tmax, u0, vrho0; abstol, krylovdim=50, maxiter=100)
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

function integrated_current(ls, ens::InitialEnsemble, tmax, current_ops, _alg::ODE; int_alg, ensemblealg=EnsembleThreads(), kwargs...)
    tspan = (0, tmax)
    alg = _alg.alg
    u0 = vecrep(first(ens.rho0s), ls)
    A = QuantumDots.LinearOperator(ls)
    # prob = ODEProblem(A, u0, tspan)
    # function prob_func(prob, i, repeat)
    #     prob.u0 .= vecrep(ens.rho0s[i], ls)
    #     prob
    # end
    # eprob = EnsembleProblem(prob;
    #     output_func=(sol, i) -> (integrated_current(sol, tmax, current_ops; int_alg, kwargs...), false),
    #     prob_func, u_init=Matrix{Float64}(undef, length(current_ops), 0),
    #     reduction=(u, data, I) -> (hcat(u, reduce(hcat, data)), false))
    # solve(eprob, alg, ensemblealg; trajectories=length(ens.rho0s), kwargs...).u
    currents = Vector{Float64}[]
    for rho0 in ens.rho0s
        u0 = vecrep(rho0, ls)
        prob = ODEProblem(A, u0, tspan)
        sol = solve(prob, alg; abstol, kwargs...)
        solInt = integrated_current(sol, tmax, current_ops; int_alg, kwargs...)
        push!(currents, solInt.u)
    end
    reduce(hcat, currents)
end
function integrated_current(ls, ens::InitialEnsemble, tmax, current_ops, _alg::IntegratedODE; ensemblealg=EnsembleThreads(), kwargs...)
    domain = (zero(tmax), tmax)
    alg = _alg.alg
    u0 = vecrep(first(ens.rho0s), ls)
    A = QuantumDots.LinearOperator(ls)
    u0 = zero(complex(vecrep(first(ens.rho0s), ls)))
    sols = []
    for rho0 in ens.rho0s
        rho0V = vecrep(rho0, ls)
        prob = SplitODEProblem{true}(A, (v, u, p, t) -> v .= rho0V, u0, domain; kwargs...)
        sol = solve(prob, alg; abstol, kwargs...)
        push!(sols, [real(sol(tmax)' * op) for op in current_ops])
    end
    reduce(hcat, sols)
    # function prob_func(prob, i, repeat)
    #     prob.u0 .= vecrep(ens.rho0s[i], ls)
    #     prob
    # end
    # eprob = EnsembleProblem(prob;
    #     output_func=(sol, i) -> ([real(sol(tmax)' * op) for op in current_ops], false),
    #     prob_func, u_init=Matrix{Float64}(undef, length(current_ops), 0),
    #     reduction=(u, data, I) -> (hcat(u, reduce(hcat, data)), false))

    # solve(eprob, alg, ensemblealg; trajectories=length(ens.rho0s), kwargs...).u
end

function current_integrand(ts, ls, vrho0, current_ops; abstol, kwargs...)
    A = QuantumDots.LinearOperator(ls)
    rhos = expv_timestep(ts, A, vrho0; tol=abstol, adaptive=true)
    [real(tr((rho)' * op)) for op in current_ops, rho in eachcol(rhos)]
end
function get_current_time_trace(ls, ens::InitialEnsemble, tmax, current_ops; alg=ROCK4(), ensemblealg=EnsembleThreads(), abstol, kwargs...)
    tspan = (0, tmax)
    u0 = vecrep(first(ens.rho0s), ls)
    A = QuantumDots.LinearOperator(ls)

    #Something wrong with threading for the ensemble? Every solution is the same. 
    # prob = ODEProblem(A, u0, tspan)
    # function prob_func(prob, i, repeat)
    #     prob.u0 .= vecrep(ens.rho0s[i], ls)
    #     prob
    # end
    ts = range(0, tmax, 200)
    # eprob = EnsembleProblem(prob;
    #     output_func=(sol, i) -> ([real(tr(sol(t)' * op)) for t in ts, op in current_ops], false),
    #     reduction=(u, data, I) -> (append!(u, data), false))
    # solve(eprob, alg, ensemblealg; trajectories=length(ens.rho0s), abstol, kwargs...)

    currents = Matrix{Float64}[]
    for rho0 in ens.rho0s
        u0 = vecrep(rho0, ls)
        prob = ODEProblem(A, u0, tspan)
        sol = solve(prob, alg; abstol, kwargs...)
        push!(currents, [real(tr(sol(t)' * op)) for t in ts, op in current_ops])
    end
    currents
end

function run_reservoir_ensemble(res::IntegratedQuantumReservoir, εs, initial_state_parameters, tmax, alg; kwargs...)
    sols = [run_reservoir(res, ε, initial_state_parameters, tmax, alg; kwargs...) for ε in εs]
    integrated = mapreduce(x -> x.integrated, vcat, sols)
    ensemble = first(sols).ensemble
    ensembleI = first(sols).ensembleI
    vecensembleI = first(sols).vecensembleI
    time_traces = map(x -> x.current, sols)

    other_keys = [key for key in keys(sols[1]) if !(key ∈ [:integrated, :ensemble, :ensembleI, :vecensembleI, :time_traces])]
    nt = NamedTuple([key => map(sol -> sol[key], sols) for key in other_keys])
    return merge((; integrated, ensemble, ensembleI, vecensembleI, time_traces), nt)
end
function integrated_current(sol, tmax, current_ops; int_alg, kwargs...)
    outsol = sol(0.0)
    # T = promote_type(eltype(outsol), eltype(eltype(current_ops)))
    function f(t, outsol)::Vector{Float64}
        [real((sol(outsol, t)' * op)) for op in current_ops]
    end
    IntegralFunction(f)
    domain = (zero(tmax), tmax)
    prob = IntegralProblem(f, domain, outsol)
    sol = solve(prob, int_alg; kwargs...)
end

function measurement_matrix(res::IntegratedQuantumReservoir, εs, tmax, alg; kwargs...)
    sols = [_measurement_matrix(res, ε, tmax, alg; kwargs...) for ε in εs]
    mat = mapreduce(sol -> sol.mat, vcat, sols)
    other_keys = [key for key in keys(sols[1]) if !(key ∈ [:mat])]
    nt = NamedTuple([key => map(sol -> sol[key], sols) for key in other_keys])
    return merge((; mat), nt)
end
function _measurement_matrix(res::IntegratedQuantumReservoir, ε, tmax, alg; lindbladian=DenseLindblad(), abstol, kwargs...)
    c = res.c
    particle_number = blockdiagonal(numberoperator(c), c)
    initials = get_initial_state(res, ε, lindbladian; abstol)

    ls = initials.IR.ls
    internal_N = QuantumDots.internal_rep(particle_number, ls)
    current_ops = map(diss -> vecrep(diss' * internal_N, ls), ls.dissipators)
    # projector = vecrep(input_N .== 1, ls)
    A = QuantumDots.LinearOperator(ls)'
    mat = __measurement_matrix(A, tmax, current_ops, alg; abstol, kwargs...)
    mat = mat' * Matrix(initials.lmap)
    return (; mat, current_ops, initials)
end
function __measurement_matrix(A, tmax, current_ops, alg::Exponentiation{EXP_krylovkit}; kwargs...)
    mapreduce(op -> krylovkit_exponentiation(A, tmax, zero(op), op; abstol), hcat, current_ops)
end
function __measurement_matrix(A, tmax, current_ops, alg::Exponentiation{EXP_sciml}; m=50, kwargs...)
    n = size(A, 1)
    wa = zeros(ComplexF64, n, 2)
    ksA = KrylovSubspace{complex(eltype(A))}(n, m)
    mapreduce(op -> sciml_exponentiation(A, tmax, op, (wa, ksA); abstol), hcat, current_ops)
end
function __measurement_matrix(A, tmax, current_ops, alg::Exponentiation{EXP_sciml_full}; kwargs...)
    Abar = [Matrix(A) I; 0I 0I]
    Ufull = exponential!(Abar * tmax)
    n = size(A, 1)
    U = Ufull[1:n, n+1:end]
    mapreduce(Base.Fix1(*, U,), hcat, current_ops)
end


function __measurement_matrix(A, tmax, current_ops, alg::IntegratedODE; abstol, kwargs...)
    sols = []
    domain = (zero(tmax), tmax)
    for op in current_ops
        prob = SplitODEProblem{true}(A, (v, u, p, t) -> v .= op, zero(complex(op)), domain; kwargs...)
        sol = solve(prob, alg.alg; abstol, kwargs...)
        push!(sols, sol(tmax))
    end
    reduce(hcat, sols)
end
function __measurement_matrix(A, tmax, current_ops, alg::ODE; int_alg, abstol, kwargs...)
    # ts = range(0, tmax, 200)
    sols = []
    domain = (zero(tmax), tmax)
    for op in current_ops
        prob = ODEProblem(A, complex(op), domain)
        sol = solve(prob, alg.alg; abstol, kwargs...)
        outsol = sol(0.0)
        probInt = IntegralProblem((u, outsol) -> sol(outsol, u), domain, outsol)
        solInt = solve(probInt, int_alg; abstol, kwargs...)
        push!(sols, solInt)
    end
    reduce(hcat, sols)
end

##
reconstructed_data(W, X, i=:) = size(W, 2) > size(X, 1) ? W[i, 1:end-1] * X + W[i, end] * ones(1, size(X, 2)) : W[i, :] * X
function fit_output_layer(training_sols; β=1e-6, fit_intercept=true)
    X = training_sols.integrated
    y = training_sols.ensemble.data
    ridge = RidgeRegression(β; fit_intercept)
    W = reduce(hcat, map(data -> fit(ridge, X', data), eachrow(y))) |> permutedims
    return W
end
get_density_matrix(obs) = get_rho_mat(obs[3:end])
function loss_function_density_matrix(W, test_sols)
    y_pred = reconstructed_data(W, test_sols.integrated)
    y_test = test_sols.ensemble.data
    losses = map(norm ∘ get_density_matrix ∘ -, eachcol(y_pred), eachcol(y_test))
    mean(losses)
end
##
μmin = -1e5
μs = -1 .* [1, 1]
T = 10
Nres_layers = 0
Nres = 50
Nmeas = 5
Nleads = length(μs) # Make this more general. Labels to connect leads.
rc = ReservoirConnections(Nres_layers, length(μs))
reservoir_parameters = [random_static_parameters(rc) for n in 1:Nres] #Many random reservoirs to get statistics of performance
qd_level_measurements = [Dict(l => 5 * (rand() - 0.5) for l in rc.labels) for i in 1:Nmeas] #Each configuration of dot levels gives a measurement
##
reservoir = initialize_reservoir(rc, reservoir_parameters[1], (1, 1, 1), (T, μs), μmin)
M_train = 20
M_val = 100
tmax = 100
abstol = 1e-12
reltol = 1e-12
grid = let n = 4
    (range(0, tmax^(1 // n), 200)) .^ n
end
# int_alg = QuadGKJL(; order=2)
# int_alg = HCubatureJL()
int_alg = GaussLegendre()
alg = DP8()
alg = Exponentiation()
lindbladian = DenseLindblad()
##
@time training_sols = run_reservoir_ensemble(reservoir, qd_level_measurements, training_parameters[1:M_train], tmax; abstol, reltol, alg=DP8(), int_alg, lindbladian);
@time training_sols2 = measurement_matrix(reservoir, qd_level_measurements, tmax; abstol, reltol);
@time training_sols3 = run_reservoir_ensemble(reservoir, qd_level_measurements, training_parameters[1:M_train], tmax; abstol, reltol, alg=Exponentiation(), int_alg, lindbladian);

inte = mapreduce(rho0 -> map(Base.Fix1(dot, rho0), eachrow(training_sols2.mat)), hcat, training_sols.vecensembleI.rho0s) |> real
training_sols.integrated
inte - training_sols.integrated .|> abs |> maximum
inte - training_sols3.integrated .|> abs |> maximum

##
@time training_sols3 = run_reservoir_ensemble(reservoir, qd_level_measurements, training_parameters[1:M_train], tmax; abstol, reltol, alg=Exponentiation(), int_alg, lindbladian);
@profview training_sols = run_reservoir_ensemble(reservoir, qd_level_measurements, training_parameters[1:3], tmax; abstol, reltol, alg=Exponentiation(), int_alg);
@profview training_sols = run_reservoir_ensemble(reservoir, qd_level_measurements, training_parameters[1:3], tmax; abstol, reltol, alg=DP8(), int_alg, ensemblealg=EnsembleSerial());

@profview run_reservoir_ensemble(reservoir, qd_level_measurements, training_parameters[1:2], tmax; abstol, reltol, alg, int_alg, ensemblealg=EnsembleSerial());

@time s1 = run_reservoir_ensemble(reservoir, qd_level_measurements, training_parameters[1:2], tmax; abstol=1e-1, reltol=1e-1, alg, int_alg, ensemblealg=EnsembleSerial());
@time s3 = run_reservoir_ensemble(reservoir, qd_level_measurements, training_parameters[1:2], tmax; abstol, reltol, alg=Sampled(grid), int_alg=SimpsonsRule(), ensemblealg=EnsembleSerial());

#@profview training_sols = run_reservoir_ensemble(reservoir, qd_level_measurements, training_parameters[1:M_train], tmax; abstol, reltol, alg, int_alg);
#@time run_reservoir_ensemble(reservoir, qd_level_measurements, validation_parameters[1:M_val], tmax; abstol, ensemblealg=EnsembleSerial());
@profview run_reservoir_ensemble(reservoir, qd_level_measurements, validation_parameters[1:M_val], tmax; abstol, alg, int_alg, ensemblealg=EnsembleSerial());
@time test_sols = run_reservoir_ensemble(reservoir, qd_level_measurements, validation_parameters[1:M_val], tmax; abstol, reltol, alg, int_alg);
@time time_sols = run_reservoir(reservoir, qd_level_measurements[1], training_parameters[1:3], tmax; abstol, reltol, time_trace=true, alg=DP8(), int_alg, ensemblealg=EnsembleSerial());
##
foreach(c -> plot(c) |> display, time_sols.current)
##
W = fit_output_layer(training_sols)
loss_function_density_matrix(W, test_sols)
##
data = []
for params in reservoir_parameters[1:20]
    reservoir = initialize_reservoir(rc, params, (1, 1, 1), (T, μs), μmin)
    @time training_sols = run_reservoir_ensemble(reservoir, qd_level_measurements, training_parameters[1:M_train], tmax; abstol)
    @time test_sols = run_reservoir_ensemble(reservoir, qd_level_measurements, validation_parameters[1:100], tmax; abstol)
    W = fit_output_layer(training_sols)
    mm = measurement_matrix(reservoir, qd_level_measurements, tmax; abstol)
    c = cond(mm.mat)
    loss = loss_function_density_matrix(W, test_sols)
    push!(data, (; loss, cond=c, mm.mat, W))
end
plot(data, ylims=(0, 0.5))
##
plot([(datacond0), data0 * norm(datacond0)])
plot([datacond, norm(datacond) * inv.(data) / 10000])
let d0 = data0, d1 = data, f = log
    plot([f.(sort(d0)) f.(sort(d1))])
end
##
data = []
for sV in range(0, 5, 5)
    localdata = []
    for params in reservoir_parameters
        reservoir = initialize_reservoir(rc, params, (1, sV, 1), (T, μs), μmin)
        @time training_sols = run_reservoir_ensemble(reservoir, qd_level_measurements, training_parameters[1:M_train], tmax; abstol)
        @time test_sols = run_reservoir_ensemble(reservoir, qd_level_measurements, validation_parameters[1:M_val], tmax; abstol)
        W = fit_output_layer(training_sols)
        push!(localdata, loss_function_density_matrix(W, test_sols))
    end
    push!(data, [mean(localdata), std(localdata)])
end
##
plot(first.(data), yerr=last.(data), ylims=(0, 0.5))

##
#plot(range(0, 1, 5), map(x -> x.mean, data), yerr=map(x -> x.std, data), ylims=(0, 0.5))

## Training
# X .+= 0randn(size(X)) * 1e-3 * mean(abs, X)
X = training_sols.integrated
y = training_sols.ensemble.data
ridge = RidgeRegression(1e-6; fit_intercept=true)
W0 = fit_output_layer(training_sols)
W1 = reduce(hcat, map(data -> fit(ridge, X', data), eachrow(y))) |> permutedims
W2 = y * X' * inv(X * X' + 1e-8 * I)
W3 = y * pinv(X)
##
titles = ["entropy of one input dot", "purity of inputs", "ρ11", "ρ22", "ρ33", "ρ44", "real(ρ23)", "imag(ρ23)"]
let is = 2:8, perm, W = W0, X = test_sols.integrated, y = test_sols.ensemble.data, b
    p = plot(; size=1.2 .* (600, 400))
    colors = cgrad(:seaborn_dark, size(y, 1))
    # colors2 = cgrad(:seaborn_dark, size(y, 2))
    colors2 = cgrad(:seaborn_bright, size(y, 1))
    y_pred = reconstructed_data(W, X)
    for i in is
        perm = sortperm(y[i, :])
        plot!(p, y_pred[i, perm]; label=titles[i] * " pred", lw=3, c=colors[i], frame=:box)
        # plot!(p, (Wi' * X[:, perm])' .+ b; label=titles[i] * " pred", lw=3, c=colors[i], frame=:box)
        plot!(y[i, perm]; label=titles[i] * " truth", lw=3, ls=:dash, c=colors2[i])
    end
    display(p)
end

##
function loss_function(W, X, y)
    loss = 0.0
    M = size(y, 2)
    for i in axes(W, 1)
        Wi, b = size(W, 2) > size(X, 1) ? (W[i, 1:end-1], W[i, end] * ones(M)) : (W[i, :], zeros(M))
        loss += sum(abs2, (Wi' * X)' .- y[i, :] .+ b)
    end
    return sqrt(loss) #/ size(X, 2)
end
loss_function(W1, training_sols.data, training_ensemble.data)
loss_function(W1, test_sols.data, test_ensemble.data)
##

function get_loss_function(c, J, V, ε, Γ, M, training_parameters, validation_parameters)
    HR = hopping_hamiltonian(c, J; labels=Rconnections)
    HI = hopping_hamiltonian(c, J; labels=Iconnections)
    HIR = hopping_hamiltonian(c, J; labels=IRconnections)
    HV = coulomb_hamiltonian(c, V; labels=hopping_labels)
    Hqd = qd_level_hamiltonian(c, ε)
    function _loss_function(v, ϵ, μs, γ, tend)
        H0 = HR + HI + v * HV + ϵ * Hqd
        H = H0 + HIR
        Γ = γ * Γ
        T = 10norm(Γ)
        input_dissipator = CombinedLead(Tuple(c[i]' for i in Ilabels); T, μ=-1e5)
        leads = Tuple(CombinedLead((c[N, k]' * Γ[k, k] + c[N, mod1(k + 1, 2)]' * Γ[k, mod1(k + 1, 2)],); T, μ=μs[k]) for k in 1:1)
        leads0 = (input_dissipator, leads...)
        reservoir = IntegratedQuantumReservoir(H0, H, leads0, leads, c, Ilabels, Rlabels)
        training_ensemble = InitialEnsemble(training_parameters[1:M], reservoir)
        test_ensemble = InitialEnsemble(validation_parameters[1:M], reservoir)
        t_obs = range(tend / 50, tend, 20)
        training_sols = time_evolve(reservoir, training_ensemble, t_obs)
        test_sols = time_evolve(reservoir, test_ensemble, t_obs)
        X = training_sols.data
        y = training_ensemble.data
        ridge = RidgeRegression(1e-6; fit_intercept=true)
        W1 = reduce(hcat, map(data -> fit(ridge, X', data), eachrow(y))) |> permutedims
        return loss_function(W1, test_sols.data, test_ensemble.data)
    end
end
##
fl = get_loss_function(c, J, V, ε, Γ, 100, training_parameters, validation_parameters)
@time fl(5, 0, 10, 1, 20)
plot([fl(v, 0.0, 0.0, 1.0, 10.0) for v in range(0, 100, 10)])
plot([fl(6.0, v, 100.0, 1.0, 10.0) for v in range(-1, 1, 10)])
plot([fl(6.0, 0.0, v, 1.0, 10.0) for v in range(0, 10, 10)])
plot([fl(6.0, 0.0, 100.0, 1.0, v) for v in range(0.5, 10, 10)])
##
using Optimization, OptimizationBBO, OptimizationOptimJL
# (v, ϵ, μs, γ, tend)
prob = OptimizationProblem((u, p) -> fl(u...), [1.0, 1.0, 1.0, 1.0, 10.0]; lb=Float64[0, -10, -10, 0.5, 1], ub=Float64[10, 10, 100, 2, 40])
sol = solve(prob, ParticleSwarm(); maxiters=1000, maxtime=20, show_trace=true)
##
prob = OptimizationProblem((u, p) -> fl(u..., 1, 40), [1.0, 1.0, 1.0]; lb=Float64[0, -10, -10], ub=Float64[10, 10, 100])
sol1 = solve(prob, ParticleSwarm(); maxiters=1000, maxtime=20, show_trace=true)
# sol2 = solve(prob, BBO_probabilistic_descent(); maxiters=1000, maxtime=20, show_trace=true)
# sol3 = solve(prob, BBO_adaptive_de_rand_1_bin_radiuslimited(); maxiters=1000, maxtime=20, show_trace=true)

##
sol = solve(OptimizationProblem((x, p) -> norm(x .^ 2), [1.0, 2.0, 3.0]), ParticleSwarm(); maxiters=100, maxtime=1, show_trace=true)
## Analyze initial state
rhod0 = diag(rho0)
@assert isapprox(tr(rho0), 1; atol=1e-3)
internal_N = QuantumDots.internal_rep(particle_number, ls0)
currents_ops = map(diss -> diss' * internal_N, ls0.dissipators)
currents = map(diss -> tr(QuantumDots.tomatrix(diss * rhointernal0, ls0) * particle_number), ls0.dissipators)
currents2 = map(op -> rhointernal0' * op, currents_ops)
@assert norm(map(-, currents, currents2)) / norm(currents) < 1e-2

##
cR = FermionBasis(Rlabels; qn)
cI = FermionBasis(Ilabels; qn)
rhoI0 = partial_trace(rho0, Ilabels, c, cI.symmetry)
rhoR0 = partial_trace(rho0, Rlabels, c, cR.symmetry)
@assert isapprox(rhoI0[1], 1; atol=1e-3)
@assert isapprox(tr(rhoI0), 1; atol=1e-3)
pretty_print(rhoI0, cI)
pretty_print(rhoR0, cR)

##
# ls = QuantumDots.LazyLindbladSystem(H, leads)
ls = QuantumDots.LindbladSystem(H, leads)
internal_N = QuantumDots.internal_rep(particle_number, ls)
current_ops = map(diss -> diss' * internal_N, ls.dissipators)
R_occ_ops = map(k -> QuantumDots.internal_rep(c[k]' * c[k], ls), Rlabels)
I_occ_ops = map(k -> c[k]' * c[k], Ilabels)

## Entanglement non-linearity
perm = sortperm(training_ensemble.data[1, :])
entropies = training_ensemble.data[1, perm];
rhos = training_ensemble.rho0s[perm];
ent1 = [input_entanglement((r1 + r2) / 2) for (n1, r1) in enumerate(rhos), (n2, r2) in enumerate(rhos)]
ent2 = [(entropies[n1] + entropies[n2]) / 2 for (n1, r1) in enumerate(rhos), (n2, r2) in enumerate(rhos)]

heatmap(ent1)
heatmap(ent2)