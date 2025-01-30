using QuantumDots, QuantumDots.BlockDiagonals
using LinearAlgebra
using Random
using OrdinaryDiffEqTsit5
using LinearSolve
using Plots
using Statistics
using MLJLinearModels
using ExponentialUtilities
using KrylovKit
using UnPack
Random.seed!(1234)
includet("..\\system.jl")
includet("src.jl")
includet("narma.jl")
includet("training.jl")
includet("plots.jl")
##
N = 4
labels = 1:N
qn = FermionConservation()
c = FermionBasis(labels; qn)
# number_operator = sum([c[l]'c[l] for l in labels])
hopping_labels = [(labels[k1], labels[k2]) for k1 in 1:length(labels), k2 in 1:length(labels) if k1 > k2] #&& is_nearest_neighbours(labels[k1], labels[k2])]

##
rand_initial_state = rand(ComplexF64, 2^length(labels), 2^length(labels)) |> (x -> x'x) |> (x -> x ./ tr(x))
##
function fully_connected_hopping(labels)
    [(labels[k1], labels[k2]) for k1 in 1:length(labels), k2 in 1:length(labels) if k1 > k2]
end
function rand_reservoir_params(fermionlabels, leadlabels=fermionlabels, hopping_labels=fully_connected_hopping(fermionlabels); Jscale=1, Vscale=1, Γscale=1, εscale=1, Γmin=0.1)
    J = Dict((k1, k2) => Jscale * 2(rand() - 0.5) for (k1, k2) in hopping_labels)
    V = Dict((k1, k2) => Vscale * rand() for (k1, k2) in hopping_labels)
    ε = Dict(l => εscale * (rand() - 0.5) for l in fermionlabels)
    Γ = Dict(l => Γscale * (rand() + Γmin) for l in leadlabels)
    (; J, V, ε, Γ)
end
function hamiltonian(params)
    Ht = hopping_hamiltonian(c, params.J)
    HV = coulomb_hamiltonian(c, params.V)
    Hqd = qd_level_hamiltonian(c, params.ε)
    Ht + HV + Hqd
end

##
anims = []
results = []
for seed in 1:1
    # seed = 3
    Random.seed!(seed)
    tfinal = 50
    tspan = (0, tfinal)
    N = 100
    ts = range(tspan..., N)

    signal = sin
    targets = [sin, (x -> sin(x - 1)), x -> sin(x - 1)^2]
    targetnames = ["sin", "sin(t-1)", "sin(t-1)^2"]
    params = rand_reservoir_params(labels)
    H = hamiltonian(params)
    temperature = 2sum(values(params.Γ))
    leads = Dict(l => NormalLead(c[l]' * params.Γ[l]; T=temperature, μ=0.0) for l in labels)
    # mask = Dict(l => l^2 * 10 * (rand() - 0.5) for l in keys(leads))
    mask = Dict(l => (l > 1) * l^2 * 0.5 for l in keys(leads))
    input = ContinuousInput(VoltageWrapper(MaskedInput(mask, signal)))
    ls = LindbladSystem(H, leads; usecache=true)
    lazyls = LazyLindbladSystem(H, leads)
    ode_kwargs = (; abstol=1e-6, reltol=1e-6)
    sol = solve(odeproblem(lazyls, input, tspan), Tsit5(); ode_kwargs...)
    sol2 = solve(odeproblem(lazyls, input, tspan, rand_initial_state), Tsit5(); ode_kwargs...)
    currents = [get_currents(sol, lazyls, input, t, number_operator) for t in ts]
    spectrum = [get_spectrum(ls, input, t) for t in ts]
    overlaps = [abs(dot(sol(t), sol2(t))) / (norm(sol(t)) * norm(sol2(t))) for t in ts]

    n_train_first = (x -> isnothing(x) ? findfirst(ts .> tfinal * 1 / 3) : x)(findfirst(overlaps .> 0.99))
    n_test_first = max(n_train_first + 1, findfirst(ts .> tfinal * 2 / 3))
    n_train = n_train_first:n_test_first-1
    n_test = n_test_first:length(ts)

    Xtrain = permutedims(stack(currents[n_train]))
    ytrain = stack([target.(ts[n_train]) for target in targets])
    Xtest = permutedims(stack(currents[n_test]))
    ytest = stack([target.(ts[n_test]) for target in targets])
    W = fit(ytrain, Xtrain)

    ztrain = predict(W, Xtrain)#[W[:, 1:end-1] * Xtrain .+ W[:, end] for W in Ws]
    ztest = predict(W, Xtest)#[W[:, 1:end-1] * Xtest .+ W[:, end] for W in Ws]
    mses = [mean((ytest .- ztest) .^ 2) for (ytest, ztest) in zip(eachcol(ytest), eachcol(ztest))]
    memory_capacities = [(cov(ztest[:], ytest[:]) / (std(ztest) * std(ytest)))^2 for (ytest, ztest) in zip(eachcol(ytest), eachcol(ztest))]
    evals = eigvals(Matrix(H))
    gapratios = map(x -> x > 1 ? inv(x) : x, diff(evals)[1:end-1] ./ diff(evals)[2:end])
    average_gapratio = mean(gapratios)
    ediffs = QuantumDots.commutator(Diagonal(evals)).diag
    ##
    result = (; seed, spectrum, currents, sol, ts, input, H, sol2, temperature, evals, params, ztrain, ztest, targets, targetnames, mses, memory_capacities, overlaps, n_train_first, n_test_first, n_train, n_test, ediffs, average_gapratio, W, leads)
    push!(results, result)
end
##
summary_gif(results[1])
##
evals = stack([eigvals(Matrix(hamiltonian(rand_reservoir_params(labels, Vscale=2)))) for _ in 1:10]);
plot(evals, markers=true)

## try discrete task
anims = []
results = []
# for seed in 1:1
seed = 3
Random.seed!(seed)
tfinal = 50
tspan = (0, tfinal)
N = 100
input_ts = range(tspan..., N + 1)[1:end-1]
time_multiplexing = 3
# measurement_ts = range(tspan..., N * time_multiplexing + 1)[1:end-1]
dt = tfinal / N
signal = [sin(t / 10) for t in input_ts]
targets = narma(5, default_narma_parameters, signal)#[sin, (x -> sin(x - 1)), x -> sin(x - 1)^2]
targetnames = ["narma"]
params = rand_reservoir_params(labels)
H = hamiltonian(params)
temperature = 2sum(values(params.Γ))
leads = Dict(l => NormalLead(c[l]' * params.Γ[l]; T=temperature, μ=0.0) for l in labels)
# mask = Dict(l => l^2 * 10 * (rand() - 0.5) for l in keys(leads))
mask = Dict(l => (l > 1) * l^2 * 0.5 for l in keys(leads))
input = DiscreteInput(VoltageWrapper(MaskedInput(mask, signal)), dt)
res = reservoir(c, H, leads, input, StationaryState())
run_reservoir(res, tspan; time_multiplexing=2)
run_reservoir(res, tspan, PiecewiseTimeSteppingMethod(EXP_KRYLOV()))
ode_kwargs = (; abstol=1e-8, reltol=1e-8)
run_reservoir(res, tspan; time_multiplexing=2) -
run_reservoir(res, tspan, ODE(Tsit5()); time_multiplexing=2, ode_kwargs...)

ls = LindbladSystem(H, leads; usecache=true)
lazyls = LazyLindbladSystem(H, leads)
to_mat = Base.Fix2(QuantumDots.tomatrix, ls)
@time sol1 = to_mat.(solve(odeproblem(ls, input, tspan), Tsit5(); saveat=measurement_ts, ode_kwargs...).u);
@time sol2 = odeproblem2(lazyls, input, tspan; tol=1e-12);
measure(rho, ls) = get_currents(rho, ls, number_operator)
@time sol2 = run_reservoir(lazyls, input, tspan, rand_initial_state, measure; time_multiplexing, tol=1e-12);
@time sol3 = solve(odeproblem(lazyls, input, tspan), Tsit5(); saveat=measurement_ts, ode_kwargs...).u;
@time sol4 = to_mat.(odeproblem2(ls, input, tspan; tol=1e-12));
# plot(map(norm, rhos - sol2))
map(rhos -> norm(map(r -> tr(r) - 1, rhos)), [sol1, sol2, sol3, sol4])
plot(map(norm, sol1 - sol2))
plot!(map(norm, sol2 - sol3))
plot!(map(norm, sol3 - sol4))
plot!(map(norm, sol4 - sol1))
plot!(map(norm, sol2 - sol4))
plot!(map(norm, sol1 - sol3))

# sol2 = solve(odeproblem(lazyls, input, tspan, rand_initial_state), Tsit5(); ode_kwargs...)
currents = [get_currents(sol, ls, input, t, number_operator) for t in ts]
currents = [get_currents(sol, lazyls, input, t, number_operator) for t in ts]
currents2 = [get_currents(sol2, lazyls, input, t, number_operator) for t in ts]
# currents2 = [get_currents(rho, lazyls) for rho in sol2]
spectrum = [get_spectrum(ls, input, t) for t in ts]
overlaps = [abs(dot(sol(t), sol2(t))) / (norm(sol(t)) * norm(sol2(t))) for t in ts]

n_train_first = (x -> isnothing(x) ? findfirst(ts .> tfinal * 1 / 3) : x)(findfirst(overlaps .> 0.99))
n_test_first = max(n_train_first + 1, findfirst(ts .> tfinal * 2 / 3))
n_train = n_train_first:n_test_first-1
n_test = n_test_first:length(ts)

Xtrain = permutedims(stack(currents[n_train]))
ytrain = stack([target.(ts[n_train]) for target in targets])
Xtest = permutedims(stack(currents[n_test]))
ytest = stack([target.(ts[n_test]) for target in targets])
W = fit(ytrain, Xtrain)

ztrain = predict(W, Xtrain)#[W[:, 1:end-1] * Xtrain .+ W[:, end] for W in Ws]
ztest = predict(W, Xtest)#[W[:, 1:end-1] * Xtest .+ W[:, end] for W in Ws]
mses = [mean((ytest .- ztest) .^ 2) for (ytest, ztest) in zip(eachcol(ytest), eachcol(ztest))]
memory_capacities = [(cov(ztest[:], ytest[:]) / (std(ztest) * std(ytest)))^2 for (ytest, ztest) in zip(eachcol(ytest), eachcol(ztest))]
##
push!(results, result)
##
##
lindresults = []
pauliresults = []
@profview Threads.@threads for seed in 1:100
    # seed = 33129
    Random.seed!(seed)
    tfinal = 40
    tspan = (0, tfinal)
    N = 100
    ts = range(tspan..., N + 1)[1:end-1]
    time_multiplexing = 2
    dt = tfinal / N
    signal_frequency = 1 / 2
    signal = [sin(t * signal_frequency) for t in ts]
    targets = Dict(["narma" => narma(5, default_narma_parameters, signal), "identity" => signal, "delay 10" => DelayedSignal(signal, 10)])
    params = rand_reservoir_params(labels)
    temperature = .1sum(values(params.Γ))
    H = hamiltonian(params)
    leads = Dict(l => NormalLead(c[l]' * params.Γ[l]; T=temperature, μ=0.0) for l in labels)
    mask = Dict(l => (l > 1) * l^2 * 0.5 for l in keys(leads))
    input = DiscreteInput(VoltageWrapper(MaskedInput(mask, signal)), dt)
    lindres = reservoir(c, H, leads, input, StationaryState())
    paulisys = PauliSystem(H, leads)
    paulires = Reservoir(c, H, leads, input, paulisys, deepcopy(paulisys), StationaryState(), nothing, CurrentMeasurements(numberoperator(c)))

    for (res, out) in zip((paulires, lindres), (pauliresults, lindresults))
        measurements = run_reservoir(res, tspan; time_multiplexing)
        simulation_results = (; measurements, res, tspan, time_multiplexing)
        task_results = task_properties(measurements, targets)
        res_props = reservoir_properties(res, tspan)
        other_data = (; params, temperature, seed, signal, input, targets, ts)
        result = merge(simulation_results, task_results, res_props, other_data)
        push!(out, result)
    end
    # result = (; seed, signal, res, params, ts, measurements, tspan, input, temperature, task_props..., res_props..., time_multiplexing)
    # push!(lindresults, lindresult)
    # push!(pauliresults, lindresult)
    # display(summary_gif2(result))
end
##
sortedlindresults = sort(lindresults, by=x -> norm(x.mses));
sortedpauliresults = sort(pauliresults, by=x -> norm(x.mses));
##
summary_gif2(first(sortedlindresults))
summary_gif2(last(sortedlindresults))
##
summary_gif2(first(sortedpauliresults))
summary_gif2(last(sortedpauliresults))
##
let sortedresults = sortedpauliresults
    plot(map(x -> norm(x.mses), sortedresults), label="mse", ylims = (0, 1))
    plot!(map(x -> norm(x.memory_capacities .- 1), sortedresults), label="memcap - 1")
    plot!(map(x -> 1 / sum(values(x.params.Γ)), sortedresults), label="1/sum of Γ")
    # plot!(map(x -> x.average_gapratio, sortedresults), label="average_gapratio")
    # plot!(map(x -> x.temperature, sortedresults), label="temperature")
    # plot!(map(x -> x.smallest_decay_rate, sortedresults), label="smallest_decay_rate")
end
let sortedresults = sortedlindresults
    plot!(map(x -> norm(x.mses), sortedresults), label="mse", ylims = (0, 1))
    plot!(map(x -> norm(x.memory_capacities .- 1), sortedresults), label="memcap - 1")
    plot!(map(x -> 1 / sum(values(x.params.Γ)), sortedresults), label="1/sum of Γ")
    # plot!(map(x -> x.average_gapratio, sortedresults), label="average_gapratio")
    # plot!(map(x -> x.temperature, sortedresults), label="temperature")
    # plot!(map(x -> x.smallest_decay_rate, sortedresults), label="smallest_decay_rate")
end
##

scatter(1:length(sortedlindresults), map(x -> x.seed, sortedlindresults))
scatter!(1:length(sortedpauliresults), map(x -> x.seed, sortedpauliresults))
scatter(map(x -> x.seed, sortedlindresults), map(x -> x.seed, sortedpauliresults))