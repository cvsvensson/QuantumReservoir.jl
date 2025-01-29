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
##
N = 4
labels = 1:N
qn = FermionConservation()
c = FermionBasis(labels; qn)
number_operator = sum([c[l]'c[l] for l in labels])
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
function summary_gif(result, Nfigs=100)
    @unpack leads, W, seed, spectrum, currents, sol, ts, input, H, sol2, temperature, evals, params, ztrain, ztest, targets, targetnames, mses, memory_capacities, overlaps, n_train_first, n_test_first, n_train, n_test, ediffs, average_gapratio = result
    xlims = maximum(abs ∘ real, first(spectrum)) .* (-1.01, 0.01)
    ylims = maximum(abs ∘ imag, first(spectrum)) .* (-1.01, 1.01)
    leadlabels = transpose(collect(keys(input(0))))
    signal = stack([collect(values(input.input.signal(t))) for t in ts])'
    pcurrent = plot(ts, stack(currents)', label=leadlabels, legendtitle="Lead", xlabel="t", ylabel="current", legendposition=:topright)

    inputsignal = [input.input.signal.signal(t) for t in ts]
    ptargets = map(eachcol(ztrain), eachcol(ztest), targets, targetnames) do ztrain, ztest, target, name
        ptarget = plot(ts, target.(ts), label="$name", xlabel="t")
        plot!(ptarget, ts, inputsignal, label="input", c=:black, linestyle=:dash)
        plot!(ptarget, ts[n_train], ztrain, label="train")
        plot!(ptarget, ts[n_test], ztest, label="test")
        ptarget
    end
    pecho = plot(ts, overlaps, xlabel="t", label="overlap of two solutions", yrange=(0, 1.01), legendposition=:bottomright)
    vline!(pecho, [ts[n_train_first]], color=:red, label="start training")
    smallest_decay_rate = mean(spec -> abs(partialsort(spec, 2, by=abs)), spectrum)
    vline!(pecho, [1 / smallest_decay_rate], label="1/(decay rate)")
    infos = (; seed, temperature=round(temperature, digits=2), average_gapratio=round(average_gapratio, digits=3), mse=round.(mses, digits=3), memory_capacity=round.(memory_capacities, digits=3))
    pinfo = plot([1:-1 for _ in infos]; framestyle=:none, la=0, label=permutedims(["$k = $v" for (k, v) in pairs(infos)]), legend_font_pointsize=10, legendposition=:top)
    pW = heatmap(log.(abs.(W')), color=:greys, yticks=(1:length(targets), targetnames), xticks=(1:length(leadlabels)+1, [leadlabels..., "bias"]), title="logabs(W)")
    indices = round.(Int, range(1, length(spectrum), Nfigs))
    anim = @animate for n in indices #(s, t) in zip(spectrum, ts)
        s = spectrum[n]
        t = ts[n]
        pspec = scatter(real(s), imag(s); xlims, ylims, size=(800, 800), ylabel="im", xlabel="re", label="eigenvalues", legendposition=:topleft)
        boltz = stack([QuantumDots.fermidirac.(ediffs, leads[l].T, input(t)[l].μ) |> sort for l in keys(leads)])
        psignal = plot(ts, signal, labels=leadlabels, xlabel="t", ylabel="voltage", legendtitle="Lead")
        vline!(psignal, [t], color=:red, label="t")
        pboltz = plot(boltz, marker=true, ylims=(0, 1), labels=leadlabels, markersize=1, markerstrokewidth=0, legendposition=:left, ylabel="boltzmann factors")
        plot(psignal, pspec, pcurrent, pboltz, pecho, pinfo, pW, ptargets..., layout=(4 + div(length(targets), 2), 2))
    end
    gif(anim, "anim.gif", fps=div(Nfigs, 5))
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
    temperature = 2norm(params.Γ)
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
temperature = 2norm(params.Γ)
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
DelayedSignal(signal, delay::Int, history=zeros(delay)) = vcat(history, signal[1:end-delay])
##
results = []
for seed in 1:100
    # seed = 33129
    Random.seed!(seed)
    tfinal = 200
    tspan = (0, tfinal)
    N = 100
    ts = range(tspan..., N + 1)[1:end-1]
    time_multiplexing = 1
    dt = tfinal / N
    signal_frequency = 1 / 20
    signal = [sin(t * signal_frequency) for t in ts]
    mask = Dict(l => (l > 1) * l^2 * 0.5 for l in keys(leads))
    input = DiscreteInput(VoltageWrapper(MaskedInput(mask, signal)), dt)
    targets = Dict(["narma" => narma(5, default_narma_parameters, signal), "identity" => signal, "delay 10" => DelayedSignal(signal, 10)])
    # targetnames = ["narma", "identity", "delay 2"]
    params = rand_reservoir_params(labels)
    H = hamiltonian(params)
    temperature = 2norm(values(params.Γ))
    leads = Dict(l => NormalLead(c[l]' * params.Γ[l]; T=temperature, μ=0.0) for l in labels)
    res = reservoir(c, H, leads, input, StationaryState())
    measurements = run_reservoir(res, tspan; time_multiplexing)
    task_props = task_properties(measurements, targets)
    res_props = reservoir_properties(res, tspan)
    result = (; seed, signal, res, params, ts, measurements, tspan, input, temperature, task_props..., res_props..., time_multiplexing)
    push!(results, result)
    # display(summary_gif2(result))
end
##
sortedresults = sort(results, by=x -> norm(x.mses));
summary_gif2(first(sortedresults))
summary_gif2(last(sortedresults))
##
plot(map(x -> norm(x.mses), sortedresults), label="mse")
plot!(map(x -> norm(x.memory_capacities .- 1), sortedresults), label="memcap - 1")
plot!(map(x -> x.average_gapratio, sortedresults), label="average_gapratio")
plot!(map(x -> x.temperature, sortedresults), label="temperature")
##
function summary_gif2(result)
    @unpack signal, res, W, seed, spectrum, measurements, ts, input, temperature, evals, params, ztrain, ztest, targets, mses, memory_capacities, n_train_first, n_test_first, n_train, n_test, ediffs, average_gapratio, time_multiplexing = result
    leads = res.leads
    xlims = maximum(abs ∘ real, first(spectrum)) .* (-1.01, 0.01)
    ylims = maximum(abs ∘ imag, first(spectrum)) .* (-1.01, 1.01)
    leadlabels = permutedims(collect(keys(input(tspan[1]))))
    multiplexedlabels = permutedims(reduce(vcat, map(n -> map(l -> string("$l,$n"), leadlabels), 1:time_multiplexing)))
    pcurrent = plot(ts, stack(measurements)', xlabel="t", ylabel="current", legend=false, marker=true)#, label=multiplexedlabels,legendtitle="Lead", legendposition=:topright#)

    ptargets = map(eachcol(ztrain), eachcol(ztest), collect(pairs(targets))) do ztrain, ztest, (name, target)
        ptarget = plot(ts, target, label=name, xlabel="t")
        plot!(ptarget, ts, signal, label="input", c=:black, linestyle=:dash)
        plot!(ptarget, ts[n_train], ztrain, label="train")
        plot!(ptarget, ts[n_test], ztest, label="test")
        ptarget
    end
    pecho = plot()#plot(ts, overlaps, xlabel="t", label="overlap of two solutions", yrange=(0, 1.01), legendposition=:bottomright)
    # vline!(pecho, [ts[n_train_first]], color=:red, label="start training")
    smallest_decay_rate = mean(spec -> abs(partialsort(spec, 2, by=abs)), spectrum)
    # vline!(pecho, [1 / smallest_decay_rate], label="1/(decay rate)")

    infos = (; seed, temperature=round(temperature, digits=2), average_gapratio=round(average_gapratio, digits=3), mse=round.(mses, digits=3), memory_capacity=round.(memory_capacities, digits=3))
    pinfo = plot([1:-1 for _ in infos]; framestyle=:none, la=0, label=permutedims(["$k = $v" for (k, v) in pairs(infos)]), legend_font_pointsize=10, legendposition=:top)
    pW = heatmap(log.(abs.(W')), color=:greys, yticks=(1:length(targets), targetnames), xticks=(1:length(multiplexedlabels)+1, [multiplexedlabels..., "bias"]), title="logabs(W)")
    # indices = 1:N#round.(Int, range(1, length(spectrum), Nfigs))
    N = length(signal)
    high_frequency_ts = range(tspan..., N * 10 + 1)[1:end-1]
    voltages = stack([[x.μ for x in values(input(t))] for t in high_frequency_ts])'
    # display(voltages)
    Nfigs = 50
    dn = max(1, div(N, Nfigs))
    anim = @animate for n in 1:dn:N-1#(s, t) in zip(spectrum, ts)
        s = spectrum[n]
        t = ts[n]
        pspec = scatter(real(s), imag(s); xlims, ylims, size=(800, 800), ylabel="im", xlabel="re", label="eigenvalues", legendposition=:topleft)
        boltz = stack([QuantumDots.fermidirac.(ediffs, leads[l].T, input(t)[l].μ) |> sort for l in keys(leads)])
        psignal = plot(high_frequency_ts, voltages, labels=leadlabels, xlabel="t", ylabel="voltage", legendtitle="Lead", legendposition=:right)
        vline!(psignal, [t], color=:red, label="t")
        vline!(psignal, [1 / smallest_decay_rate], label="t*")
        pboltz = plot(boltz, marker=true, ylims=(0, 1), labels=leadlabels, markersize=1, markerstrokewidth=0, legendposition=:left, ylabel="boltzmann factors")
        plot(psignal, pspec, pcurrent, pboltz, pecho, pinfo, pW, ptargets..., layout=(4 + div(length(targets), 2), 2))

    end
    gif(anim, "anim.gif", fps=div(length(1:dn:N-1), 5))
end
function task_properties(measurements, targets)
    n_train_first = div(N, 10)
    n_test_first = max(n_train_first + 1, Int(div(N, 10 / 7)))
    n_train = n_train_first:n_test_first-1
    n_test = n_test_first:N

    Xtrain = permutedims(stack(measurements[n_train]))
    ytrain = stack([target[n_train] for target in values(targets)])
    Xtest = permutedims(stack(measurements[n_test]))
    ytest = stack([target[n_test] for target in values(targets)])
    W = fit(ytrain, Xtrain)
    ztrain = predict(W, Xtrain)
    ztest = predict(W, Xtest)
    mses = [mean((ytest .- ztest) .^ 2) for (ytest, ztest) in zip(eachcol(ytest), eachcol(ztest))]
    memory_capacities = [(cov(ztest[:], ytest[:]) / (std(ztest) * std(ytest)))^2 for (ytest, ztest) in zip(eachcol(ytest), eachcol(ztest))]
    return (; W, mses, memory_capacities, ztrain, ztest, targets, n_train_first, n_test_first, n_train, n_test)
end
function reservoir_properties(res, tspan)
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
    return (; evals, spectrum, gapratios, average_gapratio, ediffs)
end
