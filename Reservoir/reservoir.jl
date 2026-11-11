using FermionicHilbertSpaces
using LinearAlgebra
using Random
using OrdinaryDiffEqTsit5
using Plots
using Statistics
using MLJLinearModels
using MultivariateStats
using ExponentialUtilities
using KrylovKit
using OhMyThreads
Random.seed!(1234)
includet("..\\system.jl")
includet("..\\opensystems.jl")
includet("src.jl")
includet("narma.jl")
includet("training.jl")
includet("plots.jl")
##
N = 4
labels = 1:N
@fermions f
space = hilbert_space(f, labels, NumberConservation())
hopping_labels = [(labels[k1], labels[k2]) for k1 in 1:length(labels), k2 in 1:length(labels) if k1 > k2] #&& is_nearest_neighbours(labels[k1], labels[k2])]

##
rand_initial_state = rand(ComplexF64, 2^length(labels), 2^length(labels)) |> (x -> x'x) |> (x -> x ./ tr(x))
##

##
reservoirs = []
for seed in 1:25
    Random.seed!(seed)
    # define all scales
    scales = (; Vscale=2, tscale=1, εscale=1)
    params = rand_reservoir_params(labels; scales...)
    push!(reservoirs, (; seed, params, scales))
end
leads = []
for seed in 1:25
    Random.seed!(seed)
    labels = 1:N
    temperature = 10 * rand()
    scales = (; Γscale=1, Γmin=0.1)
    Γ = Dict(l => scales.Γscale * (rand() + scales.Γmin) for l in labels)
    push!(leads, (; Γ, scales, seed, temperature, labels))
end
##
input = let
    tfinal = 40
    tspan = (0, tfinal)
    N = 100
    ts = range(tspan..., N + 1)[1:end-1]
    dt = tfinal / N
    # signal_frequency = 1 / 2
    # signal = [sin(t * signal_frequency) for t in ts]
    signal = [(-1)^rand(Bool) for _ in ts]
    mask = Dict(l => (l > 1) * (-1)^(l == 2) * l^2 * 0.5 for l in labels)
    voltage_input = DiscreteInput(VoltageWrapper(MaskedInput(mask, signal)), dt)
    (; tspan, dt, signal, voltage_input, mask, initial_state=StationaryState(), ts)
end
measurement = (; measure=CurrentMeasurements(representation(number_operator(f, labels), space)), time_multiplexing=2)
##
save_spectrum = false
# @time measurements1 = tmap(reservoirs, leads) do res, lead
#     run_reservoir(res, lead, input, measurement, Pauli(), PiecewiseTimeSteppingMethod(EXP_SCIML()); save_spectrum)
# end;
# @time measurements2 = tmap(reservoirs, leads) do res, lead
#     run_reservoir(res, lead, input, measurement, Lindblad(), PiecewiseTimeSteppingMethod(EXP_SCIML()); save_spectrum)
# end;
res_lead_combinations = collect(Iterators.product(reservoirs, leads[1:end-1]));
@time measurementslind = tmap(res_lead_combinations) do (res, lead)
    run_reservoir(res, lead, input, measurement, Lindblad(), f, space, PiecewiseTimeSteppingMethod(EXP_SCIML()); save_spectrum)
end;
@time measurementspauli = tmap(res_lead_combinations) do (res, lead)
    run_reservoir(res, lead, input, measurement, Pauli(), f, space, PiecewiseTimeSteppingMethod(EXP_SCIML()); save_spectrum)
end;
##
targets = ["narma" => narma(5, default_narma_parameters, input.signal), "identity" => input.signal, "delay 5" => DelayedSignal(input.signal, 5)]
@time task_props_lind = map(measurementslind) do m
    fit(m.measurements, targets; warmup=0.2, train=0.5)
end;
@time task_props_pauli = map(measurementspauli) do m
    fit(m.measurements, targets; warmup=0.2, train=0.5)
end;
##
fullanalysis(reservoirs[1], leads[1], input, Pauli(), f, space, measurement, targets, (; warmup=0.2, train=0.5))
##
sorted_task_props_lind = task_props_lind[sortperm(eachrow(task_props_lind), by=v -> norm(map(x -> x.mses, v))), :];
sorted_task_props_lind = sorted_task_props_lind[:, sortperm(eachcol(task_props_lind), by=v -> norm(map(x -> x.mses, v)))];
sorted_task_props_pauli = task_props_pauli[sortperm(eachrow(task_props_pauli), by=v -> norm(map(x -> x.mses, v))), :];
sorted_task_props_pauli = sorted_task_props_pauli[:, sortperm(eachcol(task_props_pauli), by=v -> norm(map(x -> x.mses, v)))];
##
heatmap(map(x -> norm(x.mses), sorted_task_props_lind), xlabel=:lead, ylabel=:reservoir, title=:lindblad, colorbar_title="MSE")
heatmap(map(x -> norm(x.mses), sorted_task_props_pauli), xlabel=:lead, ylabel=:reservoir, title=:pauli, colorbar_title="MSE")
##
res_median_performance = vec(median(map(x -> norm(x.mses), task_props_lind), dims=2)) |> sort
lead_median_performance = vec(median(map(x -> norm(x.mses), task_props_lind), dims=1)) |> sort
res_best_performance = vec(minimum(map(x -> norm(x.mses), task_props_lind), dims=2)) |> sort
lead_best_performance = vec(minimum(map(x -> norm(x.mses), task_props_lind), dims=1)) |> sort
plot(res_median_performance, label="reservoirs median mse over leads", ylabel="mse", legendposition=:bottomright)
plot!(lead_median_performance, label="leads median mse over reservoirs")
plot!(res_best_performance, label="reservoirs best mse")
plot!(lead_best_performance, label="leads best mse")
##
fullresults_lind = [(; reservoir, lead, measurements, fit) for ((reservoir, lead), measurements, fit) in zip(res_lead_combinations, measurementslind, task_props_lind)];
sorted_fullresults_lind = sort(fullresults_lind[:], by=x -> norm(x.fit.mses));
fullresults_pauli = [(; reservoir, lead, measurements, fit) for ((reservoir, lead), measurements, fit) in zip(res_lead_combinations, measurementspauli, task_props_pauli)];
sorted_fullresults_pauli = sort(fullresults_pauli[:], by=x -> norm(x.fit.mses));
##
plot(map(x -> norm(x.fit.mses), sorted_fullresults_lind), ylims=(0, :auto))
plot!(map(x -> norm(values(x.lead.Γ)), sorted_fullresults_lind))
plot!(map(x -> maximum(values(x.lead.Γ)), sorted_fullresults_lind))
## pauli
plot(map(x -> norm(x.fit.mses), sorted_fullresults_pauli), ylims=(0, :auto))
plot!(map(x -> norm(values(x.lead.Γ)), sorted_fullresults_pauli))

##
function pca_entropy(pca)
    ps = principalvars(pca)
    ps = ps / sum(ps)
    sum(map(x -> -x * log(x), ps))
end
heatmap(map(x -> pca_entropy(x.pca) |> exp, sorted_task_props_pauli), xlabel=:lead, ylabel=:reservoir, title=:pauli, colorbar_title="PCA entropy effective dimensionality")#,clims=(0, .5))
heatmap(map(x -> pca_entropy(x.pca) |> exp, sorted_task_props_lind), xlabel=:lead, ylabel=:reservoir, title=:lindblad, colorbar_title="PCA entropy effective dimensionality")#,clims=(0, .5))

evals = stack([eigvals(Matrix(representation(hamiltonian(f, rand_reservoir_params(labels, Vscale=2)), space))) for _ in 1:10]);
plot(evals, markers=true)

