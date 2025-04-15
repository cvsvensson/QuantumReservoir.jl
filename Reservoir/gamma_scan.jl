using QuantumDots, QuantumDots.BlockDiagonals
using LinearAlgebra
using Random
using OrdinaryDiffEqTsit5
using LinearSolve
using Plots
using Statistics
using MLJLinearModels
using MultivariateStats
using ExponentialUtilities
using KrylovKit
using UnPack
using OhMyThreads
using LogExpFunctions
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
hopping_labels = [(labels[k1], labels[k2]) for k1 in 1:length(labels), k2 in 1:length(labels) if k1 > k2] #&& is_nearest_neighbours(labels[k1], labels[k2])]
Random.seed!(321)
signal = [(-1)^rand(Bool) for _ in 1:400];
inputs = [
    let
        #tfinal = 40
        tspan = (0, tfinal)
        N = length(signal)
        ts = range(tspan..., N + 1)[1:end-1]
        dt = tfinal / N
        mask = Dict(l => (l > 1) * (-1)^(l == 2) * l^2 * 0.5 for l in labels)
        voltage_input = DiscreteInput(VoltageWrapper(MaskedInput(mask, signal)), dt)
        (; tspan, dt, signal, voltage_input, mask, initial_state=StationaryState(), ts)
    end
    for tfinal in range(1, 1 * 1000, 10)
];
##
Nres = 10
reservoirs = []
for seed in 1:Nres
    Random.seed!(seed)
    # define all scales
    scales = (; Vscale=2, tscale=1, εscale=1)
    params = rand_reservoir_params(labels; scales...)
    push!(reservoirs, (; seed, params, scales, qn, c))
end;
leads = []
for seed in 1:Nres
    # Random.seed!(seed)
    # seed = 1
    Random.seed!(seed)
    labels = 1:N
    temperature = 5 #* rand() + 1
    # scales = (; Γscale=20 * n / 30, Γmin=0.1)
    scales = (; Γscale=1, Γmin=0.1)
    Γ = Dict(l => scales.Γscale * (rand() + scales.Γmin) for l in labels)
    push!(leads, (; Γ, scales, seed, temperature, labels))
end;
##
measurement = (; measure=CurrentMeasurements(numberoperator(c)), time_multiplexing=2);
##
save_spectrum = false
# alg = PiecewiseTimeSteppingMethod(EXP_SCIML())
alg = PropagatorMethod()
res_lead_input_combinations = collect(Iterators.product(zip(reservoirs, leads), inputs))
@time measurementslind = tmap(res_lead_input_combinations) do ((res, lead), input)
    run_reservoir(res, lead, input, measurement, Lindblad(), c, alg; save_spectrum)
end;
@time measurementspauli = tmap(res_lead_input_combinations) do ((res, lead), input)
    run_reservoir(res, lead, input, measurement, Pauli(), c, alg; save_spectrum)
end;
@time propagators_lindblad = tmap(res_lead_input_combinations) do ((res, lead), input)
    generate_propagators(res, lead, input, Lindblad(), c)
end;
@time propagators_pauli = tmap(res_lead_input_combinations) do ((res, lead), input)
    generate_propagators(res, lead, input, Pauli(), c)
end;
@time master_matrices_lindblad = tmap(reservoirs, leads) do res, lead
    generate_master_matrices(res, lead, first(inputs), Lindblad(), c)
end;
@time master_matrices_pauli = tmap(reservoirs, leads) do res, lead
    generate_master_matrices(res, lead, first(inputs), Pauli(), c)
end;
## make non-linearity measure from commutator
algebras_lindblad = map(prop -> generate_algebra(prop, 5), propagators_lindblad);
algebras_pauli = map(prop -> generate_algebra(prop, 5), propagators_pauli);
non_linearity_lindblad = map(v -> v[2] / v[1], map(alg -> mean.(norm.(alg)), algebras_lindblad));
non_linearity_pauli = map(v -> v[2] / v[1], map(alg -> mean.(norm.(alg)), algebras_pauli));
## make memory-measure from eigenvalues
min_gaps_lindblad = map(props -> minimum(map(vals -> abs(vals[end-1] / vals[end]), eigvals.(props))), propagators_lindblad);
min_gaps_pauli = map(props -> minimum(map(vals -> abs(vals[end-1] / vals[end]), eigvals.(props))), propagators_pauli);
min_decay_rate_lindblad = map(A -> minimum(map(vals -> abs(vals[end-1]), eigvals.(A))), master_matrices_lindblad);
min_decay_rate_pauli = map(A -> minimum(map(vals -> abs(vals[end-1]), eigvals.(A))), master_matrices_pauli);
non_linearity2_pauli = map(As -> norm(As[1] * As[2] - As[2] * As[1]) / prod(norm, As), master_matrices_pauli);
non_linearity2_lindblad = map(As -> norm(As[1] * As[2] - As[2] * As[1]) / prod(norm, As), master_matrices_lindblad);

# min_gaps_lindblad = map(props -> minimum(map(vals -> exp(sum(xlogx ∘ abs, vals)), eigvals.(props))), propagators_lindblad)
# min_gaps_pauli = map(props -> minimum(map(vals -> exp(sum(xlogx ∘ abs, vals)), eigvals.(props))), propagators_pauli)
svd_entropy_lindblad = map(props -> minimum(map(s -> sum(s) * sum(map(x -> -xlogx(x), s / sum(s))), svdvals.(props))), propagators_lindblad);
svd_entropy_pauli = map(props -> minimum(map(s -> sum(s) * sum(map(x -> -xlogx(x), s / sum(s))), svdvals.(props))), propagators_pauli);
svd_entropy_lindblad = map(props -> minimum(map(s -> sum(s) * sum(map(x -> -xlogx(x), s / sum(s))), svdvals.(props))), master_matrices_lindblad);
svd_entropy_pauli = map(props -> minimum(map(s -> sum(s) * sum(map(x -> -xlogx(x), s / sum(s))), svdvals.(props))), master_matrices_pauli);

##
delays = 3:4#[1, 2, 3]
# targets = map(input -> [["delay $n" => DelayedSignal(input.signal, n) for n in delays]...], inputs) # "identity" => input.signal, "narma" => narma(5, default_narma_parameters, input.signal), 
targets = map(input -> [["delay $n" => DelayedSignal(input.signal, n) for n in delays]..., "narma" => narma(5, (; α=0.3, β=0.1, γ=0.5, δ=0), input.signal)], inputs); # "identity" => input.signal, "narma" => narma(5, default_narma_parameters, input.signal), 
@time task_props_lind = map(enumerate(eachcol(measurementslind))) do (n, m)
    map(m -> fit(m.measurements, targets[n]; warmup=0.2, train=0.5), m)
end;
@time task_props_pauli = map(enumerate(eachcol(measurementspauli))) do (n, m)
    map(m -> fit(m.measurements, targets[n]; warmup=0.2, train=0.5), m)
end;
# @time task_props_pauli = map(eachcol(measurementslind), targets) do m, targets
#     fit(m.measurements, targets; warmup=0.2, train=0.5)
# end;
##
# # plot the mses for the different reservoirs
# plot(map(x -> norm(values(x[2].Γ)), res_lead_combinations[:]), map(x -> norm(x.mses), task_props_lind[:]), xlabel="Γ", label="mse lindblad", marker=true)
# plot!(map(x -> norm(values(x[2].Γ)), res_lead_combinations[:]), map(x -> norm(x.mses), task_props_pauli[:]), xlabel="Γ", label="mse pauli", marker=true)

## 
pl = plot(; size=(800, 600), legend=:topright, frame=:box, title="MSE on delay tasks", ylabel="MSE", xlabel="dt/t̃", ylims=(-0.01, 3.2))
target_markers = [:circle, :diamond, :square, :cross]
dts = map(x -> x.dt, inputs)
effective_decay_time_lindblad = median(map(inv, min_decay_rate_lindblad))
effective_decay_time_pauli = median(map(inv, min_decay_rate_pauli))
mean_decay_time = mean((effective_decay_time_lindblad, effective_decay_time_pauli))
dts_lind = dts ./ mean_decay_time
dts_pauli = dts ./ mean_decay_time
# Γ = map(x -> norm(values(x.Γ)), leads)
for (n, kv) in enumerate(targets[1])
    marker = target_markers[n]
    mses_lind = map(task -> map(x -> x.mses[n], task), task_props_lind)
    mses_pauli = map(task -> map(x -> x.mses[n], task), task_props_pauli)
    mse_lind = map(median, mses_lind)
    mse_pauli = map(median, mses_pauli)
    # mse_lind = map(mean, eachcol(mses_lind))
    # mse_pauli = map(mean, eachcol(mses_pauli))
    # std_lind = map(std, eachcol(mses_lind))
    # std_pauli = map(std, eachcol(mses_pauli))
    std_lind = map(std, mses_lind)
    std_pauli = map(std, mses_pauli)
    quantiles = (0.25, 0.75)
    quantile_lind = map(mses -> quantile(mses, quantiles), mses_lind)
    quantile_pauli = map(mses -> quantile(mses, quantiles), mses_pauli)
    ribbon_lind = ((mse_lind .- first.(quantile_lind), last.(quantile_lind) .- mse_lind))
    ribbon_pauli = ((mse_pauli .- first.(quantile_pauli), last.(quantile_pauli) .- mse_pauli))
    # display(plot([first.(quantile_lind), last.(quantile_lind)]))
    plot!(dts_lind, mse_lind; label="$(kv[1]) lindblad", color=1, marker, ribbon=ribbon_lind)
    plot!(dts_pauli, mse_pauli; label="$(kv[1]) pauli", color=2, marker, ribbon=ribbon_pauli)
end
plot!(dts_lind, map(median, eachcol(non_linearity_lindblad)); label="non-commutativity lindblad", lw=3, color=1, ls=:dash)
plot!(dts_pauli, map(median, eachcol(non_linearity_pauli)); label="non-commutativity pauli", lw=3, color=2, ls=:dash)
# plot!(dts_lind, map(mean, eachcol(min_gaps_lindblad)); label="eigenvalue gap lindblad", lw=3, color=1, ls=:dot)
# plot!(dts_pauli, map(mean, eachcol(min_gaps_pauli)); label="eigenvalue gap pauli", lw=3, color=2, ls=:dot)
# plot!(Γ, map(mean, eachcol(svd_entropy_lindblad));  label="svd entropy lindblad")
# plot!(Γ, map(mean, eachcol(svd_entropy_pauli));  label="svd entropy pauli")
pl

##
##
svdvals_norms_lindblad = map(prop -> [norm(algebra_svdvals(generate_algebra(prop, n))) for n in 1:2], master_matrices_lindblad);
svdvals_norms_pauli = map(prop -> [norm(algebra_svdvals(generate_algebra(prop, n))) for n in 1:2], master_matrices_pauli);
heatmap(map(ss -> norm(ss), svdvals_norms_lindblad))
heatmap(map(ss -> diff(ss)[1] / ss[1], svdvals_norms_lindblad), clims=(0, 0.5))
heatmap(map(x -> x.mses[3], task_props_pauli), clims=(0, 1))
##
summary_gif(reservoirs[1], leads[1], inputs[6], Pauli(), measurement, targets[6], (; warmup=0.2, train=0.5))
summary_gif(reservoirs[1], leads[1], inputs[6], Lindblad(), measurement, targets[6], (; warmup=0.2, train=0.5))
summary_gif(reservoirs[1], leads[7], input, Pauli(), measurement, targets, (; warmup=0.2, train=0.5))
summary_gif(reservoirs[1], leads[20], input, Pauli(), measurement, targets, (; warmup=0.2, train=0.5))
# summary_gif(reservoirs[1], leads[end], input, Pauli(), measurement, targets, (; warmup=0.2, train=0.5))

##
heatmap(map(x -> norm(x.mses), task_props_lind), xlabel=:lead, ylabel=:reservoir, title=:lindblad, colorbar_title="MSE")
heatmap(map(x -> norm(x.mses), task_props_pauli), xlabel=:lead, ylabel=:reservoir, title=:pauli, colorbar_title="MSE")

## Analyse the performance of reservoirs
# The performance on delay 2 depends strongly on the reservoir at inputs[5]
let k = 5
    input = inputs[k]
    targets = [["delay $n" => DelayedSignal(input.signal, n) for n in [2]]...]
    measurements = measurementslind[:, k]
    task_props = map(m -> fit(m.measurements, targets; warmup=0.2, train=0.5), measurements)
    mses = map(x -> only(x.mses), task_props)
    perm = sortperm(mses)
    svd_entropy_lindblad = map(props -> minimum(map(s -> sum(map(x -> -xlogx(x), s / sum(s))), svdvals.(props))), master_matrices_lindblad)
    min_gaps_lindblad = map(props -> minimum(map(vals -> -real(vals[end-1]), eigvals.(props))), master_matrices_lindblad)
    non_linearity = map(As -> norm(As[1] * As[2] - As[2] * As[1]) / prod(norm, As), master_matrices_lindblad)
    non_linearity2 = non_linearity_lindblad[:, k]
    gammas = map(lead -> sum(values(lead.Γ)), leads)
    features = ["min decay rate" => min_gaps_lindblad, "non-commutativity master matrix" => non_linearity, "non-commutativity propagator" => non_linearity2, "sum of gammas" => gammas]
    smoothing = 15
    p = plot(mses[perm], label="mse", title="smoothing $smoothing", frame=:box, xlabel="reservoir")
    foreach(kv -> plot!(p, moving_average(kv[2][perm], smoothing) / mean(kv[2]), label=kv[1]), features[[1, 3, 4]])
    p
end