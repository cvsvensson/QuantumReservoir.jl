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
input = let
    tfinal = 40
    tspan = (0, tfinal)
    N = 100
    ts = range(tspan..., N + 1)[1:end-1]
    dt = tfinal / N
    signal = [(-1)^rand(Bool) for _ in ts]
    mask = Dict(l => (l > 1) * (-1)^(l == 2) * l^2 * 0.5 for l in labels)
    voltage_input = DiscreteInput(VoltageWrapper(MaskedInput(mask, signal)), dt)
    (; tspan, dt, signal, voltage_input, mask, initial_state=StationaryState(), ts)
end
##
reservoirs = []
for seed in 1:100
    Random.seed!(seed)
    # define all scales
    scales = (; Vscale=2, tscale=1, εscale=1)
    params = rand_reservoir_params(labels; scales...)
    push!(reservoirs, (; seed, params, scales, qn, c))
end
leads = []
for n in 1:30
    # Random.seed!(seed)
    seed = 1
    Random.seed!(seed)
    labels = 1:N
    temperature = 10 * rand()
    scales = (; Γscale=2.5 * n / 30, Γmin=0.1)
    Γ = Dict(l => scales.Γscale * (rand() + scales.Γmin) for l in labels)
    push!(leads, (; Γ, scales, seed, temperature, labels))
end
##

measurement = (; measure=CurrentMeasurements(numberoperator(c)), time_multiplexing=1)
##
save_spectrum = false
res_lead_combinations = collect(Iterators.product(reservoirs, leads));
@time measurementslind = tmap(res_lead_combinations) do (res, lead)
    run_reservoir(res, lead, input, measurement, Lindblad(), c, PiecewiseTimeSteppingMethod(EXP_SCIML()); save_spectrum)
end;
@time measurementspauli = tmap(res_lead_combinations) do (res, lead)
    run_reservoir(res, lead, input, measurement, Pauli(), c, PiecewiseTimeSteppingMethod(EXP_SCIML()); save_spectrum)
end;
##
targets = ["narma" => narma(5, default_narma_parameters, input.signal), "identity" => input.signal, "delay 10" => DelayedSignal(input.signal, 10)]
@time task_props_lind = map(measurementslind) do m
    fit(m.measurements, targets; warmup=0.2, train=0.5)
end;
@time task_props_pauli = map(measurementspauli) do m
    fit(m.measurements, targets; warmup=0.2, train=0.5)
end;
##
# plot the mses for the different reservoirs
plot(map(x -> norm(values(x[2].Γ)), res_lead_combinations[:]), map(x -> norm(x.mses), task_props_lind[:]), xlabel="Γ", label="mse lindblad", marker=true)
plot!(map(x -> norm(values(x[2].Γ)), res_lead_combinations[:]), map(x -> norm(x.mses), task_props_pauli[:]), xlabel="Γ", label="mse pauli", marker=true)
##
pl = plot()
target_markers = [:circle, :diamond, :square]
for (n, kv) in enumerate(targets)
    marker = target_markers[n]
    Γ = map(x -> norm(values(x.Γ)), leads)
    mses_lind = map(x -> x.mses[n], task_props_lind)
    mses_pauli = map(x -> x.mses[n], task_props_pauli)
    mse_lind = map(mean, eachcol(mses_lind))
    mse_pauli = map(mean, eachcol(mses_pauli))
    std_lind = map(std, eachcol(mses_lind))
    std_pauli = map(std, eachcol(mses_pauli))
    plot!(Γ, mse_lind; xlabel="Γ", label="$(kv[1]) lindblad", color=1, marker, ribbon=std_lind)
    plot!(Γ, mse_pauli; xlabel="Γ", label="$(kv[1]) pauli", color=2, marker, ribbon=std_pauli)
end
pl
##

summary_gif(reservoirs[1], leads[1], input, Pauli(), measurement, targets, (; warmup=0.2, train=0.5))
summary_gif(reservoirs[1], leads[7], input, Pauli(), measurement, targets, (; warmup=0.2, train=0.5))
summary_gif(reservoirs[1], leads[20], input, Pauli(), measurement, targets, (; warmup=0.2, train=0.5))
summary_gif(reservoirs[1], leads[end], input, Pauli(), measurement, targets, (; warmup=0.2, train=0.5))

##
heatmap(map(x -> norm(x.mses), task_props_lind), xlabel=:lead, ylabel=:reservoir, title=:lindblad, colorbar_title="MSE")
heatmap(map(x -> norm(x.mses), task_props_pauli), xlabel=:lead, ylabel=:reservoir, title=:pauli, colorbar_title="MSE")