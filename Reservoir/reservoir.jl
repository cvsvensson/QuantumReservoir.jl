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
include("..\\system.jl")
##
N = 4
labels = 1:N
qn = FermionConservation()
c = FermionBasis(labels; qn)
number_operator = sum([c[l]'c[l] for l in labels])
hopping_labels = [(labels[k1], labels[k2]) for k1 in 1:length(labels), k2 in 1:length(labels) if k1 > k2] #&& is_nearest_neighbours(labels[k1], labels[k2])]
##
struct ContinuousInput{I}
    input::I
end
struct DiscreteInput{I,T}
    input::I
    dt::T
end
# struct ReservoirTask1{I,T}
#     input::I
#     target::T
# end
# ReservoirTask = ReservoirTask1
# struct DelayTask{I,D}
#     input::I
#     delay::D
# end
##
struct MaskedInput{M,S}
    mask::M
    signal::S
end
(m::MaskedInput)(t) = Dict(l => m.signal(t) * v for (l, v) in pairs(m.mask))
Base.getindex(m::MaskedInput, i) = Dict(l => getindex(m.signal, i) * v for (l, v) in pairs(m.mask))
function voltage_input(signal)
    Dict(l => (; μ=v) for (l, v) in pairs(signal))
    # Dict(map((l, v) -> l => (; μ=v), labels, input))
end
struct VoltageWrapper{S}
    signal::S
end
(v::VoltageWrapper)(t) = voltage_input(v.signal(t))
Base.getindex(v::VoltageWrapper, l) = voltage_input(getindex(v.signal, l))
(c::ContinuousInput)(t) = c.input(t)
(d::DiscreteInput)(t) = d.input[1+Int(div(t, d.dt))]

function stationary_state(ls::LazyLindbladSystem; kwargs...)
    ss_prob = StationaryStateProblem(ls)
    reshape(solve(ss_prob; abstol=get(kwargs, :abstol, 1e-12)), size(ls.hamiltonian))
end
function stationary_state(ls::LindbladSystem; kwargs...)
    ss_prob = StationaryStateProblem(ls)
    solve(ss_prob; abstol=get(kwargs, :abstol, 1e-12))
end
function odeproblem(_ls, input, tspan; kwargs...)
    ls = deepcopy(_ls)
    QuantumDots.update_coefficients!(ls, input(tspan[1]))
    initial_state = collect(stationary_state(ls))
    odeproblem(ls, input, tspan, initial_state; copy=false, kwargs...)
end
function odeproblem(_ls, input::ContinuousInput, tspan, initial_state; copy=true, kwargs...)
    ls = copy ? deepcopy(_ls) : _ls
    p = (ls, input)
    ODEProblem(f_ode!, initial_state, tspan, p; kwargs...)
end
function f_ode!(du, u, (ls, input), t)
    QuantumDots.update_coefficients!(ls, input(t))
    mul!(du, ls, u)
end
##

function get_currents(sol, ls, input::ContinuousInput, t, op=number_operator)
    rho = sol(t)
    QuantumDots.update_coefficients!(ls, input(t))
    get_currents(rho, ls, op)
end
function get_currents(sol, ls, input::DiscreteInput, t, op=number_operator)
    n = 1 + Int(div(t, input.dt))
    rho = sol[n]
    QuantumDots.update_coefficients!(ls, input(t))
    get_currents(rho, ls, op)
end
function get_spectrum(ls, input, t)
    QuantumDots.update_coefficients!(ls, input(t))
    eigvals(Matrix(ls))
end
function get_currents(rho, ls, op=number_operator)
    real(QuantumDots.measure(rho, op, ls))
end
##
function MLJLinearModels.fit(target, measurements; β=1e-6, fit_intercept=true)
    ridge = RidgeRegression(β; fit_intercept)
    reduce(hcat, map(data -> fit(ridge, measurements, data), eachcol(target)))
end
function predict(W, X)
    if size(W, 1) == size(X, 2)
        return X * W
    end
    X * W[1:end-1, :] .+ ones(size(X, 1)) * W[end, :]'
end


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
includet("narma.jl")
anims = []
results = []
# for seed in 1:1
seed = 3
Random.seed!(seed)
tfinal = 50
tspan = (0, tfinal)
N = 10
ts = range(tspan..., N + 1)[1:end-1]
dt = tfinal / N
signal = [sin(t / 10) for t in ts]
targets = narma(5, default_narma_parameters, signal)#[sin, (x -> sin(x - 1)), x -> sin(x - 1)^2]
targetnames = ["narma"]
params = rand_reservoir_params(labels)
H = hamiltonian(params)
temperature = 2norm(params.Γ)
leads = Dict(l => NormalLead(c[l]' * params.Γ[l]; T=temperature, μ=0.0) for l in labels)
# mask = Dict(l => l^2 * 10 * (rand() - 0.5) for l in keys(leads))
mask = Dict(l => (l > 1) * l^2 * 0.5 for l in keys(leads))
input = DiscreteInput(VoltageWrapper(MaskedInput(mask, signal)), dt)
# input = ContinuousInput(VoltageWrapper(MaskedInput(mask, signal)))

ls = LindbladSystem(H, leads; usecache=true)
lazyls = LazyLindbladSystem(H, leads)
to_mat = Base.Fix2(QuantumDots.tomatrix, ls)
ode_kwargs = (; abstol=1e-8, reltol=1e-8)
@time sol1 = to_mat.(solve(odeproblem(ls, input, tspan), Tsit5(); saveat=ts, ode_kwargs...).u);
@time sol2 = odeproblem2(lazyls, input, tspan; tol=1e-12);
@time sol3 = solve(odeproblem(lazyls, input, tspan), Tsit5(); saveat=ts, ode_kwargs...).u;
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
evals = eigvals(Matrix(H))
gapratios = map(x -> x > 1 ? inv(x) : x, diff(evals)[1:end-1] ./ diff(evals)[2:end])
average_gapratio = mean(gapratios)
ediffs = QuantumDots.commutator(Diagonal(evals)).diag
##
result = (; seed, spectrum, currents, sol, ts, input, H, sol2, temperature, evals, params, ztrain, ztest, targets, targetnames, mses, memory_capacities, overlaps, n_train_first, n_test_first, n_train, n_test, ediffs, average_gapratio, W, leads)
push!(results, result)
# end

##
function odeproblem2(_ls, input, tspan; kwargs...)
    ls = deepcopy(_ls)
    QuantumDots.update_coefficients!(ls, input(tspan[1]))
    initial_state = collect(stationary_state(ls))
    odeproblem2(ls, input, tspan, initial_state; copy=false, kwargs...)
end
function odeproblem2(_ls, input::DiscreteInput, tspan, initial_state; copy=true, kwargs...)
    ls = copy ? deepcopy(_ls) : _ls
    dt = input.dt
    ts = range(tspan..., step=dt)
    rhos = [initial_state]
    # infos = []
    for t in ts[1:end-1]
        QuantumDots.update_coefficients!(ls, input(t))
        rho, info = exponentiate(ls, dt, rhos[end]; kwargs...)
        # rho2 = expv(dt, ls, rhos[end])
        # println(norm(rho-rho2))
        push!(rhos, rho)
        # push!(infos, info)
    end
    return rhos
end
function odeproblem(_ls, input::DiscreteInput, tspan, initial_state; copy=true, kwargs...)
    ls = copy ? deepcopy(_ls) : _ls
    p = (ls, input)
    ODEProblem(f_ode!, initial_state, tspan, p; kwargs...)
end
##
lsc = deepcopy(ls)
[norm(QuantumDots.update_coefficients!(lsc, input(t)).total) for t in range(tspan..., 100N)] |> plot