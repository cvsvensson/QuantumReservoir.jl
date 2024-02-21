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
includet("integrated_current_misc.jl")
#includet("gpu.jl")
Random.seed!(1234)
# density_matrix_distribution = DensityMatrixDistribution(Ginibre(2, 2))
# training_input_states = [cat([0;;], rand(density_matrix_distribution), [0;;]; dims=(1, 2)) for i in 1:1000];
# validation_input_states = [cat([0;;], rand(density_matrix_distribution), [0;;]; dims=(1, 2)) for i in 1:1000];
##
reconstructed_data(W, X, i=:) = size(W, 2) > size(X, 1) ? W[i, 1:end-1] * X + W[i, end] * ones(1, size(X, 2)) : W[i, :] * X
function fit_output_layer(X, y; β=1e-6, fit_intercept=true)
    # X = training_sols.integrated
    # y = training_sols.ensemble.data
    ridge = RidgeRegression(β; fit_intercept)
    W = reduce(hcat, map(data -> fit(ridge, X', data), eachrow(y))) |> permutedims
    return W
end
get_density_matrix(obs) = get_rho_mat(obs[3:end])
function loss_function_density_matrix(W, X_test, y_test)
    y_pred = reconstructed_data(W, X_test)
    losses = map(norm ∘ get_density_matrix ∘ -, eachcol(y_pred), eachcol(y_test))
    mean(losses)
end
##
μmin = -1e5
μs = -1 .* [1, 1]
T = 10
Nres_layers = 1
Nres = 50
rc = ReservoirConnections(Nres_layers, length(μs))
Nleads = length(μs) # Make this more general. Labels to connect leads.
training_input_states = random_inputs(rc, 1000)
validation_input_states = random_inputs(rc, 1000)
##
Nmeas = 5
reservoir_parameters = [random_parameters(rc, Nmeas) for n in 1:Nres] #[random_static_parameters(rc) for n in 1:Nres] #Many random reservoirs to get statistics of performance
# N1_projector = cat([0;;], one(rand(4, 4)), [0;;]; dims=(1, 2))[2:5, :]
##
M_train = 10
M_val = 100
tmax = 100
abstol = 1e-8
reltol = 1e-6
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
@time reservoir = prepare_reservoir(rc, reservoir_parameters[1], (T, μs); abstol);
@time mm = measurement_matrix(reservoir, tmax, alg; abstol);
@time training_sols = run_reservoir_trajectories(reservoir, training_input_states[1:M_train], tmax, alg; time_trace=false, abstol);
@time val_sols = run_reservoir_trajectories(reservoir, validation_input_states[1:M_val], tmax, Exponentiation(EXP_krylovkit()); time_trace=false, abstol);
@time val_sols = run_reservoir_trajectories(reservoir, validation_input_states[1:M_val], tmax, Exponentiation(EXP_sciml()); time_trace=false, abstol);
@benchmark val_sols1 = run_reservoir_trajectories(reservoir, validation_input_states[1:M_val], tmax, ODE(DP8()); int_alg, time_trace=false, ensemblealg = EnsembleThreads(), abstol);
@benchmark val_sols2 = run_reservoir_trajectories(reservoir, validation_input_states[1:M_val], tmax, IntegratedODE(DP8()); int_alg, time_trace=false, ensemblealg = EnsembleThreads(), abstol)

mm' * stack(map(rho -> vecrep(rho, reservoir), training_input_states[1:M_train])) - training_sols.integrated |> norm
##
traj = run_reservoir_trajectories(reservoir, training_input_states[1:2], tmax, alg; time_trace=true, abstol)
foreach(c -> plot(c') |> display, traj.currents)
##
y_train = stack(get_training_data.(training_input_states[1:size(training_sols.integrated, 2)]))
y_val = stack(get_training_data.(validation_input_states[1:size(val_sols.integrated, 2)]))
X_train = training_sols.integrated
X_val = val_sols.integrated
W = fit_output_layer(X_train, y_train)
loss_function_density_matrix(W, X_val, y_val)
##
data = []
@showprogress dt = 1 desc = "Running reservoir..." for params in reservoir_parameters[1:50]
    alg = Exponentiation()
    reservoir = prepare_reservoir(rc, params, (T, μs); abstol)
    mm = measurement_matrix(reservoir, tmax, alg; abstol)
    training_sols = run_reservoir_trajectories(reservoir, training_input_states[1:M_train], tmax, alg; time_trace=false, abstol)
    val_sols = run_reservoir_trajectories(reservoir, validation_input_states[1:M_val], tmax, alg; time_trace=false, abstol)

    y_train = stack(get_training_data.(training_input_states[1:M_train]))
    y_val = stack(get_training_data.(validation_input_states[1:M_val]))
    X_train = training_sols.integrated
    X_val = val_sols.integrated
    W = fit_output_layer(X_train, y_train)
    loss = loss_function_density_matrix(W, X_val, y_val)
    push!(data, (; loss, cond=cond(mm), mm, W, y_train, y_val, X_train, X_val))
end
##
data0 = deepcopy(data)
##
sp0 = sortperm(map(d -> d.loss, data0))
sp1 = sortperm(map(d -> d.loss, data1))

plot(map(d -> log(d.loss), data0[sp0]));
plot!(map(d -> log(d.loss), data1[sp1]))

plot(map(d -> log(d.cond), data0[sp0]));
plot!(map(d -> log(d.cond), data1[sp1]))
proj = cat([0;;], one(rand(4, 4)), [0;;]; dims=(1, 2))[2:5, :]

##
subtract_mean!(x) = (x .-= mean(x); x)
##
# let data = data0[sp0]
let data = data1[sp1]
    plot(map(d -> log(d.loss), data) |> subtract_mean!, lw = 2, label="loss")
    plot!(map(d -> log(d.cond), data) |> subtract_mean!, label="cond")
    plot!(map(d -> -log(minimum(svd(d.mm).S)), data) |> subtract_mean!, label="min(svd(m))")
    plot!(map(d -> log(cond(proj * d.mm)), data) |> subtract_mean!, label="cond(proj*m)", ls = :dash)
    plot!(map(d -> -log(minimum(svd(proj * d.mm).S)), data) |> subtract_mean!, label="min(svd(proj*m))", ls = :dash)
end
##
let f = d -> log(cond(proj * d.mm))
    plot(map(f, data0[sp0]))
    plot!(map(f, data1[sp1]))
end
let f = d -> log(svd(proj * d.mm).S |> minimum)
    plot(map(f, data0[sp0]))
    plot!(map(f, data1[sp1]))
end
let f = d -> log.(svd(d.mm).S)
    heatmap(stack(map(f, data0[sp0])); clim=(-12, 1), c=:viridis) |> display
    heatmap(stack(map(f, data1[sp1])); clim=(-12, 1), c=:viridis)
end
let f = d -> log.(svd(d.mm).S)
    plot(stack(map(f, data0[sp0])); ylims=(-12, 1), c=:viridis) |> display
    plot(stack(map(f, data1[sp1])); ylims=(-12, 1), c=:viridis)
end


##
let d0 = data0, d1 = data1, f = log
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
is = 3:8
X = X_train
y = y_train[is, :]
ridge = RidgeRegression(1e-6; fit_intercept=true)
W0 = fit_output_layer(X, y, β=1e-6)
W1 = reduce(hcat, map(data -> fit(ridge, X', data), eachrow(y))) |> permutedims
W2 = y * X' * inv(X * X' + 1e-8 * I)
W3 = y * pinv(X)
##
titles = ["entropy of one input dot", "purity of inputs", "ρ11", "ρ22", "ρ33", "ρ44", "real(ρ23)", "imag(ρ23)"]
colors = cgrad(:seaborn_dark, length(titles))
colors2 = cgrad(:seaborn_bright, length(titles))
let perm, W = W0, X = X_val, y = y_val[is, :], b
    p = plot(; size=1.2 .* (600, 400))
    # colors2 = cgrad(:seaborn_dark, size(y, 2))
    y_pred = reconstructed_data(W, X)
    for (n, i) in enumerate(is)
        perm = sortperm(y[n, :])
        plot!(p, y_pred[n, perm]; label=titles[i] * " pred", lw=3, c=colors[i], frame=:box)
        # plot!(p, (Wi' * X[:, perm])' .+ b; label=titles[i] * " pred", lw=3, c=colors[i], frame=:box)
        plot!(y[n, perm]; label=titles[i] * " truth", lw=3, ls=:dash, c=colors2[i])
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
