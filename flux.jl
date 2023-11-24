using Flux, Statistics, ProgressMeter, Plots

function get_model(y, W)
    n = size(W, 2) - 1
    m = size(W, 1)
    return Chain(
        Dense(W[:, 1:end-1], W[:, end], tanh),
        # Dense(m => m, tanh; bias=true),
        Dense(m => size(y, 1)))
end
function get_model2(y, W)
    n = size(W, 2) - 1
    m = size(W, 1)
    m = 3
    return Chain(
        # Dense(n => m),
        Dense(n => m, tanh; bias=true),
        # Dense(n => size(y, 1), tanh; bias=true),)
        Dense(m => size(y, 1)))
end

# X2 = Float32.([X; ones(M)'])
X2 = Float32.(X)
y2 = Float32.(y[1:2, :])
model = get_model2(y2, Float32.(W1))
loader = Flux.DataLoader((X2, y2), batchsize=20, shuffle=true);
# optim = Flux.setup(Flux.Adam(0.001), model)
optim = Flux.setup(OptimiserChain(WeightDecay(1.0f-12), Adam(0.01)), model)
##
# Training loop, using the whole data set 1000 times:
losses = []
@showprogress for epoch in 1:5_000
    for (x, y) in loader
        _loss, grads = Flux.withgradient(model) do m
            # Evaluate model and loss inside gradient context:
            y_hat = m(x)
            sum(abs2.(y_hat .- y))
        end
        Flux.update!(optim, model, grads[1])
        push!(losses, _loss)  # logging, outside gradient context
    end
end

##
plot(losses; xaxis=(:log10, "iteration"),
    yaxis="loss", label="per batch");
n = length(loader)
plot!(n:n:length(losses), mean.(Iterators.partition(losses, n)),
    label="epoch mean", dpi=200)

##
titles = ["entropy of one input dot", "purity of inputs", "ρ11", "ρ22", "ρ33", "ρ44", "real(ρ23)", "imag(ρ23)", "n1", "n2"]
let is = 1:2, perm, X = test_sols.data, y = test_ensemble.data, b
    p = plot(; size=1.2 .* (600, 400))
    colors = cgrad(:seaborn_dark, size(y, 1))
    # colors2 = cgrad(:seaborn_dark, size(y, 2))
    colors2 = cgrad(:seaborn_bright, size(y, 1))
    for i in is
        perm = sortperm(y[i, :])
        yout = model(X)[i, perm]
        plot!(p, yout; label=titles[i] * " pred", lw=3, c=colors[i], frame=:box)
        plot!(y[i, perm]; label=titles[i] * " truth", lw=3, ls=:dash, c=colors2[i])
    end
    display(p)
end