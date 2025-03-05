function MLJLinearModels.fit(target, measurements; β=1e-6, fit_intercept=true)
    ridge = RidgeRegression(β; fit_intercept)
    reduce(hcat, map(data -> MLJLinearModels.fit(ridge, measurements, data), eachcol(target)))
end
function predict(W, X)
    if size(W, 1) == size(X, 2)
        return X * W
    end
    X * W[1:end-1, :] .+ ones(size(X, 1)) * W[end, :]'
end


function fit(measurements, targets; warmup, train)
    N = length(measurements)
    stacked_measurements = stack(measurements)
    n_train_first = Int(div(N, 1/warmup))
    n_test_first = max(n_train_first + 1, Int(div(N, 1/(warmup+train))))
    n_train = n_train_first:n_test_first-1
    n_test = n_test_first:N

    Xtrain = permutedims(stacked_measurements[:,n_train])
    ytrain = stack([target[n_train] for (name,target) in targets])
    Xtest = permutedims(stacked_measurements[:,n_test])
    ytest = stack([target[n_test] for (name,target) in targets])
    W = MLJLinearModels.fit(ytrain, Xtrain)
    ztrain = predict(W, Xtrain)
    ztest = predict(W, Xtest)
    mses = [mean((ytest .- ztest) .^ 2) for (ytest, ztest) in zip(eachcol(ytest), eachcol(ztest))]
    memory_capacities = [(cov(ztest[:], ytest[:]) / (std(ztest) * std(ytest)))^2 for (ytest, ztest) in zip(eachcol(ytest), eachcol(ztest))]
    pca = MultivariateStats.fit(PCA, stacked_measurements)
    return (; W, mses, memory_capacities, ztrain, ztest, targets, n_train_first, n_test_first, n_train, n_test, pca)
end

DelayedSignal(signal, delay::Int, history=zeros(delay)) = vcat(history, signal[1:end-delay])
