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
