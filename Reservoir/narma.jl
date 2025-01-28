struct Narma{I,P}
    input::I
    params::P
end
##
function narma(n, p, input, initial=zeros(n))
    y = deepcopy(initial)
    N = length(input) - n + 1
    for i in 1:N
        _y = @view input[i:i+n-1]
        _u = @view input[i:i+n-1]
        next_y = narma_step(_y, _u, p)
        push!(y, next_y)
    end
    return y
end
#https://arxiv.org/pdf/1906.04608v3
default_narma_parameters = (; α=0.3, β=0.05, γ=1.5, δ=0.1)
function narma_step(y, u, p)
    (; α, β, γ, δ) = p
    α * last(y) + β * last(y) * sum(y) + γ * last(u) * first(u) + δ
end
##
plot(narma(10, default_narma_parameters, 1 .+ 0.5 * rand(100)))
