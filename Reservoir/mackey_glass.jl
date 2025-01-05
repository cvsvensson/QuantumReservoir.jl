using DelayDiffEq
##
default_mackey_glass_parameters = (; γ=1, β=2, τ=2, n=9.65)
function f_mackey_glass(u, h, p, t)
    (; β, γ, τ, n) = p

    z = h(p, t - τ)
    β * z / (1 + z^n) - γ * u
    # 0.2 * z / (1 + z^10) - 0.1 * u
end
function h_mackey_glass(p, t)
    t ≤ 0 || error("history function is only implemented for t ≤ 0")
    0.5
end
function mackey_glass_problem(p, tspan)
    DDEProblem(f_mackey_glass, h_mackey_glass, tspan, p; constant_lags=[p.τ])
end
function mackey_glass_solution(p=default_mackey_glass_parameters, tspan=(0, 100), alg=MethodOfSteps(Vern9(); fpsolve=NLFunctional(; max_iter=1000)); reltol=1e-14, abstol=1e-14, kwargs...)
    solve(mackey_glass_problem(p, tspan), alg; reltol, abstol, kwargs...)
end
##
@time sol = mackey_glass_solution();