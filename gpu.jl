using CUDA

struct GPU <: AbstractPU end
function time_evolve(proc::GPU, rho, A, args...; occ_ops, current_ops, kwargs...)
    time_evolve(cu(rho), MatrixOperator(cu(A.A)), map(arg -> Float32.(arg), args)...; current_ops=cu.(current_ops), occ_ops=cu.(occ_ops), kwargs...)
end