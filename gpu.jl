using CUDA
# This is much slower than the CPU version, but it works.
struct GPU <: AbstractPU end
function _time_evolve(proc::GPU, rho, A, args...; occ_ops, current_ops, kwargs...)
    _time_evolve(cu(rho), MatrixOperator(cu(A.A)), map(arg -> Float32.(arg), args)...; current_ops=cu.(current_ops), occ_ops=cu.(occ_ops), kwargs...)
end

function time_evolve(proc::GPU, rho, ls::LindbladSystem, args...; kwargs...)
    A = QuantumDots.LinearOperator(ls)
    _time_evolve(proc, LinearOperatorRep(rho, ls), A, args...; kwargs...)
end