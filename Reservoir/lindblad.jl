

struct Lindblad{H,L,EL,C}
    H::H
    lead_ops::L
    vectorizer::V
    cache::C
end
function ratetransform!(out, cache::Matrix, op, eigvecs::Matrix, T, μ)
    mul!(cache, eigvecs', op)
    mul!(out, cache, eigvecs)
    QuantumDots.ratetransform_energy_basis!(out, diagham.values, T, μ)
    mul!(cache, eigvecs', out)
    mul!(out, cache, eigvecs)
end

# function ratetransform!(out, energy_basis_op, T, μ)
#     _ratetransform_energy_basis!(out, diagham.values, T, μ)
# end

function _ratetransform_energy_basis!(out, op, energies::AbstractVector, T, μ)
    for I in CartesianIndices(op)
        n1, n2 = Tuple(I)
        δE = energies[n1] - energies[n2]
        out[n1, n2] = sqrt(fermidirac(δE, T, μ)) * op[n1, n2]
    end
    return op
end

function superoperator!(superop, lead_op_energy_basis, diagham, T, μ, rate, vectorizer, cache)
    _ratetransform_energy_basis!(leadopcache, lead_op_energy_basis, diagham.values, T, μ)
    return QuantumDots.dissipator!(superop, cache.opcache, rate, vectorizer, cache.kroncache, cache.mulcache)
end


function lindblad_matrix!(total, unitary, dissipators)
    fill!(total, zero(eltype(total)))
    total .+= unitary
    for d in dissipators
        total .+= d.superop
    end
    return total
end

function Lindblad(H::BlockDiagonal, leads)
    diagham = diagonalize(hamiltonian)
    matrixdiagham = QuantumDots.DiagonalizedHamiltonian(collect(diagham.values), collect(diagham.vectors), collect(hamiltonian))
    commutator_hamiltonian = QuantumDots.commutator(hamiltonian, vectorizer)
    unitary = -1im * commutator_hamiltonian
    # cache = usecache ? LindbladCache(unitary, hamiltonian) : nothing
    cache = (; superopcache=similar(unitary), matrixopcache=similar(matrixdiagham.hamiltonian), blockopcache=similar(hamiltonian), kroncache=similar(unitary), mulcache=similar(unitary))

    dissipators = map((lead, rate) -> (; superop=superoperator!(similar(unitary), lead, matrixdiagham, rate, vectorizer, cache), rate, lead), leads, rates)
    total = similar(unitary)
    lindblad_matrix!(total, unitary, dissipators)
    LindbladSystem(total, unitary, dissipators, vectorizer, hamiltonian, cache)

end
function superoperator!(lead_op, diagham, T, μ, rate, vectorizer, cache)
    op = ratetransform!(lead_op, cache.matrixcache, diagham, T, μ)
    return dissipator!(cache.superopcache, op, rate, vectorizer, cache.kroncache, cache.mulcache)
end
function update_lindblad_system!(L::LindbladSystem, p, tmp=L.cache)
    # println("__")
    _newdissipators = map(lp -> first(lp) => update_dissipator(L.dissipators[first(lp)], last(lp), tmp), collect(pairs(p)))
    newdissipators = merge(L.dissipators, _newdissipators)
    # newdissipators = mergewith!(pairs(L.dissipators), pairs(p)) do d, p
    #     QuantumDots.update_coefficients(d, p, tmp)
    # end
    # # println(newdissipators)
    # println(newdissipators2)
    # d = mergewith((d1, d2) -> norm(d1.superop - d2.superop), pairs(newdissipators), newdissipators2)
    # println(d)
    total = lindblad_matrix!(L.total, L.unitary, newdissipators)
    LindbladSystem(total, L.unitary, newdissipators, L.vectorizer, L.hamiltonian, L.cache)
end

function update_dissipator!(d, p, tmp=d.cache)
    rate = get(p, :rate, d.rate)
    newlead = QuantumDots.update_lead(d.lead, p)
    newsuperop = superoperator!(newlead, d.ham, rate, d.vectorizer, tmp)
    LindbladDissipator(newsuperop, rate, newlead, d.ham, d.vectorizer, d.cache)
end