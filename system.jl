struct ReservoirConnections{L,C,Cl,B}
    labels::L
    Ilabels::L
    Ihalflabels::L
    Rlabels::L
    hopping_labels::C
    Iconnections::C
    Rconnections::C
    IRconnections::C
    lead_connections::Cl
    bases::B
end
function ReservoirConnections(N, M=1; qn=QuantumDots.fermionnumber)
    labels = vec(Base.product(0:N, 1:2) |> collect)
    hopping_labels = [(labels[k1], labels[k2]) for k1 in 1:length(labels), k2 in 1:length(labels) if k1 > k2 && is_nearest_neighbours(labels[k1], labels[k2])]
    Ilabels = filter(x -> first(x) <= 0, labels)
    Rlabels = filter(x -> first(x) > 0, labels)
    Ihalflabels = filter(x -> isone(x[2]), Ilabels)
    Iconnections = filter(k -> first(k[1]) + first(k[2]) == 0, hopping_labels)
    Rconnections = filter(k -> first(k[1]) + first(k[2]) > 1, hopping_labels)
    IRconnections = filter(k -> abs(first(k[1]) - first(k[2])) == 1, hopping_labels)
    lead_connections = [(m, [(N, k) for k in 1:2]) for m in 1:M]

    cI = FermionBasis(Ilabels; qn)
    cR = FermionBasis(Rlabels; qn)
    cIR = wedge(cI, cR)

    return ReservoirConnections(labels, Ilabels, Ihalflabels, Rlabels, hopping_labels, Iconnections, Rconnections, IRconnections, lead_connections, (; cI, cR, cIR))
end

hopping_hamiltonian(c::FermionBasis{0}, J; labels=keys(J)) = [0;;]
function hopping_hamiltonian(c, J; labels=keys(J))
    T = valtype(J)
    H = deepcopy(zero(T) * first(c))
    length(J) == 0 && return H
    for (k1, k2) in labels
        H .+= J[(k1, k2)] * c[k1]'c[k2] + hc
    end
    return blockdiagonal(H, c)
end
coulomb_hamiltonian(c::FermionBasis{0}, V; labels=keys(V)) = [0;;]
function coulomb_hamiltonian(c, V; labels=keys(V))
    T = valtype(V)
    H = deepcopy(zero(T) * first(c))
    length(V) == 0 && return H
    for (k1, k2) in labels
        H .+= V[(k1, k2)] * c[k1]'c[k1] * c[k2]'c[k2]
    end
    return blockdiagonal(H, c)
end
qd_level_hamiltonian(c::FermionBasis{0}, ε; labels=keys(ε)) = [0;;]
function qd_level_hamiltonian(c, ε; labels=keys(ε))
    T = valtype(ε)
    H = deepcopy(zero(T) * first(c))
    length(ε) == 0 && return H
    for l in labels
        H .+= ε[l] * c[l]'c[l]
    end
    return blockdiagonal(H, c)
end

is_nearest_neighbours(k1, k2) = k1 != k2 && all(map((l1, l2) -> abs(l1 - l2) ∈ (0, 1), k1, k2))
# function random_nearest_hoppings(spatial_labels, s=1)
#     couplings = [(k1, k2) => s * rand() * is_nearest_neighbours(k1, k2) for k1 in spatial_labels, k2 in spatial_labels]
#     Dict(couplings)
# end

function one_hoppings(labels, s=1)
    couplings = [(k1, k2) => s * 1.0 for (k1, k2) in labels]
    Dict(couplings)
end
function random_hoppings(labels, s=1)
    couplings = [(k1, k2) => s * rand() for (k1, k2) in labels]
    Dict(couplings)
end
function random_nearest_hoppings(labels, s=1)
    couplings = [(k1, k2) => s * rand() for (k1, k2) in labels if is_nearest_neighbours(k1, k2)]
    Dict(couplings)
end
