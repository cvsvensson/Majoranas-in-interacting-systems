using Test
using UnPack
using ManybodyMajoranas
using LinearAlgebra
using Random: seed!

@testset "Effective Hamiltonian" begin
    seed!(2)
    ## system, bath
    qn = ParityConservation()
    S = 1:2
    B = 3:4
    R = 2:2
    spaces = hilbert_spaces(S, R, B, qn)

    ## hamiltonians
    hams = (; hS0=random_hamiltonian(spaces.HS), hS=random_hamiltonian(spaces.HS), hB=random_hamiltonian(spaces.HB), hRB=random_hamiltonian(spaces.HRB))

    canon_hams = canonicalize_hamiltonians(hams, spaces)
    @test ManybodyMajoranas.iscanonical(canon_hams, spaces)
    @test norm(values(canonicalize_hamiltonians(canon_hams, spaces)) .- values(canon_hams)) < 1e-10

    ## effective hamiltonian and majoranas
    vals, vecs = blockeigen(canon_hams.hS0, spaces.HS)
    gs_odd = vecs[:, 1]
    gs_even = vecs[:, div(size(vecs, 2), 2)+1]
    q, p = 1, Inf
    #q, p = 2, 2
    reduced = reduced_majoranas_properties(gs_even, gs_odd, spaces.HS, spaces.HR; q)
    γSmin, γSmax = γS = reduced[[:γmin, :γmax]]
    heff = effective_hamiltonian(canon_hams, spaces, γS)

    spaces_gs = hilbert_spaces((:gs,), (:gs,), B, qn)
    hams_gs = (; hS0=heff.heffgs, hS=heff.heffgs, hB=canon_hams.hB, hRB=heff.heffgsB)
    heff2 = effective_hamiltonian(hams_gs, spaces_gs, heff.γgs)
    @test heff.effops.Fmin ≈ heff2.effops.Fmin
    @test heff.effops.Fmax ≈ heff2.effops.Fmax
    @test heff.effops.G ≈ heff2.effops.G
    @test heff.effops.ε ≈ heff2.effops.ε

    ## Bounds
    odd_coupling, even_coupling = decompose_coupling(canon_hams.hRB, spaces.HRB, spaces.HR, spaces.HB)
    @test even_coupling + odd_coupling ≈ canon_hams.hRB

    @test norm(svdvals(heff.effops.Fmin), p) <= reduced.LFmin * norm(svdvals(odd_coupling), p) + 10eps()
    @test norm(svdvals(heff.effops.Fmax), p) <= reduced.LFmax * norm(svdvals(odd_coupling), p) + 10eps()
    @test norm(svdvals(heff.effops.G), p) <= reduced.LD * norm(svdvals(even_coupling), p) + 10eps()

    γgsBmin, γgsBmax = [embed(γ, spaces_gs.Hgs => spaces_gs.HgsB) for γ in heff.γgs]
    hs_from_tos = [(heff.heffgs, spaces.Hgs => spaces.HgsB), (heff.heffgsB, spaces.HgsB => spaces.HgsB), (canon_hams.hB, spaces.HB => spaces.HgsB)]
    hgsBcoupled = sum(embed(h, from_to) for (h, from_to) in hs_from_tos)

    vals, vecs = blockeigen(hgsBcoupled, spaces_gs.HgsB)
    n = div(size(vecs, 2), 2)
    gs_odd = vecs[:, 1]
    gs_even = vecs[:, n+1]
    odd_eigenstates, even_eigenstates = vecs.blocks
    overlaps1 = odd_eigenstates' * γgsBmin[1:n, n+1:2n] * even_eigenstates
    overlaps2 = odd_eigenstates' * 1im * γgsBmax[1:n, n+1:2n] * even_eigenstates # check the sign
    δEs = map(es -> -(es...), Base.product(vals[1:n], vals[n+1:2n]))
    ε = heff.effops.ε

    LHSs = [abs(δE * overlap1 - overlap2 * ε) for (δE, overlap1, overlap2) in zip(δEs, overlaps1, overlaps2)]
    Neven = norm(svdvals(even_coupling), p)
    Nodd = norm(svdvals(odd_coupling), p)

    RHS = (reduced.LD * Neven + reduced.LFmin * Nodd)
    @test all(LHSs .< RHS)

end


@testset "Effective ham vs original" begin
    seed!(2)

    qn = ParityConservation()
    S = 1:2
    B = 3:4
    R = 2:2
    spaces = hilbert_spaces(S, R, B, qn)
    @unpack HS, HR, HSB, HRB, HB = spaces
    q = 2

    @fermions f
    hS0 = random_hamiltonian(HS)
    hB0 = 0 * random_hamiltonian(HB)
    hRB0 = 0 * random_hamiltonian(HRB)

    hams = (; hS0=hS0, hS=hS0, hB=hB0, hRB=hRB0)
    canon_hams = canonicalize_hamiltonians(hams, spaces)
    @unpack hS, hRB, hB = canon_hams
    hcanon = embed(hS, HS => HSB) + embed(hRB, HRB => HSB) + embed(hB, HB => HSB)
    h = embed(hS0, HS => HSB) + embed(hRB0, HRB => HSB) + embed(hB0, HB => HSB)
    @test norm(hcanon - h) < 1e-10

    vals, vecs = blockeigen(hS, HS)
    gs_odd = vecs[:, 1]
    gs_even = vecs[:, div(size(vecs, 2), 2)+1]
    reduced = reduced_majoranas_properties(gs_even, gs_odd, HS, HR; q)
    (γSmin, γSmax) = γS = reduced[[:γmin, :γmax]]
    heff = effective_hamiltonian(canon_hams, spaces, γS)
    ε = heff.effops.ε

    γSBmin, γSBmax = map(γ -> embed(γ, HS => HSB), γS)
    P = embed(γSmin^2, HS => HSB)
    @test P^2 ≈ P
    G = embed(heff.effops.G, HB => HSB)
    Fmin = embed(heff.effops.Fmin, HB => HSB)
    Fmax = embed(heff.effops.Fmax, HB => HSB)
    h = ((ε * I + G) * 1im * γSBmin * γSBmax + γSBmin * Fmin + γSBmax * Fmax) / 2 + embed(hB, HB => HSB)
    @test P * h * P ≈ h
    h2 = embed(hS, HS => HSB) + embed(hRB, HRB => HSB) + embed(hB, HB => HSB)
end

