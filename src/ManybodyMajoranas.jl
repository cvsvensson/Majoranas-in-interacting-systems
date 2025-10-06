module ManybodyMajoranas
using LinearAlgebra, BlockDiagonals, SparseArrays, ArnoldiMethod
using RandomMatrices
using UnPack
using Reexport
using Roots

@reexport using FermionicHilbertSpaces
import FermionicHilbertSpaces: AbstractHilbertSpace, AbstractFockHilbertSpace, FockHilbertSpace, SymmetricFockHilbertSpace, FockSymmetry, dim, project_on_parities

export schatten_norm, conjugate_norm
export kitaev_hamiltonian, sweet_spot_μ, optimized_Δ
export ground_states_arnoldi, blockeigen, reduced_majoranas_properties
export effective_hamiltonian, canonicalize_hamiltonians, hilbert_spaces, random_hamiltonian, decompose_coupling
export FrobeniusGauge, EigGauge
include("src.jl")

end