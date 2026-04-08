module ManybodyMajoranas
using LinearAlgebra, BlockDiagonals, SparseArrays, ArnoldiMethod
using RandomMatrices
using UnPack
using Reexport
using Roots
using LowRankMatrices

@reexport using FermionicHilbertSpaces
import FermionicHilbertSpaces: AbstractHilbertSpace, AbstractFockHilbertSpace, FockHilbertSpace, SymmetricFockHilbertSpace, FockSymmetry, dim, project_on_parities

export schatten_norm, conjugate_norm
export kitaev_hamiltonian, sweet_spot_μ, frustration_free_μ
export ground_states_arnoldi, blockeigen, reduced_majoranas_properties, calculate_bounds
export effective_hamiltonian, canonicalize_hamiltonians, hilbert_spaces, random_hamiltonian, decompose_coupling
export FrobeniusGauge, EigGauge
export good_majoranas_parameters, global_parameters, degenerate_good_majorana_parameters, bad_majoranas_parameters

include("src.jl")
include("parameters.jl")

## "Trick" LSP so that stuff works in scripts files
@static if false
    include("../scripts/kitaev_analytics.jl")
end

end
