
function conjugate_norm(q)
    if q == 1
        return Inf
    elseif q == 2
        return 2
    elseif q == Inf
        return 1
    else
        return 1 / (1 - 1 / q)
    end
end

schatten_norm(m, p) = norm(svdvals(Matrix(m)), p)


function effective_operators((γmin, γmax), hS, hSB, spaces)
    @unpack HS, HSB, HB = spaces
    γSBmin = embed(γmin, HS => HSB; complement=HB)
    γSBmax = embed(γmax, HS => HSB; complement=HB)
    δρ = 1im * γSBmax * γSBmin
    Fmin = partial_trace(γSBmin * hSB, HSB => HB)
    Fmax = partial_trace(γSBmax * hSB, HSB => HB)
    G = partial_trace(δρ * hSB, HSB => HB) |> Hermitian
    ε = real(tr(1im * γmax * γmin * hS))
    return (; Fmin, Fmax, G, ε)
end

function effective_hamiltonian_parts((γmin, γmax), effective_operators, HS, HSB, HB)
    @unpack Fmin, Fmax, G, ε = effective_operators
    tp = tensor_product((HS, HB) => HSB)
    hgsB = (tp(γmin, Fmin) + tp(γmax, Fmax) + tp(1im * γmax * γmin, G)) / 2
    hgs = ε * 1im * γmax * γmin / 2
    return hgs, hgsB
end

function canonicalize_hamiltonians(hamiltonians, spaces)
    @unpack hS, hS0, hB, hRB = hamiltonians
    @unpack HS, HB, HR, HSB, HRB = spaces
    # diffHR = HS.ham - HS0.ham
    diffhR = partial_trace(hS - hS0, HS => HR) * dim(HR) / dim(HS)
    hS = hS - embed(diffhR, HR => HS)
    hRB = hRB + embed(diffhR, HR => HRB)
    # remove part of hRB only in B
    diffhB = partial_trace(hRB, HRB => HB) / dim(HR)
    hRB = hRB - embed(diffhB, HB => HRB)
    hB = hB + diffhB
    # make them traceless
    hRB = hRB - tr(hRB) * I / dim(HRB)
    return (; hamiltonians..., hS, hS0, hB, hRB)
end

function iscanonical(hamiltonians, spaces)
    @unpack hS, hS0, hB, hRB = hamiltonians
    @unpack HS, HB, HR, HRB, HSB = spaces
    hRBinB_norm = norm(partial_trace(hRB, HRB => HB))
    if hRBinB_norm > 1e-8
        println("hRB in B norm: $hRBinB_norm")
        return false
    end
    partial_trace(hS, HS => HR) ≈ partial_trace(hS0, HS => HR) || (println(norm(partial_trace(hS - hS0, HS => HR))); return false) # The part of HS in R is zero. That part is included in HRB
    return true
end
#=remove_trace(H) = H - I * tr(H) / size(H, 1)=#

function hilbert_spaces(S, R, B, qn=ParityConservation())
    all(r in S for r in R) || throw(ArgumentError("R must be a subset of S"))
    isdisjoint(S, B) || throw(ArgumentError("S and B must be disjoint"))
    HS = hilbert_space(S, qn)
    HB = hilbert_space(B, qn)
    HSB = hilbert_space(union(S, B), qn)

    HR = hilbert_space(R, qn)
    HRB = hilbert_space(union(R, B), qn)

    gs_label = 0
    (gs_label in S || gs_label in B) && throw(ArgumentError("The label 0 is reserved for the ground state space"))
    Hgs = hilbert_space((gs_label,), qn)
    HgsB = hilbert_space(union((gs_label,), B), qn)
    (; HS, HB, HR, HSB, HRB, Hgs, HgsB)
end

function effective_hamiltonian(hamiltonians, spaces, (γmin, γmax); check_canonical=true)
    @unpack hS, hS0, hB, hRB = hamiltonians
    @unpack HS, HB, HR, HSB, HRB, Hgs, HgsB = spaces
    if check_canonical
        iscanonical(hamiltonians, spaces) || throw(ArgumentError("Hamiltonians are not canonical"))
    end
    hSB = embed(hRB, HRB => HSB)
    effops = effective_operators((γmin, γmax), hS, hSB, spaces)
    @fermions f
    γmings = matrix_representation(f[0] + hc, Hgs)
    γmaxgs = matrix_representation(1im * f[0] + hc, Hgs)
    heffgs, heffgsB = effective_hamiltonian_parts((γmings, γmaxgs), effops, Hgs, HgsB, HB)
    heffS, heffSB = effective_hamiltonian_parts((γmin, γmax), effops, HS, HSB, HB)
    hs_from_tos = [(heffgs, Hgs => HgsB),
        (heffgsB, HgsB => HgsB),
        (hB, HB => HgsB)]
    total_ham = sum(embed(h, from_to) for (h, from_to) in hs_from_tos)
    return (; total_ham, heffS, heffSB, heffgs, heffgsB, effops, γgs=(γmings, γmaxgs))
end


function random_hamiltonian(H::AbstractFockHilbertSpace, β=2)
    ensemble = GaussianHermite(β)
    Hermitian(rand(ensemble, dim(H)))
end
function random_hamiltonian(H::SymmetricFockHilbertSpace, β=2)
    ensemble = GaussianHermite(β)
    mats = [rand(ensemble, length) for (length) in (length.(H.symmetry.qntofockstates))]
    Hermitian(cat(mats..., dims=(1, 2)))
end

struct Rank1Matrix{T} <: AbstractMatrix{T}
    vec1::Vector{T}
    vec2::Vector{T}
end
Base.getindex(m::Rank1Matrix, i::Int, j::Int) = m.vec1[i] * conj(m.vec2[j])
Base.size(m::Rank1Matrix) = (length(m.vec1), length(m.vec2))

function abs_sign_mat!(m::Hermitian{T}; cutoff=10eps(real(T))) where T
    vals, vecs = eigen!(m)
    absvals::Vector{T} = abs.(vals)
    signvals::Vector{T} = map(v -> abs(v) > cutoff ? sign(v) : zero(T), vals)
    return vecs * Diagonal(absvals) * vecs', vecs * Diagonal(signvals) * vecs'
end

abstract type AbstractGauge end
struct LFMinimization <: AbstractGauge end
struct FrobeniusGauge <: AbstractGauge end
struct EigGauge <: AbstractGauge end

function optimal_gauge(oeR, ::LFMinimization, q)
    function root_function(θ)
        γR = Hermitian(exp(1im * θ) * oeR + hc)
        γRtilde = Hermitian(exp(1im * (θ + pi / 2)) * oeR + hc)
        absγR, sgnγR = Hermitian.(abs_sign_mat!(γR; cutoff=0))
        real(tr(sgnγR * absγR^(q - 1) * γRtilde))
    end
    θs = find_zeros(root_function, -pi / 2, pi / 2)
    LFs = [norm(svdvals(exp(1im * θ) * oeR + hc), q) for θ in θs]
    θmin = θs[argmin(LFs)]
end
function optimal_gauge(oeR, ::FrobeniusGauge, q)
    γ = Hermitian(oeR + hc)
    γtilde = Hermitian(1im * oeR + hc)
    θ = 1 / 2 * atan(2 * real(tr(γ * γtilde)) / (norm(γ)^2 - norm(γtilde)^2))

    if norm(exp(1im * θ) * oeR + hc) > norm(exp(1im * (θ + pi / 2)) * oeR + hc)
        θ = minimum(abs, (θ + pi / 2, θ - pi / 2))
    end
    θ

end
function optimal_gauge(oeR, ::EigGauge, q)
    @warn "EigGauge does not minimize anything, the eigenvectors returned from the eigensolver are used."
    return 0
end

function reduced_majoranas_properties(e, o, H::AbstractHilbertSpace, Hsub::AbstractHilbertSpace, gauge=FrobeniusGauge(); q=1, opt_kwargs=Dict())
    eo = Rank1Matrix(e, o)
    ee = Rank1Matrix(e, e)
    oo = Rank1Matrix(o, o)
    eoR = partial_trace(eo, H => Hsub)
    eeR = partial_trace(ee, H => Hsub)
    ooR = partial_trace(oo, H => Hsub)
    θmin = optimal_gauge(eoR, gauge, q, opt_kwargs...)
    θmax = θmin + pi / 2
    γRmin, γRmax = (exp(1im * θmin) * eoR + hc, exp(1im * θmax) * eoR + hc)
    LFmin, LFmax = map(γ -> norm(svdvals(γ), q), (γRmin, γRmax))
    MR = sqrt(abs(tr(γRmax^2 - γRmin^2))^2 + 4 * abs(tr(γRmax * γRmin))^2) / abs(tr(γRmax^2 + γRmin^2))
    cR = abs(tr(γRmax^2 + γRmin^2)) / 2
    LD = norm(svdvals(ooR - eeR), q)
    return (; LFmin, LFmax, LD, θmin, θmax, γRmin, γRmax, MR, cR)
end

blockdiagonal(m, H::SymmetricFockHilbertSpace) = blockdiagonal(m, H.symmetry)
function blockdiagonal(m::AbstractMatrix, sym::FockSymmetry)
    blockinds = [map(state -> sym.state_indexdict[state], block) for block in values(sym.qntofockstates)]
    BlockDiagonal([m[block, block] for block in blockinds])
end
blockdiagonal(m::Hermitian, sym::FockSymmetry) = Hermitian(blockdiagonal(m.data, sym))
blockeigen(m, H::AbstractHilbertSpace) = BlockDiagonals.eigen_blockwise(blockdiagonal(m, H))
blockeigen(m::SparseArrays.SparseMatrixCSC, H::AbstractHilbertSpace) = BlockDiagonals.eigen_blockwise(blockdiagonal(Matrix(m), H))
blockeigen(m::Hermitian{<:Any,<:SparseArrays.SparseMatrixCSC}, H::AbstractHilbertSpace) = BlockDiagonals.eigen_blockwise(blockdiagonal(Hermitian(Matrix(m)), H))

function ground_states_arnoldi(m::SparseArrays.SparseMatrixCSC, H::AbstractHilbertSpace; kwargs...)
    ms = blocks(blockdiagonal(m, H))
    eigens = map(ms) do m
        decomp, history = partialschur(Hermitian(m), nev=1, which=:SR; kwargs...)
        # @show history
        partialeigen(decomp)
    end
    vals = map(first ∘ first, eigens)
    vecs = [e[2][:, 1] for e in eigens]
    vals, vecs
end

function decompose_coupling(hRB::AbstractMatrix, HRB, HR, HB)
    proj(parities...) = FermionicHilbertSpaces.project_on_parities(hRB, HRB, (HR, HB), parities)
    hodd = proj(-1, -1)
    heven = proj(1, 1)
    return hodd, heven
end


hopping(t, f1, f2) = t * f1'f2 + hc
pairing(Δ, f1, f2) = Δ * f1' * f2' + hc
numberop(f) = f'f
coulomb(f1, f2) = f1' * f1 * f2' * f2
_kitaev_2site(f1, f2; t, Δ, U) = hopping(t, f1, f2) + U * coulomb(f1, f2) + pairing(Δ, f1, f2)
_kitaev_1site(f; μ) = μ * numberop(f)

getvalue(v::Union{<:AbstractVector,<:Tuple,<:StepRange}, i, N; size=1) = v[i]
getvalue(x::Number, i, N; size=1) = 1 <= i <= N + 1 - size ? x : zero(x)

function kitaev_hamiltonian(c, H::AbstractHilbertSpace; μ, t, Δ, U)
    labels = keys(H)
    N = length(labels)
    h1s = (_kitaev_1site(c[lab]; μ=getvalue(μ, ind, N)) for (ind, lab) in enumerate(labels))
    h2s = (_kitaev_2site(c[lab], c[labels[mod1(ind + 1, N)]]; t=getvalue(t, ind, N; size=2), Δ=getvalue(Δ, ind, N; size=2), U=getvalue(U, lab, N; size=2)) for (ind, lab) in enumerate(labels))
    sum(h1s) + sum(h2s)
end


function sweet_spot_μ(; U, N)
    μ = fill(-U / 2, N)
    if N > 2
        μ[2:end-1] .= -U
    end
    return μ
end

function frustration_free_μ(; t, Δ, U, N)
    μstar = sqrt((U + 2t)^2 - 4 * Δ^2)
    μ = fill(-U / 2 - μstar / 2, N)
    if N > 2
        μ[2:end-1] .= -U - μstar
    end
    return μ
end

function kitaev_at_sweetspot(c, H; U, t)
    N = length(H)
    μ = sweet_spot_μ(; U, N)
    Δ = t + U / 2
    return kitaev_hamiltonian(c, H; μ, t, Δ, U)
end

function degeneracy_fun_Δ(c, H; μ, t, Δ, U, kwargs...)
    arnoldis = [ArnoldiWorkspace(ones(dim(H) ÷ 2), 20) for n in 1:2]
    function deg(dΔ)
        h = matrix_representation(kitaev_hamiltonian(c, H; μ, t, Δ=Δ .+ only(dΔ), U), H)
        hs = blocks(blockdiagonal(h, H))
        vals = [partialschur!(h, arnoldi; nev=1, which=:SR, start_from=1, kwargs...) for (h, arnoldi) in zip(hs, arnoldis)]
        real(vals[1][1].eigenvalues[1] - vals[2][1].eigenvalues[1])
    end
    return deg
end

function optimized_Δ(H; μ, t, Δ, U, tol)
    @fermions c
    deg_fun = degeneracy_fun_Δ(c, H; μ, t, Δ, U, tol)
    prob = ZeroProblem(deg_fun, zero(Δ))
    solve(prob; rtol=tol)
end
