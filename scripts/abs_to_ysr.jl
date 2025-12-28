using DrWatson
@quickactivate :ManybodyMajoranas
using Folds
using UnPack, CairoMakie, MakiePublication, LaTeXStrings

function symbolic_hamiltonian(; μ, EZ, U, Δ0, t, tso)
    @fermions c

    # Hamiltonian for quantum dots (L and R) - no superconducting pairing
    H_dots = map((:L, :R)) do l
        nup = c[l, :↑]'c[l, :↑]
        ndn = c[l, :↓]'c[l, :↓]
        (μ[l] + EZ[l]) * nup + (μ[l] - EZ[l]) * ndn + U[l] * nup * ndn
    end

    # Hamiltonian for hybrid segment (H) - includes superconducting pairing
    nup_H = c[:H, :↑]'c[:H, :↑]
    ndn_H = c[:H, :↓]'c[:H, :↓]
    H_hybrid = μ.H * (nup_H + ndn_H) +
               (Δ0 * c[:H, :↑]c[:H, :↓] + hc)

    # Tunnel coupling with spin-orbit coupling
    H_tunnel = t * (c[:H, :↑]'c[:L, :↑] + c[:H, :↓]'c[:L, :↓]) +
               tso * t * (c[:H, :↓]'c[:L, :↑] - c[:H, :↑]'c[:L, :↓]) +
               t * (c[:R, :↑]'c[:H, :↑] + c[:R, :↓]'c[:H, :↓]) +
               tso * t * (c[:R, :↓]'c[:H, :↑] - c[:R, :↑]'c[:H, :↓]) +
               hc

    sum(H_dots) + H_hybrid + H_tunnel
end

spatial_labels = (:L, :H, :R)
spins = (:↑, :↓)
labels = Base.product(spatial_labels, spins)
H = hilbert_space(labels, ParityConservation())

##
Δ0 = 1
params = (; EZ=(; L=1.5Δ0, R=1.5Δ0),
    U=(; L=5Δ0, R=5Δ0),
    t=0.25 * Δ0, tso=0.3,
    Δ0)
##
using Optimization, OptimizationBBO
function find_sweet_spot(ham0, perts; HS, q, Epenalty=1e6, optkwargs...)
    HRs = [subregion([(n, σ)], HS) for n in (:L, :R), σ in spins]
    function cost_function(xs, p)
        ham = ham0 + sum(xs .* perts)
        vals, vecs = blockeigen(ham, HS)
        n2 = length(vals) ÷ 2 + 1
        Ediff = (vals[n2] - vals[1])^2
        reduceds = [reduced_majoranas_properties(vecs[:, 1], vecs[:, n2], HS, HR; q) for HR in HRs]
        sum(x -> x.LD^2, reduceds) + Ediff * Epenalty
    end
    x0 = -[0.1Δ0, 0.5Δ0]
    prob = OptimizationProblem(cost_function, x0, nothing; optkwargs...)# lb=[-1.0, -1.0], ub=[1.0, 1.0])
    sol = solve(prob, BBO_adaptive_de_rand_1_bin_radiuslimited())
end
##
sweet_spots = map([0.25, 0.5, 1, 1.1]) do x
    _params = (; params..., μ=(; L=params.EZ.L, H=0Δ0, R=params.EZ.R), t=x * Δ0)
    symham = symbolic_hamiltonian(; _params...)
    ham0 = matrix_representation(symham, H)
    num_ops = [matrix_representation(c[n, :↑]'c[n, :↑] + c[n, :↓]'c[n, :↓], H) for n in (:L, :H, :R)]
    perts = [num_ops[1] + num_ops[3], num_ops[2]]
    sol = find_sweet_spot(ham0, perts; HS=H, q=2, Epenalty=1e5, lb=Δ0 .* [-2.5 - x, -1], ub=Δ0 .* [0, 0])
    ham = ham0 + sum(sol .* perts)
    vals, vecs = eigen!(Hermitian(Matrix(ham)))
    modes = keys(H)
    reduced = Dict(m => reduced_majoranas_properties(vecs[:, 1], vecs[:, 2], H, subregion([m], H); q=2) for m in keys(H))
    (dμ=sol, params=_params, vals, vecs, reduced)
end
map(x -> [sqrt(2) * x.reduced[m].LD / (x.reduced[m].LFmin * x.reduced[m].LFmax) for m in keys(H)[[1, 3, 4, 6]]], sweet_spots)
map(x -> [x.reduced[m].LD for m in keys(H)[[1, 3, 4, 6]]], sweet_spots)
map(x -> [x.reduced[m].LFmin for m in keys(H)[[1, 3, 4, 6]]], sweet_spots)
map(x -> [x.reduced[m].LFmax for m in keys(H)[[1, 3, 4, 6]]], sweet_spots)
##
symham = symbolic_hamiltonian(; params..., μ=(; L=params.EZ.L, H=0Δ0, R=params.EZ.R), t=1.3Δ0)
ham0 = matrix_representation(symham, H)
num_ops = [matrix_representation(c[n, :↑]'c[n, :↑] + c[n, :↓]'c[n, :↓], H) for n in (:L, :H, :R)]
perts = [num_ops[1] + num_ops[3], num_ops[2]]
Edata = Folds.map(Base.product(range(-10, 1, 50), range(-2, 2, 62))) do xs
    ham = ham0 + sum(xs .* perts)
    vals, vecs = blockeigen(ham, H)
    n2 = length(vals) ÷ 2 + 1
    vals[1] - vals[n2]
end
fig, ax, hm = heatmap(Edata'; colormap=:vik, colorrange=maximum(abs, Edata) .* (-1, 1) .* 0.1)
Colorbar(fig[1, 2], hm)
fig