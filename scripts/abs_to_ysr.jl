using DrWatson
@quickactivate :ManybodyMajoranas
using Folds
using LinearAlgebra
using UnPack, CairoMakie, MakiePublication, LaTeXStrings

@fermions c

function symbolic_hamiltonian(; μ, EZ, U, Δ0, t, tso)
    #=@fermions c=#

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

function charge_diff(o, e, Nj)
    return e' * Nj * e - o' * Nj * o
end
using Optimization, OptimizationBBO
function find_sweet_spot(ham0, perts; HS, q, Epenalty=1e6, optkwargs...)
    HRs = [subregion([(n, σ)], HS) for n in (:L, :R), σ in spins]
    function cost_function(xs, p)
        ham = ham0 + sum(xs .* perts)
        vals, vecs = blockeigen(ham, HS)
        n2 = length(vals) ÷ 2 + 1
        Ediff = (vals[n2] - vals[1])^2
        NL = matrix_representation(c[:L, :↑]'c[:L, :↑] + c[:L, :↓]'c[:L, :↓], HS)
        δQL = charge_diff(vecs[:, 1], vecs[:, n2], NL)
        δQL^2 + Ediff * Epenalty
        # reduceds = [reduced_majoranas_properties(vecs[:, 1], vecs[:, n2], HS, HR; q) for HR in HRs]
        # sum(x -> x.LD^2, reduceds) + Ediff * Epenalty
    end
    x0 = -[0.1Δ0, 0.5Δ0]
    prob = OptimizationProblem(cost_function, x0, nothing; optkwargs...)# lb=[-1.0, -1.0], ub=[1.0, 1.0])
    sol = solve(prob, BBO_adaptive_de_rand_1_bin_radiuslimited())
end
##

outer_sites = [[(n, :↑), (n, :↓)] for n in (:L, :R)]
outer_spins = [[(n, σ)] for n in (:L, :R) for σ in (:↑, :↓)]
subregions = vcat(outer_sites, outer_spins)
function quality_measures(reduced_dict)
    Qespin = sqrt(sum(reduced_dict[label].LD^2 for label in outer_spins))
    Qesite = sqrt(sum(reduced_dict[label].LD^2 for label in outer_sites))
    Qospin = sqrt(sum(reduced_dict[label].LFmin^2 for label in [[(:L, :↑)], [(:L, :↓)]]))
    Qosite = reduced_dict[[(:L, :↑), (:L, :↓)]].LFmin
    (; Qespin, Qesite, Qospin, Qosite)
end

ts = range(0.1, 1.2, 20)
q = 2
sweet_spots = Folds.map(ts) do x
    _params = (; params..., μ=(; L=params.EZ.L, H=0Δ0, R=params.EZ.R), t=x * Δ0)
    symham = symbolic_hamiltonian(; _params...)
    ham0 = matrix_representation(symham, H)
    num_ops = [matrix_representation(c[n, :↑]'c[n, :↑] + c[n, :↓]'c[n, :↓], H) for n in (:L, :H, :R)]
    perts = [num_ops[1] + num_ops[3], num_ops[2]]
    sol = find_sweet_spot(ham0, perts; HS=H, q, Epenalty=1e3, lb=Δ0 .* [-1.3, -1], ub=Δ0 .* [0, 0.0])
    ham = ham0 + sum(sol .* perts)
    vals, vecs = eigen!(Hermitian(Matrix(ham)))
    reduced = Dict(m => reduced_majoranas_properties(vecs[:, 1], vecs[:, 2], H, subregion(m, H); q) for m in subregions)
    δQs = [charge_diff(vecs[:, 1], vecs[:, 2], num_ops[i]) for i in (1, 3)]
    Egap = vals[3] - vals[2]
    δE = vals[2] - vals[1]
    (dμ=sol, params=_params, vals, vecs, reduced, δQs, Egap, δE)
end

### Quality measures at sweet spots
qms = map(x -> quality_measures(x.reduced), sweet_spots)
Egaps = map(x -> x.Egap, sweet_spots)
δEs = map(x -> x.δE, sweet_spots)
@assert all(abs.(δEs) .< 1e-10)
δQs = map(x -> abs(x.δQs[1]), sweet_spots)
@assert all(abs.(δQs) .< 1e-10)

figsize = 150 .* (1.5, 1)
fig = with_theme(theme_aps(linestyles=[nothing, :dash, :dot])) do
    fig = Figure(
        size=figsize,
        figure_padding=(20, 10, 10, 10),
    )
    ax = Axis(
        fig[1, 1];
        xlabel=L"t/\Delta_H",
    )
    colors = [Cycled(2), Cycled(4), Cycled(1)]
    ax.xgridvisible = false
    ax.ygridvisible = false
    lines!(ax, ts, [q.Qospin for q in qms];
        label=L"Q_o",
        color=colors[2])
    lines!(ax, ts, [q.Qespin for q in qms];
        label=L"Q_e", linestyle=:dash,
        color=colors[1])
    lines!(ax, ts, [abs(x.δQs[1]) for x in sweet_spots];
        label=L"\left|\partial_{\varepsilon_a} \delta E\right|",
        linestyle=(:dot, :dense),
        color=colors[3])
    Legend(
        fig[1, 1], ax;
        tellheight=false,
        tellwidth=false,
        margin=(10, 10, 10, 10),
        rowgap=-2,
        labelsize=10,
        halign=:left, valign=:top,
    )
    text!(fig.scene, 0.01, 0.85; text=LaTeXString("(b)"), space=:relative, fontsize=10)
    fig
end
##
save(plotsdir("ABS_to_YSR_sweet_spots_Qs.pdf"), fig)
save(plotsdir("ABS_to_YSR_sweet_spots_Qs.png"), fig, px_per_unit=1.3)
