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
sweet_spots = Folds.map(ts) do x
    _params = (; params..., μ=(; L=params.EZ.L, H=0Δ0, R=params.EZ.R), t=x * Δ0)
    symham = symbolic_hamiltonian(; _params...)
    ham0 = matrix_representation(symham, H)
    num_ops = [matrix_representation(c[n, :↑]'c[n, :↑] + c[n, :↓]'c[n, :↓], H) for n in (:L, :H, :R)]
    perts = [num_ops[1] + num_ops[3], num_ops[2]]
    sol = find_sweet_spot(ham0, perts; HS=H, q=2, Epenalty=1e3, lb=Δ0 .* [-1.3, -1], ub=Δ0 .* [0, 0.0])
    ham = ham0 + sum(sol .* perts)
    vals, vecs = eigen!(Hermitian(Matrix(ham)))
    reduced = Dict(m => reduced_majoranas_properties(vecs[:, 1], vecs[:, 2], H, subregion(m, H); q=2) for m in subregions)
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
        size = figsize,
        figure_padding = (20, 10, 10, 10),
    )
    ax = Axis(
        fig[1, 1];
        xlabel = L"t/\Delta_H",
    )
    colors = [Cycled(2), Cycled(4), Cycled(1)]
    ax.xgridvisible = false
    ax.ygridvisible = false
    lines!(ax, ts, [q.Qospin for q in qms];
           label = L"Q_o",
           color = colors[2])
    lines!(ax, ts, [q.Qespin for q in qms];
           label = L"Q_e", linestyle = :dash,
           color = colors[1])
    lines!(ax, ts, [abs(x.δQs[1]) for x in sweet_spots];
           label = L"\left|\partial_{\varepsilon_a} \delta E\right|",
           linestyle = (:dot, :dense),
           color = colors[3])
    Legend(
        fig[1, 1], ax;
        tellheight = false,
        tellwidth = false,
        margin = (10, 10, 10, 10),
        rowgap = -2,
        labelsize = 10,
        halign = :left, valign = :top,
    )
    text!(fig.scene, 0.01, 0.85; text=LaTeXString("(b)"), space=:relative, fontsize=10)
    fig
end

save("../ABS_to_YSR_sweet_spots_Qs.pdf", fig)







axgap = Axis(f[1, 2]; xlabel=L"t/\Delta_0")
g1 = lines!(axgap, ts, Egaps, label=L"E_g/\Delta_0");
g2 = lines!(axgap, ts, map(x -> x.δE, sweet_spots), label=L"\delta E", linestyle=:dash);
axislegend(axgap; position=:lt)
ax_log = Axis(
    f[2, 1];
    xlabel=L"t/\Delta_0",
    yscale=log10
)
lines!(ax_log, ts, [q.Qespin for q in qms])
lines!(ax_log, ts, [q.Qesite for q in qms])
lines!(ax_log, ts, [q.Qospin for q in qms], linestyle=:dash)
lines!(ax_log, ts, [q.Qosite for q in qms], linestyle=:dash)
#=lines!(ax_log, ts, [abs(x.δQs[1]) for x in sweet_spots], linestyle=:dot)=#
axgap_log = Axis(
    f[2, 2];
    xlabel=L"t/\Delta_0",
    yscale=log10
)
lines!(axgap_log, ts, Egaps)
lines!(axgap_log, ts, map(x -> abs(x.δE), sweet_spots), linestyle=:dash)
f

# save("../ABS_to_YSR_sweet_spots_Qs_optLDspin.png", f)

### Detailed sweet-spot properties
ints = mapreduce(x -> [sqrt(2) * x.reduced[m].LD / (x.reduced[m].LFmin * x.reduced[m].LFmax) for m in outer_spins], hcat, sweet_spots)
LDs = map(x -> [x.reduced[m].LD for m in outer_spins], sweet_spots)
LFs = map(x -> [x.reduced[m].LFmin for m in outer_spins], sweet_spots)
map(x -> [x.reduced[m].LFmax for m in outer_spins], sweet_spots)
δQs = map(x -> abs(x.δQs[1]), sweet_spots)
@assert all(abs.(δQs) .< 1e-9)
δEs = map(x -> x.δE, sweet_spots)
@assert all(abs.(δEs) .< 1e-9)
μshifts = mapreduce(x -> x.dμ.u, hcat, sweet_spots)

f = Figure();
axδQ = Axis(f[1, 1]; xlabel=L"t/\Delta_0", yscale=log10)
lines!(axδQ, ts, replace(δQs, 0 => NaN), label=L"δn_L = δn_R");
axislegend(axδQ; position=:rb)
axδE = Axis(f[1, 2]; xlabel=L"t/\Delta_0", yscale=log10)
lines!(axδE, ts, replace(δEs, 0 => NaN), label="δE");
axislegend(axδE; position=:rb)
axμ = Axis(f[2, 1]; xlabel=L"t/\Delta_0")
series!(axμ, ts, μshifts, labels=[L"dμ_D" L"dμ_H"]);
axislegend(axμ; position=:lb)
axints = Axis(f[2, 2]; xlabel=L"t/\Delta_0")
series!(axints, ts, ints);
f

gs = sweet_spots[1].vecs[:, 1]
gs' * matrix_representation(c[:L, :↑]'c[:L, :↑], H) * gs
gs' * matrix_representation(c[:L, :↓]'c[:L, :↓], H) * gs

### Majorana wavefunctions at sweet spots
LFmaxup = map(x -> x.reduced[[(:L, :↑)]].LFmax, sweet_spots)
LFmaxdn = map(x -> x.reduced[[(:L, :↓)]].LFmax, sweet_spots)
LFminup = map(x -> x.reduced[[(:L, :↑)]].LFmin, sweet_spots)
LFmindn = map(x -> x.reduced[[(:L, :↓)]].LFmin, sweet_spots)
MPup = map(x -> x.reduced[[(:L, :↑)]].MR, sweet_spots)
MPdn = map(x -> x.reduced[[(:L, :↓)]].MR, sweet_spots)
f = Figure();
ax = Axis(f[1, 1]; xlabel=L"t/\Delta_0")
lines!(ax, ts, LFmaxup, label=L"\tilde\gamma_{L\uparrow }", color=:blue);
lines!(ax, ts, LFmaxdn, label=L"\tilde\gamma_{L\downarrow }", color=:red);
lines!(ax, ts, LFminup, label=L"\gamma_{L\uparrow}", linestyle=:dash, color=:blue);
lines!(ax, ts, LFmindn, label=L"\gamma_{L\downarrow}", linestyle=:dash, color=:red);
# lines!(ax, ts, MPup, label=L"M_{L\uparrow}", linestyle=:dot, color=:blue);
# lines!(ax, ts, MPdn, label=L"M_{L\downarrow}", linestyle=:dot, color=:red);
axislegend(ax; position=:lt)
f

save("../ABS_to_YSR_sweet_spots_majorana_wavefunctions.png", f)




### charge-stability diagram around delft sweet spot
fig = Figure();
ax = Axis(fig[1, 1]; xlabel=L"\delta\varepsilon_L/Δ_0", ylabel=L"\delta\varepsilon_R/Δ_0")
dϵs = range(-7, 7, 100)
num_ops = [matrix_representation(c[n, :↑]'c[n, :↑] + c[n, :↓]'c[n, :↓], H) for n in (:L, :H, :R)]
perts = [num_ops[1], num_ops[3]]
ss = sweet_spots[end]
symham = symbolic_hamiltonian(; ss.params..., 
                              μ=(; L=ss.params.μ.L + ss.dμ[1], H=ss.params.μ.H + ss.dμ[2], R=ss.params.μ.R + ss.dμ[1]))
ham0 = matrix_representation(symham, H)
Edata = Folds.map(Base.product(dϵs, dϵs)) do xs
    ham = ham0 + sum(xs .* perts)
    vals, vecs = blockeigen(ham, H)
    n2 = length(vals) ÷ 2 + 1
    vals[1] - vals[n2]
end
heatmap!(ax, dϵs, dϵs, Edata'; colormap=:vik, colorrange=maximum(abs, Edata) .* (-1, 1) .* 0.1)
#=fig, ax, hm = heatmap(Edata'; colormap=:vik, colorrange=maximum(abs, Edata) .* (-1, 1) .* 0.1);=#
#=Colorbar(fig[1, 2], hm)=#
fig

save("../charge_stability_delft_sweet_spot.png", fig)

### check splitting under zeeman variations
dEzs = range(-0.1, 0.1, 100)
pert_norm = 0.0
Edata = Folds.map(dEzs) do dEz
    pert = matrix_representation(sum(dEz * (c[n, :↑]'c[n, :↑] - c[n, :↓]'c[n, :↓]) for n in (:L, :R)), H)
    ham = ham0 + pert
    vals, vecs = blockeigen(ham, H)
    n2 = length(vals) ÷ 2 + 1
    vals[1] - vals[n2]
end

pert_norm = 4 * schatten_norm(matrix_representation(c[:L, :↑]'c[:L, :↑], H), Inf)^2
f = Figure();
ax = Axis(f[1, 1]; xlabel=L"\delta E_Z/Δ_0")
lines!(ax, dEzs, Edata, label=L"\delta E");
lines!(ax, dEzs, qms[end].Qespin * sqrt(pert_norm) * collect(dEzs), label="bound");
axislegend(ax; position=:lt)
f

save("../zeeman_perturbation_sweet_spot.png", f)





##
symham = symbolic_hamiltonian(; params..., μ=(; L=params.EZ.L, H=0Δ0, R=params.EZ.R), t=1Δ0)
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
