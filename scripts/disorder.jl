using DrWatson
@quickactivate :ManybodyMajoranas
using LinearAlgebra, Folds
using UnPack, CairoMakie, MakiePublication, LaTeXStrings
using Random: seed!

##
@fermions f
N = 8
S = 1:N
B = length(S) .+ (1:1)
R = last(S):last(S)
r = only(R)
b = only(B)
qn = ParityConservation()
spaces = hilbert_spaces(S, R, B, qn)
@unpack HS, HB, HR, HSB, HRB = spaces
##
params = energy_splitting_parameters(HS)
# params = (; params..., μ=round.(params.μ; digits=4))
symham = kitaev_hamiltonian(f, HS; params...)
hS = matrix_representation(symham, HS)
vals, vecs = blockeigen(Hermitian(hS), HS)
nSB = div(size(vecs, 2), 2)
gs_odd = vecs[:, 1]
gs_even = vecs[:, nSB+1]
exc_gap = minimum([abs(vals[2] - vals[1]), abs(vals[nSB+2] - vals[nSB+1])]) # excitation gap
gapratio = abs((vals[1] - vals[nSB+1]) / exc_gap)
@assert gapratio < 1e-8 "Not degenerate: gapratio = $gapratio"
q = Inf
using LowRankMatrices
δρ = LowRankMatrix(gs_even, gs_even) - LowRankMatrix(gs_odd, gs_odd)
Qe = sqrt(sum(norm(svdvals(partial_trace(δρ, HS => subregion(k:k, HS))), q)^2 for k in S))
Qeabs = sum(norm(svdvals(partial_trace(δρ, HS => subregion(k:k, HS))), q) for k in S)

##
seed!(2)
M = 1000
σs = range(0, 1, length=40)
εs = randn(N, M)

## Calculate energies and bounds for random perturbations
@time energy_splittings = stack(σs) do σ
    Folds.map(eachcol(εs)) do ε
        hdissym = kitaev_hamiltonian(f, HS; μ=ε * σ, t=0, Δ=0, U=0)
        hdis = matrix_representation(hdissym, HS)
        vals, vecs = ground_states_arnoldi(hS + hdis, HS)
        only(diff(vals))
    end
end;
##
using Statistics
rms = dropdims(mapslices(mean, energy_splittings .^ 2, dims=1); dims=1) .|> sqrt
absmeans = dropdims(mapslices(mean, abs.(energy_splittings), dims=1); dims=1)
rms_data = (; y=rms, x=σs, Q=Qe, prefactor=1, label=L"\sqrt{\overline{|\delta E|^2}}")
abs_data = (; y=absmeans, x=σs, Q=Qeabs, prefactor=sqrt(2 / pi), label=L"\overline{|\delta E|}")
rms_data2 = (; y=rms, x=σs, Q=Qe, prefactor=sqrt(N), label=rms_data.label)
wsave(datadir("int_kitaev_disorder_$N.jld2"), Dict("energy_splittings" => energy_splittings, "σs" => σs, "εs" => εs, "params" => params, "rms" => rms_data, "abs" => abs_data, "rms2" => rms_data2))
##
disorder_data = wload(datadir("int_kitaev_disorder_$N.jld2"))

## plotting
disorder_fig = with_theme(theme_aps()) do
    normalization = disorder_data["params"].t
    data = disorder_data["rms2"]
    σs = data.x
    xs = σs / normalization
    ys = data.y / normalization
    Q = data.Q / normalization
    fig = Figure(size=110 .* (2, 1), figure_padding=6)
    ax = Axis(fig[1, 1], xlabel=L"\sigma/t", ylabel=L"E/t", xlabelpadding=1)
    ylims!(ax, 0, 1.1 * maximum(ys))
    xlims!(ax, 0, maximum(xs))
    # lines!(ax, xs, ys; label=L"\overline{|\delta E|}", color=Cycled(1), linewidth=3)
    lines!(ax, xs, ys; label=data.label, color=Cycled(1), linewidth=1.5)
    lines!(ax, σs, σs * Q * data.prefactor / normalization, label="Eq. (38)"; linestyle=:dash, color=Cycled(4), linewidth=1.5)
    # axislegend(ax, position=(:left, :top), labelsize=8)
    Legend(
        fig[1, 1], ax;
        tellheight = false,
        tellwidth = false,
        margin = (10, 10, 10, 10),
        rowgap = -4,
        labelsize = 8,
        halign = :left, valign = :top,
    )
    fig
end

##
save(plotsdir("disorder_$N.pdf"), disorder_fig)
save(plotsdir("disorder_$N.png"), disorder_fig, px_per_unit=2.5)
