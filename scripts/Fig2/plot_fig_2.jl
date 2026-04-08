using DrWatson
@quickactivate :ManybodyMajoranas
using LaTeXStrings, UnPack, CairoMakie, MakiePublication
## Load data
phase_diagram = wload(datadir("int_kitaev_phase_diagram_$N.jld2"))
wavefunctions = wload(datadir("int_kitaev_wavefunctions_$N.jld2"))

## Plot
fig_aps = with_theme(theme_aps(markers=[:circle, :diamond, :utriangle], linestyles=[nothing, :dash])) do
    x = phase_diagram["Us"]
    y = phase_diagram["δμs"]
    bounds = phase_diagram["bounds"]
    xparam = :U
    yparam = :δμ
    colorrange = (0, 1)
    colormap = cgrad(:viridis)
    limits = (nothing, (0, 1.4))

    fig = Figure(figure_padding=(13, 8, 1, 5), size=220 .* (1.2, 1))
    ga = fig[2, 1] = GridLayout()
    ax1 = Axis(ga[1, 1]; xlabel=L"U/t", ylabel=L"%$yparam/t", title=L"\sqrt{1-M_\text{half}}")
    ax2 = Axis(ga[1, 2]; xlabel=L"U/t", title=L"||(i\gamma \tilde{\gamma})_\text{half}||_2")
    linkaxes!(ax1, ax2)
    hideydecorations!(ax2; minorticks=false, ticks=false)
    clips = (; highclip=last(colormap))
    hm1 = heatmap!(ax1, x, y, map(d -> sqrt(1 - d.MR), bounds); colorrange, colormap, colorscale=identity)
    heatmap!(ax2, x, y, map(d -> d.LD, bounds); colorrange, colormap, clips...)
    Colorbar(ga[1, 3], hm1; ticks=LogTicks(WilkinsonTicks(4, k_min=1, k_max=4)), minorticksvisible=false)

    rowsize!(fig.layout, 1, Relative(0.45))
    gb = fig[1, 1] = GridLayout()
    yscale = identity
    ax = Axis(gb[1, 1]; yscale, xlabel=L"j", limits, xticks=1:N)
    text!(fig.scene, 0.002, 0.92; text=LaTeXString("(a)"), space=:relative)
    text!(fig.scene, 0.002, 0.5; text=LaTeXString("(b)"), space=:relative)

    colors = [Cycled(2), Cycled(1)]
    heatmap_markersize = 12
    scatter_markersize = 8
    params = wavefunctions["good"].params
    params2 = wavefunctions["bad"].params
    scatter!(ax1, params[xparam], params[yparam]; color=colors[1], marker=:x, strokewidth=1, markersize=heatmap_markersize)

    lines_strokewidth = 0.2
    scatter!(ax1, params2[xparam], params2[yparam]; color=colors[2], marker=:+, strokewidth=1, markersize=heatmap_markersize)
    scatterlines!(ax, 1:N, map(d -> d.LFmin, wavefunctions["bad"].wavefunction), label=L"||\gamma_{j}||_2", marker=:+, linestyle=:solid, color=colors[2], markersize=scatter_markersize, strokewidth=lines_strokewidth)
    scatterlines!(ax, 1:N, map(d -> d.LFmax, wavefunctions["bad"].wavefunction), label=L"||\tilde{\gamma}_{j}||_2", marker=:+, linestyle=:dash, color=colors[2], markersize=scatter_markersize, strokewidth=lines_strokewidth)

    scatterlines!(ax, 1:N, map(d -> d.LFmin, wavefunctions["good"].wavefunction), label=L"||\gamma_{j}||_2", marker=:x, linestyle=:solid, color=colors[1], markersize=scatter_markersize, strokewidth=lines_strokewidth)
    scatterlines!(ax, 1:N, map(d -> d.LFmax, wavefunctions["good"].wavefunction), label=L"||\tilde{\gamma}_{j}||_2", marker=:x, linestyle=:dash, color=colors[1], markersize=scatter_markersize, strokewidth=lines_strokewidth)
    Legend(gb[1, 2],
        [LineElement(color=:black, linestyle=:solid),
            LineElement(color=:black, linestyle=:dash)],
        [L"||\gamma_{j}||_2", L"||\tilde{\gamma}_{j}||_2"], rowgap=5, labelsize=14)

    rowgap!(fig.layout, 1, Relative(0.02))
    colsize!(gb, 2, Relative(0.22))
    fig
end
##
save(plotsdir("int_kitaev_phase_diagram_wavefunctions_$(N).pdf"), fig_aps)
save(plotsdir("int_kitaev_phase_diagram_wavefunctions_$(N).png"), fig_aps, px_per_unit=1.2)