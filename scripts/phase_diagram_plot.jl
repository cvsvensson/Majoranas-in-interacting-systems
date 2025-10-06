using DrWatson
@quickactivate :ManybodyMajoranas
using LaTeXStrings, UnPack, CairoMakie, MakiePublication

include("parameters.jl")
##
N = 6
qn = ParityConservation()
HS = hilbert_space(1:N, qn)
HR = hilbert_space(1:div(N, 2), qn)
@fermions f

## Makie plot
@unpack bounds, Us, δμs = wload(datadir("int_kitaev_phase_diagram_frob_$N.jld2"))
## Pick a point in the phase diagram and plot the majorana wavefunctions
HS = hilbert_space(1:N, qn)
params = example_point_parameters(HS)
symham = kitaev_hamiltonian(f, HS; params[[:U, :Δ, :μ, :t]]...)
ham = matrix_representation(symham, HS)
isreal(ham) && (ham = real(ham))

vals, vecs = ground_states_arnoldi(ham, HS)
gs_odd = vcat(vecs[1], zero(vecs[2]))
gs_even = vcat(zero(vecs[1]), vecs[2])
q = 2
gauge = EigGauge()
wavefunction_data = map(R -> reduced_majoranas_properties(gs_even, gs_odd, HS, hilbert_space([R]), gauge; q), 1:N)


##
fig_aps = with_theme(theme_aps(markers=[:circle, :diamond, :utriangle], linestyles=[nothing, :dash])) do
    # maxval = max(maximum(d -> d.LD, bounds))
    # minval = min(minimum(d -> d.LD, bounds))
    # colorrange = (minval, maxval)
    colorrange = (0, 1)
    colormap = cgrad(:viridis)
    limits = (nothing, (0, 1.2))

    fig = Figure(figure_padding=(13, 8, 1, 5), size=220 .* (1.2, 1))
    ga = fig[2, 1] = GridLayout()
    ax1 = Axis(ga[1, 1]; xlabel=L"U/t", ylabel=L"\delta\mu/t", title=L"\sqrt{1-M_\text{half}}")
    ax2 = Axis(ga[1, 2]; xlabel=L"U/t", title=L"||(\gamma \tilde{\gamma})_\text{half}||_2")
    linkaxes!(ax1, ax2)
    hideydecorations!(ax2; minorticks=false, ticks=false)
    clips = (; highclip=last(colormap))
    hm1 = heatmap!(ax1, Us, δμs, map(d -> sqrt(1 - d.MR), bounds); colorrange, colormap, colorscale=identity)
    heatmap!(ax2, Us, δμs, map(d -> d.LD, bounds); colorrange, colormap, clips...)
    scatter!(ax1, params.U, params.δμ; color=:red, marker=:x, strokewidth=1, markersize=10)
    Colorbar(ga[1, 3], hm1; ticks=LogTicks(WilkinsonTicks(4, k_min=1, k_max=4)), minorticksvisible=false)

    rowsize!(fig.layout, 1, Relative(0.45))
    gb = fig[1, 1] = GridLayout()
    yscale = identity
    ax = Axis(gb[1, 1]; yscale, xlabel=L"n", limits)
    text!(fig.scene, 0.002, 0.92; text=LaTeXString("(a)"), space=:relative)
    text!(fig.scene, 0.002, 0.5; text=LaTeXString("(b)"), space=:relative)
    scatterlines!(ax, 1:N, map(d -> d.LFmin, wavefunction_data), label=L"||\gamma_{n}||_2", marker=:circle, linestyle=:dash)
    scatterlines!(ax, 1:N, map(d -> d.LFmax, wavefunction_data), label=L"||\tilde{\gamma}_{n}||_2", marker=:diamond, linestyle=:dot)
    scatterlines!(ax, 1:N, map(d -> d.MR, wavefunction_data), label=L"M_n", marker=:utriangle, linestyle=:solid)
    # scatterlines!(ax, 1:N, map(d -> d.LD, wavefunction_data), label=L"||(\gamma \tilde{\gamma})_{n}||_1", marker=:utriangle, linestyle=:solid)
    axislegend(ax, position=(0.5, 1.5), labelsize=9)
    rowgap!(fig.layout, 1, Relative(0.02))
    fig
end
##
save(plotsdir("int_kitaev_phase_diagram_and_wavefunctions_$(N)_aps_mp.pdf"), fig_aps, px_per_unit=40)
#save(plotsdir("int_kitaev_phase_diagram_and_wavefunctions_$(N)_aps_mp.png"), fig_aps, px_per_unit=10)
