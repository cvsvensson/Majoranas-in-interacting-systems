using DrWatson
@quickactivate :ManybodyMajoranas
using LinearAlgebra, Folds
using UnPack, CairoMakie, MakiePublication, LaTeXStrings

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
params = degenerate_good_majorana_parameters(HS)
symham = kitaev_hamiltonian(f, HS; params...)
hS = matrix_representation(symham, HS)
vals, vecs = blockeigen(Hermitian(hS), HS)
nSB = div(size(vecs, 2), 2)
gs_odd = vecs[:, 1]
gs_even = vecs[:, nSB+1]
exc_gap = minimum([abs(vals[2] - vals[1]), abs(vals[nSB+2] - vals[nSB+1])]) # excitation gap
gapratio = abs((vals[1] - vals[nSB+1]) / exc_gap)
@assert gapratio < 1e-8 "Not degenerate: gapratio = $gapratio"
q = 2
reduced = reduced_majoranas_properties(gs_even, gs_odd, HS, HR, FrobeniusGauge(); q);
u = 1
## vary dot level
λstrong = 0.5 * global_parameters.t
θ = pi / 6
tc = λstrong * cos(θ)
Δc = λstrong * sin(θ)
Uc = λstrong * u
hRBsym = tc * f[r]' * f[b] + Δc * f[r] * f[b] + hc +
         Uc * f[b]' * f[b] * f[r]' * f[r]
hRB = matrix_representation(hRBsym, spaces.HRB)
ϵs_strong = 2 * λstrong * range(-1, 1, 100)
@time energy_splitting_data_strong = Folds.map(ϵs_strong) do ϵ
    hB = matrix_representation(ϵ * f[only(B)]' * f[only(B)], spaces.HB)
    hamiltonians = (; hS0=hS, hS=hS, hB=hB, hRB=hRB)
    calculate_bounds(reduced, hamiltonians, spaces, q)
end;
##
λweak = 0.01 * global_parameters.t
tc = λweak * cos(θ)
Δc = λweak * sin(θ)
Uc = λweak * u
hRBsym = tc * f[r]' * f[b] + Δc * f[r] * f[b] + hc +
         Uc * f[b]' * f[b] * f[r]' * f[r]
hRB = matrix_representation(hRBsym, spaces.HRB)
ϵs_weak = 2 * λweak * range(-1, 1, 100)
@time energy_splitting_data_weak = Folds.map(ϵs_weak) do ϵ
    hB = matrix_representation(ϵ * f[only(B)]' * f[only(B)], spaces.HB)
    hamiltonians = (; hS0=hS, hS=hS, hB=hB, hRB=hRB)
    calculate_bounds(reduced, hamiltonians, spaces, q)
end;
##
δEs_weak = [abs(d.full.δEs[1, 1]) for d in energy_splitting_data_weak]
δEs_strong = [abs(d.full.δEs[1, 1]) for d in energy_splitting_data_strong]
npbounds_strong = [d.full.np_bounds.np_bound[1, 1] for d in energy_splitting_data_strong]
pbounds_strong = [d.full.p_bounds.p_bound for d in energy_splitting_data_strong]
npbounds_weak = [d.full.np_bounds.np_bound[1, 1] for d in energy_splitting_data_weak]
pbounds_weak = [d.full.p_bounds.p_bound for d in energy_splitting_data_weak]
##
my_theme = copy(theme_aps())
# my_theme.Axis.yminorticksvisible = false
# my_theme.Axis.xminorticksvisible = false
energy_splitting_fig = with_theme(my_theme) do
    # fig = Figure(size=140 .* (1.8, 1), figure_padding=3)
    fig = Figure(size=280 .* Tuple(normalize([2.4, 1])), figure_padding=3)
    grid = fig[1, 1] = GridLayout()
    titlesize = 12
    #    minorticks = (; xminorticksvisible=false, yminorticksvisible=false)
    # Weak coupling subplot
    ax_weak = Axis(grid[1, 1];
        xlabel=L"\varepsilon_d / λ",
        ylabel=L"E / λ",
        limits=(nothing, (0, 1.15 * pbounds_weak[1] / λweak)),
        title=L"λ = t/100",
        titlesize)
    colors = [Cycled(2), Cycled(4), Cycled(1), Cycled(6)]

    simple = lines!(ax_weak, ϵs_weak ./ λweak, pbounds_weak ./ λweak,
        label=LaTeXString("Simple bound"); linestyle=(:dot, :dense), color=colors[1])
    detailed = lines!(ax_weak, ϵs_weak ./ λweak, npbounds_weak ./ λweak,
        label=LaTeXString("Detailed bound"); linestyle=:dash, color=colors[2])
    energy = lines!(ax_weak, ϵs_weak ./ λweak, δEs_weak ./ λweak,
        label=L"|\delta E| / λ"; linestyle=nothing, color=colors[3])

    # text!(fig.scene, 0.03, 0.84; text=LaTeXString("\\frac{E}{λ}"),
    #     space=:relative, fontsize=10)
    # axislegend(ax_weak; position=(1.05, 0.75))

    # Strong coupling subplot
    ax_strong = Axis(grid[1, 2];
        xlabel=L"\varepsilon_d / λ",
        limits=(nothing, (0, 1.15 * pbounds_strong[1] / λstrong)),
        title=L"λ = t/2", titlesize)
    hideydecorations!(ax_strong; ticks=false, minorticks=false)

    lines!(ax_strong, ϵs_strong ./ λstrong, pbounds_strong ./ λstrong,
        label=LaTeXString("Detailed bound"); linestyle=(:dot, :dense), color=colors[1])
    lines!(ax_strong, ϵs_strong ./ λstrong, npbounds_strong ./ λstrong,
        label=LaTeXString("Simple bound"); linestyle=:dash, color=colors[2])
    # lines!(ax_strong, ϵs_strong ./ λstrong, pbounds_strong ./ λstrong,
    #     label=LaTeXString("Eq. (36)"); linestyle=(:dot, :dense), color=colors[1])
    # lines!(ax_strong, ϵs_strong ./ λstrong, npbounds_strong ./ λstrong,
    #     label=LaTeXString("Eq. (35)"); linestyle=:dash, color=colors[2])
    lines!(ax_strong, ϵs_strong ./ λstrong, δEs_strong ./ λstrong,
        label=L"|\delta E| / λ"; linestyle=nothing, color=colors[3])
    # axislegend(ax_strong; position=(1.1, 0.85), labelhalign=:left)

    Legend(grid[1, 3],
        [[energy], [simple, detailed]],
        [[L"|\delta E| / λ"], ["Simple", "Detailed"]],
        ["", "Bounds"]; labelsize=10, titlegap=0, titlesize=10, titlefont=:regular, groupgap=5,
        patchlabelgap=4)
    colsize!(grid, 3, Relative(0.27))
    fig
end

##
save(plotsdir("energy_splitting_comparison_$N.pdf"), energy_splitting_fig)
save(plotsdir("energy_splitting_comparison_$N.png"), energy_splitting_fig, px_per_unit=1.3)

##
reduced.LFmin
reduced.LD