using DrWatson
@quickactivate :ManybodyMajoranas
using LinearAlgebra, Folds
using UnPack, CairoMakie, MakiePublication, LaTeXStrings
function calculate_bounds(hamiltonians, spaces, q)
    @unpack HS = spaces
    canon_hams = canonicalize_hamiltonians(hamiltonians, spaces)
    @unpack hS0, hS, hB, hRB = canon_hams
    _, vecs = ground_states_arnoldi(hS0, HS)
    gs_odd = vcat(vecs[1], zero(vecs[2]))
    gs_even = vcat(zero(vecs[1]), vecs[2])

    reduced = reduced_majoranas_properties(gs_even, gs_odd, HS, HR, FrobeniusGauge(); q)
    return calculate_bounds(reduced, hamiltonians, spaces, q)
end

function calculate_bounds(reduced, hamiltonians, spaces, q)
    @unpack HS, HB, HR, HSB, HRB = spaces
    p = conjugate_norm(q)
    canon_hams = canonicalize_hamiltonians(hamiltonians, spaces)
    @unpack hS0, hS, hB, hRB = canon_hams
    heff = effective_hamiltonian(canon_hams, spaces, (reduced.γmin, reduced.γmax))
    if abs(heff.effops.ε) > 1e-6
        @warn "ε > 1e-6" (heff.effops.ε)
    end
    odd_coupling, even_coupling = decompose_coupling(hRB, HRB, HR, HB)
    vals_full, vecs_full = blockeigen(Hermitian(embed(hS0, HS => HSB; complement=HB) + embed(hRB, HRB => HSB) + embed(hB, HB => HSB; complement=HS)), HSB)
    vals, vecs = blockeigen(Hermitian(heff.total_ham), spaces.HgsB)
    γeff = [embed(γ, spaces.Hgs => spaces.HgsB) for γ in heff.γgs]
    γfull = [embed(γ, spaces.HS => spaces.HSB) for γ in (reduced.γmin, reduced.γmax)]
    # γgsBmin, _ = [embed(γ, spaces.Hgs => spaces.HgsB) for γ in heff.γgs]
    eff = calculate_bounds(reduced, vals, vecs, odd_coupling, even_coupling, γeff, spaces.Hgs, spaces.HB, spaces.HgsB, q, p)
    full = calculate_bounds(reduced, vals_full, vecs_full, odd_coupling, even_coupling, γfull, spaces.HS, spaces.HB, spaces.HSB, q, p)
    (; heff, eff, full)
end

function calculate_bounds(reduced, vals, vecs, odd_coupling, even_coupling, (γmin, γmax), HS, HB, HSB, q, p; dn=4)
    n = div(size(vecs, 2), 2) # number of odd/even states
    dn = min(dn, n)
    even_norm = schatten_norm(even_coupling, p)
    odd_norm = schatten_norm(odd_coupling, p)
    odd_states, even_states = map(states -> states[:, 1:dn], vecs.blocks)
    overlaps1 = abs.(odd_states' * γmin[1:n, n+1:2n] * even_states)
    OEs = [E * O' for (O, E) in Base.product(eachcol(vecs[:, 1:dn]), eachcol(vecs[:, n+1:n+dn]))]
    OEBnorms = [schatten_norm(partial_trace(OE, HSB => HB), q) for OE in OEs]
    pairs = map(CartesianIndex, enumerate(map(v -> argmax(v), eachrow(overlaps1))))
    δEs = map(es -> -(es...), Base.product(vals[1:dn], vals[n+1:n+dn]))
    # δEs_full = map(es -> -(es...), Base.product(vals_full[1:n], vals_full[div(length(vals_full), 2).+(1:n)]))
    np_bound = (reduced.LD * even_norm .+ reduced.LFmin * odd_norm * OEBnorms) ./ overlaps1
    p_bound = sqrt(dim(HB)) * (reduced.LD * even_norm + reduced.LFmin * odd_norm)
    (; δEs, p_bound, np_bound, even_norm, odd_norm, vals, vecs, pairs, overlaps1, OEBnorms)
end

##
@fermions f
N = 14
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
@profview @time energy_splitting_data_strong = Folds.map(ϵs_strong) do ϵ
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
npbounds_strong = [d.full.np_bound[1, 1] for d in energy_splitting_data_strong]
pbounds_strong = [d.full.p_bound for d in energy_splitting_data_strong]
npbounds_weak = [d.full.np_bound[1, 1] for d in energy_splitting_data_weak]
pbounds_weak = [d.full.p_bound for d in energy_splitting_data_weak]
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
save(plotsdir("energy_splitting_comparison_legend_$N.pdf"), energy_splitting_fig)
save(plotsdir("energy_splitting_comparison_legend_$N.png"), energy_splitting_fig, px_per_unit=2.5)

##
reduced.LFmin
reduced.LD