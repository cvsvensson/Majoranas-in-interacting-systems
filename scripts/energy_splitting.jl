using DrWatson
@quickactivate :ManybodyMajoranas
using Folds # parallell calculations
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
    @assert abs(heff.effops.ε) < 1e-6 # zero in this script
    vals_full, _ = blockeigen(embed(hS0, HS => HSB) + embed(hRB, HRB => HSB) + embed(hB, HB => HSB), HSB)
    vals, vecs = blockeigen(heff.total_ham, spaces.HgsB)
    n = div(size(vecs, 2), 2) # number of odd/even states
    odd_coupling, even_coupling = decompose_coupling(hRB, HRB, HR, HB)
    even_norm = schatten_norm(even_coupling, p)
    odd_norm = schatten_norm(odd_coupling, p)
    odd_states, even_states = vecs.blocks
    γgsBmin, _ = [embed(γ, spaces.Hgs => spaces.HgsB) for γ in heff.γgs]
    overlaps1 = abs.(odd_states' * γgsBmin[1:n, n+1:2n] * even_states)
    OEs = [E * O' for (O, E) in Base.product(eachcol(vecs[:, 1:n]), eachcol(vecs[:, n+1:end]))]
    OEBnorms = [schatten_norm(partial_trace(OE, spaces.HgsB => spaces.HB), q) for OE in OEs]
    pairs = map(CartesianIndex, enumerate(map(v -> argmax(v), eachrow(overlaps1))))
    δEs = map(es -> -(es...), Base.product(vals[1:n], vals[n+1:2n]))
    δEs_full = abs(vals_full[1] - vals_full[div(length(vals_full), 2)+1])
    np_bound = (reduced.LD * even_norm .+ reduced.LFmin * odd_norm * OEBnorms) ./ overlaps1
    p_bound = sqrt(dim(spaces.HB)) * (reduced.LD * even_norm + reduced.LFmin * odd_norm)
    # analytics = analytical_bounds(reduced, OEBnorms, overlaps1, pairs)
    (; heff, δEs, δEs_full, p_bound, np_bound, even_norm, odd_norm, vals, vecs, pairs, overlaps1, OEBnorms)
end

function analytical_bounds(reduced, OEBnorms, overlaps1, pairs)
    simple = reduced.LD + 2 * reduced.LFmin
    complicated = (reduced.LD + 2 * reduced.LFmin * OEBnorms[pairs[1]]) / (sqrt(2) * overlaps1[pairs[1]])
    return (; simple, complicated)
end
include("parameters.jl")
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
@time params = example_point_parameters(HS; tol=1e-6)
symham = kitaev_hamiltonian(f, HS; params[[:μ, :t, :U]]..., Δ=params[:Δdeg])
hS = matrix_representation(symham, HS)
vals, vecs = blockeigen(hS, HS)
nSB = div(size(vecs, 2), 2)
gs_odd = vecs[:, 1]
gs_even = vecs[:, nSB+1]
exc_gap = minimum([abs(vals[2] - vals[1]), abs(vals[nSB+2] - vals[nSB+1])]) # excitation gap
gapratio = abs((vals[1] - vals[nSB+1]) / exc_gap)
@assert gapratio < 1e-8 "Not degenerate: gapratio = $gapratio"

## vary dot level
λ = 0.01 * global_parameters.t
θ = pi / 6
tc = λ * cos(θ)
Δc = λ * sin(θ)
Uc = λ
hRBsym = tc * f[r]' * f[b] + Δc * f[r] * f[b] + hc +
         Uc * f[b]' * f[b] * f[r]' * f[r]
hRB = matrix_representation(hRBsym, spaces.HRB)
ϵs = 2 * λ * range(-1, 1, 200)
reduced = reduced_majoranas_properties(gs_even, gs_odd, HS, HR, FrobeniusGauge(); q=2)
energy_splitting_data = Folds.map(ϵs) do ϵ
    hB = matrix_representation(ϵ * f[only(B)]' * f[only(B)], spaces.HB)
    hamiltonians = (; hS0=hS, hS=hS, hB=hB, hRB=hRB)
    calculate_bounds(reduced, hamiltonians, spaces, 2)
end
##
δEs = [abs(d.δEs[1, 1]) for d in energy_splitting_data]
normalization = λ
npbounds = [d.np_bound[1, 1] for d in energy_splitting_data]
pbounds = [d.p_bound for d in energy_splitting_data]
energy_splitting_fig = with_theme(theme_aps()) do
    fig = Figure(size=150 .* (1.5, 1), figure_padding=5)
    ax = Axis(fig[1, 1]; xlabel=L"\varepsilon_d/ λ", limits=(nothing, (0, 1.1 * pbounds[1] / normalization)))#, ylabel=L"\delta E / \max{\delta E}")
    lines!(ax, ϵs ./ λ, pbounds ./ normalization, label=LaTeXString("Eq. (27)"); linestyle=:dot, color=:black)
    lines!(ax, ϵs ./ λ, npbounds ./ normalization, label=LaTeXString("Eq. (26)"); linestyle=:dash, color=:black)
    lines!(ax, ϵs ./ λ, δEs ./ normalization, label=L"|\delta E| / λ"; linestyle=nothing, color=:black)
    axislegend(ax)
    fig
end
##
save(plotsdir("energy_splitting_comparison_dΔ.pdf"), energy_splitting_fig, px_per_unit=40)