using DrWatson
@quickactivate :ManybodyMajoranas
using Folds
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
    (; heff, δEs, δEs_full, p_bound, np_bound, even_norm, odd_norm, vals, vecs, pairs, overlaps1, OEBnorms)
end

##
@fermions f
N = 5
S = 1:N
B = length(S) .+ (1:1)
R = S[end-1:end]
# r = only(R)
# b = only(B)
qn = ParityConservation()
spaces = hilbert_spaces(S, R, B, qn)
@unpack HS, HB, HR, HSB, HRB = spaces
##
N = length(keys(HS))
t = global_parameters.t
U = 2t
Δ = t + U / 3
μ = frustration_free_μ(; U, Δ, global_parameters.t, N)
params = (; U, Δ, μ, t)# params = (; params..., μ=round.(params.μ; digits=4))
symham = kitaev_hamiltonian(f, HS; params...)
hS = matrix_representation(symham, HS)
vals, vecs = blockeigen(hS, HS)
nSB = div(size(vecs, 2), 2)
gs_odd = vecs[:, 1]
gs_even = vecs[:, nSB+1]
exc_gap = minimum([abs(vals[2] - vals[1]), abs(vals[nSB+2] - vals[nSB+1])]) # excitation gap
gapratio = abs((vals[1] - vals[nSB+1]) / exc_gap)
@assert gapratio < 1e-8 "Not degenerate: gapratio = $gapratio"

reduced = reduced_majoranas_properties(gs_even, gs_odd, HS, HR, FrobeniusGauge(); q=1);
## vary dot level
hRB = Hermitian(randn(dim(spaces.HRB), dim(spaces.HRB)))
hB = Hermitian(randn(ComplexF64, dim(spaces.HB), dim(spaces.HB)))
# ϵs = (x -> [-reverse(x)..., x...])(exc_gap * logrange(1e-3, 1e0, 30))
ϵs = 4exc_gap * range(-1, 1, 30)
canon_hams = canonicalize_hamiltonians((; hS0=hS, hS=hS, hB=hB, hRB=hRB), spaces)
@unpack hB, hRB, hS, hS0 = canon_hams
hBodd = FermionicHilbertSpaces.project_on_parity(hB, spaces.HB, -1)
# hRB = 1im * tensor_product((real(reduced.γRmin), hBodd), (spaces.HR, spaces.HB), spaces.HRB) |> real
hRB = hRB / norm(hRB)
@time energy_splitting_data = Folds.map(ϵs) do ϵ
    hamiltonians = (; hS0=hS, hS=hS, hB=0.5 * real(hB) / norm(real(hB)), hRB=ϵ * hRB)
    calculate_bounds(reduced, hamiltonians, spaces, 1)
end;
##
δEs = [abs(d.δEs[1, 1]) for d in energy_splitting_data]
δEs_full = [abs(d.δEs_full[1, 1]) for d in energy_splitting_data]
normalization = map(abs, ϵs)
npbounds = [d.np_bound[1, 1] for d in energy_splitting_data]
pbounds = [d.p_bound for d in energy_splitting_data]
energy_splitting_fig = with_theme(theme_aps()) do
    fig = Figure(size=150 .* (1.5, 1), figure_padding=5)
    ax = Axis(fig[1, 1]; xlabel=L"||H_{RB}||/ E_{exc}", limits=(nothing, nothing))
    colors = [Cycled(2), Cycled(4), Cycled(1)]
    lines!(ax, ϵs / exc_gap, pbounds ./ normalization, label=LaTeXString("Eq. (36)"); linestyle=(:dot, :dense), color=colors[1])
    lines!(ax, ϵs / exc_gap, npbounds ./ normalization, label=LaTeXString("Eq. (35)"); linestyle=:dash, color=colors[2])
    lines!(ax, ϵs / exc_gap, δEs ./ normalization, label=L"|\delta E| / ||H_{RB}||"; linestyle=nothing, color=colors[3])
    lines!(ax, ϵs / exc_gap, δEs_full ./ normalization, label=L"|\delta Efull| / ||H_{RB}||"; linestyle=nothing, color=:black)
    text!(fig.scene, 0.03, 0.84; text=LaTeXString("\\frac{E}{||H_{RB}||}"), space=:relative, fontsize=10)
    # text!(ax, 0.2, 0.7; text=L"Q_o = %$(round(reduced.LFmin; digits =3))", space=:relative)
    # text!(ax, 0.2, 0.55; text=L"Q_e = %$(round(reduced.LD; digits =3))", space=:relative)
    axislegend(ax; position=(0.9, 0.75))
    fig
end
##
save(plotsdir("energy_splitting_comparison_rand_$N.pdf"), energy_splitting_fig)
save(plotsdir("energy_splitting_comparison_rand_$N.png"), energy_splitting_fig, px_per_unit=1.2)

##
reduced.LFmin
reduced.LD