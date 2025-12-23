using DrWatson
@quickactivate :ManybodyMajoranas
using Folds
using UnPack, CairoMakie, MakiePublication, LaTeXStrings

function symbolic_hamiltonian(; μ, EZ, U, Γ, wL, wSOL, wR, wSOR)
    @fermions c
    His = map((:L, :H, :R)) do l
        nup = c[l, :↑]'c[l, :↑]
        ndn = c[l, :↓]'c[l, :↓]
        Hi = (μ[l] + EZ[l]) * nup + (μ[l] - EZ[l]) * ndn +
             U[l] * nup * ndn +
             (Γ[l] * c[l, :↑]'c[l, :↓]' + hc)
    end
    HT = wL * (c[:H, :↑]'c[:L, :↑] + c[:H, :↓]'c[:L, :↓]) +
         wSOL * (c[:H, :↓]'c[:L, :↑] - c[:H, :↑]'c[:L, :↓]) +
         wR * (c[:R, :↑]'c[:H, :↑] + c[:R, :↓]'c[:H, :↓]) +
         wSOR * (c[:R, :↓]'c[:H, :↑] - c[:R, :↑]'c[:H, :↓]) +
         hc
    sum(His) + HT
end
##
spatial_labels = (:L, :H, :R)
spins = (:↑, :↓)
labels = Base.product(spatial_labels, spins)
H = hilbert_space(labels, ParityConservation())

##
Γ = 1
params = (; EZ=(; L=1.5Γ, R=1.5Γ, H=0.5Γ),
    U=(; L=5Γ, R=5Γ, H=0Γ),
    wL=Γ, wSOL=Γ / 2, wR=0.75Γ, wSOR=0.75Γ / 2,
    Γ=(; L=0Γ, R=Γ, H=Γ))
sweet_spot = (; μ=(; H=0.617Γ, L=2.290Γ, R=-5.879Γ))
##
symham = symbolic_hamiltonian(; params..., sweet_spot...)
ham = matrix_representation(symham, H)

## Energy spectrum 
VHs = range(-1Γ, 1Γ, 100)
ham0 = symbolic_hamiltonian(; params..., sweet_spot...)
Hs = FermionicHilbertSpaces.sectors(H)
ham0s = [Matrix(matrix_representation(symham, H)) for H in Hs]
@fermions c
perts = [matrix_representation(c[:H, :↑]'c[:H, :↑] + c[:H, :↓]'c[:H, :↓], H) for H in Hs]
vals = stack(VHs) do VH
    _vals = mapreduce(vcat, ham0s, perts, Hs) do ham0, pert, H
        eigvals!(Hermitian(ham0 + VH * pert), 1:3)
    end
    _vals .- minimum(_vals)
end
##
fig = Figure()
ax = Axis(fig[1, 1];)
for es in eachrow(vals)
    lines!(ax, VHs, es)
end
fig
##
sweet_ground_states = map(ham0s, Hs) do ham, Hsec
    vecs = eigvecs(Hermitian(ham))
    inds = FermionicHilbertSpaces.indices(Hsec, H)
    gs = zeros(dim(H))
    gs[inds] .= vecs[:, 1]
    gs
end
sublabels_iter = [#((:R, :↑), (:R, :↓)),
    ((:R, :↑),),
    ((:R, :↓),),
    # ((:L, :↑), (:L, :↓)),
    ((:L, :↑),),
    ((:L, :↓),),
    #((:H, :↑),),
    #((:H, :↓),)
]
reduced = map(sublabels_iter) do sublabels
    (; reduced_majoranas_properties(sweet_ground_states..., H, subregion(sublabels, H))..., space=first(first(sublabels)), spin=map(last, sublabels))
end;
##
lds = map(x -> string("LD[", x.space, x.spin..., "]") => round(x.LD; sigdigits=3), reduced)
lfs = map(x -> string("γ[", x.space, x.spin..., "]") => round(x.LFmin; sigdigits=3), reduced)
# map(x -> x.LFmax, reduced)
## Tune left dot chemical potential
VLPs = 2 * range(-1Γ, 1Γ, 100)
ham0 = symbolic_hamiltonian(; params..., sweet_spot...)
Hs = FermionicHilbertSpaces.sectors(H)
ham0s = [Matrix(matrix_representation(ham0, H)) for H in Hs]
@fermions c
perts = [matrix_representation(c[:L, :↑]'c[:L, :↑] + c[:L, :↓]'c[:L, :↓], H) for H in Hs]
vals = stack(VLPs) do V
    _vals = mapreduce(vcat, ham0s, perts, Hs) do ham0, pert, H
        eigvals!(Hermitian(ham0 + V * pert), 1:3)
    end
    _vals .- minimum(_vals)
end
fig = Figure()
ax = Axis(fig[1, 1];)
for es in eachrow(vals)
    lines!(ax, VHs, es)
end
fig

## Scan VH, calculate reduced properties
VHs = -sweet_spot.μ.H .+ 3 * range(-1Γ, 1Γ, 100)
ham0 = symbolic_hamiltonian(; params..., sweet_spot...)
# ham0 = symbolic_hamiltonian(; params..., μ=(; H=0.617Γ, L=2.290Γ, R=-5.879Γ))
Hs = FermionicHilbertSpaces.sectors(H)
ham0s = [Matrix(matrix_representation(ham0, H)) for H in Hs]
@fermions c
perts = [matrix_representation(c[:H, :↑]'c[:H, :↑] + c[:H, :↓]'c[:H, :↓], H) for H in Hs]
fermionops = fermions(H)
sublabels_iter = [((:R, :↑),),
    ((:R, :↓),),
    ((:L, :↑),),
    ((:L, :↓),),
    ((:H, :↑),),
    ((:H, :↓),)]
data = map(VHs) do V
    sweet_ground_states = map(ham0s, perts, Hs) do ham0, pert, Hsec
        vecs = eigvecs(Hermitian(ham0 + V * pert))
        inds = FermionicHilbertSpaces.indices(Hsec, H)
        gs = zeros(dim(H))
        gs[inds] .= vecs[:, 1]
        gs
    end
    o, e = sweet_ground_states
    us = Dict(k => dot(o, v', e) for (k, v) in fermionops)
    vs = Dict(k => dot(o, v, e) for (k, v) in fermionops)
    subinfo = map(sublabels_iter) do sublabels
        (; reduced_majoranas_properties(sweet_ground_states..., H, subregion(sublabels, H))..., space=first(first(sublabels)), spin=map(last, sublabels))
    end
    (; subinfo, us, vs)
end
## Crossing plots
μLs = range(5, -13, 40)
μRs = range(10, -2, 42)
ham0 = symbolic_hamiltonian(; params..., μ=(L=0, H=0, R=0))
Hs = FermionicHilbertSpaces.sectors(H)
ham0s = [Matrix(matrix_representation(symham, H)) for H in Hs]
@fermions c
hμs = [matrix_representation(c[l, :↑]'c[l, :↑] + c[l, :↓]'c[l, :↓], H) for H in Hs, l in spatial_labels]
μH = 0Γ
@time gaps = Folds.map(Base.product(μLs, μRs)) do (μL, μR)
    μs = (μL, μH, μR)
    _vals = map(ham0s, eachrow(hμs), Hs) do ham0, hμs, H
        eigvals!(Hermitian(ham0 + sum(hμs .* μs)), 1:1) |> only
    end
    _vals |> diff |> only
end
fig, ax, hm = heatmap(μLs, μRs, gaps; colormap=:vik, colorrange=maximum(abs, gaps) .* (-1, 1))
Colorbar(fig[1, 2], hm)
fig


##


## Scan VH, reproduce YSR plots
VHs = -sweet_spot.μ.H .+ 3 * range(-1Γ, 1Γ, 100)
# ham0 = symbolic_hamiltonian(; params..., sweet_spot...)
ham0 = symbolic_hamiltonian(; params..., μ=(; H=0.617Γ, L=2.290Γ, R=-5.879Γ + 1e6))
Hs = FermionicHilbertSpaces.sectors(H)
ham0s = [Matrix(matrix_representation(ham0, H)) for H in Hs]
@fermions c
perts = [matrix_representation(c[:H, :↑]'c[:H, :↑] + c[:H, :↓]'c[:H, :↓], H) for H in Hs]
fermionops = fermions(H)
sublabels_iter = [((:R, :↑),),
    ((:R, :↓),),
    ((:L, :↑),),
    ((:L, :↓),),
    ((:H, :↑),),
    ((:H, :↓),)]
ham_nLs = [matrix_representation(c[:L, :↑]'c[:L, :↑] + c[:L, :↓]'c[:L, :↓], H) for H in Hs]
using Roots
function find_zero_Ediff(hams, hVs, bracket)
    length(hams) == length(hVs)
    f = x -> map((ham, hV) -> eigvals!(Hermitian(ham + x * hV), 1:1) |> only, hams, hVs) |> diff |> only
    find_zero(f, bracket)
end
μLs = []
ediffs = []
data = map(VHs) do V
    hams = ham0s .+ V .* perts
    μL = find_zero_Ediff(hams, ham_nLs, (-2.5, 5))
    push!(μLs, μL)
    sweet_ground_states = map(hams, Hs, ham_nLs) do ham, Hsec, ham_nL
        ham = Hermitian(ham + μL * ham_nL)
        vals, vecs = eigen(ham)
        push!(ediffs, vals[1])
        inds = FermionicHilbertSpaces.indices(Hsec, H)
        gs = zeros(dim(H))
        gs[inds] .= vecs[:, 1]
        gs
    end
    o, e = sweet_ground_states
    us = Dict(k => dot(o, v', e) for (k, v) in fermionops)
    vs = Dict(k => dot(o, v, e) for (k, v) in fermionops)
    subinfo = map(sublabels_iter) do sublabels
        (; reduced_majoranas_properties(sweet_ground_states..., H, subregion(sublabels, H))..., space=first(first(sublabels)), spin=map(last, sublabels))
    end
    (; subinfo, us, vs)
end
##
fig = Figure(; size=(1000, 300))
ax = Axis(fig[1, 2]; aspect=1.25, title="weight")
ax2 = Axis(fig[1, 1]; aspect=1.25, title="charge")
ax3 = Axis(fig[1, 3]; aspect=1.25, title="normalized charge")
ylims!(ax, 0, 1)
ylims!(ax2, -0.6, 0.95)
ylims!(ax3, -1.05, 1.05)
colors_red = cgrad(:Reds, 5, categorical=true)[3:end]
colors_blue = cgrad(:Blues, 5, categorical=true)[3:end]
colors_green = cgrad(:Greens, 5, categorical=true)[3:end]
colors = [colors_red[1:2]..., colors_blue[1:2]..., colors_green[1:2]...]
linewidth = 4
μs = -(VHs .+ sweet_spot.μ.H)
uis = Dict(l => map(x -> sum(σ -> abs2(x.us[l, σ]), spins), data) for l in spatial_labels)
vis = Dict(l => map(x -> sum(σ -> abs2(x.vs[l, σ]), spins), data) for l in spatial_labels)
colors = Dict(:L => Cycled(1), :H => Cycled(2), :R => Cycled(3))
foreach(spatial_labels[1:2]) do l
    color = colors[l]
    label = string(l)
    kwargs = (; color, linewidth, label)
    lines!(ax, μs, uis[l] + vis[l]; kwargs...)
    lines!(ax2, μs, -(vis[l] - uis[l]); kwargs...)
    lines!(ax3, μs, -(vis[l] .- uis[l]) ./ (uis[l] + vis[l]); kwargs...)
end
foreach((ax, ax2, ax3)) do ax
    vlines!(ax, [-sweet_spot.μ.H]; color=:grey, linestyle=:dash)
    axislegend(ax)
    tightlimits!(ax)
end
fig
##

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
    np_bound_even = (reduced.LD * even_norm) ./ overlaps1
    np_bound_odd = (reduced.LFmin * odd_norm * OEBnorms) ./ overlaps1
    p_bound = sqrt(dim(spaces.HB)) * (reduced.LD * even_norm + reduced.LFmin * odd_norm)
    p_bound_LD = sqrt(dim(spaces.HB)) * (reduced.LD * even_norm)
    p_bound_LF = sqrt(dim(spaces.HB)) * (reduced.LFmin * odd_norm)
    (; heff, δEs, δEs_full, p_bound, np_bound, even_norm, odd_norm, vals, vecs, pairs, overlaps1, OEBnorms, p_bound_LD, p_bound_LF, np_bound_odd, np_bound_even)
end

##
S = collect(labels) |> vec
# R = ((:R, :↑),)
R = ((:R, :↓),)
B = ((:B, 0),)
qn = ParityConservation()
spaces = hilbert_spaces(S, R, B, qn)

symham = symbolic_hamiltonian(; params..., sweet_spot...)
hS = matrix_representation(symham, spaces.HS)
vals, vecs = blockeigen(hS, spaces.HS)
nSB = div(size(vecs, 2), 2)
gs_odd = vecs[:, 1]
gs_even = vecs[:, nSB+1]
##
λ = 0.5 * global_parameters.t
θ = pi / 6
tc = λ * cos(θ)
Δc = λ * sin(θ)
Uc = λ
@fermions f
hRBsym = sum(tc * f[l]' * f[only(B)] + Δc * f[l] * f[only(B)] + hc +
             Uc * f[only(B)]' * f[only(B)] * f[l]' * f[l] for l in R)
hRB = matrix_representation(hRBsym, spaces.HRB)
ϵs = 2 * λ * range(-1, 1, 50)
reduced = reduced_majoranas_properties(gs_even, gs_odd, spaces.HS, spaces.HR, FrobeniusGauge(); q=2);
@time energy_splitting_data = Folds.map(ϵs) do ϵ
    hB = matrix_representation(ϵ * f[only(B)]' * f[only(B)], spaces.HB)
    hamiltonians = (; hS0=hS, hS=hS, hB=hB, hRB=hRB)
    calculate_bounds(reduced, hamiltonians, spaces, 2)
end;
##
δEs = [abs(d.δEs[1, 1]) for d in energy_splitting_data]
δEs_full = [abs(d.δEs_full[1, 1]) for d in energy_splitting_data]
normalization = λ
npbounds = [d.np_bound[1, 1] for d in energy_splitting_data]
npbounds_even = [d.np_bound_even[1, 1] for d in energy_splitting_data]
npbounds_odd = [d.np_bound_odd[1, 1] for d in energy_splitting_data]
pbounds = [d.p_bound for d in energy_splitting_data]
pbounds_LD = [d.p_bound_LD for d in energy_splitting_data]
pbounds_LF = [d.p_bound_LF for d in energy_splitting_data]

energy_splitting_fig = with_theme(theme_aps()) do
    fig = Figure(size=150 .* (1.5, 1), figure_padding=5)
    ax = Axis(fig[1, 1]; xlabel=L"\varepsilon_d/ λ", limits=(nothing, (0, 1.1 * pbounds[1] / normalization)), title="R=$R")
    colors = [Cycled(2), Cycled(4), Cycled(1)]
    lines!(ax, ϵs ./ λ, pbounds ./ normalization, label=LaTeXString("Eq. (36)"); linestyle=(:dot, :dense), color=Cycled(5))

    lines!(ax, ϵs ./ λ, pbounds_LD ./ normalization, label=LaTeXString("even"); linestyle=:dash, color=Cycled(5))
    lines!(ax, ϵs ./ λ, pbounds_LF ./ normalization, label=LaTeXString("odd"); linestyle=:solid, color=Cycled(5))

    lines!(ax, ϵs ./ λ, npbounds ./ normalization, label=LaTeXString("Eq. (35)"); linestyle=:dot, color=colors[2])
    lines!(ax, ϵs ./ λ, npbounds_even ./ normalization, label=LaTeXString("even"); linestyle=:dash, color=colors[2])
    lines!(ax, ϵs ./ λ, npbounds_odd ./ normalization, label=LaTeXString("odd"); linestyle=:solid, color=colors[2])
    lines!(ax, ϵs ./ λ, δEs ./ normalization, label=L"|\delta Eeff| / λ"; linestyle=nothing, color=:black)

    # lines!(ax, ϵs ./ λ, δEs_full ./ normalization, label=L"|\delta Efull| / λ"; linestyle=nothing, color=:black)
    text!(fig.scene, 0.03, 0.84; text=LaTeXString("\\frac{E}{λ}"), space=:relative, fontsize=10)
    # axislegend(ax; position=(0.9, 0.75))
    Legend(fig[1, 2], ax)
    fig
end
extra_label = last(only(R)) == :↑ ? :up : :dn
save(plotsdir("energy_splitting_spin$(extra_label)_$(first(first(R))).png"), energy_splitting_fig, px_per_unit=6.2)


##

using Roots

ham0 = symbolic_hamiltonian(; params..., μ=(; H=0.617Γ, L=2.290Γ, R=-5.879Γ))
Hs = FermionicHilbertSpaces.sectors(H)
ham0s = [Matrix(matrix_representation(ham0, H)) for H in Hs]
@fermions c
perts = [matrix_representation(c[:H, :↑]'c[:H, :↑] + c[:H, :↓]'c[:H, :↓], H) for H in Hs]
fermionops = fermions(H)
sublabels_iter = [((:R, :↑),),
    ((:R, :↓),),
    ((:L, :↑),),
    ((:L, :↓),)]
ham_nLs = [matrix_representation(c[:L, :↑]'c[:L, :↑] + c[:L, :↓]'c[:L, :↓], H) for H in Hs]
μLs = []
ediffs = []
dμs = range(-0.01, 0.01, 20)
data = map(dμs) do V
    hams = ham0s .+ V .* perts
    μL = find_zero_Ediff(hams, ham_nLs, (-0.5, 0.5))
    push!(μLs, μL)
    sweet_ground_states = map(hams, Hs, ham_nLs) do ham, Hsec, ham_nL
        ham = Hermitian(ham + μL * ham_nL)
        vals, vecs = eigen(ham)
        push!(ediffs, vals[1])
        inds = FermionicHilbertSpaces.indices(Hsec, H)
        gs = zeros(dim(H))
        gs[inds] .= vecs[:, 1]
        gs
    end
    o, e = sweet_ground_states
    us = Dict(k => dot(o, v', e) for (k, v) in fermionops)
    vs = Dict(k => dot(o, v, e) for (k, v) in fermionops)
    subinfo = map(sublabels_iter) do sublabels
        (; reduced_majoranas_properties(sweet_ground_states..., H, subregion(sublabels, H))..., space=first(first(sublabels)), spin=map(last, sublabels))
    end
    (; subinfo, us, vs)
end
plot(dμs, map(x -> norm(map(y -> y.LD, x.subinfo)), data))