using LinearAlgebra

struct Lead{T1,T2,T3,C}
    T::T1              # Temperature
    μ::T2              # Chemical potential
    Γ::T3              # Tunneling rate
    c::C   # Annihilation operator in many-body eigenbasis
end
fermi(ε, T) = T > 0 ? 1 / (1 + exp(ε / T)) : Float64(ε < 0)

"""
    compute_rates(E, leads)
Returns W_in[lead_idx][k, n] and W_out[lead_idx][k, n]
"""
function compute_rates(E::Vector, leads)
    N = length(E)
    W_in = [zeros(N, N) for _ in leads]
    W_out = [zeros(N, N) for _ in leads]
    compute_rates!((W_in, W_out), E, leads)
end
function compute_rates!((W_in, W_out), E::Vector, leads)
    N = length(E)

    for (α, (l, W_in, W_out)) in enumerate(zip(leads, W_in, W_out))
        c_dag = l.c'
        for n in 1:N, k in 1:N
            ΔE = E[k] - E[n]
            W_in[k, n] = l.Γ * abs2(c_dag[k, n]) * fermi(ΔE + l.μ, l.T)
            W_out[k, n] = l.Γ * abs2(l.c[k, n]) * fermi(ΔE - l.μ, l.T)
        end
    end
    return W_in, W_out
end

function solve_steady_state(W_in, W_out)
    N = size(W_in[1], 1)
    # Total transition matrix W_kn (rate from n to k)
    W = sum(W_in) + sum(W_out)
    for n in 1:N
        W[n, n] = -sum(W[:, n]) # Column sum must be 0
    end

    # Solve WP = 0 with sum(P) = 1
    A = copy(W)
    A[1, :] .= 1.0
    b = zeros(N)
    b[1] = 1.0
    return A \ b
end

function get_current(P, W_in_α, W_out_α)
    sum(P[n] * (-sum(W_in_α[:, n]) + sum(W_out_α[:, n])) for n in eachindex(P))
end

"""
    conductance_matrix(E, leads; dμ=1e-6)
Returns matrix G where G[α, β] = dI_α / dμ_β
"""
function conductance_matrix(E::Vector, leads; dμ=1e-6)
    L = length(leads)

    # Baseline current
    W_in, W_out = compute_rates(E, leads)
    P = solve_steady_state(W_in, W_out)
    I0 = [get_current(P, W_in[α], W_out[α]) for α in 1:L]

    G = zeros(L, L)
    for β in 1:L
        # Perturb chemical potential of lead β
        leads_perturbed = [n == β ? Lead(l.T, l.μ + dμ, l.Γ, l.c) : l
                           for (n, l) in enumerate(leads)]

        Wi_p, Wo_p = compute_rates(E, leads_perturbed)
        P_p = solve_steady_state(Wi_p, Wo_p)

        for α in 1:L
            I_p = get_current(P_p, Wi_p[α], Wo_p[α])
            G[α, β] = (I_p - I0[α]) / dμ
        end
    end
    return G
end

function conductance_groups(E::Vector, leads, groups; dμ=1e-6)
    Ng = length(groups)

    # baseline rates and steady state
    W_in, W_out = compute_rates(E, leads)
    P0 = solve_steady_state(W_in, W_out)

    # baseline group currents
    I0 = zeros(Ng)
    for g in 1:Ng
        for α in groups[g]
            I0[g] += get_current(P0, W_in[α], W_out[α])
        end
    end

    G = zeros(Ng, Ng)

    for gβ in 1:Ng
        # build perturbed lead list
        leads_perturbed = copy(leads)
        for α in groups[gβ]
            l = leads[α]
            leads_perturbed[α] = Lead(l.T, l.μ + dμ, l.Γ, l.c)
        end

        Wi_p, Wo_p = compute_rates!((W_in, W_out), E, leads_perturbed)
        P_p = solve_steady_state(Wi_p, Wo_p)

        for gα in 1:Ng
            I_p = 0.0
            for α in groups[gα]
                I_p += get_current(P_p, Wi_p[α], Wo_p[α])
            end
            G[gα, gβ] = (I_p - I0[gα]) / dμ
        end
    end
    return G
end

##
params = (; EZ=(; L=1.5Γ, R=1.5Γ, H=0.5Γ),
    U=(; L=5Γ, R=5Γ, H=0Γ),
    wL=Γ, wSOL=Γ / 2, wR=0.75Γ, wSOR=0.75Γ / 2,
    Γ=(; L=0Γ, R=Γ, H=Γ))
sweet_spot = (; μ=(; H=0.617Γ, L=2.290Γ, R=-5.879Γ))
symham = symbolic_hamiltonian(; params..., sweet_spot...)
ham = matrix_representation(symham, H)
vals, vecs = eigen(Matrix(ham))
@fermions c
cLup = vecs' * matrix_representation(c[:L, :↑], H) * vecs'
cRup = vecs' * matrix_representation(c[:R, :↑], H) * vecs
cLdn = vecs' * matrix_representation(c[:L, :↓], H) * vecs
cRdn = vecs' * matrix_representation(c[:R, :↓], H) * vecs

##
leads = [
    Lead(0.01, 0.0, 0.2, cLup),  # Left lead
    Lead(0.01, 0.0, 0.2, cLdn),  # Left lead
    Lead(0.01, 0.0, 0.2, cRup),   # Right lead
    Lead(0.01, 0.0, 0.2, cRdn)   # Right lead
]
G = conductance_matrix(vals, leads, dμ=1e-5)
println("Conductance Matrix (G_αβ = dI_α/dμ_β):")
G

## Scan μL, voltages
μLs = 2.5 * range(-1, 1, 141)
Vs = 2 * range(-1, 1, 142)
@time Gs = stack(μLs) do μL
    params = (; EZ=(; L=1.5Γ, R=1.5Γ, H=0.5Γ),
        U=(; L=5Γ, R=5Γ, H=0Γ),
        wL=Γ, wSOL=Γ / 2, wR=0.75Γ, wSOR=0.75Γ / 2,
        Γ=(; L=0Γ, R=Γ, H=Γ))
    sweet_spot = (; μ=(; H=0.617Γ, L=2.290Γ + μL, R=-5.879Γ))
    symham = symbolic_hamiltonian(; params..., sweet_spot...)
    ham = matrix_representation(symham, H)
    vals, vecs = eigen(Matrix(ham))
    @fermions c
    cLup = vecs' * matrix_representation(c[:L, :↑], H) * vecs
    cRup = vecs' * matrix_representation(c[:R, :↑], H) * vecs
    cLdn = vecs' * matrix_representation(c[:L, :↓], H) * vecs
    cRdn = vecs' * matrix_representation(c[:R, :↓], H) * vecs

    Folds.map(Vs) do V
        γ = 1
        T = 0.03
        leads = [
            Lead(T, V, γ, cLup),  # Left lead
            Lead(T, V, γ, cLdn),  # Left lead
            Lead(T, zero(V), γ, cRup),   # Right lead
            Lead(T, zero(V), γ, cRdn)   # Right lead
        ]
        G = conductance_matrix(vals, leads, dμ=1e-3 * T)
        # G = conductance_groups(vals, leads, [[1, 2], [3, 4]])
    end
end;
##
fig = Figure(; size=(1200, 1200))
g = fig[1, 1] = GridLayout()
for (n, k) in Base.product(1:2, 1:2)
    ax = Axis(g[n, k], aspect=1)
    dat = map(G -> sum(G[(2n-1):2n, k+2]), Gs)
    heatmap!(ax, VRs, μLs, dat', colormap=:vik, colorrange=maximum(abs, dat) .* (-1, 1))
    # heatmap!(ax, VRs, μLs, dat', colormap=:vik, colorrange=(-0.05, 0.05))
end
fig
##
fig = Figure(; size=(1200, 1200))
g = fig[1, 1] = GridLayout()
for (n, k) in Base.product(1:2, 1:2)
    ax = Axis(g[n, k], aspect=1)
    dat = map(G -> sum(G[n, k]), Gs)
    heatmap!(ax, VRs, μLs, dat, colormap=:vik, colorrange=maximum(abs, dat) .* (-1, 1))
    # heatmap!(ax, VRs, μLs, dat', colormap=:vik, colorrange=(-0.05, 0.05))
end
fig
##
fig = Figure(; size=(1200, 1200))
g = fig[1, 1] = GridLayout()
for (n, k) in Base.product(1:4, 1:4)
    ax = Axis(g[n, k], aspect=1)
    dat = map(G -> sum(G[n, k]), Gs)
    heatmap!(ax, VRs, μLs, dat, colormap=:vik, colorrange=maximum(abs, dat) .* (-1, 1))
    # heatmap!(ax, VRs, μLs, dat', colormap=:vik, colorrange=(-0.05, 0.05))
end
fig