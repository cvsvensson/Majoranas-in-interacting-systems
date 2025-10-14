using DrWatson, LinearAlgebra
using FermionicHilbertSpaces: complementary_subsystem
@quickactivate :ManybodyMajoranas
@fermions f
N = 3
S = 1:N
HS = hilbert_space(S, ParityConservation())
t = 1.0
U = 2.0
Δ = t + U/2
μ = ManybodyMajoranas.frustration_free_μ(; N, t, Δ, U)
symham = kitaev_hamiltonian(f, HS; μ, t, Δ, U)
ham = matrix_representation(symham, HS)
vals, vecs = ground_states_arnoldi(ham, HS)
@test abs(vals[2] - vals[1]) < 1e-10
o = vcat(vecs[1], zero(vecs[2]))
e = vcat(zero(vecs[1]), vecs[2])
y = o*e' + hc
yt = 1im*o*e' + hc

maj_labels = vec(permutedims(collect(Base.product(S, (:+, :-)))))
fmats = fermions(HS)
Γmats = reduce(vcat, [[fmats[j] + hc, 1im * fmats[j] + hc] for j in S])
Γ = Dict(zip(maj_labels, Γmats))
@test Γ[1, :+] ≈ fmats[1]' + fmats[1]
@test Γ[N, :-] ≈ 1im * (fmats[N] - fmats[N]')
Q = 1 / 2^(N - 1) * prod(I - 1im * Γ[j, :-] * Γ[j + 1, :+] for j in 1:N-1)
@test Q ≈ e * e' + o * o'
@test y ≈ Γ[1, :+] * Q
@test yt ≈ Γ[N, :-] * Q

Hjs = [hilbert_space(j:j, ParityConservation()) for j in S]
yjs = [partial_trace(y, HS => Hj) for Hj in Hjs]
@test yjs[1] ≈ partial_trace(Γ[1, :+], HS => Hjs[1]) * dim(Hjs[1]) / dim(HS)
@test all(norm.(yjs[2:N]) .< 1e-10)
ytjs = [partial_trace(yt, HS => Hj) for Hj in Hjs]
@test ytjs[N] ≈ partial_trace(Γ[N, :-], HS => Hjs[N]) * dim(Hjs[N]) / dim(HS)
@test all(norm.(ytjs[1:N - 1]) .< 1e-10)
