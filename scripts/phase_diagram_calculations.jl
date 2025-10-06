using DrWatson
@quickactivate :ManybodyMajoranas
using LaTeXStrings, LinearAlgebra, Folds
function calculate_ground_state_properties(ham, HS, HR, q, gauge=FrobeniusGauge(); kwargs...)
    vals, vecs = ground_states_arnoldi(ham, HS)
    gs_odd = vcat(vecs[1], zero(vecs[2]))
    gs_even = vcat(zero(vecs[1]), vecs[2])
    (; reduced_majoranas_properties(gs_even, gs_odd, HS, HR, gauge; q, kwargs...)..., vals)
end

include("parameters.jl")
##
N = 6
qn = ParityConservation()
HS = hilbert_space(1:N, qn)
HR = hilbert_space(1:div(N, 2), qn)
@fermions f

## 
Us = range(-3, 5, 200)
δμs = range(0, 5, 200)

@time bounds = Folds.map(Base.product(Us, δμs)) do (U, δμ)
    Δ = global_parameters.t + U / 2
    μ0 = sweet_spot_μ(; U, N)
    μ = μ0 .+ δμ
    params = (; t=global_parameters.t, Δ, μ, U)
    symham = kitaev_hamiltonian(f, HS; params...)
    gauge = FrobeniusGauge()
    ham = matrix_representation(symham, HS)
    isreal(ham) && (ham = real(ham))
    calculate_ground_state_properties(ham, HS, HR, 2, gauge)[[:LFmin, :LFmax, :LD, :θmin, :θmax, :vals, :MR, :cR]]
end;
## Save data
wsave(datadir("int_kitaev_phase_diagram_frob_$N.jld2"), Dict("bounds" => bounds, "Us" => Us, "δμs" => δμs))
