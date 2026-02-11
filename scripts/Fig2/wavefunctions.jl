using DrWatson
@quickactivate :ManybodyMajoranas

function calculate_wavefunctions(params, HS; q=2, gauge=EigGauge())
    @fermions f
    symham = kitaev_hamiltonian(f, HS; params[[:U, :Δ, :μ, :t]]...)
    ham = matrix_representation(symham, HS)
    isreal(ham) && (ham = real(ham))
    vals, vecs = ground_states_arnoldi(ham, HS)
    gs_odd = vcat(vecs[1], zero(vecs[2]))
    gs_even = vcat(zero(vecs[1]), vecs[2])
    wavefunction = map(R -> reduced_majoranas_properties(gs_even, gs_odd, HS, hilbert_space([R]), gauge; q), keys(HS))
    (; wavefunction, vals, vecs, params)
end

##
N = 8
qn = ParityConservation()
HS = hilbert_space(1:N, qn)
good = calculate_wavefunctions(good_majoranas_parameters(HS), HS)
bad = calculate_wavefunctions(bad_majoranas_parameters(HS), HS)
##
wsave(datadir("int_kitaev_wavefunctions_$N.jld2"), Dict("good" => good, "bad" => bad))
##
params = good_majoranas_parameters(HS)
symham = kitaev_hamiltonian(f, HS; params[[:U, :Δ, :μ, :t]]...)
ham = matrix_representation(symham, HS)
isreal(ham) && (ham = real(ham))

vals, vecs = ground_states_arnoldi(ham, HS)
gs_odd = vcat(vecs[1], zero(vecs[2]))
gs_even = vcat(zero(vecs[1]), vecs[2])
q = 2
gauge = EigGauge()
@time wavefunction_data = map(R -> reduced_majoranas_properties(gs_even, gs_odd, HS, hilbert_space([R]), gauge; q), 1:N)

params2 = bad_majoranas_parameters(HS)
symham2 = kitaev_hamiltonian(f, HS; params2[[:U, :Δ, :μ, :t]]...)
ham2 = matrix_representation(symham2, HS)
isreal(ham2) && (ham2 = real(ham2))

vals2, vecs2 = ground_states_arnoldi(ham2, HS)
gs_odd2 = vcat(vecs2[1], zero(vecs2[2]))
gs_even2 = vcat(zero(vecs2[1]), vecs2[2])
wavefunction_data2 = map(R -> reduced_majoranas_properties(gs_even2, gs_odd2, HS, hilbert_space([R]), gauge; q), 1:N)

