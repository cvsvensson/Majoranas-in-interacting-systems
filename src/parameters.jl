
global_parameters = (; t=1.0)
function example_point_parameters(HS; tol=1e-6, kwargs...)
    N = length(keys(HS))
    t = global_parameters.t
    U = 2.5
    δμ = 3.0
    μ0 = sweet_spot_μ(; U, N)
    μ = μ0 .+ δμ
    Δ = t + U / 2
    Δ0 = 2 / 3 * Δ
    dΔ = optimized_Δ(HS; μ, t, Δ=Δ0, U, tol, kwargs...)
    println("Optimized dΔ: $dΔ")
    Δdeg = Δ0 + dΔ
    println("Optimized Δ: $Δdeg")
    (; U, Δ, Δdeg, μ, t, δμ)
end