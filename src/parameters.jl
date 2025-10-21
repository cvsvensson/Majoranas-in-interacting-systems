
global_parameters = (; t=1.0)
function good_majoranas_parameters(HS)
    N = length(keys(HS))
    t = global_parameters.t
    U = 1.5
    δμ = 1.2
    μ0 = sweet_spot_μ(; U, N)
    μ = μ0 .+ δμ
    Δ = t + U / 2
    (; U, Δ, μ, t, δμ)
end

function energy_splitting_parameters(HS;)
    N = length(keys(HS))
    t = global_parameters.t
    U = 2t
    Δ = t
    μ = frustration_free_μ(; U, Δ, global_parameters.t, N)
    (; U, Δ, μ, t)
end

function bad_majoranas_parameters(HS)
    N = length(keys(HS))
    t = global_parameters.t
    U = 1.5
    δμ = 4.0
    μ0 = sweet_spot_μ(; U, N)
    μ = μ0 .+ δμ
    Δ = t + U / 2
    (; U, Δ, μ, t, δμ)
end