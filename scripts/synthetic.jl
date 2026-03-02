#---- Setup ----
using Random: seed!; seed!(6)

using Distributions: InverseGamma

using StochasticDiffEq
using BNP

using Statistics: quantile

using DataFrames: DataFrame
using CSV: write

#---- Constants ----
const p1 = create_bparams(
  1_000, # Burn-in iterations
  200_000, # Gibbs sampler iterations
  1.0, # Time span
  4_000, # Number of samples
  160, # Number of blocks
)
const p2 = create_bparams(
  1_000, # Burn-in iterations
  200_000, # Gibbs sampler iterations
  1.0, # Time span
  4_000, # Number of samples
  320, # Number of blocks
)
const p3 = create_bparams(
  1_000, # Burn-in iterations
  200_000, # Gibbs sampler iterations
  1.0, # Time span
  4_000, # Number of samples
  80, # Number of blocks
)

# Model hyperparameters
const α₁ = 0.1
const hiper = ARWMInfo(InverseGamma(0.3, 0.3), 100) # α ~ IG(0.3,0.3); σ₀=100

#---- Aux functions ----
b(u, p, t) = -10 * u + 20

const t_j = (0.1, 0.13, 0.15, 0.23, 0.25, 0.4, 0.44, 0.65, 0.76, 0.78, 0.81)
const h_j = (4.0, -5.0, 3.0, -4.0, 5.0, -4.2, 2.1, 4.3, -3.1, 2.1, -4.2)

function K(a::T, b::T)::T where {T}
  return a >= b
end

s(u, p, t) = 10 + 3.655606 * sum(@. h_j * K(t, t_j))
s(t) = s(0, 0, t)

#---- Main ----
function (@main)(args)
  # SDE simulation
  dt = 1 // (p1.n - 1)
  tspan = (0.0, p1.T)

  prob = SDEProblem(b, s, 0, tspan)
  sol = solve(prob, EM(), dt=dt)

  # Writing sample path
  path_path = joinpath("data", "synthetic_path.csv") |> abspath
  path_df = DataFrame(sol)
  path_df[!, :s] .= s.(sol.t)
  write(path_path, path_df)

  #---- Using N=80 ----
  # Gibbs sampler
  gibbs = zeros(p1.N, p1.burn + p1.k)
  group_and_sample!(gibbs, sol.u; α₁=α₁, params=p1, ainfo=hiper)

  # Getting samples
  θ = zeros(p1.k, p1.N)
  permutedims!(θ, gibbs[:, (p1.burn+1):end], (2, 1))

  # Getting ξ lower and upper bounds
  ξ_bounds = mapslices(x -> quantile(x, [0.025, 0.975]), θ; dims=1) .|> sqrt

  # Volatility
  t = LinRange(0, p1.T, p1.N + 1) |> collect
  pop!(t)

  vol_path = joinpath("data", "synthetic_vol.csv") |> abspath
  vol_df = DataFrame(ξ_bounds', ["lower", "upper"])
  vol_df[!, :t] .= t
  push!(vol_df, (ξ_bounds[:, end]..., 1.0))

  write(vol_path, vol_df)

  #---- Using N=160 ----
  # Gibbs sampler
  gibbs = zeros(p2.N, p2.burn + p2.k)
  group_and_sample!(gibbs, sol.u; α₁=α₁, params=p2, ainfo=hiper)

  # Getting samples
  θ = zeros(p2.k, p2.N)
  permutedims!(θ, gibbs[:, (p2.burn+1):end], (2, 1))

  # Getting ξ lower and upper bounds
  ξ_bounds = mapslices(x -> quantile(x, [0.025, 0.975]), θ; dims=1) .|> sqrt

  # Volatility
  t = LinRange(0, p2.T, p2.N + 1) |> collect
  pop!(t)

  vol_path = joinpath("data", "synthetic_vol_double.csv") |> abspath
  vol_df = DataFrame(ξ_bounds', ["lower", "upper"])
  vol_df[!, :t] .= t
  push!(vol_df, (ξ_bounds[:, end]..., 1.0))

  write(vol_path, vol_df)

  #---- Using N=16 ----
  # Gibbs sampler
  gibbs = zeros(p3.N, p3.burn + p3.k)
  group_and_sample!(gibbs, sol.u; α₁=α₁, params=p3, ainfo=hiper)

  # Getting samples
  θ = zeros(p3.k, p3.N)
  permutedims!(θ, gibbs[:, (p3.burn+1):end], (2, 1))

  # Getting ξ lower and upper bounds
  ξ_bounds = mapslices(x -> quantile(x, [0.025, 0.975]), θ; dims=1) .|> sqrt

  # Volatility
  t = LinRange(0, p3.T, p3.N + 1) |> collect
  pop!(t)

  vol_path = joinpath("data", "synthetic_vol_half.csv") |> abspath
  vol_df = DataFrame(ξ_bounds', ["lower", "upper"])
  vol_df[!, :t] .= t
  push!(vol_df, (ξ_bounds[:, end]..., 1.0))

  write(vol_path, vol_df)

  return 0
end
