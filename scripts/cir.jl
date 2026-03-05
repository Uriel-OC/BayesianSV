#---- Setup ----
using Random: seed!; seed!(10)

using Distributions: InverseGamma

using StochasticDiffEq
using BayesianSV

using Statistics: quantile

using DataFrames: DataFrame
using CSV: write

#---- Constants ----
const p1 = create_bparams(
  1_000, # Burn-in iterations
  200_000, # Gibbs sampler iterations
  1.0, # Time span
  4_000, # Number of samples
  80, # Number of blocks
)
const p2 = create_bparams(
  1_000, # Burn-in iterations
  200_000, # Gibbs sampler iterations
  1.0, # Time span
  4_000, # Number of samples
  160, # Number of blocks
)

# Model hyperparameters
const α₁ = 0.1
const hiper = ARWMInfo(InverseGamma(0.3, 0.3), 100) # α ~ IG(0.3,0.3); σ₀=100

#---- Aux functions ----
b(u, p, t) = p[1] - p[2] * u

s(u, p, t) = p[3] * sqrt(u)

#---- Main ----
function (@main)(args)
  # SDE simulation
  dt = 1 // (p1.n - 1)
  tspan = (0.0, p1.T)
  cir_params = (6.0, 3.0, 2.0)

  prob = SDEProblem(b, s, 1.0, tspan, cir_params)
  sol = solve(prob, EM(), dt=dt)

  # Writing sample path
  path_path = joinpath(DATA_PATH, "cir_path.csv")
  path_df = DataFrame(sol)
  @. path_df[!, :s] = cir_params[3] * sqrt(sol.u)
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

  vol_path = joinpath(DATA_PATH, "cir_vol.csv")
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

  vol_path = joinpath(DATA_PATH, "cir_vol_double.csv")
  vol_df = DataFrame(ξ_bounds', ["lower", "upper"])
  vol_df[!, :t] .= t
  push!(vol_df, (ξ_bounds[:, end]..., 1.0))

  write(vol_path, vol_df)

  return 0
end
