#---- Setup ----
using Random: seed!; seed!(14);

using Distributions: InverseGamma

using BayesianSV

using Statistics: quantile

using Dates: Date, Week
using DataFrames: DataFrame
using CSV: write

#---- Constants ----
const burn = 1_000 # Burn-in iterations
const k = 200_000 # Gibbs sampler iterations

# Model hyperparameters
const α₁ = 0.1
const hiper = ARWMInfo(InverseGamma(0.3, 0.3), 100) # α ~ IG(0.3,0.3); σ₀=100

#---- Aux funtions ----
function get_log_returns(x::AbstractVector{T})::Vector{T} where {T<:AbstractFloat}
  c = first(x)
  return @. log(x / c)
end

#---- Main ----
function (@main)(args)
  # Data
  data_path = joinpath(DATA_PATH, "DWJ.txt")
  prices = parse.(Float64, readlines(data_path))

  # Missing variables
  n = length(prices)
  T = n / 52 # Weeks -> Years

  # Get logreturns
  returns = get_log_returns(prices)

  # Compute difference
  diff_lrt = diff(returns)
  pushfirst!(diff_lrt, 0)

  # Period of DWJ data
  period = Date(1971, 7, 2):Week(1):Date(1974, 8, 2)

  # Index data
  out_path = joinpath(DATA_PATH, "DWJ_data.csv")
  data_df = DataFrame(
    :date => period,
    :price => prices,
    :logreturn => returns
  )
  write(out_path, data_df)

  #---- Using N=13 ----
  p1 = create_bparams(burn, k, T, n, 13)

  # Gibbs sampler
  gibbs = zeros(p1.N, burn + k)
  group_and_sample!(gibbs, diff_lrt; α₁=α₁, params=p1, ainfo=hiper)

  # Getting samples
  θ = zeros(k, p1.N)
  permutedims!(θ, gibbs[:, (burn+1):end], (2, 1))

  ξ_bounds = mapslices(x -> quantile(x, [0.025, 0.975]), θ; dims=1) .|> sqrt

  # Volatility
  vol_df = DataFrame(ξ_bounds', ["lower", "upper"])
  vol_df[!, :t] .= [period[p1.m*i] for i in 1:p1.N]
  push!(vol_df, (ξ_bounds[:, end]..., last(period)))

  vol_path = joinpath(DATA_PATH, "DWJ_estimate.csv")
  write(vol_path, vol_df)

  #---- Using N=26 ----
  p2 = create_bparams(burn, k, T, n, 26)

  # Gibbs sampler
  gibbs = zeros(p2.N, burn + k)
  group_and_sample!(gibbs, diff_lrt; α₁=α₁, params=p2, ainfo=hiper)

  # Getting samples
  θ = zeros(k, p2.N)
  permutedims!(θ, gibbs[:, (burn+1):end], (2, 1))

  ξ_bounds = mapslices(x -> quantile(x, [0.025, 0.975]), θ; dims=1) .|> sqrt

  # Period of DWJ data
  period = Date(1971, 7, 2):Week(1):Date(1974, 8, 2)

  # Volatility
  vol_df = DataFrame(ξ_bounds', ["lower", "upper"])
  vol_df[!, :t] .= [period[p2.m*i] for i in 1:p2.N]
  push!(vol_df, (ξ_bounds[:, end]..., last(period)))

  vol_path_double = joinpath(DATA_PATH, "DWJ_estimate_double.csv")
  write(vol_path_double, vol_df)

  return 0
end
