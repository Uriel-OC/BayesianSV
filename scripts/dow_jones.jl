#---- Setup ----
using Random: seed!; seed!(14);

using Distributions: InverseGamma

using BayesianSV

using MCMCChains

using Statistics: quantile

using Dates: Date, Week
using DataFrames: DataFrame
using CSV: write

#---- Constants ----
const nc = 4 # Number of chains
const burn = 1_000 # Burn-in iterations
const k = 50_000 # Gibbs sampler iterations

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

  # Initialise chain
  θ = Chains(zeros(p1.k, p1.N, nc), map(Symbol, 1:p1.N))
  # Initialise Gibbs
  gibbs = zeros(p1.N, burn + k)
  for i in 1:nc
    # Gibbs sampler
    group_and_sample!(gibbs, diff_lrt; α₁=α₁, params=p1, ainfo=hiper)
    # Getting samples
    θ[:, :, i] = permutedims(gibbs[:, (p1.burn+1):end])
  end

  # Compute convergence test and save results
  conv_t = ess_rhat(θ; kind=:tail, tail_prob=0.05)
  write(joinpath(DATA_PATH, "DWJ_conv.csv"), conv_t)

  # Get lower and upper bounds
  θ_bounds = quantile(θ; q=[0.025, 0.975])

  ξ_bounds = zeros(p1.N, 2)
  @. ξ_bounds[:, 1] = sqrt(θ_bounds.nt[2])
  @. ξ_bounds[:, 2] = sqrt(θ_bounds.nt[3])

  # Volatility
  vol_df = DataFrame(ξ_bounds, ["lower", "upper"])
  vol_df[!, :t] .= [period[p1.m*i] for i in 1:p1.N]
  push!(vol_df, (ξ_bounds[end, :]..., last(period)))

  vol_path = joinpath(DATA_PATH, "DWJ_estimate.csv")
  write(vol_path, vol_df)

  #---- Using N=26 ----
  p2 = create_bparams(burn, k, T, n, 26)

  # Initialise chain
  θ = Chains(zeros(p2.k, p2.N, nc), map(Symbol, 1:p2.N))
  # Initialise Gibbs
  gibbs = zeros(p2.N, burn + k)
  for i in 1:nc
    # Gibbs sampler
    group_and_sample!(gibbs, diff_lrt; α₁=α₁, params=p2, ainfo=hiper)
    # Getting samples
    θ[:, :, i] = permutedims(gibbs[:, (p2.burn+1):end])
  end

  # Compute convergence test and save results
  conv_t = ess_rhat(θ; kind=:tail, tail_prob=0.05)
  write(joinpath(DATA_PATH, "DWJ_conv_double.csv"), conv_t)

  # Get lower and upper bounds
  θ_bounds = quantile(θ; q=[0.025, 0.975])

  ξ_bounds = zeros(p2.N, 2)
  @. ξ_bounds[:, 1] = sqrt(θ_bounds.nt[2])
  @. ξ_bounds[:, 2] = sqrt(θ_bounds.nt[3])

  # Period of DWJ data
  period = Date(1971, 7, 2):Week(1):Date(1974, 8, 2)

  # Volatility
  vol_df = DataFrame(ξ_bounds, ["lower", "upper"])
  vol_df[!, :t] .= [period[p2.m*i] for i in 1:p2.N]
  push!(vol_df, (ξ_bounds[end, :]..., last(period)))

  vol_path_double = joinpath(DATA_PATH, "DWJ_estimate_double.csv")
  write(vol_path_double, vol_df)

  return 0
end
