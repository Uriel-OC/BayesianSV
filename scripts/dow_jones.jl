#---- Setup ----
using Random: seed!; seed!(4)

using BNP

using Statistics: quantile

using Dates: Date, Week
using DataFrames: DataFrame
using CSV: write

#---- Variables ----
const N = 26 # Blocks number
const burn = 1_000 # Burn-in iterations
const k = 200_000 # Gibbs sampler iterations

# Model hyperparameters
const α₁ = 0.1
const α = α_ζ = 4.0

#---- Aux funtions ----
function get_returns(x::AbstractVector{T})::Vector{T} where {T<:AbstractFloat}
  vec1 = @view x[1:end-1]
  vec2 = @view x[2:end]

  return @. (vec2 - vec1) / vec1
end

#---- Main ----
function (@main)(args)
  # Data
  data_path = joinpath("data", "DWJ.txt") |> abspath
  prices = parse.(Float64, readlines(data_path))

  # Missing variables
  n = length(prices)
  m = fld(n, N) # Partition size
  r = n - N * m # Remainder
  T = n / 52 # Weeks -> Years

  # Get returns
  returns = get_returns(prices)
  returns = [0; returns]

  # Gibbs sampler
  gibbs = zeros(burn + k, N)

  sample!(gibbs, returns; burn, k, T, n, N, m, r, α₁, α, α_ζ)

  # Getting samples
  θ = zeros(k, N)
  θ .= gibbs[(burn+1):end, :]
  ξ_bounds = sqrt.(mapslices(x -> quantile(x, [0.025, 0.975]), θ; dims=1))

  t = LinRange(0, T, n)
  marks = map(x -> x * T / n, 0:m:n)

  lower = map(t -> estimate(t, ξ_bounds[1, :], marks), t)
  upper = map(t -> estimate(t, ξ_bounds[2, :], marks), t)

  dates = Date(1971, 7, 2):Week(1):Date(1974, 8, 2)

  path = joinpath("data", "DWJ_vol.csv") |> abspath
  df = DataFrame(date=dates)
  df[!, "price"] .= prices
  df[!, "return"] .= returns
  df[!, "lower"] .= lower
  df[!, "upper"] .= upper
  write(path, df)

  return 0
end
