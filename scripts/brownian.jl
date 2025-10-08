#---- Setup ----
using Random
Random.seed!(3)

using Distributions: Normal
using DataFrames: DataFrame
using CSV: write

#---- Constants ----
const Δt = 0.01
const σ = 1
const r = 0.1

#---- Main ----
function (@main)(args)
  t = 0:Δt:3
  N = length(t)
  steps = rand(Normal(), N - 1)

  # Brownian
  W = [0; cumsum(steps) * sqrt(Δt)]
  # Exp brownian
  S = similar(W)
  @. S = exp((r - σ^2 / 2) * t + σ * W)

  path = joinpath("data", "brownian.csv") |> abspath
  df = DataFrame([t W S], ["t", "W", "S"])
  write(path, df)

  return 0
end
