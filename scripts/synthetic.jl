#---- Setup ----
using Random: seed!; seed!(4)

using StochasticDiffEq
using BNP

using Statistics: quantile

using DataFrames: DataFrame
using CSV: write

#---- Constants ----
const n = 4_000 # Number of samples
const T = 1.0 # Time span
const N = 160 # Number of blocks
const m = fld(n, N) # Partition size
const r = n - N * m # Remainder
const burn = 1_000 # Burn-in iterations
const k = 200_000 # Gibbs sampler iterations

# Model hyperparameters
const α₁ = 0.1
const α = α_ζ = 20.0

#---- Aux functions ----
b(u, p, t) = -10 * u + 20

const t_j = [0.1, 0.13, 0.15, 0.23, 0.25, 0.4, 0.44, 0.65, 0.76, 0.78, 0.81]
const h_j = [4.0, -5.0, 3.0, -4.0, 5.0, -4.2, 2.1, 4.3, -3.1, 2.1, -4.2]

@inline K(x) = (1 + sign(x)) / 2

s(u, p, t) = 10 + 3.655606 * sum(@. h_j * K(t - t_j))
s(t) = s(0,0,t)

#---- Main ----
function (@main)(args)
  # SDE simulation
  dt = 1 // (n - 1)
  tspan = (0.0, T)

  prob = SDEProblem(b, s, 0, tspan)
  sol = solve(prob, EM(), dt=dt)

  # Gibbs sampler
  gibbs = zeros(burn + k, N)
  sample!(gibbs, sol.u; burn, k, T, n, N, m, r, α₁, α, α_ζ)

  # Getting samples
  θ = zeros(k, N)
  θ .= gibbs[(burn+1):end, :]
  ξ_bounds = sqrt.(mapslices(x -> quantile(x, [0.025, 0.975]), θ; dims=1))

  marks = map(x -> x * T / n, 0:m:n)

  lower = map(t -> estimate(t, ξ_bounds[1, :], marks), sol.t)
  upper = map(t -> estimate(t, ξ_bounds[2, :], marks), sol.t)

  # Output
  path = joinpath("data", "synthetic.csv") |> abspath
  df = DataFrame(
    t = sol.t,
    X = sol.u,
    s = s.(sol.t),
    lower=lower,
    upper=upper
  )
  write(path, df)

  return 0
end
