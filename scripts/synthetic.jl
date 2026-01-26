#---- Setup ----
using Random: seed!; seed!(4);

using StochasticDiffEq
using BNP

using Statistics: quantile

using DataFrames: DataFrame
using CSV: write

#---- Constants ----
const params = create_bparams(
  1_000, # Burn-in iterations
  200_000, # Gibbs sampler iterations
  1.0, # Time span
  4_000, # Number of samples
  160, # Number of blocks
)

# Model hyperparameters
const hparams = create_const_hparams(0.1, 20.0) # α₁ = 0.1; α = α_ζ = 20.0

#---- Aux functions ----
b(u, p, t) = -10 * u + 20

const t_j = (0.1, 0.13, 0.15, 0.23, 0.25, 0.4, 0.44, 0.65, 0.76, 0.78, 0.81)
const h_j = (4.0, -5.0, 3.0, -4.0, 5.0, -4.2, 2.1, 4.3, -3.1, 2.1, -4.2)

@inline K(x) = (1 + sign(x)) / 2

s(u, p, t) = 10 + 3.655606 * sum(@. h_j * K(t - t_j))
s(t) = s(0, 0, t)

#---- Main ----
function (@main)(args)
  # SDE simulation
  dt = 1 // (params.n - 1)
  tspan = (0.0, params.T)

  prob = SDEProblem(b, s, 0, tspan)
  sol = solve(prob, EM(), dt=dt)

  # Gibbs sampler
  gibbs = zeros(params.N, params.burn + params.k)
  group_and_sample!(gibbs, sol.u; params, hparams)

  # Getting samples
  θ = zeros(params.k, params.N)
  permutedims!(θ, gibbs[:, (params.burn+1):end], (2, 1))

  # Getting ξ lower and upper bounds
  ξ_bounds = sqrt.(mapslices(x -> quantile(x, [0.025, 0.975]), θ; dims=1))

  #-- Writing --
  # Path
  path_path = joinpath("data", "synthetic_path.csv") |> abspath
  path_df = DataFrame(
    t=sol.t,
    X=sol.u,
    s=s.(sol.t)
  )
  write(path_path, path_df)
  # Volatility
  t = LinRange(0, params.T, params.N + 1) |> collect
  pop!(t)

  vol_path = joinpath("data", "synthetic_vol.csv") |> abspath
  vol_df = DataFrame(
    t=t,
    lower=ξ_bounds[1, :],
    upper=ξ_bounds[2, :]
  )
  push!(vol_df, (1.0, ξ_bounds[1, end], ξ_bounds[2, end]))

  write(vol_path, vol_df)

  return 0
end
