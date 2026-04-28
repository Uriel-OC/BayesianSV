#---- Setup ----
using Random: seed!; seed!(22)

using Distributions: InverseGamma

using BayesianSV

using MCMCChains

using Statistics: quantile

using CSV
using DataFrames, DataFramesMeta
using Dates

#---- Constants ----
const nc = 4 # Number of chains
const burn = 1_000 # Burn-in iterations
const k = 50_000 # Gibbs sampler iterations

const T = 5.0 # The scripts uses data of 5 years

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
  file = joinpath(DATA_PATH, "DJA.csv")
  in_df = CSV.read(file, DataFrame)

  # Create returns DataFrame
  returns_df = DataFrame(date=in_df.date)

  # Compute log returns
  tmp = get_log_returns(in_df.price)

  # Compute difference
  diff_lrt = diff(tmp)
  pushfirst!(diff_lrt, 0)

  returns_df[!, :logreturn] .= tmp
  returns_df[!, :diff] .= diff_lrt

  # Save returns_df
  returns_path = joinpath(DATA_PATH, "DJA_returns.csv")
  CSV.write(returns_path, returns_df)

  # Missing variables
  n = nrow(in_df)

  #---- Using quarterly log-returns ----
  p1 = create_params(burn, k, T, n, 20)

  # Get quarterly returns sum
  group_df = @chain returns_df begin
    @transform(:qt = floor.(:date, Dates.Month(3)))
    @transform(:ym = yearmonth.(:qt))
    @groupby(:ym)
    @combine(:z = sum(:diff .^ 2), :m = length(:logreturn))
    @select(:z, :m)
  end
  Z = group_df.z
  m = group_df.m

  # Initialise chain
  θ = Chains(zeros(p1.k, p1.N, nc), map(Symbol, 1:p1.N))
  # Initialise Gibbs
  gibbs = zeros(p1.N, burn + k)
  for i in 1:nc
    # Gibbs sampler
    sample!(gibbs, Z, m; α₁=α₁, params=p1, ainfo=hiper)
    # Getting samples
    θ[:, :, i] = permutedims(gibbs[:, (p1.burn+1):end])
  end

  # Compute convergence test and save results
  conv_t = ess_rhat(θ; kind=:tail, tail_prob=0.05)
  CSV.write(joinpath(DATA_PATH, "DJA_qt_conv.csv"), conv_t)

  # Get lower and upper bounds
  θ_bounds = quantile(θ; q=[0.025, 0.975])

  ξ_bounds = zeros(p1.N, 2)
  @. ξ_bounds[:, 1] = sqrt(θ_bounds.nt[2])
  @. ξ_bounds[:, 2] = sqrt(θ_bounds.nt[3])

  # Writing estimation DataFrame
  qt_out_df = DataFrame(ξ_bounds, ["lower", "upper"])
  qt_out_df[!, :date] .= Date(2019):Month(3):Date(2023, 12)
  push!(qt_out_df, (ξ_bounds[end, :]..., last(in_df.date)))

  qt_vol_path = joinpath(DATA_PATH, "DJA_qt_vol.csv")
  CSV.write(qt_vol_path, qt_out_df)

  #---- Using monthly log-returns
  p2 = create_params(burn, k, T, n, 60)

  # Get monthly returns sum
  group_df = @chain returns_df begin
    @transform(:ym = yearmonth.(:date))
    @groupby(:ym)
    @combine(:z = sum(:diff .^ 2), :m = length(:logreturn))
    @select(:z, :m)
  end
  Z = group_df.z
  m = group_df.m

  # Initialise chain
  θ = Chains(zeros(p2.k, p2.N, nc), map(Symbol, 1:p2.N))
  # Initialise Gibbs
  gibbs = zeros(p2.N, burn + k)
  for i in 1:nc
    # Gibbs sampler
    sample!(gibbs, Z, m; α₁=α₁, params=p2, ainfo=hiper)
    # Getting samples
    θ[:, :, i] = permutedims(gibbs[:, (p2.burn+1):end])
  end

  # Compute convergence test and save results
  conv_t = ess_rhat(θ; kind=:tail, tail_prob=0.05)
  CSV.write(joinpath(DATA_PATH, "DJA_month_conv.csv"), conv_t)

  # Get lower and upper bounds
  θ_bounds = quantile(θ; q=[0.025, 0.975])

  ξ_bounds = zeros(p2.N, 2)
  @. ξ_bounds[:, 1] = sqrt(θ_bounds.nt[2])
  @. ξ_bounds[:, 2] = sqrt(θ_bounds.nt[3])

  # Writing estimation DataFrame
  month_out_df = DataFrame(ξ_bounds, ["lower", "upper"])
  month_out_df[!, :date] .= Date(2019):Month(1):Date(2023, 12)
  push!(month_out_df, (ξ_bounds[end, :]..., last(in_df.date)))

  month_vol_path = joinpath(DATA_PATH, "DJA_month_vol.csv")
  CSV.write(month_vol_path, month_out_df)

  return 0
end
