#---- Setup ----
using Random: seed!; seed!(6)

using Distributions: InverseGamma
using DataFrames: DataFrame
using CSV: write

#---- Constants ----
const N = 400

#---- Aux functions ----
@inline rinvgamma(α, β) = rand(InverseGamma(α, β))

function sample_IG_chain!(θ::AbstractVector{T}, α::T, α_ζ::T) where {T<:AbstractFloat}
  # Prealocate output
  ζ = similar(θ)

  # Set constants
  @inbounds θ[1] = 500.0

  # Compute chain
  @inbounds for i in 2:lastindex(θ)
    ζ[i] = rinvgamma(α_ζ, α_ζ / θ[i-1])
    θ[i] = rinvgamma(α, α / ζ[i])
  end

  nothing
end

#---- Main ----
function (@main)(args)
  θ = zeros(N)
  df = DataFrame(n=1:N)
  α_arr = Float64[20, 30, 40]

  path = joinpath("data", "IG.csv") |> abspath

  for (α, β) in zip(α_arr, reverse(α_arr))
    for i in 1:5
      sample_IG_chain!(θ, α, β)
      df[!, "$α-$β-$i"] .= θ
    end
  end
  write(path, df)

  return 0
end

