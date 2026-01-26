module BNP

export create_params, create_bparams, create_const_hparams, create_hparams, group_and_sample!, sample!

#---- Parameters creator helper functions ----
import Distributions: ContinuousUnivariateDistribution

BlockParams = @NamedTuple begin
    burn::Integer
    k::Integer
    T::Float64
    n::Integer
    N::Integer
    m::Integer
    r::Integer
end

function create_bparams(burn, k, T, n, N)::BlockParams
  m = fld(n, N)
  return (
    burn=burn,
    k=k,
    T=T,
    n=n,
    N=N,
    m=m,
    r=n - N * m
  )
end

Params = @NamedTuple begin
    burn::Integer
    k::Integer
    T::Float64
    n::Integer
    N::Integer
end

function create_params(burn, k, T, n, N)::Params
  return (
    burn=burn,
    k=k,
    T=T,
    n=n,
    N=N,
  )
end

ConstHyperParams = @NamedTuple{α_1::Float64, α::Float64, α_ζ::Float64}

function create_const_hparams(α_1::T, α::T, α_ζ::T)::ConstHyperParams where {T<:Real}
  return (α_1=α_1, α=α, α_ζ=α_ζ)
end
create_const_hparams(α_1::Real, α::Real, α_ζ::Real) = create_const_hparams(promote(α_1, α, α_ζ)...)
create_const_hparams(α_1::Integer, α::Integer, α_ζ::Integer) = create_const_hparams(float(α_1), float(α), float(α_ζ))
create_const_hparams(α_1, α) = create_const_hparams(α_1, α, α)

HyperParams = @NamedTuple{α_1::Float64, α::ContinuousUnivariateDistribution, α_ζ::ContinuousUnivariateDistribution}
function create_hparams(α_1, α::T, α_ζ::T) where {T<:ContinuousUnivariateDistribution}
  return (α_1=α_1, α=α, α_ζ=α_ζ)
end
create_hyperparams(α_1, α) = create_hyperparams(α_1, α, α)

#---- Model functions ----
import Distributions: InverseGamma

@inline rinvgamma(α, β) = rand(InverseGamma(α, β))

function IG_chain!(θ::AbstractVector{T}, ζ::AbstractVector{T}, α₁::T, α::T, α_ζ::T) where {T<:AbstractFloat}
  @assert length(θ) == length(ζ) "The vectors don't have the same length"

  @inbounds θ[1] = rinvgamma(α₁, α₁)

  @inbounds for i in 2:lastindex(θ)
    ζ[i] = rinvgamma(α_ζ, α_ζ / θ[i-1])
    θ[i] = rinvgamma(α, α / ζ[i])
  end

  nothing
end

function sample!(gibbs::AbstractMatrix{R}, Z::AbstractVector{R}; params::BlockParams, hparams::ConstHyperParams) where {R<:AbstractFloat}
  @assert size(gibbs)[1] == length(Z) "The matrix have more rows than expected"

  burn, k, T, n, N, m, r = params
  α₁, α, α_ζ = hparams

  ζ = zeros(R, N) # Preallocation

  ## Model
  IG_chain!(view(gibbs, :, 1), ζ, α₁, α, α_ζ)

  # Gibbs
  @inbounds for j in 2:(burn+k)
    gibbs[1, j] = rinvgamma(α₁ + α_ζ + m / 2, α₁ + α_ζ / ζ[2] + n * Z[1] / 2T)
    for i in 2:N-1
      gibbs[i, j] = rinvgamma(α + α_ζ + m / 2, α / ζ[i] + α_ζ / ζ[i+1] + n * Z[i] / 2T)
      ζ[i] = rinvgamma(α_ζ + α, α_ζ / gibbs[i-1, j] + α / gibbs[i, j])
    end
    gibbs[N, j] = rinvgamma(α + (m + r) / 2, α / ζ[N] + n * Z[N] / 2T)
    ζ[N] = rinvgamma(α_ζ + α, α_ζ / gibbs[N-1, j] + α / gibbs[N, j])
  end

  nothing
end

function group_and_sample!(gibbs::AbstractMatrix{R}, X::AbstractVector{R}; params::BlockParams, hparams::ConstHyperParams) where {R<:AbstractFloat}
  ## Data transformation
  Y = diff(X)
  Z = map(x -> sum(x .^ 2), Iterators.partition(Y, params.m))

  sample!(gibbs, Z; params, hparams)

  nothing
end

function sample!(gibbs::AbstractMatrix{R}, Z::AbstractVector{R}, m::AbstractVector{R}; params::Params, hparams::ConstHyperParams) where {R<:AbstractFloat}
  @assert size(gibbs)[1] == length(Z) "The matrix have more rows than expected"

  burn, k, T, n, N = params
  α₁, α, α_ζ = hparams

  ζ = zeros(R, N) # Preallocation

  ## Model
  IG_chain!(view(gibbs, :, 1), ζ, α₁, α, α_ζ)

  # Gibbs
  @inbounds for j in 2:(burn+k)
    gibbs[1,j] = rinvgamma(α₁ + α_ζ + m[1] / 2, α₁ + α_ζ / ζ[2] + n * Z[1] / 2T)
    for i in 2:N
      gibbs[i,j] = rinvgamma(α + α_ζ + m[i] / 2, α / ζ[i] + α_ζ / ζ[i+1] + n * Z[i] / 2T)
      ζ[i] = rinvgamma(α_ζ + α, α_ζ / gibbs[i-1,j] + α / gibbs[i,j])
    end
  end

  nothing
end

end
