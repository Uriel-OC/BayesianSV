module BayesianSV

export ARWMInfo, DATA_PATH, create_params, create_bparams, group_and_sample!, sample!

const PROJECT_PATH = Base.current_project() |> dirname
const DATA_PATH = joinpath(PROJECT_PATH, "data")

using Distributions

#---- Parameters creator helper functions ----
"Alias type that represents all block-related parameters assuming same length intervals"
const BlockParams = @NamedTuple begin
  burn::Int
  k::Int
  T::Float64
  n::Int
  N::Int
  m::Int
  r::Int
end

"'Constructor' of `BlockParams` type"
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

"Alias type that represents all block-related parameters assuming different interval sizes"
const Params = @NamedTuple begin
  burn::Int
  k::Int
  T::Float64
  n::Int
  N::Int
end

"'Constructor' of `Params` type"
function create_params(burn, k, T, n, N)::Params
  return (
    burn=burn,
    k=k,
    T=T,
    n=n,
    N=N,
  )
end

"Struct keeping relevant information about for the adaptive random walk Metropolis algorithm"
struct ARWMInfo{D<:ContinuousUnivariateDistribution,T<:AbstractFloat}
  Q::D
  σ₀::T
  ε::T
  A::T
  τ::T
  ARWMInfo{D,T}(Q::D, σ₀::T, ε::T, A::T, τ::T) where {D,T} = new{D,T}(Q, σ₀, ε, A, τ)
end

function ARWMInfo(Q::D, σ₀::T; ε::T=1e-4, A::T=1e3, τ::T=0.234) where {D,T<:AbstractFloat}
  @assert minimum(Q) >= zero(T) "The target distribution must be positive valued"

  return ARWMInfo{D,T}(Q, σ₀, ε, A, τ)
end

ARWMInfo(Q::D, σ₀::Integer) where {D} = ARWMInfo(Q::D, float(σ₀))

#---- Auxiliary functions ----
function params_iterator(θ::AbstractVector{T}, ζ::AbstractVector{T}) where {T}
  i = lastindex(θ)

  return zip(view(θ, 1:i-1), view(θ, 2:i), view(ζ, 2:i))
end

#---- Adaptive random walk Metropolis functions ----
rnorm(μ, σ) = rand(Normal(μ, σ))
runif() = rand(Uniform())

import SpecialFunctions: gamma

function update_hyperparam(x::T, σ::T, θ::AbstractVector{T}, ζ::AbstractVector{T}; info::ARWMInfo{D,T}, i::Integer, N::Integer) where {D,T}
  # Constants
  c1 = sum(w -> inv(w[3]) * (inv(w[1]) + inv(w[2])), params_iterator(θ, ζ))
  c2 = prod(w -> w[1] * w[2] * w[3]^2, params_iterator(θ, ζ))

  # Hyperparameter unnormalised full conditional density
  f(u::BigFloat) = pdf(info.Q, u) * (u^u / gamma(u))^(2N - 2) * exp(-u * c1) * c2^-u
  f(u::T) = f(BigFloat(u))

  y = zero(T)
  while y <= zero(T)
    y = rnorm(x, σ) # Get sample
  end

  accept::T = min(f(y) / f(x), one(T)) # Calculate acceptance

  # Update σ
  new_σ = clamp(σ + (accept - info.τ)info.σ₀ / i, info.ε, info.A)

  # Throw sample
  if runif() <= accept
    return y, new_σ
  else
    return x, new_σ
  end
end

#---- Model functions ----
rinvgamma(α, β) = rand(InverseGamma(α, β))

function IG_chain!(θ::AbstractVector{T}, ζ::AbstractVector{T}, α₁::T, α::T) where {T}
  @assert length(θ) == length(ζ) "The parameter vectors don't have the same length"

  @inbounds θ[1] = rinvgamma(α₁, α₁)

  @inbounds for i in 2:lastindex(θ)
    ζ[i] = rinvgamma(α, α / θ[i-1])
    θ[i] = rinvgamma(α, α / ζ[i])
  end

  nothing
end

"""

Function that populates the matrix `gibbs` with a full run of the Gibbs sampler
used to estimate the stochastic volatility

The keyword arguments needed are:
- α₁     | The model α₁ hyperparameter
- params | The block related parameters
- ainfo  | The adaptive random walk Metropolis parameters used to update the
           hyperparameter α

It assumes all ``Bₖ``'s, ``k < N``, have the same length and the sum of the
squared differences was already computed and are stored in `Z`

"""
function sample!(gibbs::AbstractMatrix{R}, Z::AbstractVector{R}; α₁::R, params::BlockParams, ainfo::ARWMInfo{D,R}) where {D,R}
  @assert size(gibbs)[1] == length(Z) "The matrix have more rows than expected"

  (; burn, k, T, n, N, m, r) = params

  ζ = zeros(R, N) # Preallocation

  ## Model
  σ = ainfo.σ₀
  α = rand(ainfo.Q)
  IG_chain!(view(gibbs, :, 1), ζ, α₁, α)

  # Gibbs
  @inbounds for j in 2:(burn+k)
    α, σ = update_hyperparam(α, σ, view(gibbs, :, j - 1), ζ; info=ainfo, i=j, N=N)

    gibbs[1, j] = rinvgamma(α₁ + α + m / 2, α₁ + α / ζ[2] + n * Z[1] / 2T)
    for i in 2:N-1
      gibbs[i, j] = rinvgamma(2α + m / 2, α / ζ[i] + α / ζ[i+1] + n * Z[i] / 2T)
      ζ[i] = rinvgamma(2α, α / gibbs[i-1, j] + α / gibbs[i, j])
    end
    gibbs[N, j] = rinvgamma(α + (m + r) / 2, α / ζ[N] + n * Z[N] / 2T)
    ζ[N] = rinvgamma(2α, α / gibbs[N-1, j] + α / gibbs[N, j])
  end

  nothing
end

"""

Function that compute the squared differences of `X`, then sum them accordingly
and finally populates the matrix `gibbs` with a run of the Gibbs sampler used
to estimate the stochastic volatility

The keyword arguments needed are:
- α₁     | The model α₁ hyperparameter
- params | The block-related parameters
- ainfo  | The adaptive random walk Metropolis parameters used to update the
           hyperparameter α

It assumes all ``Bₖ``'s, ``k < N``, have the same length

"""
function group_and_sample!(gibbs::AbstractMatrix{R}, X::AbstractVector{R}; α₁::R, params::BlockParams, ainfo::ARWMInfo{D,R}) where {D,R}
  ## Data transformation
  Y = diff(X)
  pushfirst!(Y, 0)

  Z = map(x -> sum(x .^ 2), Iterators.partition(Y, params.m))
  while length(Z) > params.N
    Z[end-1] += pop!(Z)
  end

  sample!(gibbs, Z; α₁, params, ainfo)

  nothing
end

"""

Function that populates the matrix `gibbs` with a full run of the Gibbs sampler
used to estimate the stochastic volatility

The keyword arguments needed are:
- α₁     | The model α₁ hyperparameter
- params | The block-related parameters
- ainfo  | The adaptive random walk Metropolis parameters used to update the
           hyperparameter α

It assumes: all ``Bₖ``'s, ``k ≤ N``, may have differents lengths; the sum of the
squared differences have to be already computed and are stored in `Z`; the
number of how many differences fall in each ``Bₖ`` are stored in `m`

"""
function sample!(gibbs::AbstractMatrix{R}, Z::AbstractVector{R}, m::AbstractVector{<:Integer}; α₁::R, params::Params, ainfo::ARWMInfo{D,R}) where {D,R}
  @assert size(gibbs)[1] == length(Z) "The matrix have more rows than expected"

  (; burn, k, T, n, N) = params

  ζ = zeros(R, N) # Preallocation

  ## Model
  σ = ainfo.σ₀
  α = rand(ainfo.Q)
  IG_chain!(view(gibbs, :, 1), ζ, α₁, α)

  # Gibbs
  @inbounds for j in 2:(burn+k)
    α, σ = update_hyperparam(α, σ, view(gibbs, :, j - 1), ζ; info=ainfo, i=j, N=N)

    gibbs[1, j] = rinvgamma(α₁ + α + m[1] / 2, α₁ + α / ζ[2] + n * Z[1] / 2T)
    for i in 2:(N-1)
      gibbs[i, j] = rinvgamma(2α + m[i] / 2, α / ζ[i] + α / ζ[i+1] + n * Z[i] / 2T)
      ζ[i] = rinvgamma(2α, α / gibbs[i-1, j] + α / gibbs[i, j])
    end
    gibbs[N, j] = rinvgamma(α + m[N] / 2, α / ζ[N] + n * Z[N] / 2T)
    ζ[N] = rinvgamma(2α, α / gibbs[N-1, j] + α / gibbs[N, j])
  end

  nothing
end

end
