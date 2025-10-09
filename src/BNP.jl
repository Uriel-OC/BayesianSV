module BNP

export sample!, estimate

using Distributions: InverseGamma

@inline rinvgamma(α, β) = rand(InverseGamma(α, β))

function IG_chain!(θ::AbstractVector{T}, ζ::AbstractVector{T}, α₁::T, α::T, α_ζ::T) where {T<:AbstractFloat}
  @inbounds θ[1] = rinvgamma(α₁, α₁)

  @inbounds for i in 2:lastindex(θ)
    ζ[i] = rinvgamma(α_ζ, α_ζ / θ[i-1])
    θ[i] = rinvgamma(α, α / ζ[i])
  end

  nothing
end

function sample!(gibbs::AbstractMatrix{R}, X::AbstractVector{R}; burn::S, k::S, T::R, n::S, N::S, m::S, r::S, α₁::R, α::R, α_ζ::R) where {R<:AbstractFloat,S<:Integer}
  ## Data transformation
  Y = diff(X)
  Z = map(x -> sum(x .^ 2), Iterators.partition(Y, m))

  ζ = zeros(R, N) # Preallocation

  ## Model
  IG_chain!(view(gibbs, 1, :), ζ, α₁, α, α_ζ)

  # Gibbs
  @inbounds for j in 2:(burn+k)
    gibbs[j, 1] = rinvgamma(α₁ + α_ζ + m / 2, α₁ + α_ζ / ζ[2] + n * Z[1] / 2T)
    for i in 2:N-1
      gibbs[j, i] = rinvgamma(α + α_ζ + m / 2, α / ζ[i] + α_ζ / ζ[i+1] + n * Z[i] / 2T)
      ζ[i] = rinvgamma(α_ζ + α, α_ζ / gibbs[j, i-1] + α / gibbs[j, i])
    end
    gibbs[j, N] = rinvgamma(α + (m + r) / 2, α / ζ[N] + n * Z[N] / 2T)
    ζ[N] = rinvgamma(α_ζ + α, α_ζ / gibbs[j, N-1] + α / gibbs[j, N])
  end

  nothing
end

function estimate(t::T,x::AbstractVector{T},marks::AbstractVector{T}) where {T<:AbstractFloat}
  index = searchsortedlast(marks,t)
  return index <= lastindex(x) ? x[index] : last(x)
end

end
