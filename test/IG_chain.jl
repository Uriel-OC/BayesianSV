using Distributions: InverseGamma

const N = 1_000_000

function IG_chain!(θ::Vector{T}, α::T, α_ζ::T) where {T<:AbstractFloat}
  ζ = similar(θ)

  @inbounds θ[1] = 500.0

  for i in 2:N
    @inbounds ζ[i] = rand(InverseGamma(α_ζ, α_ζ / θ[i-1]))
    @inbounds θ[i] = rand(InverseGamma(α, α / ζ[i]))
  end

  nothing
end

using CairoMakie

θ = zeros(N);

f = Figure();
ax = Axis(f[1, 1], title="Inverse-Gamma chain", yscale=log10)

for _ in 1:6
  IG_chain!(θ, 30.0, 30.0)
  lines!(ax, 1:N, θ)
end
