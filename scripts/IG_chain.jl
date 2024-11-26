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

θ₁ = zeros(N);
θ₂ = zeros(N);
θ₃ = zeros(N);
θ₄ = zeros(N);
θ₅ = zeros(N);
θ₆ = zeros(N);

IG_chain!(θ₁, 30.0, 30.0)
IG_chain!(θ₂, 30.0, 30.0)
IG_chain!(θ₃, 30.0, 30.0)
IG_chain!(θ₄, 30.0, 30.0)
IG_chain!(θ₅, 30.0, 30.0)
IG_chain!(θ₆, 30.0, 30.0)

f = Figure();
ax = Axis(f[1, 1], title="Inverse-Gamma chain", yscale=log10)
lines!(ax, 1:N, θ₁)
lines!(ax, 1:N, θ₂)
lines!(ax, 1:N, θ₃)
lines!(ax, 1:N, θ₄)
lines!(ax, 1:N, θ₅)
lines!(ax, 1:N, θ₆)
