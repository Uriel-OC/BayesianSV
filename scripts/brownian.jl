#---- Setup ----
using Random
Random.seed!(10)

using Distributions: Normal
using DataFrames, CSV

#---- Main ----
# using CairoMakie
# f = Figure()
# ax = Axis(f[1, 1], title="Movimiento browniano", xlabel=L"t")
# lines!(ax, x, brownian, color=:blue)

function main()
  Δt = 0.01
  x = collect(0:Δt:3)
  N = length(x)
  steps = rand(Normal(), N - 1)

  brownian = [0; cumsum(steps) * sqrt(Δt)]

  path = joinpath("data", "brownian.csv") |> abspath
  df = DataFrame("x" => x, "y" => brownian)
  CSV.write(path, df)
end

if abspath(PROGRAM_FILE) == @__FILE__
  main()
end
