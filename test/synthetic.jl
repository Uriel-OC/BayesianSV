using DifferentialEquations

b(u, p, t) = -10 * u + 20

t_j = [0.1, 0.13, 0.15, 0.23, 0.25, 0.4, 0.44, 0.65, 0.76, 0.78, 0.81]
h_j = [4.0, -5.0, 3.0, -4.0, 5.0, -4.2, 2.1, 4.3, -3.1, 2.1, -4.2]

K(x) = (1 + sign(x)) / 2

s(u, p, t) = 10 + 3.655606 * sum(@. h_j * K(t - t_j))
s(t) = s(0,0,t)

dt = 1//1000
tspan = (0.0, 1.0)

prob = SDEProblem(b, s, 0, tspan)

sol = solve(prob, EM(), dt=dt)

using CairoMakie

f = Figure();
ax = Axis(f[1,1], title="Test", xlabel="x", ylabel="y")
lines!(ax, sol.t,sol.u)
lines!(ax, sol.t, s.(sol.t))
