using DifferentialEquations
using Plots

# SDEs of form du = f(u,p,t)dt + g(u,p,t)dW

"""
    example1(dt=0.01)

SDE of form du = dW.
This is is the simplest SDE possible,
just pure stochastic Brownian motion.
"""
function example1(dt=0.01)
    # parameters
    u₀ = 0
    tspan = (0.0, 1.0)

    # ODE
    f(u, p, t) = 0
    g(u, p, t) = 1

    prob = SDEProblem(f, g, u₀, tspan)
    sol = solve(prob, EM(), dt=dt)

    plot(sol)
end
