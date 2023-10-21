using DifferentialEquations: EM, SDEProblem, solve, vecvec_to_mat
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


"""
    example2(dt=0.01)

This one has potential 🤣.
"""
function example2(dt=0.01)
    # parameters
    u₀ = 0
    tspan = (0.0, 1.0)

    # ODE
    f(u, p, t) = -u
    g(u, p, t) = 1

    prob = SDEProblem(f, g, u₀, tspan)
    sol = solve(prob, EM(), dt=dt)

    plot(sol)
end

"""
Now we are in 2D.
"""
function example3(dt=0.01, potential=true)
    # parameters
    u₀ = [0; 0]
    tspan = (0.0, 1.0)

    # ODE
    f(u, p, t) = -u*potential
    g(u, p, t) = 1

    prob = SDEProblem(f, g, u₀, tspan)
    sol = solve(prob, EM(), dt=dt)

    usol = vecvec_to_mat(sol.u)
    p = plot(usol[:,1], usol[:,2], show=true)
    display(p)
    return usol
end
