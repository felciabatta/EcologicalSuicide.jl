# SDEs of form du = f(u,p,t)dt + g(u,p,t)dW

using DifferentialEquations:
    EM, SDEProblem, SKenCarp, SOSRA, solve, vecvec_to_mat
using Distances: euclidean, pairwise
using NaNStatistics: nansum
using Plots

@userplot ParticlePlot
@recipe function f(pp::ParticlePlot)
    sol, idx = pp.args
    seriestype --> :scatter
    xlims --> (-10, 10)
    ylims --> (-10, 10)
    show --> true
    marker --> 12.5
    aspect_ratio --> :equal
    legend --> false
    size --> (540, 540)
    dpi --> 100.1
    sol[idx:idx, 1, :], sol[idx:idx, 2, :]
end

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
    f(u, p, t) = -u * potential
    g(u, p, t) = 1

    prob = SDEProblem(f, g, u₀, tspan)
    sol = solve(prob, EM(), dt=dt)

    usol = vecvec_to_mat(sol.u)
    p = plot(usol[:, 1], usol[:, 2], show=true)
    display(p)
    return usol
end

"""
    ∂ᵣwca(r, σ=1, ε=1)

TBW
"""
function ∂ᵣwca(r, σ=1, ε=1)
    q = σ ./ r
    return @. -24ε * (2q^12 - q^6) / r
end

"""
    ∇wca(x, σ=1, ε=1)

TBW
"""
function ∇wca(x, σ=1, ε=1)
    xij = x .- (xᵥ = permutedims(x, (3, 2, 1)))
    r = pairwise(euclidean, xᵥ[:, :], dims=1)
    n = size(r)[1]
    r = reshape(r, n, 1, n)

    x̂ij = xij ./ r
    # foreach(normalize!, eachrow(𝐫)) # NOT WORK FOR 3D apparently
    return nansum(∂ᵣwca(r, σ, ε) .* x̂ij, dims=1)
end

"""
    example4(
    n=2;
    tspan=(0.0, 1.0),
    x₀=10 * 2(rand(1, 2, n) .- 0.5),
    dt=0.01,
    animate=false,
    potential=false,
    solver=EM(),
)

Now we have multiple particles.
f: deterministic; global potential + interactions
g: stochastic; brownian motion

x₀: particles in dims=3; x,y in dims=2
"""
function example4(
    n=2;
    tspan=(0.0, 1.0),
    x₀=10 * 2(rand(1, 2, n) .- 0.5),
    dt=0.01,
    animate=false,
    potential=false,
    solver=SOSRA(),
)

    μ, kᵦ, T = 1, 1 / 2, 1
    γ = √(2μ * kᵦ * T)

    # ODE
    f(x, p, t) = (-∇wca(x) - x * potential) / μ
    g(x, p, t) = γ / μ

    prob = SDEProblem(f, g, x₀, tspan)
    sol = solve(prob, solver, dt=dt)
    println("SDE Solved!")
    usol = reduce(vcat, sol.u)
    nsol, = size(usol)
    if animate
        anim = @animate for i ∈ 1:nsol
            particleplot(usol, i)
        end every 40
        display(anim)
        mov(anim, "particle.mov", fps=30)
        return usol, anim
    else
        p = particleplot(usol, 1:size(usol)[1])
        display(p)
        return usol, p
    end
end
