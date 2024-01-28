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
    f(u, p, t) = -u * potential
    g(u, p, t) = 1

    prob = SDEProblem(f, g, u₀, tspan)
    sol = solve(prob, EM(), dt=dt)

    usol = vecvec_to_mat(sol.u)
    p = plot(usol[:, 1], usol[:, 2], show=true)
    display(p)
    return usol
end

cat3(x...) = cat(x..., dims=3)


"""
    example4(
    n=2;
    tspan=(0.0, 1.0),
    width=20,
    dt=0.01,
    animate=false,
    potential=false,
    solver=SOSRA(),
)

Now we have multiple particles.

# Arguments
- `f`: deterministic; global potential + interactions
- `g`: stochastic; brownian motion
- `x₀`: particles in dims=3; x, y in dims=2

"""
function example4(
    K=20;
    tspan=(0.0, 1.0),
    width=20,
    dt=0.01,
    animate=false,
    potential=false,
    solver=SOSRA(),
)

    # rand starting pos
    u₀ = width .* (rand(1, 2, K) .- 0.5)

    # ODE # ∇C_simple(u, -5, 5, -10, 10)
    f(u, p, t) = (-∇wca(u) - u .* potential)
    g(u, p, t) = 1

    prob = SDEProblem(f, g, u₀, tspan)
    sol = solve(prob, solver, dt=dt)
    println("SDE Solved!")
    usol = reduce(vcat, sol.u)

    x = y = -width/2:1:width/2
    V(x, y) = (x^2 + y^2) / 2
    z = @. V(x', y)
    if animate
        particle_animate(usol, x, y, z; save=true, width=width)
    else
        p = particleplot(usol, :; boxwidth=width)
        potentialplot!(p, x, y, z)
        display(p)
        return usol, p
    end
end


"""
    example5(
    K::Integer=10;
    tspan=(0.0, 1.0),
    width::Real=20,
    N::Integer=50,
    dt::Real=0.01,
    savedt::Real=5dt,
    visuals::Union{Symbol, Nothing}=nothing,
    solver::AbstractSDEAlgorithm=SOSRA(),
    Dp=1, Dc=1, μ=1, γ=1,
)

Add diffusion with particle sources.

# Arguments
- `f`: deterministic; global potential + interactions
- `g`: stochastic; brownian motion
- `x₀`: particles in dims=3; x, y in dims=2

"""
function example5(
    K::Integer=10;
    tspan=(0.0, 1.0),
    width::Real=20,
    N::Integer=50,
    dt::Real=0.01,
    savedt::Real=5dt,
    visuals::Union{Symbol, Nothing}=nothing,
    solver::AbstractSDEAlgorithm=SOSRA(),
    Dp=1, Dc=1, μ=1, γ=1,
)

    # INITIAL CONDITION
    p₀ = width .* (rand(1, 2, K) .- 0.5)
    c₀ = zeros(N, N)
    u₀ = [p₀[:]; c₀[:]]

    # ODE
    f, g, xgrid = active_brownian_system(K, width, N, Dp=Dp, Dc=Dc, μ=μ, γ=γ)

    prob = SDEProblem(f, g, u₀, tspan)
    sol = solve(prob, solver, dt=dt, saveat=savedt, dense=false)
    println("SDE Solved!")
    usol = reduce(cat3, sol.u)
    psol = reshape(usol[1:K*2, :, :], 2, K, :)
    psol = range_mod.(psol, -width / 2, width / 2)
    csol = reshape(usol[K*2+1:end, :, :], N, N, :)

    z = permutedims(csol, (2, 1, 3))
    if visuals == :animate
        particle_animate(psol, xgrid, xgrid, z; Δf=1, save=true, width=width)
        return psol, csol, sol
    elseif visuals == :plot
        p = particleplot(psol, :; boxwidth=width)
        potentialplot!(p, x, y, z)
        display(p)
        return psol, csol, sol, p
    end
end
