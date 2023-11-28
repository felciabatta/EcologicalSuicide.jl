# SDEs of form du = f(u,p,t)dt + g(u,p,t)dW

using DifferentialEquations:
    EM, SDEProblem, SKenCarp, SOSRA, solve, vecvec_to_mat
using Distances: euclidean, pairwise
using NaNStatistics: nansum
using Plots

@userplot ParticlePlot
@recipe function f(pp::ParticlePlot; boxwidth=20)
    sol, idx = pp.args
    # idx = (idx isa Int) ? (idx:idx) : idx
    seriestype --> :scatter
    xlims --> (-1, 1) .* boxwidth ./ 2
    ylims --> (-1, 1) .* boxwidth ./ 2
    show --> true
    marker --> 280 / boxwidth
    markeralpha --> 0.5
    markercolor --> :black
    markerstrokewidth --> 0
    # margin --> (1, :mm)
    aspect_ratio --> :equal
    legend --> false
    size --> (540, 540)
    dpi --> 100.1
    sol[[1], :, idx], sol[[2], :, idx]
end

@userplot PotentialPlot
@recipe function f(pp::PotentialPlot)
    x, y, z, idx = pp.args
    seriestype --> :contourf
    color --> :RdPu
    levels --> 20
    # cbar --> false
    lw --> 0.2
    alpha --> 0.9
    z_order --> :back
    x, y, z[:, :, idx]
end

function particle_animate(
    usol,
    x=0,
    y=0,
    z=0;
    Δf=40,
    fps=30,
    save=false,
    width=20,
)
    nsol = size(usol)[3]
    anim = @animate for i ∈ 1:nsol
        if !any((x, y, z) .== false)
            potentialplot(x, y, z, i)
        end
        particleplot!(usol, i; boxwidth=width)
    end every Δf
    display(anim)
    if save
        mov(anim, "particle.mov", fps=fps)
    end
    return usol, anim
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
    r_crit = 2^(1 / 6) * σ
    cutoff_filter = r .< r_crit
    q = σ ./ r
    return @. -24ε * (2q^12 - q^6) / r * cutoff_filter
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

# TODO: add WCA force for boundaries, to prevent escap!

function ∇C_simple(x, x₁, x₂, c₁, c₂)
    [(c₂ - c₁) / (x₂ - x₁);; 0]
end


"""
    δₓ(x, σ=1)

Dirac delta Gaussian with radius σ.

# Arguments
- `x`: Array of N-dimensional vectors, where `size(x)[2] == N`.

"""
function δₓ(x, σ=1)
    gauss = @. √(1 / (2π * σ^2)) * exp(-x^2 / (2σ^2))
    prod(gauss; dims=2)
end


"""
    idx_mod(i, N)

Modular arithmetic for indexes: a shifted mod function, starting at `i=1`.

# Arguments
- `i`: index
- `N`: modulo order

# Examples
    idx_mod(3, 9) == 3
    idx_mod(1, 9) == 1
    idx_mod(9, 9) == 1

"""
function idx_mod(i, N)
    mod(i - 1, N - 1) + 1
end


"""
    diffusion(c, x, p; Dp=1, Dc=1, μ=1, γ=1)

# Arguments
- `c`: matrix of concentrations, size `N*M`; assume `N==M`.
- `x`: vector of discretized grid points, length `N`; assumes `x==y`.
- `p`: array of `K` particle positions, size `1*2*K` or `K*2`.

"""
function diffusion(c, x, p; Dp=1, Dc=1, μ=1, γ=1)
    N, M = size(c)
    cₓₓ = zeros(N, M)  # diffusion term
    Σδ = zeros(N, M)  # particle sources
    for i ∈ 1:N, j ∈ 1:M
        # enforce periodic boundary
        i₋, i₊, j₋, j₊ = @. idx_mod([i j] + [-1; 1], [N M])
        # diffusion term
        cₓₓ[i, j] = c[i₊, j] + c[i₋, j] - 4c[i, j] + c[i, j₊] + c[i, j₋]

        # rel position p to xij
        Δr = @. x[[i j]] - p
        # delta source term
        Σδ[i, j] = sum(δₓ(Δr))
    end

    # constants
    Δx = x[2] - x[1]
    α = 2Dp / μ^2

    # sum equation terms
    (Dc * Δx^-2 * cₓₓ - γ * c + Σδ) / α
end

function abs_reshape(u, K, N, dim)
    p = reshape(u[1:dim*K], 1, dim, K)
    c = reshape(u[dim*K+1:end], N, N)
    return p, c
end

function active_brownian_system(K, width, N, dim=2; Dp=1, Dc=1, μ=1, γ=1)
    xgrid = range(-width, width, N)
    grhs = [ones(Int, K * dim); zeros(Int, N^2)]
    function f(u, par, t)
        p, c = abs_reshape(u, K, N, dim)
        fparticles = (-∇wca(p) .- (∇c = 0))
        fdiffusion = diffusion(c, xgrid, p; Dp=Dp, Dc=Dc, μ=μ, γ=γ)
        return [fparticles[:]; fdiffusion[:]]
    end

    g(u, par, t) = grhs

    return f, g, xgrid
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
    example4(
    n=2;
    tspan=(0.0, 1.0),
    width=20,
    dt=0.01,
    animate=false,
    potential=false,
    solver=SOSRA(),
)

Add diffusion with particle sources.

# Arguments
- `f`: deterministic; global potential + interactions
- `g`: stochastic; brownian motion
- `x₀`: particles in dims=3; x, y in dims=2

"""
function example5(
    K=10;
    tspan=(0.0, 1.0),
    width=20,
    N=50,
    dt=0.01,
    savedt=5dt,
    animate=false,
    solver=SOSRA(),
    Dp=1, Dc=1, μ=1, γ=1,
)

    # INITIAL CONDITION
    p₀ = width .* (rand(1, 2, K) .- 0.5)
    c₀ = zeros(N, N)
    u₀ = [p₀[:]; c₀[:]]

    # ODE
    f, g, xgrid = active_brownian_system(K, width, N, Dp=Dp, Dc=Dc, μ=μ, γ=γ)

    prob = SDEProblem(f, g, u₀, tspan)
    sol = solve(prob, solver, dt=dt, saveat=savedt, dense=false )
    println("SDE Solved!")
    usol = reduce(cat3, sol.u)
    psol = reshape(usol[1:K*2, :, :], 2, K, :)
    csol = reshape(usol[K*2+1:end, :, :], N, N, :)

    z = permutedims(csol, (2, 1, 3))
    if animate
        particle_animate(psol, xgrid, xgrid, z; Δf=1, save=true, width=width)
        return psol, csol, sol
    else
        p = particleplot(psol, :; boxwidth=width)
        potentialplot!(p, x, y, z)
        display(p)
        return psol, csol, sol, p
    end
end
