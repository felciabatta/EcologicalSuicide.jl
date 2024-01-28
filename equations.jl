"""
    ṗ()

Right-hand-side for the particle SDE.
"""
function ṗ()
    return
end

"""
    ċ()

Right-hand-side for chemical diffusion PDE.
"""
function ċ()
    return
end

"""
    active_brownian_system(K, width, N, dim=2; Dp=1, Dc=1, μ=1, γ=1)

TBW
"""
function active_brownian_system(K, width, N, dim=2; Dp=1, Dc=1, μ=1, γ=1)
    xgrid = range(-width / 2, width / 2, N)
    grhs = [ones(Int, K * dim); zeros(Int, N^2)]
    function f(u, par, t)
        p, c = abs_reshape(u, K, N, dim)
        # enforce periodic boundary on particles
        p = range_mod.(p, -width / 2, width / 2)
        fparticles = (-∇wca(p))
        fdiffusion = diffusion(c, xgrid, p; Dp=Dp, Dc=Dc, μ=μ, γ=γ)
        return [fparticles[:]; fdiffusion[:]]
    end

    g(u, par, t) = grhs

    return f, g, xgrid
end
