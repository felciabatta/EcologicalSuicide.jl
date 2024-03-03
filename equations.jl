"""
    ṗ!(wcache, p, abp)

Right-hand-side for the particle SDE.
"""
function ṗ(wcaforce::T, chemotax) where {T <: Real}
    return @fastmath -wcaforce + chemotax
end

"""
    ċ(c::T, cₓₓ::T, Σδ::T, chem, abp) where {T<: Real}

Right-hand-side for scalar chemical diffusion PDE.
"""
function ċ(
    c::T,
    cₓₓ::T,
    Σδ::T,
    chem::DiffusiveChemicalSpec,
    abp::ActiveBrownianParticleSpec,
) where {T <: Real}
    Dc, φout = chem.Dc, chem.φout
    return @fastmath (Dc * cₓₓ - φout * c + Σδ) / abp.α
end

function ṗ!(dp::AbstractArray, wcache::WCACache)
    wcaforce = wcache.r
    @inbounds for i ∈ eachindex(dp)
        dp[i] = ṗ(wcaforce[i], 0)
    end
end

"""
    ċ!(dcache, c, p, chem, abp, gauss, grid)

Right-hand-side for chemical diffusion PDE.
"""
function ċ!(
    dc::AbstractArray,
    dcache::DiffusionCache,
    c::AbstractArray,
    chem::DiffusiveChemicalSpec,
    abp::ActiveBrownianParticleSpec,
)
    cₓₓ, Σδ = dcache.cₓₓ, dcache.Σδ
    @inbounds for i ∈ eachindex(dc)
        dc[i] = ċ(c[i], cₓₓ[i], Σδ[i], chem, abp)
    end
end

function activebrownian!(du, u, par, t)
    # CONVENIENCE POINTERS
    (; wcache, dcache, abp, chem, gauss, grid) = par
    p, c = u.x
    dp, dc = du.x
    # p = u[1]
    # c = u[2]
    # dp = du[1]
    # dc = du[2]

    range_mod!(p, grid)

    # COMPUTE TERMS
    ∇wca!(wcache, p, abp)
    diffuse!(dcache, c, p, gauss, grid)

    # SUMMATE TERMS
    ṗ!(dp, wcache)
    ċ!(dc, dcache, c, chem, abp)
    return nothing
end

function randombrownian!(du, u, par, t)
    dp, dc = du.x
    fill!(dp, 1)
    fill!(dc, 0)
    return nothing
end
