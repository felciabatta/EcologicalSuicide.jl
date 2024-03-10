"""
Equation terms used for particle dynamics.
"""

# abstract type AbstractCache end

struct WCACache{T <: Real, N, M}
    r::Array{T, N}              # ∇wca return array
    pij::Array{T, M}            # displacement array
    dij::Array{T, N}            # distances array
    cutoff::Array{T, N}         # rcrit cutoff array
    q::Array{T, N}              # ∂wca quotient term
end

"""
    WCACache(ndim::Integer, nparticle::Integer)

Create a WCACache
"""
WCACache(ndim::Integer, nparticle::Integer) = WCACache(
    zeros(Float64, ndim, nparticle),
    zeros(Float64, ndim, nparticle, nparticle),
    ((); dij = zeros(Float64, nparticle, nparticle)),
    fill!(similar(dij), 0),
    fill!(similar(dij), 0),
)

function ∂wca!(ch::WCACache, abp::ActiveBrownianParticleSpec)
    # convenience pointers
    cutoff, dij, q = ch.cutoff, ch.dij, ch.q
    @. cutoff = @fastmath dij < abp.rcrit
    @. q = @fastmath (abp.σ / dij)^6
    @. dij = @fastmath -24abp.ε * 2q * (q - 1) / dij * cutoff
    return nothing
end

function distances!(ch::WCACache)
    pij, dij = ch.pij, ch.dij
    @inbounds for i ∈ axes(pij, 2), j ∈ axes(pij, 3)
        @views @fastmath dij[i, j] = √(pij[1, i, j]^2 + pij[2, i, j]^2)
    end
end

"""
    displacements!(p::AbstractArray)

Compute pairwise displacement vectors.
"""
function displacements!(r::AbstractArray, p::AbstractArray)
    @inbounds for dim ∈ axes(p, 1), i ∈ axes(p, 2), j ∈ axes(p, 2)
        @views r[dim, i, j] = p[dim, j] - p[dim, i]
    end
end

function normalize!(ch::WCACache)
    dij, pij = ch.dij, ch.pij
    @inbounds for dim ∈ axes(pij, 1), i ∈ axes(pij, 2), j ∈ axes(pij, 3)
        @views pij[dim, i, j] = pij[dim, i, j] / dij[i, j]
    end
end

function nansumforces!(ch::WCACache)
    r, mag, dirvec = ch.r, ch.dij, ch.pij
    fill!(r, 0)
    @inbounds for dim ∈ axes(r, 1), i ∈ axes(r, 2), j ∈ axes(r, 2)
        @views r[dim, i] +=
            isnan(mag[i, j]) ? 0 : @fastmath mag[i, j] * dirvec[dim, i, j]
    end
    return nothing
end

function ∇wca!(ch::WCACache, p::AbstractArray, abp::ActiveBrownianParticleSpec)
    # convenience pointers
    pij = ch.pij
    # pairwise displacement vectors
    displacements!(pij, p)
    # pairwise distances
    distances!(ch)
    # normalize displacements
    normalize!(ch)
    # find force magnitudes (∂/∂r)
    ∂wca!(ch, abp)
    # sum of magnitudes*direction vectors
    nansumforces!(ch)
    return nothing
end

#############################################################################

#############################################################################

function ∇C_simple(x, x₁, x₂, c₁, c₂)
    return [(c₂ - c₁) / (x₂ - x₁);; 0]
end

"""
    ∇c(c::Matrix{Real}, p; x)

# Overview
Calculate `∇c` at some arbitrary point `p`.

- `c`: a discretized scalar field.
- `p`:
- `x`: equally spaced grid points.

# Implementation Details
1. Interpolate `c`.
2. Get interpolated gradient for `p`.

"""
function ∇c(c::AbstractArray, p; grid)
    x = grid.xrange
    # interpolate
    c_itp = cubic_spline_interpolation((x, x), c)
    # for specific p, first
    return gradient(c_itp, p...)
end
