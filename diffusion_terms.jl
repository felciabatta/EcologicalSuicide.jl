"""
Equation terms used in diffusion.
"""

"""
    δsource(x::T, gauss::GaussianSpec{T}) where {T <: Real}

 1D Gaussian δ-function.

 # Arguments
 - `x` : the input coordinate
 - `gauss` : `GaussianSpec` paramater spec object
"""
function δsource(x::Real, gauss::GaussianSpec)
    @fastmath gauss.linear_coef * exp(-gauss.exponent_coef * x^2)
end

function δsource!(
    r::AbstractArray,
    x::AbstractArray,
    gauss::GaussianSpec,
)
    @inbounds for i ∈ axes(x, 2)
        @views @fastmath r[i] =
            δsource(x[1, i], gauss) * δsource(x[2, i], gauss)
    end
end

function δtrain!(
    r::AbstractArray,
    x::AbstractArray,
    λ::AbstractArray,
    gauss::GaussianSpec,
    # grid::PeriodicLattice, # TODO: check @btime on this
)
    # λ = grid.λrange  # TODO: check @btime on this
    fill!(r, 0)
    @inbounds for i ∈ axes(x, 2), n ∈ axes(λ, 1)
        @views @fastmath r[i] +=
            δsource(x[1, i] + λ[n], gauss) * δsource(x[2, i] + λ[n], gauss)
    end
end

function Σδtrain(
    x::AbstractArray,
    gauss::GaussianSpec{T},
    grid::PeriodicLattice,
) where {T}
    λ = grid.λrange
    Σδ::T = 0
    @inbounds for i ∈ axes(x, 2), n ∈ axes(λ, 1)
        @views @fastmath Σδ +=
            δsource(x[1, i] + λ[n], gauss) * δsource(x[2, i] + λ[n], gauss)
    end
    return Σδ
end

"""
    ddiff1(c, c₋, c₊, Δx)

 2nd order central difference in 1D.
"""
function ddiff1(c, c₋, c₊, Δx)
    return (c₋ + c₊ - 2c) * Δx^-2
end

function ddiff2!(
    r::AbstractArray,
    c₀::AbstractArray,
    ci₋::AbstractArray,
    ci₊::AbstractArray,
    cj₋::AbstractArray,
    cj₊::AbstractArray;
    grid::PeriodicLattice,
)
    return r .= (ci₋ + ci₊ + cj₋ + cj₊ - 4c₀) * grid.Δx_invsq
end

function ddiff2(
    c₀::T,
    ci₋::T,
    ci₊::T,
    cj₋::T,
    cj₊::T;
    grid::PeriodicLattice,
) where {T <: Real}
    return @fastmath (ci₋ + ci₊ + cj₋ + cj₊ - 4c₀) * grid.Δx_invsq
end

struct DiffusionCache{T <: Real, N}
    cₓₓ::Array{T, N}            # Laplacian matrix
    Σδ::Array{T, N}             # source summation matrix
    Δr::Array{T, N}             # source displacement vectors
end

"""
    WCACache(ndim::Integer, nparticle::Integer)

Create a DiffusionCache
"""
DiffusionCache(grid::PeriodicLattice, nparticle::Integer) = DiffusionCache(
    zeros(eltype(grid.x), (grid.npoints, grid.npoints)),
    zeros(eltype(grid.x), (grid.npoints, grid.npoints)),
    zeros(eltype(grid.x), (grid.ndim, nparticle)),
)

function relposition!(
    ch::DiffusionCache,
    xij::AbstractArray,
    p::AbstractArray,
)
    Δr = ch.Δr
    @inbounds for I ∈ CartesianIndices(Δr)
        dim, n = Tuple(I)
        @fastmath @views Δr[dim, n] = xij[dim] - p[dim, n]
    end
end

function diffuse!(
    ch::DiffusionCache,
    c::AbstractArray,    # chemical field
    p::AbstractArray,    # particle vectors
    gauss::GaussianSpec,
    grid::PeriodicLattice,
)
    # convenience pointers
    x, N, ij = grid.x, grid.npoints, grid.ij
    cₓₓ, Σδ, Δr = ch.cₓₓ, ch.Σδ, ch.Δr

    # declare local variables
    local i₋::Int, i₊::Int, j₋::Int, j₊::Int  # indexes

    @inbounds for I ∈ CartesianIndices(c)
        i, j = Tuple(I)
        # enforce periodic boundary
        i₋, i₊, j₋, j₊ = idx_pm(i, j, N)

        # Laplacian term
        @views cₓₓ[I] =
            ddiff2(c[I], c[i₋, j], c[i₊, j], c[i, j₋], c[i, j₊]; grid)

        # rel position p to xi
        @. ij = i, j
        @views relposition!(ch, x[ij], p)

        # periodic delta source term
        Σδ[I] = Σδtrain(Δr, gauss, grid)
    end
end
