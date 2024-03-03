"""
Types for conveniently constructing model parameter specifications.
Not intended for storing dependent variables,
just static parameters or mutable array caches.
"""

const KBOLTZ::Float64   = 1.380649e-23      # Boltzmann constant
const T_FREEZE::Float64 = 273.15            # Kelvin
const T_ROOM::Float64   = T_FREEZE + 20

struct ActiveBrownianParticleSpec{T <: Real}
    μ::T        # mobility of particle
    Dp::T       # collective particle diffusion rate
    α::T        # non-dimensionalised diffusion term 2Dp/μ²
    ϰ::T        # chemotaxic strength
    σ::T        # particle diameter
    rcrit::T    # critical wca distance
    ε::T        # wca field strength
end

function ActiveBrownianParticleSpec(μ::Real=1, ϰ::Real=1)
    return ActiveBrownianParticleSpec(
        promote(
            μ,
            # ((); Dp = μ * KBOLTZ * T_ROOM),
            ((); Dp = 1),
            2Dp / μ^2,
            ϰ,
            ((); σ = 1),
            2^(1 / 6) * σ,
            1,
        )...,
    )
end

struct DiffusiveChemicalSpec{T <: Real}
    Dc::T       # diffusion constant
    φout::T     # outflow rate
end

DiffusiveChemicalSpec() = DiffusiveChemicalSpec(ones(2)...)

struct GaussianSpec{T <: Real}
    σ::T                # Gaussian width
    linear_coef::T      # amplitude normalisation 1/√(2π)σ
    exponent_coef::T    # input scale factor 1/2σ^2
end

"""
    GaussianSpec(σ::Real=1)

Parameter specs for a Gaussian function.

Used for avoiding repeated calculation of coefficients.

# Arguments
- `σ` : standard deviation
"""
GaussianSpec(σ::Real=1) =
    GaussianSpec(promote(σ, 1 / √(2π)σ, 1 / 2σ^2)...)

abstract type Lattice end

struct PeriodicLattice{T <: Real, I <: Integer} <: Lattice
    xlims::NTuple{2, T}
    npoints::I
    ndim::I
    ij::Vector{I}                               # convenient grid vector index
    nperiods::U where {U <: Union{I, Nothing}}  # for periodic source
    xrange::AbstractRange
    x::Vector{T}
    xlength::T
    Δx::T
    Δx_inv::T
    Δx_invsq::T
    λrange::Vector{T}                           # for periodic source
end

"""
    PeriodicLattice(
    xlims::NTuple{2, T},
    nsteps::Integer,
    ndims::Integer=2,
) where {T <: AbstractFloat}

TBW
"""
function PeriodicLattice(
    xlims::NTuple{2, T},
    nsteps::I,
    ndim::I=2,
    nperiods::Union{I, Nothing}=nothing,
) where {T <: AbstractFloat, I <: Integer}
    return PeriodicLattice(
        xlims,
        ((); npoints = nsteps + 1),
        ndim,
        zeros(I, 2),
        nperiods,
        ((); xrange=range(xlims..., npoints)),
        Vector(xrange),
        ((); xlength = xlims[2] - xlims[1]),
        ((); Δx = xlength / nsteps),
        Δx^-1,
        Δx^-2,
        Vector((-nperiods:nperiods) * xlength),
    )
end

struct ActiveBrownianSystemSpec{T <: Real, I <: Integer}
    nparticle::I
    g::V where {V <: AbstractVector{A} where {A <: AbstractArray{T}}}
end

function ActiveBrownianSystemSpec(nparticle::Integer, grid::PeriodicLattice)
    g = [ones(grid.ndim, nparticle), zeros(grid.npoints, grid.npoints)]
    return ActiveBrownianSystemSpec(nparticle, g)
end
