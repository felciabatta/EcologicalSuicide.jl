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
function ∇wca(x::Array{<:Real, 3}, σ=1, ε=1)
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
function ∇c(c::Matrix{<:Real}, p; x)
    # interpolate
    c_itp = cubic_spline_interpolation((x, x), c)
    # for specific p, first
    ∇c_itp = gradient(c_itp, p...)
end
