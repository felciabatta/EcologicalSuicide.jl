"""
    δsource(x::Real, σ::Real=1)::AbstractFloat

1D Gaussian δ-function.
"""
function δsource(x::Real, σ::Real=1)::AbstractFloat
    return @fastmath √(1 / (2π * σ^2))exp(-x^2 / (2σ^2))
end


"""
    δsource(x; σ=1)

ND Gaussian δ-function.

# Arguments
- `x`: Array of N-dimensional vectors, where `size(x)[2] == N`.

"""
function δsource(x::AbstractArray, args...; dims::Integer=2)
    gauss = @. δsource(x, args...)
    prod(gauss; dims)
end

function δsource!(r::AbstractArray, x::AbstractArray, args...)
    gauss::typeof(x) = @. δsource(x, args...)
    return @fastmath prod!(r, gauss)
end


function polycos_n(x::Number, L::Real, n::Integer)
    ((x / (n * L) + 1) * (x / (n * L) - 1))^2
end

function polycos(x::Number, L::Real=1, N::Integer=2)
    out = -x^2
    for n ∈ 1:N
        out *= polycos_n(x, L, n)
    end
    return out
end


function polycos(x::Number, L::Real=1, N::Integer=2)
    out = -x^2
    prod(polycos_n(x, L, 1:N))
    for n ∈ 1:N
        out *= polycos_n(x, L, n)
    end
    return out
end

"""
    δtrain(x::Real, period::Real, σ::Real=1)

1D Gaussian δ-train, approximated with a cosine exponent.
"""
function δtrain(x::Real, period::Real, σ::Real=1)
    A = @fastmath √(1 / (2π * σ^2))
    k = @fastmath (period / (2π * σ))^2
    return @fastmath A * exp(k * cos(2π * x / period) - 1)
end


"""
    δtrain(x::AbstractArray, args...; dims=2)

ND Gaussian δ-train, approximated with a cosine exponent.
"""
function δtrain(x::AbstractArray, args...; dims=2)
    gauss = @. δtrain(x, args...)
    return @fastmath prod(gauss; dims)
end


function δtrainB(x::AbstractArray, args...; dims=2)
    temp = typeof(x)(undef, size(x)...)
    broadcast!(δtrain, temp, x, args...)
    return @fastmath prod(temp; dims)
end


function δtrain!(
    r::AbstractArray{T, N},
    x::AbstractArray{T, N},
    args...,
) where {T <: Real, N}
    temp = typeof(x)(undef, size(x)...)
    broadcast!(δtrain, temp, x, args...)
    return @fastmath prod!(r, temp)
end


"""
    periodic(
    f::Function,
    period::Real,
    x,
    args...;
    N::Integer=1,
    kwargs...,
)

Evaluate the periodic version of function `f` at `x`.

# Arguments
- `f`: function, must take position `x` as input.
"""
function periodic(
    f::Function,
    period::Real,
    x::Real,
    args...;
    N::Integer=1,
    kwargs...,
)
    pcomponents = [f(x + period * n, args...; kwargs...) for n ∈ -N:N]
    return sum(pcomponents)
end

function periodic(
    f::Function,
    period,
    x::AbstractArray,
    args...;
    N=1,
    kwargs...,
)
    pcomponents =
        [f(x .+ period .* [n m], args...; kwargs...) for n ∈ -N:N, m ∈ -N:N]
    return sum(pcomponents)
end

"""
    diffusion(c, x, p; Dp=1, Dc=1, μ=1, γ=1)

# Arguments
- `c`: matrix of concentrations, size `N*M`; assume `N==M`.
- `x`: vector of discretized grid points, length `N`; assumes `x==y`.
- `p`: array of `K` particle positions, size `1*2*K` or `K*2`.

"""
function diffusion(c::Matrix{<:AbstractFloat}, x, p; Dp=1, Dc=1, μ=1, γ=1)
    N::Matrix{Int} = collect(size(c))'
    L = x[end] - x[1] # period
    local i₋::Int, i₊::Int, j₋::Int, j₊::Int # indexes
    ij::Matrix{Int} = zeros(Int, 1, 2)
    pm1::Matrix{Int} = [1 -1]'
    cₓₓ = zeros(N...)  # diffusion term
    Δr::typeof(p) = typeof(p)(undef, size(p)...)
    δij::typeof(p) = typeof(p)(undef, 1, 1, size(p, 3))
    Σδ = zeros(N...)  # particle sources
    for i ∈ 1:N[1], j ∈ 1:N[2]
        ij = [i j]
        # enforce periodic boundary
        i₋, i₊, j₋, j₊ = @. idx_mod(ij - pm1, N)
        # diffusion term
        cₓₓ[i, j] = c[i₊, j] + c[i₋, j] - 4c[i, j] + c[i, j₊] + c[i, j₋]
        # TODO: USE @VIEWS
        # rel position p to xij
        Δr = @. x[ij] - p
        # periodic delta source term

        Σδ[i, j] = @fastmath δsource!(δij, Δr)
    end

    # pde constants
    Δx = x[2] - x[1]
    α = 2Dp / μ^2

    # sum equation terms
    (Dc * Δx^-2 * cₓₓ - γ * c + Σδ) / α
end
