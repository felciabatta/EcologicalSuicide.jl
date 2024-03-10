import Base.fill!
# import Base.zero

function fill!(dest::A, a::A) where {A <: Array}
    @inbounds for i ∈ eachindex(dest)
        dest[i] = a[i]
    end
end

function fill!(f::Function, a::A, b::A) where {A <: Array}
    @inbounds for i ∈ eachindex(a)
        a[i] = f(b[i])
    end
end

import Base.size_to_strides
import Base.strides
strides(A::RecursiveArrayTools.ArrayPartition) = size_to_strides(1, size(A)...)

import Base.pointer
pointer(A::RecursiveArrayTools.ArrayPartition) = pointer(A.x[1])

# struct GenericZero{T} end

# function zero(::Type{Matrix{T}}) where {T <: Real}
#     return GenericZero{Matrix{T}}()
# end

# function fill!(dest::Vector{Matrix{T}}, ::GenericZero{Matrix{T}}) where {T}
#     @inbounds for i ∈ eachindex(dest)
#         dest[i] .= 0
#     end
# end

# zero(x::Vector{Matrix{T}}) where {T <: Real} = [zero(el) for el in x]

"""
    idx_mod(i::Integer, N::Integer)

Modulo arithmetic for indexes: a shifted mod function, starting at `i=1`.

# Arguments
- `i`: index
- `N`: modulo order

# Examples
    idx_mod(3, 9) == 3
    idx_mod(1, 9) == 1
    idx_mod(9, 9) == 1

"""
function idx_mod(i::T, N::T)::T where {T <: Integer}
    return mod(i - 1, N - 1) + 1
end

function idx_mod(inds::NTuple{M, T}, N::T) where {T <: Integer, M}
    return (idx_mod(i, N) for i ∈ inds)
end

function idx_limit(i::T, N::T)::T where {T <: Integer}
    return (i == N + 1) ? 1 : (i == 0) ? N : i
end

function idx_limit(inds::NTuple{M, T}, N::T) where {T <: Integer, M}
    return (idx_limit(i, N) for i ∈ inds)
end

function idx_pm(i::T, j::T, N::T) where {T <: Integer}
    return idx_limit((i - 1, i + 1, j - 1, j + 1), N)
end

function range_mod(x, a, b)
    @assert a < b
    return mod(x - a, b - a) + a
end

function range_mod!(x::AbstractArray, a, b)
    @inbounds for I ∈ CartesianIndices(x)
        x[I] = range_mod(x[I], a, b)
    end
end

function range_mod!(x::AbstractArray, grid::PeriodicLattice)
    return range_mod!(x, grid.xlims...)
end

cat3(x...) = cat(x...; dims=3)
