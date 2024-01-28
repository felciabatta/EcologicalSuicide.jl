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
function idx_mod(i::Integer, N::Integer)::Integer
    mod(i - 1, N - 1) + 1
end

function range_mod(x::Real, a::Real, b::Real)
    @assert a < b
    return mod(x - a, b - a) + a
end

function abs_reshape(u, K::Integer, N::Integer, dim::Integer)
    p = reshape(u[1:dim*K], 1, dim, K)
    c = reshape(u[dim*K+1:end], N, N)
    return p, c
end
