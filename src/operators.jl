using TensorKit
using LinearAlgebra: I, norm, exp

# Helpers: raw d×d matrices in the Fock basis

"""
    _a_matrix(d::Int) → Matrix{Float64}

Annihilation operator ⟨n−1|a|n⟩ = √n in the Fock basis of dimension `d`.

Arguments
- `d::Int` : Fock-space truncation (number of levels)

Returns
Real `d × d` annihilation matrix.
"""
function _a_matrix(d::Int)
    a = zeros(Float64, d, d)
    for n in 1:d-1
        a[n, n+1] = sqrt(Float64(n))
    end
    return a
end

"""
    _phi_matrix(d::Int) → Matrix{Float64}

Field quadrature  φ = (a + a†)/√2  in the Fock basis of dimension `d`.

Arguments
- `d::Int` : Fock-space truncation

Returns
Real `d × d` field matrix.
"""
function _phi_matrix(d::Int)
    a = _a_matrix(d)
    return (a + a') / sqrt(2.0)
end

"""
    _pi_matrix(d::Int) → Matrix{ComplexF64}

Conjugate momentum  Π = i(a† − a)/√2.  Hermitian, purely imaginary coefficients.

Arguments
- `d::Int` : Fock-space truncation

Returns
Complex `d × d` momentum matrix.
"""
function _pi_matrix(d::Int)
    a = _a_matrix(d)
    return im * (a' - a) / sqrt(2.0)
end


# TensorMap constructors

"""
    boson_space(d::Int) → ComplexSpace

Returns the physical Hilbert space `ℂ^d` for a bosonic site truncated to
`d` Fock levels.

Arguments
- `d::Int` : number of Fock levels

Returns
Physical space `ℂ^d`.
"""
boson_space(d::Int) = ℂ^d

"""
    fock_a(d::Int) → TensorMap{ComplexSpace,1,1}

Annihilation operator as a TensorMap  V ← V  where V = ℂ^d.

Arguments
- `d::Int` : Fock-space truncation

Returns
Annihilation operator as a `TensorMap`.
"""
function fock_a(d::Int)
    V = boson_space(d)
    return TensorMap(_a_matrix(d), V ← V)
end

"""
    fock_adag(d::Int) → TensorMap{ComplexSpace,1,1}

Creation operator  a†  as a TensorMap  V ← V.

Arguments
- `d::Int` : Fock-space truncation

Returns
Creation operator as a `TensorMap`.
"""
fock_adag(d::Int) = fock_a(d)'

"""
    fock_phi(d::Int) → TensorMap{ComplexSpace,1,1}

Field quadrature  φ = (a + a†)/√2  as a TensorMap.

Arguments
- `d::Int` : Fock-space truncation

Returns
Field operator as a `TensorMap`.
"""
function fock_phi(d::Int)
    V = boson_space(d)
    return TensorMap(_phi_matrix(d), V ← V)
end

"""
    fock_pi(d::Int) → TensorMap{ComplexSpace,1,1}

Conjugate momentum  Π = i(a† − a)/√2  as a TensorMap.

Arguments
- `d::Int` : Fock-space truncation

Returns
Momentum operator as a `TensorMap`.
"""
function fock_pi(d::Int)
    V = boson_space(d)
    return TensorMap(_pi_matrix(d), V ← V)
end

"""
    fock_pi2(d::Int) → TensorMap{ComplexSpace,1,1}

Π²  as a TensorMap.  Real and positive semidefinite.

Arguments
- `d::Int` : Fock-space truncation

Returns
Squared momentum operator as a `TensorMap`.
"""
function fock_pi2(d::Int)
    V = boson_space(d)
    Π = _pi_matrix(d)
    return TensorMap(Π * Π, V ← V)
end

"""
    fock_phi2(d::Int) → TensorMap{ComplexSpace,1,1}

φ²  as a TensorMap.  Real and positive semidefinite.

Arguments
- `d::Int` : Fock-space truncation

Returns
Squared field operator as a `TensorMap`.
"""
function fock_phi2(d::Int)
    V = boson_space(d)
    φ = _phi_matrix(d)
    return TensorMap(φ * φ, V ← V)
end

"""
    fock_number(d::Int) → TensorMap{ComplexSpace,1,1}

Number operator  N = a†a  as a TensorMap.

Arguments
- `d::Int` : Fock-space truncation

Returns
Number operator as a `TensorMap`.
"""
function fock_number(d::Int)
    V = boson_space(d)
    a = _a_matrix(d)
    return TensorMap(a' * a, V ← V)
end

"""
    fock_cosp(d::Int; β::Real=1.0) → TensorMap{ComplexSpace,1,1}

Cosine potential  I − cos(βφ)  as a TensorMap.

The matrix exponential is used to compute `cos(βφ)` exactly in the Fock basis:
    cos(βφ) = Re[exp(iβφ)]

Arguments
- `d::Int`    : Fock-space truncation
- `β::Real`   : coupling constant of the cosine potential

Returns
`I − cos(βφ)` as a `TensorMap`.
"""
function fock_cosp(d::Int; β::Real=1.0)
    V = boson_space(d)
    φ = _phi_matrix(d)
    cos_bphi = real(exp(im * β * φ))
    return TensorMap(Matrix{Float64}(I, d, d) - cos_bphi, V ← V)
end

"""
    fock_cosphi(d::Int; β::Real=1.0) → TensorMap{ComplexSpace,1,1}

Pure cosine  cos(βφ)  as a TensorMap.

Arguments
- `d::Int`    : Fock-space truncation
- `β::Real`   : coupling constant of the cosine potential

Returns
`cos(βφ)` as a `TensorMap`.
"""
function fock_cosphi(d::Int; β::Real=1.0)
    V = boson_space(d)
    φ = _phi_matrix(d)
    return TensorMap(real(exp(im * β * φ)), V ← V)
end

"""
    fock_id(d::Int) → TensorMap{ComplexSpace,1,1}

Identity operator on `ℂ^d` as a TensorMap.

Arguments
- `d::Int` : Fock-space truncation

Returns
Identity operator as a `TensorMap`.
"""
function fock_id(d::Int)
    V = boson_space(d)
    return TensorMap(Matrix{Float64}(I, d, d), V ← V)
end
