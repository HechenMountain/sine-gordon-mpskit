using MPSKit, TensorKit
using LinearAlgebra: kron

# Helper: build a two-site operator from two d×d matrices via Kronecker product.
#
# Convention: for expectation_value(ψ, (j, j+1) => TensorMap(kron(A,B), V⊗V, V⊗V)),
# the first kron argument A acts on site j+1 (right) and the second argument B
# acts on site j (left).  The arguments are named mat_L/mat_R here for the
# intended physical sites (j, j+1), but must be passed in reversed order:
#   _twosite_op(mat_R_physical, mat_L_physical, V)
function _twosite_op(mat_L::AbstractMatrix, mat_R::AbstractMatrix, V::VectorSpace)
    return TensorMap(ComplexF64.(kron(mat_L, mat_R)), V ⊗ V, V ⊗ V)
end

# InfiniteMPOHamiltonian  (for VUMPS on the uniform infinite chain)

"""
    sine_gordon_mpo(; d, a, m, β) → InfiniteMPOHamiltonian

Constructs the lattice sine-Gordon Hamiltonian as an `InfiniteMPOHamiltonian`
with a one-site unit cell.

Arguments
- `d`  : Fock-space truncation (number of bosonic modes per site)
- `a`  : lattice spacing
- `m`  : mass parameter
- `β`  : coupling constant (β² < 8π is the semiclassical regime)

Returns
An `InfiniteMPOHamiltonian` suitable for `find_groundstate` with VUMPS.
"""
function sine_gordon_mpo(; d::Int=30, a::Real=1.0, m::Real=1.0, β::Real=1.0)
    V = boson_space(d)

    # On-site operators
    Pi2   = fock_pi2(d)       # Π²
    Phi2  = fock_phi2(d)      # φ²
    CosP  = fock_cosp(d; β)   # I − cos(βφ)  (cosine potential)

    # Coupling coefficients
    t_kin  = 1.0 / (2.0 * a)    # Πⱼ² / (2a)
    t_on   = 1.0 / a            # (1/a) φⱼ²
    t_pot  = a * m^2 / β^2      # (am²/β²)(I − cos βφ)

    # Combined on-site operator
    h_onsite = t_kin * Pi2 + t_on * Phi2 + t_pot * CosP

    # NN coupling: −(1/a) φⱼ φⱼ₊₁  as a single two-site operator
    phi_mat = _phi_matrix(d)
    h_nn = (-1.0 / a) * _twosite_op(phi_mat, phi_mat, V)

    # Hamiltonian constructor: InfiniteMPOHamiltonian(physical_spaces, terms...)
    #
    # Each term is a Pair from a site-index tuple to a TensorMap operator:
    #   (i,)     => O     — on-site operator at unit-cell site i
    #   (i, i+1) => O     — two-site operator coupling sites i and i+1
    #
    H = InfiniteMPOHamiltonian(fill(V, 1),
        (1,) => h_onsite,       # on-site term for the single unit-cell site
        (1, 2) => h_nn,         # NN coupling across unit-cell boundary
    )

    return H
end


# FiniteMPOHamiltonian  (for finite-chain DMRG / soliton construction)

"""
    sine_gordon_finite_mpo(N::Int; d::Int=30, a::Real=1.0, m::Real=1.0,
                           β::Real=1.0, ϕ_L::Real=0.0, ϕ_R::Real=0.0)
        → FiniteMPOHamiltonian

Full-chain Hamiltonian MPO on `N` sites for finite DMRG / TDVP, with
Dirichlet "ghost-site" boundary conditions at field values `ϕ_L`, `ϕ_R`.

Arguments
- `N::Int`    : number of lattice sites
- `d::Int`    : Fock-space truncation
- `a::Real`   : lattice spacing
- `m::Real`   : mass parameter
- `β::Real`   : coupling constant
- `ϕ_L::Real` : ghost-site field value at the left end  (vacuum of site 1)
- `ϕ_R::Real` : ghost-site field value at the right end (vacuum of site N)

Returns
`FiniteMPOHamiltonian` on `N` sites with the specified ghost-site boundaries.
"""
function sine_gordon_finite_mpo(N::Int; d::Int=30, a::Real=1.0, m::Real=1.0, β::Real=1.0,
                                 ϕ_L::Real=0.0, ϕ_R::Real=0.0)
    V = boson_space(d)

    Pi2  = fock_pi2(d)
    Phi2 = fock_phi2(d)
    Phi  = fock_phi(d)
    Id   = fock_id(d)
    CosP = fock_cosp(d; β)

    t_kin = 1.0 / (2.0 * a)
    t_on  = 1.0 / a
    t_pot = a * m^2 / β^2

    h_onsite = t_kin * Pi2 + t_on * Phi2 + t_pot * CosP

    # Endpoint on-site operators: bulk term + ghost-bond completion
    # (1/a)φ² − (ϕ/a)φ + (1/2a)ϕ² = (1/2a)φ² + (1/2a)(φ − ϕ)².
    h_site1 = h_onsite + (-ϕ_L / a) * Phi + (ϕ_L^2 / (2a)) * Id
    h_siteN = h_onsite + (-ϕ_R / a) * Phi + (ϕ_R^2 / (2a)) * Id

    phi_mat = _phi_matrix(d)
    h_nn = (-1.0 / a) * _twosite_op(phi_mat, phi_mat, V)

    # FiniteMPOHamiltonian: same tuple-Pair syntax as the infinite case.
    H = FiniteMPOHamiltonian(fill(V, N),
        (1,) => h_site1,                            # left endpoint (ghost at ϕ_L)
        [(i,) => h_onsite for i in 2:N-1]...,       # bulk on-site
        (N,) => h_siteN,                            # right endpoint (ghost at ϕ_R)
        [(i, i + 1) => h_nn for i in 1:N-1]...,     # NN coupling
    )

    return H
end


# Pinned soliton/antisoliton Hamiltonian  (for IBC construction)

"""
    soliton_pinned_mpo(N; d, a, m, β, λ_pin) → FiniteMPOHamiltonian

Full-chain Hamiltonian MPO for a soliton state with boundary-pinning terms.
The first site is pinned to vacuum 0 (φ = 0) and the last site to vacuum 1
(φ = 2π/β) via a quadratic penalty:

    H_pin = λ_pin [φ₁² + (φ_N − 2π/β)²]

This enforces the Q = +1 topological sector during DMRG.

Arguments
- `N`     : number of lattice sites
- `d`     : Fock-space truncation
- `a`     : lattice spacing
- `m`     : mass parameter
- `β`     : coupling constant
- `λ_pin` : boundary-pinning strength

Returns
`FiniteMPOHamiltonian` with soliton boundary-pinning terms.
"""
function soliton_pinned_mpo(N::Int; d::Int=30, a::Real=1.0, m::Real=1.0, β::Real=1.0,
                          λ_pin::Real=100.0)
    V = boson_space(d)

    Pi2  = fock_pi2(d)
    Phi2 = fock_phi2(d)
    Phi  = fock_phi(d)
    CosP = fock_cosp(d; β)

    t_kin = 1.0 / (2.0 * a)
    t_on  = 1.0 / a
    t_pot = a * m^2 / β^2

    ϕ_right = 2π / β

    # On-site Hamiltonian for bulk sites
    h_onsite = t_kin * Pi2 + t_on * Phi2 + t_pot * CosP

    # On-site for site 1: bulk + pinning to φ=0 → λ φ²
    h_site1 = h_onsite + λ_pin * Phi2

    # On-site for site N: bulk + pinning to φ=2π/β → λ(φ−2π/β)²
    #   = λ φ² − 2λ(2π/β) φ + const (drop constant)
    h_siteN = h_onsite + λ_pin * Phi2 + (-2 * λ_pin * ϕ_right) * Phi

    phi_mat = _phi_matrix(d)
    h_nn = (-1.0 / a) * _twosite_op(phi_mat, phi_mat, V)

    H = FiniteMPOHamiltonian(fill(V, N),
        (1,) => h_site1,
        [(i,) => h_onsite for i in 2:N-1]...,
        (N,) => h_siteN,
        [(i, i + 1) => h_nn for i in 1:N-1]...,
    )

    return H
end

"""
    antisoliton_pinned_mpo(N; d, a, m, β, λ_pin) → FiniteMPOHamiltonian

Full-chain Hamiltonian MPO for an anti-soliton state with boundary-pinning.
Site 1 is pinned to vacuum 1 (φ = 2π/β) and site N to vacuum 0 (φ = 0).
Enforces the Q = −1 topological sector.

Arguments
- `N`     : number of lattice sites
- `d`     : Fock-space truncation
- `a`     : lattice spacing
- `m`     : mass parameter
- `β`     : coupling constant
- `λ_pin` : boundary-pinning strength

Returns
`FiniteMPOHamiltonian` with anti-soliton boundary-pinning terms.
"""
function antisoliton_pinned_mpo(N::Int; d::Int=30, a::Real=1.0, m::Real=1.0, β::Real=1.0,
                               λ_pin::Real=100.0)
    V = boson_space(d)

    Pi2  = fock_pi2(d)
    Phi2 = fock_phi2(d)
    Phi  = fock_phi(d)
    CosP = fock_cosp(d; β)

    t_kin = 1.0 / (2.0 * a)
    t_on  = 1.0 / a
    t_pot = a * m^2 / β^2

    ϕ_left = 2π / β

    h_onsite = t_kin * Pi2 + t_on * Phi2 + t_pot * CosP

    # Site 1 pinned to vacuum 1 → λ(φ−2π/β)²
    h_site1 = h_onsite + λ_pin * Phi2 + (-2 * λ_pin * ϕ_left) * Phi

    # Site N pinned to vacuum 0 → λ φ²
    h_siteN = h_onsite + λ_pin * Phi2

    phi_mat = _phi_matrix(d)
    h_nn = (-1.0 / a) * _twosite_op(phi_mat, phi_mat, V)

    H = FiniteMPOHamiltonian(fill(V, N),
        (1,) => h_site1,
        [(i,) => h_onsite for i in 2:N-1]...,
        (N,) => h_siteN,
        [(i, i + 1) => h_nn for i in 1:N-1]...,
    )

    return H
end
