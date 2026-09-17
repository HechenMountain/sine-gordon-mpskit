using MPSKit, TensorKit
using TensorKit: truncrank

# VUMPS vacuum

"""
    find_vacuum(; d, a, m, β, χ, tol, maxiter, verbosity) → NamedTuple

Find the uniform infinite-MPS ground state (vacuum) of the lattice sine-Gordon
model using the VUMPS algorithm.

Arguments
- `d`          : Fock-space truncation (bosonic modes per site)
- `a`          : lattice spacing
- `m`          : boson mass
- `β`          : coupling constant
- `χ`          : bond dimension of the infinite MPS
- `tol`        : VUMPS convergence tolerance
- `maxiter`    : maximum VUMPS iterations
- `verbosity`  : 0 = silent, 1 = per-iteration, 2 = detailed

Returns
Named tuple with fields:
- `psi`    : converged `InfiniteMPS`
- `env`    : VUMPS environments
- `energy` : real energy per site ε₀ for the one-site unit cell
- `H`      : the `InfiniteMPOHamiltonian` used
"""
function find_vacuum(;
    d::Int        = 30,
    a::Real       = 1.0,
    m::Real       = 1.0,
    β::Real       = 1.0,
    χ::Int        = 50,
    tol::Real     = 1e-6,
    maxiter::Int  = 100,
    verbosity::Int = 1,
)
    V = boson_space(d)
    D = ℂ^χ

    # Build the Hamiltonian
    H = sine_gordon_mpo(; d, a, m, β)

    # Build a product-state seed for VUMPS (bond dimension 1).
    D1 = ℂ^1                                        # Domain
    v0 = zeros(ComplexF64, d); v0[1] = 1.0          # Fock vacuum |0⟩
    A_seed = TensorMap(reshape(v0, 1, d, 1), D1 ⊗ V ← D1) # Define TensorMap for (2→1) tensors
    ψ_seed = InfiniteMPS([A_seed])

    # Expand the bond dimension from 1 to χ by injecting random noise.
    alg_expand = RandExpand(; trscheme=truncrank(χ))
    ψ₀ = changebonds(ψ_seed, alg_expand)

    # VUMPS parameters
    alg = VUMPS(; tol=tol, maxiter=maxiter, verbosity=verbosity)

    # Optimize
    ψ, envs, ε = find_groundstate(ψ₀, H, alg)

    # Energy per site
    e0 = real(expectation_value(ψ, H, envs))

    return (; psi=ψ, env=envs, energy=e0, H=H)
end

# Displaced vacuum (vacuum 1, ⟨φ⟩ ≈ 2π/β)

"""
    find_vacuum1(; d, a, m, β, χ, tol, maxiter, verbosity) → NamedTuple

Find vacuum 1 of the sine-Gordon model: the degenerate ground state with
⟨φ⟩ ≈ 2π/β.

Arguments
- `d`         : Fock-space truncation
- `a`         : lattice spacing
- `m`         : boson mass
- `β`         : coupling constant
- `χ`         : bond dimension of the infinite MPS
- `tol`       : VUMPS convergence tolerance
- `maxiter`   : maximum VUMPS iterations
- `verbosity` : VUMPS output level

Returns
Named tuple with the same `psi`, `env`, `energy`, and `H` fields as `find_vacuum`.
"""
function find_vacuum1(;
    d::Int         = 30,
    a::Real        = 1.0,
    m::Real        = 1.0,
    β::Real        = 1.0,
    χ::Int         = 50,
    tol::Real      = 1e-6,
    maxiter::Int   = 100,
    verbosity::Int = 1,
)
    V = boson_space(d)
    D = ℂ^χ

    H = sine_gordon_mpo(; d, a, m, β)

    # ── Seed near vacuum 1 ────────────────────────────────────────────────────
    # Displacement that maps vacuum 0 → vacuum 1: α = (2π/β)/√2
    # so that ⟨φ⟩ = √2 α = 2π/β.
    α_disp = (2π / β) / √2.0
    v_disp = ComplexF64.(_coherent_state_vec(α_disp, d))

    # Build a product-state seed displaced to the vacuum 1 minimum.
    #
    # Same (1×d×1) TensorMap construction as in find_vacuum, but the Fock-space
    # vector now encodes a coherent state at displacement α = (2π/β)/√2,
    # so that ⟨φ⟩ = √2 α = 2π/β — the minimum of the cosine potential in
    # the vacuum-1 sector.
    D1 = ℂ^1
    A_seed = TensorMap(reshape(v_disp, 1, d, 1), D1 ⊗ V ← D1)
    ψ_seed = InfiniteMPS([A_seed])

    # Expand to the target bond dimension and run VUMPS.
    alg_vumps  = VUMPS(; tol=tol, maxiter=maxiter, verbosity=verbosity)
    alg_expand = RandExpand(; trscheme=truncrank(χ))

    ψ_seed, envs_seed, _ = find_groundstate(ψ_seed, H, alg_vumps)
    ψ = changebonds(ψ_seed, alg_expand)
    ψ, envs, ε = find_groundstate(ψ, H, alg_vumps)

    e1 = real(expectation_value(ψ, H, envs))
    return (; psi=ψ, env=envs, energy=e1, H=H)
end


# Energy density helper

"""
    bulk_energy(ψ::InfiniteMPS, H::InfiniteMPOHamiltonian) → Float64

Compute the bulk energy density ε0 = ⟨H⟩/N for the uniform infinite MPS `ψ`.

Arguments
- `ψ::InfiniteMPS`             : uniform infinite-MPS vacuum state
- `H::InfiniteMPOHamiltonian`  : Hamiltonian MPO

Returns
Bulk energy density as a real scalar.
"""
function bulk_energy(ψ::InfiniteMPS, H::InfiniteMPOHamiltonian)
    return real(expectation_value(ψ, H))
end

"""
    vacuum_phi(ψ::InfiniteMPS, d::Int) → Float64

Expectation value ⟨φ⟩ in the uniform vacuum state (should be ≈ 0 for vacuum 0).

Arguments
- `ψ::InfiniteMPS` : uniform infinite-MPS vacuum
- `d::Int`         : Fock-space truncation

Returns
Real expectation value ⟨φ⟩.
"""
function vacuum_phi(ψ::InfiniteMPS, d::Int)
    Phi = fock_phi(d)
    # Measure ⟨φ⟩ in the uniform vacuum state.
    return real(expectation_value(ψ, (1,) => Phi))
end
