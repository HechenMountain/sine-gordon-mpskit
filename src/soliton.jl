using MPSKit, TensorKit
using LinearAlgebra: norm


# Classical soliton profiles  (seed for coherent-state initial MPS)

"""
    soliton_profile(j::Real, x0::Real; a::Real=1.0, m::Real=1.0, β::Real=1.0) → Float64

Classical soliton solution φ_soliton(j) = (4/β) arctan(exp(am(j − x₀))).

Arguments
- `j::Real`   : lattice site
- `x0::Real`  : soliton centre position
- `a::Real`   : lattice spacing
- `m::Real`   : mass parameter
- `β::Real`   : coupling constant

Returns
Classical soliton field at site `j`.
"""
soliton_profile(j::Real, x0::Real; a::Real=1.0, m::Real=1.0, β::Real=1.0) =
    (4 / β) * atan(exp(a * m * (j - x0)))

"""
    antisoliton_profile(j::Real, x0::Real; a::Real=1.0, m::Real=1.0, β::Real=1.0) → Float64

Classical anti-soliton  φ_AK(j) = 2π/β − φ_K(j).

Arguments
- `j::Real`   : lattice site
- `x0::Real`  : anti-soliton centre position
- `a::Real`   : lattice spacing
- `m::Real`   : mass parameter
- `β::Real`   : coupling constant

Returns
Classical anti-soliton field at site `j`.
"""
antisoliton_profile(j::Real, x0::Real; a::Real=1.0, m::Real=1.0, β::Real=1.0) =
    (2π / β) - soliton_profile(j, x0; a=a, m=m, β=β)


# Coherent-state vector helper

"""
    _coherent_state_vec(α, d) → Vector{Float64}

Fock-space amplitudes for a real coherent state with displacement `α`,
truncated to dimension `d` and normalised: |α⟩ ∝ Σₙ (α/√2)ⁿ/√(n!) |n⟩.

Arguments
- `α` : real coherent-state displacement
- `d` : Fock-space truncation

Returns
Normalized real vector of `d` Fock amplitudes.
"""
function _coherent_state_vec(α::Real, d::Int)
    v = zeros(Float64, d)
    v[1] = exp(-α^2 / 2)
    for n in 1:d-1
        v[n+1] = v[n] * α / √Float64(n)
    end
    nrm = norm(v)
    nrm > 0 && (v ./= nrm)
    return v
end

"""
    _complex_coherent_vec(α, d) → Vector{ComplexF64}

Fock-space amplitudes for a complex coherent state |α⟩ truncated to `d` levels.

Note: currently unused — retained for potential future use with complex
displacements (e.g. boosted soliton initial states).

Arguments
- `α` : complex coherent-state displacement
- `d` : Fock-space truncation

Returns
Normalized complex vector of `d` Fock amplitudes.
"""
function _complex_coherent_vec(α::Number, d::Int)
    v = zeros(ComplexF64, d)
    v[1] = exp(-abs2(α) / 2)
    for n in 1:d-1
        v[n+1] = v[n] * α / √Float64(n)
    end
    nrm = norm(v)
    nrm > 0 && (v ./= nrm)
    return v
end


# Coherent-state initial FiniteMPS builders

"""
    soliton_initial_fmps(N; d, a, m, β, x0) → FiniteMPS

Product FiniteMPS of real coherent states matching the classical soliton profile,
used to seed finite DMRG in the Q = +1 topological sector.

Arguments
- `N`  : number of lattice sites
- `d`  : Fock-space truncation
- `a`  : lattice spacing
- `m`  : mass parameter
- `β`  : coupling constant
- `x0` : soliton centre position

Returns
Product `FiniteMPS` in the soliton sector.
"""
function soliton_initial_fmps(N::Int; d::Int=30, a::Real=1.0, m::Real=1.0, β::Real=1.0,
                             x0::Real=N/2)
    V = boson_space(d)
    D = oneunit(V)          # ℂ^1 — trivial bond for a product MPS
    tensors = [begin
        ϕ_j = soliton_profile(j, x0; a=a, m=m, β=β)
        α_j = ϕ_j / √2.0
        v = ComplexF64.(_coherent_state_vec(α_j, d))
        TensorMap(reshape(v, 1, d, 1), D ⊗ V ← D)
    end for j in 1:N]
    return FiniteMPS(tensors)
end

"""
    antisoliton_initial_fmps(N; d, a, m, β, x0) → FiniteMPS

Product FiniteMPS matching the classical anti-soliton profile (Q = −1 sector).

Arguments
- `N`  : number of lattice sites
- `d`  : Fock-space truncation
- `a`  : lattice spacing
- `m`  : mass parameter
- `β`  : coupling constant
- `x0` : anti-soliton centre position

Returns
Product `FiniteMPS` in the anti-soliton sector.
"""
function antisoliton_initial_fmps(N::Int; d::Int=30, a::Real=1.0, m::Real=1.0, β::Real=1.0,
                                 x0::Real=N/2)
    V = boson_space(d)
    D = oneunit(V)
    tensors = [begin
        ϕ_j = antisoliton_profile(j, x0; a=a, m=m, β=β)
        α_j = ϕ_j / √2.0
        v = ComplexF64.(_coherent_state_vec(α_j, d))
        TensorMap(reshape(v, 1, d, 1), D ⊗ V ← D)
    end for j in 1:N]
    return FiniteMPS(tensors)
end


# IBC soliton / anti-soliton via boundary-pinned finite DMRG

"""
    soliton_mps(; d, a, m, β, N_w, N_buf, χ, λ_pin, nsweeps, verbosity) → NamedTuple

Construct a soliton state using the IBC (Infinite Boundary Condition) approach:
boundary-pinned finite DMRG on `N_total = N_w + 2 N_buf` sites.

The physical window is sites `N_buf+1 : N_buf+N_w`.

Arguments
- `d`         : Fock-space truncation
- `a`         : lattice spacing
- `m`         : mass parameter
- `β`         : coupling constant
- `N_w`       : physical window length
- `N_buf`     : buffer length on each side
- `χ`         : requested bond dimension (currently unused)
- `λ_pin`     : boundary-pinning strength
- `nsweeps`   : maximum DMRG sweeps
- `verbosity` : DMRG output level

Returns
Named tuple with fields:
- `psi`        : full DMRG MPS on `N_total` sites
- `phi_window` : ⟨φ⟩ at each window site (length `N_w`)
- `phi_all`    : ⟨φ⟩ at every site (length `N_total`)
- `energy`     : ⟨H_phys⟩ (physical Hamiltonian, no pinning)
- `N_total`, `N_buf`, `N_w`
"""
function soliton_mps(;
    d::Int         = 30,
    a::Real        = 1.0,
    m::Real        = 1.0,
    β::Real        = 1.0,
    N_w::Int       = 60,
    N_buf::Int     = 10,
    χ::Int         = 50,
    λ_pin::Real    = 100.0,
    nsweeps::Int   = 20,
    verbosity::Int = 1,
)
    N_total = N_w + 2 * N_buf
    x0 = N_total / 2

    # Pinned Hamiltonian for DMRG
    H_pinned = soliton_pinned_mpo(N_total; d=d, a=a, m=m, β=β, λ_pin=λ_pin)

    # Initial state: classical soliton profile
    ψ₀ = soliton_initial_fmps(N_total; d=d, a=a, m=m, β=β, x0=x0)

    # DMRG
    alg = DMRG(; tol=1e-10, maxiter=nsweeps,
                 verbosity=verbosity)
    ψ, _ = find_groundstate(ψ₀, H_pinned, alg)

    # Physical Hamiltonian (no pinning) for energy measurement
    H_phys = sine_gordon_finite_mpo(N_total; d=d, a=a, m=m, β=β)
    E = real(dot(ψ, H_phys, ψ))

    # Measure ⟨φ⟩ at each site.
    Phi = fock_phi(d)
    φ_all = real.([expectation_value(ψ, (j,) => Phi) for j in 1:N_total])
    φ_window = φ_all[N_buf+1 : N_buf+N_w]

    return (;
        psi       = ψ,
        phi_window = φ_window,
        phi_all   = φ_all,
        energy    = E,
        N_total   = N_total,
        N_buf     = N_buf,
        N_w       = N_w,
    )
end

"""
    antisoliton_mps(; kwargs...) → NamedTuple

Anti-soliton variant: swaps boundary pinning so that site 1 is in vacuum 1 and
site N is in vacuum 0 (Q = −1 topological sector).

Arguments
- `d`         : Fock-space truncation
- `a`         : lattice spacing
- `m`         : mass parameter
- `β`         : coupling constant
- `N_w`       : physical window length
- `N_buf`     : buffer length on each side
- `χ`         : requested bond dimension (currently unused)
- `λ_pin`     : boundary-pinning strength
- `nsweeps`   : maximum DMRG sweeps
- `verbosity` : DMRG output level

Returns
Named tuple with the same fields as `soliton_mps`.
"""
function antisoliton_mps(;
    d::Int         = 30,
    a::Real        = 1.0,
    m::Real        = 1.0,
    β::Real        = 1.0,
    N_w::Int       = 60,
    N_buf::Int     = 10,
    χ::Int         = 50,
    λ_pin::Real    = 100.0,
    nsweeps::Int   = 20,
    verbosity::Int = 1,
)
    N_total = N_w + 2 * N_buf
    x0 = N_total / 2

    H_pinned = antisoliton_pinned_mpo(N_total; d=d, a=a, m=m, β=β, λ_pin=λ_pin)
    ψ₀ = antisoliton_initial_fmps(N_total; d=d, a=a, m=m, β=β, x0=x0)

    alg = DMRG(; tol=1e-10, maxiter=nsweeps,
                 verbosity=verbosity)
    ψ, _ = find_groundstate(ψ₀, H_pinned, alg)

    H_phys = sine_gordon_finite_mpo(N_total; d=d, a=a, m=m, β=β)
    E = real(dot(ψ, H_phys, ψ))

    Phi = fock_phi(d)
    φ_all = real.([expectation_value(ψ, (j,) => Phi) for j in 1:N_total])
    φ_window = φ_all[N_buf+1 : N_buf+N_w]

    return (;
        psi        = ψ,
        phi_window = φ_window,
        phi_all    = φ_all,
        energy     = E,
        N_total    = N_total,
        N_buf      = N_buf,
        N_w        = N_w,
    )
end


# Topological charge and soliton mass

"""
    topological_charge(φ_window; β) → Float64

Topological charge Q = (β/2π)(φ_N − φ_1) computed from the window field
expectation values.  Should be ≈ +1 for a soliton and ≈ −1 for an anti-soliton.

Arguments
- `φ_window` : field expectation values across the physical window
- `β`        : coupling constant

Returns
Topological charge computed from the endpoint fields.
"""
function topological_charge(φ_window::AbstractVector; β::Real=1.0)
    return (β / (2π)) * (φ_window[end] - φ_window[1])
end

"""
    soliton_rest_mass(E_soliton, N_w, e_vac) → Float64

Soliton rest mass  M = E_window − N_w · ε₀  where `ε₀` is the bulk energy density
per site and `E_window` is the energy of the soliton in the window.
Classical prediction: M_cl = 8m/β².

Arguments
- `E_soliton` : soliton energy in the window
- `N_w`       : number of window sites
- `e_vac`     : bulk vacuum energy per site

Returns
Vacuum-subtracted soliton energy.
"""
soliton_rest_mass(E_soliton::Real, N_w::Int, e_vac::Real) = E_soliton - N_w * e_vac
