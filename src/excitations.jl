using MPSKit, TensorKit
using Statistics: mean

# Excitation ansatz: soliton, anti-soliton, and breathers

"""
    find_soliton_excitation(H, ψ_vac0, ψ_vac1; momenta, num, verbosity)
    → (Es, qps)

Find the soliton excitation spectrum using the quasiparticle (excitation-tensor)
ansatz.

Arguments
- `H`         : `InfiniteMPOHamiltonian` (from `sine_gordon_mpo`)
- `ψ_vac0`    : `InfiniteMPS` — left vacuum, ⟨φ⟩ ≈ 0
- `ψ_vac1`    : `InfiniteMPS` — right vacuum, ⟨φ⟩ ≈ 2π/β
- `momenta`   : momenta at which to compute the excitation (default: 20 pts)
- `num`       : number of excitation branches per momentum (default 1 = lowest)
- `verbosity` : output level for the optimiser

Returns
- `Es`  : excitation energies, one row per momentum and one column per branch
- `qps` : quasiparticle states with the same indexing
"""
function find_soliton_excitation(
    H::InfiniteMPOHamiltonian, ψ_vac0::InfiniteMPS, ψ_vac1::InfiniteMPS;
    momenta::AbstractVector{<:Real} = range(-π, π; length=20),
    num::Int   = 1,
    verbosity::Int = 0,
)
    alg = QuasiparticleAnsatz(; verbosity=verbosity)
    Es, qps = excitations(H, alg, collect(Float64, momenta), ψ_vac0,
                           environments(ψ_vac0, H), ψ_vac1;
                           num=num)
    return Es, qps
end

"""
    find_antisoliton_excitation(H, ψ_vac1, ψ_vac0; momenta, num, verbosity)
    → (Es, qps)

Anti-soliton excitation with the left and right vacua swapped.

Arguments
- `H`         : `InfiniteMPOHamiltonian`
- `ψ_vac1`    : left vacuum, ⟨φ⟩ ≈ 2π/β
- `ψ_vac0`    : right vacuum, ⟨φ⟩ ≈ 0
- `momenta`   : momenta at which to compute the excitation
- `num`       : number of excitation branches per momentum
- `verbosity` : output level for the optimiser

Returns
- `Es`  : excitation energies, one row per momentum
- `qps` : quasiparticle states
"""
function find_antisoliton_excitation(
    H::InfiniteMPOHamiltonian, ψ_vac1::InfiniteMPS, ψ_vac0::InfiniteMPS;
    momenta::AbstractVector{<:Real} = range(-π, π; length=20),
    num::Int   = 1,
    verbosity::Int = 0,
)
    alg = QuasiparticleAnsatz(; verbosity=verbosity)
    Es, qps = excitations(H, alg, collect(Float64, momenta), ψ_vac1,
                           environments(ψ_vac1, H), ψ_vac0;
                           num=num)
    return Es, qps
end

"""
    find_breather_excitations(H, ψ_vac; momenta, num, verbosity) → (Es, qps)

Find the `num` lightest breather (vacuum-sector) excitations using the
quasiparticle ansatz.

Breathers are soliton–antisoliton bound states with vacuum quantum numbers, so both
the left and right asymptotic states are the same vacuum `ψ_vac`.

Arguments
- `H`         : `InfiniteMPOHamiltonian`
- `ψ_vac`     : `InfiniteMPS` — vacuum ground state (⟨φ⟩ ≈ 0)
- `momenta`   : momenta at which to compute excitations (default: [0.0])
- `num`       : number of breather branches to compute (default 3)
- `verbosity` : output level for the optimiser

Returns
- `Es`  : excitation energies, matrix of size (length(momenta), num)
- `qps` : quasiparticle states with the same indexing
"""
function find_breather_excitations(
    H::InfiniteMPOHamiltonian, ψ_vac::InfiniteMPS;
    momenta::AbstractVector{<:Real} = [0.0],
    num::Int       = 3,
    verbosity::Int = 0,
)
    alg = QuasiparticleAnsatz(; verbosity=verbosity)
    Es, qps = excitations(H, alg, collect(Float64, momenta), ψ_vac,
                          environments(ψ_vac, H); num=num)
    return Es, qps
end

# Dispersion-relation helpers

"""
    soliton_dispersion(Es::AbstractVector{<:Real},
                    momenta::AbstractVector{<:Real}) → NamedTuple

Fit excitation energies `Es` to the lattice dispersion relation

    E(p)² = M² + C sin²(p/2)

where `p` is the Bloch momentum (dimensionless, Brillouin zone p ∈ [−π, π]).
Both `M` (soliton rest mass) and `C` (lattice curvature) are fitted via
least squares on E².

Arguments
- `Es::AbstractVector{<:Real}`      : excitation energies at each momentum
- `momenta::AbstractVector{<:Real}` : Bloch momenta (same length as `Es`)

Returns
- `mass`      : fitted soliton rest mass
- `C`         : fitted lattice curvature
- `c2_eff`    : effective lattice velocity squared, `C / 4`
- `E_fit`     : fitted energies at the input momenta
- `residuals` : `Es - E_fit`
- `momenta`   : input momenta converted to `Float64`
"""
function soliton_dispersion(Es::AbstractVector{<:Real}, momenta::AbstractVector{<:Real})
    ps   = collect(Float64, momenta)
    Esq  = Es .^ 2
    sin2 = sin.(ps ./ 2) .^ 2

    # Two-parameter OLS:  E²(p) = M² + C sin²(p/2)
    n      = length(ps)
    x_mean = mean(sin2)
    y_mean = mean(Esq)
    C      = sum((sin2 .- x_mean) .* (Esq .- y_mean)) /
             sum((sin2 .- x_mean) .^ 2)
    M2     = y_mean - C * x_mean

    if M2 < 0
        @warn "soliton_dispersion: M² = $(round(M2; sigdigits=4)) < 0; " *
              "excitations may not follow the soliton dispersion."
        M2 = 0.0
    end

    M_fit     = sqrt(M2)
    c2_eff    = C / 4.0       # effective lattice velocity squared
    E_fit     = sqrt.(max.(M2 .+ C .* sin2, 0.0))
    residuals = Es .- E_fit

    return (; mass=M_fit, C, c2_eff, E_fit, residuals, momenta=ps)
end
