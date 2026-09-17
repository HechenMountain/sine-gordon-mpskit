using SpecialFunctions: gamma

"""
    C_RENORM

Lattice ⇄ CFT scheme-matching constant `C = 8 / (2 e^{-γ}) = 4 e^γ`
relating the bare lattice cosine to the conformally-normalized
vertex operator.
"""
const C_RENORM = 4 * exp(Base.MathConstants.eulergamma)

"""
    bare_mass(a, β; m_ref=1.0, a_ref=1.0)

Bare lattice mass to use in the Hamiltonian so that the renormalized coupling is
held fixed in the continuum limit: `m_bare(a) = m_ref · (a_ref / a)^α`,
`α = β²/(8π)`.

Arguments
- `a`     : lattice spacing
- `β`     : coupling constant
- `m_ref` : bare mass at the reference spacing
- `a_ref` : reference lattice spacing

Returns
Bare lattice mass at spacing `a`.
"""
bare_mass(a::Real, β::Real; m_ref::Real=1.0, a_ref::Real=1.0) =
    m_ref * (a_ref / a)^(β^2 / (8π))

"""
    renormalized_mass(m_bare, a, β)

Renormalized mass entering Zamolodchikov's formula,
`m_ren = m_bare · (a / C_RENORM)^α`, `α = β²/(8π)`.

Arguments
- `m_bare` : bare lattice mass
- `a`      : lattice spacing
- `β`      : coupling constant

Returns
Renormalized mass in the conformal normalization.
"""
renormalized_mass(m_bare::Real, a::Real, β::Real) =
    m_bare * (a / C_RENORM)^(β^2 / (8π))

"""
    M_zamolodchikov(m_ren, β)

Exact Zamolodchikov soliton mass from the renormalized mass `m_ren`.

Arguments
- `m_ren` : renormalized mass
- `β`     : coupling constant

Returns
Exact soliton mass.
"""
function M_zamolodchikov(m_ren::Real, β::Real)
    α      = β^2 / (8π)
    ξ      = β^2 / (8π - β^2)
    prefac = (2 / sqrt(π)) * gamma(ξ / 2) / gamma((1 + ξ) / 2)
    brak   = (π * m_ren^2 / (2β^2)) * (gamma(1 - α) / gamma(α))
    return prefac * brak^(1 / (2 - 2α))
end

"""
    breather_masses_zamolodchikov(M_sol, β)

Breather (soliton–anti-soliton bound state) masses `M_n = 2 M_sol sin(nπξ/2)`,
`n = 1, …, ⌊1/ξ⌋`, `ξ = β²/(8π−β²)`.  Returns an empty vector for `ξ ≥ 1`
(`β² ≥ 4π`), where no bound states exist.

Arguments
- `M_sol` : soliton mass
- `β`     : coupling constant

Returns
Vector of breather masses ordered by breather number.
"""
function breather_masses_zamolodchikov(M_sol::Real, β::Real)
    ξ   = β^2 / (8π - β^2)
    N_B = floor(Int, 1 / ξ)
    return [2 * M_sol * sin(n * π * ξ / 2) for n in 1:N_B]
end


# Lattice geometry — holding the physical wavepacket setup fixed as a → 0
#
#     box length     L = N·a       ⇒   N(a) = N_ref · (a_ref / a)   (rounded)
#     envelope width W = σ·a       ⇒   σ(a) = σ_ref · (a_ref / a)
#     momentum       k = p / a     ⇒   p(a) = p_ref · (a / a_ref)
# Fractional positions (x_K = N÷3, …) follow N and are taken as integers in the calling scripts.

"""
    n_sites(a, N_ref; a_ref=1.0) → Int

Number of lattice sites at spacing `a` that holds the physical box length
`L = N·a` fixed at its reference value `N_ref·a_ref`:
`N(a) = round(N_ref · a_ref / a)`.

Arguments
- `a`     : lattice spacing
- `N_ref` : reference number of sites
- `a_ref` : reference lattice spacing

Returns
Rounded number of sites at spacing `a`.
"""
n_sites(a::Real, N_ref::Integer; a_ref::Real=1.0) = round(Int, N_ref * a_ref / a)

"""
    lattice_width(a, σ_ref; a_ref=1.0) → Float64

Gaussian envelope width (lattice units) at spacing `a` that holds the physical
width `σ·a` fixed: `σ(a) = σ_ref · a_ref / a`.

Arguments
- `a`     : lattice spacing
- `σ_ref` : reference width in lattice units
- `a_ref` : reference lattice spacing

Returns
Gaussian width in lattice units at spacing `a`.
"""
lattice_width(a::Real, σ_ref::Real; a_ref::Real=1.0) = σ_ref * a_ref / a

"""
    lattice_momentum(a, p_ref; a_ref=1.0) → Float64

Lattice momentum at spacing `a` that holds the physical momentum `k = p/a`
fixed: `p(a) = p_ref · a / a_ref`  (convention `p_lat = k·a`, matching the
dispersion grid in continuum_scan.jl).

Arguments
- `a`     : lattice spacing
- `p_ref` : reference lattice momentum
- `a_ref` : reference lattice spacing

Returns
Lattice momentum at spacing `a`.
"""
lattice_momentum(a::Real, p_ref::Real; a_ref::Real=1.0) = p_ref * a / a_ref
