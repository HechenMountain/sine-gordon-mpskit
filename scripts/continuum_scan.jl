using MKL

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

include(joinpath(@__DIR__, "..", "src", "sine_gordon.jl"))
MPSKit.Defaults.set_scheduler!(:dynamic)

# VUMPS / QP parameters
const d_soliton      = 90
const χ_vac       = 64        # VUMPS bond dimension (used for both vacua and QP)
const tol_vac     = 1e-6      # VUMPS convergence tolerance
const maxiter_vac = 200       # maximum VUMPS iterations
const N_BREATHERS_MAX = 3     # breather branches to compute per task

const RENORM_MASS = true
const m_ref   = 1.0                 # bare mass at a = a_ref  (physical reference)
const a_ref   = 1.0                 # reference lattice spacing

# Dispersion scan: momenta passed to find_soliton_excitation.
# p = 0 is always included; DISP_NPTS symmetric non-zero points span the
# physical momentum range k ∈ [-DISP_KPHYS_MAX, +DISP_KPHYS_MAX] (units of m).
# Lattice momenta are p = k·a, so finer lattices use smaller lattice momenta,
# keeping all runs in the same continuum regime — this lets the dispersion
# relation E(k) itself be checked for continuum convergence across lattice spacings.
const DISP_KPHYS_MAX = 1.2   # max physical momentum for dispersion fit (units of m_soliton)
const DISP_NPTS      = 10     # number of non-zero dispersion points (symmetric about 0)
disp_momenta(a) = vcat([0.0], collect(range(-DISP_KPHYS_MAX * a, DISP_KPHYS_MAX * a; length=DISP_NPTS)))

# Scan parameters

const β_list  = [1.0, 1.5, 2.0, 2.5, 3.0]   # β values to scan  (β=0.1 underflows: exp(-π²/β²)→0)
const a_list  = [2.0, 1.9, 1.8, 1.7, 1.6, 1.5, 1.4, 1.3, 1.2, 1.1, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]

# Renormalization (bare_mass, renormalized_mass, M_zamolodchikov,
# breather_masses_zamolodchikov, C_RENORM) lives in src/renormalization.jl.

# Flat task list: one entry per (β, a) combination — 5 × 10 = 50 total.
const task_list = [(β=β, a=a) for β in β_list for a in a_list]
const N_tasks   = length(task_list)

# Optional: run only one entry (1-based CLI argument, or SLURM_ARRAY_TASK_ID).
const _task_str = isempty(ARGS) ? get(ENV, "SLURM_ARRAY_TASK_ID", "") : ARGS[1]
const tasks_run = isempty(_task_str) ? task_list : [task_list[parse(Int, _task_str)]]


# Logging

_logtag = length(tasks_run) == 1 ?
    @sprintf("continuum_scan_a%.3f_beta%.3f", tasks_run[1].a, tasks_run[1].β) :
    "continuum_scan"
init_log(_logtag)


# Runner functions

"""
    run_vacua(; a, β, m) → NamedTuple

Find both degenerate vacua via VUMPS.
Returns `(psi0, psi1, H, e0, e1, phi0, phi1, delta_e)`.
"""
function run_vacua(; a::Float64, β::Float64, m::Float64)
    tprint("  [VUMPS] vacuum 0  (⟨φ⟩ ≈ 0) ...")
    vac0 = find_vacuum(;
        d=d_soliton, a=a, m=m, β=β, χ=χ_vac,
        tol=tol_vac, maxiter=maxiter_vac, verbosity=0,
    )
    φ0 = vacuum_phi(vac0.psi, d_soliton)
    tprint(@sprintf("    ε₀ = %.10g   ⟨φ⟩ = %.6g", vac0.energy, φ0))

    tprint(@sprintf("  [VUMPS] vacuum 1  (⟨φ⟩ ≈ %.4g) ...", 2π / β))
    vac1 = find_vacuum1(;
        d=d_soliton, a=a, m=m, β=β, χ=χ_vac,
        tol=tol_vac, maxiter=maxiter_vac, verbosity=0,
    )
    φ1 = vacuum_phi(vac1.psi, d_soliton)
    tprint(@sprintf("    ε₁ = %.10g   ⟨φ⟩ = %.6g  (target: %.4g)",
                    vac1.energy, φ1, 2π / β))

    Δe = abs(vac0.energy - vac1.energy)
    tprint(@sprintf("    Δε = |ε₀−ε₁| = %.2e  %s",
                    Δe, Δe < 1e-2 ? "OK" : "WARNING: large vacuum splitting"))

    return (; psi0=vac0.psi, psi1=vac1.psi, H=vac0.H,
              e0=vac0.energy, e1=vac1.energy,
              phi0=φ0, phi1=φ1, delta_e=Δe)
end


"""
    run_qp_soliton(H, ψ_vac0, ψ_vac1; a, β, m) → NamedTuple

Compute the soliton rest mass and dispersion using the QuasiparticleAnsatz.

Calls `find_soliton_excitation` once over `disp_momenta(a)`, which includes
p = 0 as its first entry.  M_QP = E(p=0) is read directly from that result;
no second call is needed.  Returns `(M_QP, M_fit, C, c2_eff, Es, momenta)`.
"""
function run_qp_soliton(H, ψ_vac0::InfiniteMPS, ψ_vac1::InfiniteMPS;
                     a::Float64, β::Float64, m::Float64)
    momenta = disp_momenta(a)   # momenta[1] = 0.0 by construction
    tprint(@sprintf("  [QP] soliton excitation + dispersion (%d momenta, k_phys ∈ [0, ±%.3g]) ...",
                    length(momenta), DISP_KPHYS_MAX))
    Es_disp, _ = find_soliton_excitation(H, ψ_vac0, ψ_vac1;
                                      momenta=momenta, num=1, verbosity=0)
    Es_vec = real.(Es_disp[:, 1])
    M_QP   = Es_vec[1]   # p = 0 is first entry
    tprint(@sprintf("    M_QP(p=0) = %.8g", M_QP))

    disp = soliton_dispersion(Es_vec, momenta)
    tprint(@sprintf("    M_fit = %.8g   C = %.4g   c²_eff = %.4g   max|res| = %.2e",
                    disp.mass, disp.C, disp.c2_eff, maximum(abs, disp.residuals)))

    return (; M_QP=M_QP, M_fit=disp.mass, C=disp.C, c2_eff=disp.c2_eff,
              Es=Es_vec, momenta=momenta)
end


# Breather runner

"""
    run_breather(H, ψ_vac; β, m_ren) → NamedTuple

Compute the `N_BREATHERS_MAX` lightest breather masses using the
QuasiparticleAnsatz in the vacuum sector (single-vacuum form).

Returns `(; M_QP, M_Zam, ratio, num)` where each vector has length
`num = min(N_BREATHERS_MAX, N_B)` and `N_B = ⌊1/ξ⌋`, ξ = β²/(8π−β²).
"""
function run_breather(H, ψ_vac::InfiniteMPS; β::Float64, m_ren::Float64)
    ξ   = β^2 / (8π - β^2)
    N_B = floor(Int, 1 / ξ)
    num = min(N_BREATHERS_MAX, N_B)

    if num == 0
        tprint("  [QP] no bound-state breathers for β=$(β) (ξ=$(round(ξ; digits=4)) ≥ 1)")
        return (; M_QP=Float64[], M_Zam=Float64[], ratio=Float64[], num=0)
    end

    tprint(@sprintf("  [QP] breather excitations  (num=%d of N_B=%d, ξ=%.4g) ...",
                    num, N_B, ξ))
    Es_br, _ = find_breather_excitations(H, ψ_vac; momenta=[0.0], num=num, verbosity=0)
    M_QP     = [real(Es_br[1, n]) for n in 1:num]

    M_sol = M_zamolodchikov(m_ren, β)
    M_Zam = breather_masses_zamolodchikov(M_sol, β)[1:num]

    for n in 1:num
        tprint(@sprintf("    n=%d:  M_QP=%.6g  M_Zam=%.6g  ratio=%.4f",
                        n, M_QP[n], M_Zam[n], M_QP[n] / M_Zam[n]))
    end
    return (; M_QP=M_QP, M_Zam=M_Zam, ratio=M_QP ./ M_Zam, num=num)
end


# File I/O helpers

"Write scalar results for one (a, β, m_bare) point to a single-row TSV."
function write_scalars(path, a, β, m_bare, m_ref_val, vr, kr)
    M_cl  = 8m_ref_val / β^2   # classical mass uses physical m_ref, not m_bare
    M_Zam = M_zamolodchikov(renormalized_mass(m_bare, a, β), β)   # m_ren via eq:mren
    open(path, "w") do io
        println(io, "a\tbeta\tm_bare\tm_ref\te_vumps0\te_vumps1\tdelta_e\t" *
                    "phi_vac0\tphi_vac1\t" *
                    "M_QP\tM_fit\tC\tc2_eff\tM_cl\tM_QP_over_Mcl\tM_Zam\tM_QP_over_MZam")
        @printf(io,
            "%.6g\t%.6g\t%.10g\t%.6g\t%.10g\t%.10g\t%.2e\t%.6g\t%.6g\t%.10g\t%.10g\t%.6g\t%.6g\t%.10g\t%.6g\t%.10g\t%.6g\n",
            a, β, m_bare, m_ref_val,
            vr.e0, vr.e1, vr.delta_e,
            vr.phi0, vr.phi1,
            kr.M_QP, kr.M_fit, kr.C, kr.c2_eff,
            M_cl, kr.M_QP / M_cl, M_Zam, kr.M_QP / M_Zam)
    end
end

"Write E(p) dispersion data for one (a, β, m) point."
function write_dispersion(path, a, β, m, kr)
    open(path, "w") do io
        println(io, "p_lat\tp_phys\tE_QP\tE_fit")
        disp_E_fit = sqrt.(max.(kr.M_fit^2 .+ kr.C .* sin.(kr.momenta ./ 2).^2, 0.0))
        for i in eachindex(kr.momenta)
            @printf(io, "%.6g\t%.6g\t%.10g\t%.10g\n",
                    kr.momenta[i], kr.momenta[i] / a, kr.Es[i], disp_E_fit[i])
        end
    end
end

"Write breather masses for one (a, β) point."
function write_breathers(path, a, β, br)
    open(path, "w") do io
        println(io, "n\tM_QP\tM_Zam\tratio")
        for n in 1:br.num
            @printf(io, "%d\t%.10g\t%.10g\t%.6g\n",
                    n, br.M_QP[n], br.M_Zam[n], br.ratio[n])
        end
    end
end


# Main scan

# Date-stamped run directory: data/continuum_scan/YYYYMMDD[_n]/ (see helpers.jl)
out_dir = make_dated_run_dir(joinpath(@__DIR__, "..", "data", "continuum_scan"))

tprint("═"^75)
tprint(" MPSKit continuum-limit scan  (soliton mass via QuasiparticleAnsatz)")
tprint(@sprintf("  d=%-4d  χ_vac=%-4d  tol_vac=%.1e  maxiter_vac=%-4d",
    d_soliton, χ_vac, tol_vac, maxiter_vac))
tprint(@sprintf("  RENORM_MASS = %s  →  m_bare(a, β) = m_ref·(a_ref/a)^{β²/(8π)},  C = %.4f",
    RENORM_MASS, C_RENORM))
tprint(@sprintf("  m_ref = %.4g,  a_ref = %.4g", m_ref, a_ref))
tprint(@sprintf("  output dir: %s", out_dir))
tprint(@sprintf("  β values: %s", join(β_list, ", ")))
tprint(@sprintf("  a values: %s", join(a_list, ", ")))
tprint(@sprintf("  N_tasks total: %d  (running: %d)", N_tasks, length(tasks_run)))
tprint(@sprintf("  dispersion: k_phys ∈ [±%.3g]  (%d non-zero pts)",
    DISP_KPHYS_MAX, DISP_NPTS))
tprint("═"^75)

summary_rows = NamedTuple[]

for task in tasks_run
    a_soliton = task.a
    β_cur  = task.β
    m_cur  = RENORM_MASS ? bare_mass(a_soliton, β_cur; m_ref=m_ref, a_ref=a_ref) : float(m_ref)
    M_cl   = 8m_ref / β_cur^2   # classical soliton mass (physical m_ref, not m_bare)

    tprint("\n── β = $β_cur,  a = $a_soliton,  m_bare = $(round(m_cur; digits=6)) " *
           "─"^max(0, 40 - length("── β = $β_cur,  a = $a_soliton ")))

    # Step 1: both degenerate vacua
    t_vac = @elapsed vr = run_vacua(; a=a_soliton, β=β_cur, m=m_cur)
    tprint(@sprintf("  [time] vacua = %.1f s", t_vac))

    # Step 2: soliton mass via QuasiparticleAnsatz
    t_qp = @elapsed kr = run_qp_soliton(vr.H, vr.psi0, vr.psi1; a=a_soliton, β=β_cur, m=m_cur)
    tprint(@sprintf("  [time] QP soliton + dispersion = %.1f s", t_qp))
    M_Zam = M_zamolodchikov(renormalized_mass(m_cur, a_soliton, β_cur), β_cur)   # m_ren via eq:mren
    tprint(@sprintf("  M_QP/M_cl = %.6g   M_QP/M_Zam = %.6g   (M_cl = %.6g, M_Zam = %.6g, m_bare = %.6g)",
                    kr.M_QP / M_cl, kr.M_QP / M_Zam, M_cl, M_Zam, m_cur))

    # Step 3: breather masses via QuasiparticleAnsatz (vacuum sector, p = 0)
    t_br = @elapsed br = run_breather(vr.H, vr.psi0; β=β_cur, m_ren=renormalized_mass(m_cur, a_soliton, β_cur))
    tprint(@sprintf("  [time] breather = %.1f s", t_br))

    # Write output files — tag omits m (it is stored inside the TSV)
    tag             = @sprintf("a%.3f_beta%.3f", a_soliton, β_cur)
    scalar_file     = joinpath(out_dir, "scalars_$(tag).tsv")
    dispersion_file = joinpath(out_dir, "dispersion_$(tag).tsv")
    breather_file   = joinpath(out_dir, "breathers_$(tag).tsv")

    write_scalars(scalar_file, a_soliton, β_cur, m_cur, m_ref, vr, kr)
    write_dispersion(dispersion_file, a_soliton, β_cur, m_cur, kr)
    write_breathers(breather_file, a_soliton, β_cur, br)

    tprint("  → $(basename(scalar_file))")
    tprint("  → $(basename(dispersion_file))")
    tprint("  → $(basename(breather_file))")

    push!(summary_rows, (
        β        = β_cur,
        a        = a_soliton,
        m_bare   = m_cur,
        M_cl     = M_cl,
        M_Zam    = M_Zam,
        e0       = vr.e0,
        delta_e  = vr.delta_e,
        M_QP     = kr.M_QP,
        M_fit    = kr.M_fit,
        C        = kr.C,
        c2_eff   = kr.c2_eff,
    ))
end

# ─── Summary table ────────────────────────────────────────────────────────────
tprint("\n" * "═"^117)
tprint(" SCAN SUMMARY  (QuasiparticleAnsatz soliton mass, RENORM_MASS = $RENORM_MASS)")
tprint("═"^117)
tprint(@sprintf("  %-5s  %-5s  %-10s  %-14s  %-10s  %-14s  %-12s  %-12s  %-8s  %-8s",
    "β", "a", "m_bare", "ε0", "Δε(vac)", "M_QP(p=0)", "M_QP/M_cl", "M_QP/M_Zam", "C", "c²_eff"))
tprint("-"^117)
for r in summary_rows
    tprint(@sprintf("  %-5.3f  %-5.3f  %-10.6g  %-14.8g  %-10.2e  %-14.8g  %-12.6g  %-12.6g  %-8.4f  %-8.4f",
        r.β, r.a, r.m_bare, r.e0, r.delta_e, r.M_QP, r.M_QP / r.M_cl, r.M_QP / r.M_Zam, r.C, r.c2_eff))
end
tprint("═"^117)

close_log()
