using MKL
using ThreadPinning
let _p = get(ENV, "JULIA_PIN", "cores")
    _p == "none" || pinthreads(Symbol(_p))   # NERSC packs 4 procs/node → set JULIA_PIN=affinitymask
end

include(joinpath(@__DIR__, "..", "src", "sine_gordon.jl"))

# Parameters
# Every knob below is read from the environment via `env_get(NAME, default)`,
# with the previous hard-coded value as the default — so a job is fully specified
# at submit time and this file is never edited to change a parameter point:
#
#   sbatch --export=ALL,A_VAL=0.75,D_VAL=30,CHI=80,CHI_VAC=64 \
#          scripts/run_soliton_scattering.slurm
#
# This matters for the resume path below: the parameter tag names the checkpoint
# file, so editing a parameter here between two requeues of the SAME job would
# change the tag, miss the checkpoint and silently restart the evolution.
# Renormalization / continuum reference  (a_ref is also the reference spacing for
# the geometry rescaling below, so mass and geometry stay in the same convention).
const RENORM_MASS = env_get("RENORM_MASS", true)
const a_ref       = env_get("A_REF", 1.0)
const m_ref       = env_get("M_REF", 1.0)

# Lattice parameters
const a_val = env_get("A_VAL", 1.0)   # actual lattice spacing
const d_val = env_get("D_VAL", 25)    # Fock-space truncation (modes per site)

# Reference system, defined at a_ref.  The lattice values below are rescaled to
# a_val so the PHYSICAL box length L = N·a, envelope width σ·a and momentum p/a
# stay fixed as a → 0 (see renormalization.jl: n_sites / lattice_width /
# lattice_momentum).  At a_val = a_ref they reduce to (N, σ, p0) = (120, 8.0, 0.8).
const N_ref  = env_get("N_REF", 120)        # sites at a_ref  ⇒  box length L = N_ref·a_ref
const σ_ref  = env_get("SIGMA_REF", 8.0)    # Gaussian envelope width at a_ref (lattice units)
const p0_ref = env_get("P0_REF", 0.8)       # central wavepacket momentum at a_ref

const N_val  = n_sites(a_val, N_ref; a_ref)            # integer; holds L = N·a fixed
const σ_val  = lattice_width(a_val, σ_ref; a_ref)      # holds physical width σ·a fixed
const p0_val = lattice_momentum(a_val, p0_ref; a_ref)  # holds physical momentum p/a fixed

# Model parameters
const β_val = env_get("BETA", 2.5)
const m_val = RENORM_MASS ? bare_mass(a_val, β_val; m_ref=m_ref, a_ref=a_ref) : float(m_ref)

# Vacuum parameters
const χ_vac   = env_get("CHI_VAC", 15)      # Bond dimension for vacuum iMPS
const TOL_VAC = env_get("TOL_VAC", 1e-5)    # VUMPS convergence tolerance

# Wavepacket parameters
const χ_val    = env_get("CHI", 64)         # Max. bond dimension (raise for small a)
const X_K_FRAC = env_get("X_K_FRAC", 1/3)   # soliton position as a fraction of the chain
const x_K      = round(Int, N_val * X_K_FRAC)  # Initial soliton position (sites)
const x_AK     = N_val - x_K                   # Initial antisoliton position (sites)

# TDVP parameters
const T_val      = env_get("T_FINAL", 60.0)   # Total simulation time
const dt_val     = env_get("DT", 0.5)         # TDVP time step
const SAVE_EVERY = env_get("SAVE_EVERY", 1)   # save measurements every N TDVP steps
const TWO_SITE   = env_get("TWO_SITE", true)  # false = single-site TDVP (fast); true = TDVP2 (adapts χ)

# Run directory / resume
# To continue a cancelled run, set the RESUME_DIR environment variable to its run
# folder; the latest state checkpoint there is reloaded and TDVP continues.
# Otherwise `resolve_run_dir` returns the folder this SLURM job already owns (so
# an automatic requeue picks up where it left off) or a fresh dated one.  
# A resume then auto-triggers exactly when a state checkpoint
# for this parameter tag lives in the resolved folder.  The tag encodes
# β/p0/m/a/χ/d/N/dt, so changing any of those yields a distinct tag and never
# resumes onto an incompatible checkpoint.
const STATE_EVERY = env_get("STATE_EVERY", 1)  # save full MPS state every N TDVP steps

# File tag that uniquely identifies this parameter point (also names checkpoints).
tag = @sprintf("b%.2f_p%.2f_m%.2f_a%.2f_chi%d_d%d_N%d_dt%.3f",
               β_val, p0_val, m_val, a_val, χ_val, d_val, N_val, dt_val)

const RUN_DIR    = resolve_run_dir(joinpath(DATA_DIR, "soliton_scattering"))
const STATE_PATH = joinpath(RUN_DIR, "ckpt_state_$(tag).jld2")
const RESUMING   = isfile(STATE_PATH)

# Logging

log_path = init_log("scattering")
vcapture() do
    # Diagnostic only — ThreadPinning.threadinfo()'s topology visualization can
    # throw on some nodes (SysInfo hyperthread classification, e.g. Perlmutter's
    # 256-logical-CPU layout). Keep it non-fatal so a logging glitch never kills
    # the run.
    try
        threadinfo(; slurm=true, blas=true, hints=true)
    catch e
        println("threadinfo() failed (non-fatal): ", e)
    end
end
vprint("Run dir: $RUN_DIR")

vprint("═"^80)
vprint("  Soliton–anti-soliton scattering  ")
vprint("-"^80)
vprint("  Model parameters      : β = $β_val, m_bare = $m_val")
vprint("  Renormalization       : RENORM_MASS = $RENORM_MASS, m_ref = $m_ref, a_ref = $a_ref")
vprint("  Lattice parameters    : a = $a_val, N = $N_val, d = $d_val")
vprint("  Reference system      : N_ref = $N_ref, σ_ref = $σ_ref, p0_ref = $p0_ref")
vprint("  Vacuum parameters     : χ_v = $χ_vac, tol = $TOL_VAC")
vprint("  Wavepacket parameters : χ = $χ_val, p0 = $p0_val, σ = $σ_val, x_K = $x_K, x_AK = $x_AK")
vprint("  TDVP parameters       : T = $T_val, dt = $dt_val, save_every = $SAVE_EVERY, two_site = $TWO_SITE")
vprint("  Resume                : state_every = $STATE_EVERY, resuming = $RESUMING")
vprint("═"^80)

if RESUMING
    # Resume — skip vacuum + wavepacket setup, reload state from the checkpoint
    vprint("\n[resume] State checkpoint found — skipping vacuum + wavepacket setup.")
    vprint("[resume] Continuing TDVP from $STATE_PATH")
    flush(stdout)
    result = @timed run_scattering(;
        d       = d_val,
        a       = a_val,
        m       = m_val,
        β       = β_val,
        χ       = χ_val,
        dt      = dt_val,
        T_final = T_val,
        save_every = SAVE_EVERY,
        two_site   = TWO_SITE,
        verbosity  = 1,
        checkpoint_dir = RUN_DIR,
        checkpoint_tag = tag,
        state_every    = STATE_EVERY,
        resume         = true,
    )
    elapsed = result.time
    result  = result.value
    vprint(@sprintf("  TDVP (resumed) completed in %.4f seconds.", elapsed))
else
    # Step 1 — Find vacua
    vprint("\n[1/3] Finding vacua ...")

    vprint("  Vacuum 0 (⟨φ⟩ ≈ 0) ...")
    flush(stdout)
    result = @timed find_vacuum(; d=d_val, a=a_val, m=m_val, β=β_val, χ=χ_vac,
                                  tol=TOL_VAC, verbosity=0)
    vac0    = result.value
    elapsed = result.time

    vprint(@sprintf("  E₀ = %.8f   ⟨φ⟩ = %.4f, computed in %.4f seconds\n", vac0.energy, vacuum_phi(vac0.psi, d_val), elapsed))

    vprint(@sprintf("  Vacuum 1 (⟨φ⟩ ≈ 2π/β = %.4f) ...", 2π/β_val))
    result = @timed find_vacuum1(; d=d_val, a=a_val, m=m_val, β=β_val, χ=χ_vac,
                                tol=TOL_VAC, verbosity=0)
    vac1 = result.value
    elapsed = result.time

    vprint(@sprintf("  E₁ = %.8f   ⟨φ⟩ = %.4f, computed in %.4f seconds\n", vac1.energy, vacuum_phi(vac1.psi, d_val), elapsed))

    dE_vac = abs(vac0.energy - vac1.energy)
    vprint(@sprintf("  ΔE(vac0,vac1) = %.2e  %s\n", dE_vac,
            dE_vac < 1e-2 ? "OK" : "WARNING: large vacuum splitting — check χ_vac"))

    # Step 2 — Build two-particle wavepacket
    vprint("\n[2/3] Building soliton–anti-soliton wavepacket ...")
    flush(stdout)
    result = @timed soliton_antisoliton_wavepacket(;
        H       = vac0.H,
        ψ_vac0  = vac0.psi,
        ψ_vac1  = vac1.psi,
        N=N_val, d=d_val, a=a_val, m=m_val, β=β_val,
        x_K, x_AK, p0=p0_val, σ=σ_val, χ=χ_val,
        verbosity=1,
    )
    wp = result.value
    elapsed = result.time
    ψ       = wp.psi
    H_finite = wp.H_finite

    χ_max_init = maximum(dim, right_virtualspace.(Ref(ψ), 1:N_val))
    vprint(@sprintf("  FiniteMPS: N=%d  χ_max=%d\n", length(ψ), χ_max_init))

    # Sanity check: initial energy
    E_init = real(sum(expectation_value(ψ, H_finite)))
    vprint(@sprintf("  E_total(t=0) = %.8f\n", E_init))
    vprint(@sprintf("  Wavepacket built in %.4f seconds.", elapsed))
    @assert isfinite(E_init) "Energy is NaN/Inf — state construction failed"
    flush(stdout)

    # Vacuum energy-density baseline ε_vac(j) — computed here from vac0 and
    # carried into the resume checkpoint via `extras`, so a resumed run needs no
    # VUMPS.  Both ends sit in vacuum 0 here (ϕ_L = ϕ_R = 0).
    ε_vac = vacuum_energy_density(vac0.psi, N_val; d=d_val, a=a_val, m=m_val, β=β_val)

    # Step 3 — Run scattering (TDVP time evolution)
    vprint("\n[3/3] Running TDVP time evolution ...")
    flush(stdout)
    result = @timed run_scattering(ψ, H_finite;
        d       = d_val,
        a       = a_val,
        m       = m_val,
        β       = β_val,
        χ       = χ_val,
        dt      = dt_val,
        T_final = T_val,
        save_every = SAVE_EVERY,
        two_site   = TWO_SITE,
        verbosity  = 1,
        checkpoint_dir = RUN_DIR,
        checkpoint_tag = tag,
        state_every    = STATE_EVERY,
        extras         = (; ε_vac),
    )
    elapsed = result.time
    result  = result.value
    vprint(@sprintf("  TDVP completed in %.4f seconds.", elapsed))
end


# Save results

vprint("\nSaving results ...")

# Scalar time series: one row per saved time step
save_data("scalars_$tag.tsv",
    ["time", "E_total", "P_total"],
    hcat(result.times, result.total_energy, result.total_momentum);
    dir=RUN_DIR)

# Energy density: rows = sites (1:N), columns = saved time snapshots
# Header encodes the time values; first column is the site index j.
save_data("energy_density_$tag.tsv",
    vcat(["j"], [@sprintf("t=%.4f", t) for t in result.times]),
    hcat(collect(1:N_val), result.energy_density);
    dir=RUN_DIR)

# Vacuum energy-density baseline ε_vac(j): subtract from ε(j,t) in the plots to
# remove the static bulk offset and the boundary spike, leaving the soliton excess
# energy + dynamics.  Sourced from the run result (carried through the resume
# checkpoint), so it is identical on fresh and resumed runs.  Both ends sit in
# vacuum 0 here (ϕ_L = ϕ_R = 0).
# Name prefixed "vac_" so it does NOT match the energy_density_*.tsv glob the
# notebooks use to load the (sites × times) grid.
save_data("vac_energy_density_$tag.tsv",
    ["j", "e_vac"],
    hcat(collect(1:N_val), result.extras.ε_vac);
    dir=RUN_DIR)

# Field profile: ⟨φ(j, t)⟩, same layout as energy density
save_data("field_profile_$tag.tsv",
    vcat(["j"], [@sprintf("t=%.4f", t) for t in result.times]),
    hcat(collect(1:N_val), result.field_profile);
    dir=RUN_DIR)

# Topological charge density: ρ(j, j+1, t) = β/(2πa)⟨φ_{j+1} − φ_j⟩
# Bond-local: N−1 rows (j = 1, ..., N−1), same time columns as above.
save_data("charge_density_$tag.tsv",
    vcat(["j"], [@sprintf("t=%.4f", t) for t in result.times]),
    hcat(collect(1:N_val-1), result.charge_density);
    dir=RUN_DIR)

# Parameter record for this run
save_data("params_$tag.tsv",
    ["param", "value"],
    [
        "beta"         β_val;
        "p0"           p0_val;
        "p0_ref"       p0_ref;
        "m"            m_val;
        "a"            a_val;
        "a_ref"        a_ref;
        "chi"          χ_val;
        "d"            d_val;
        "N"            N_val;
        "dt"           dt_val;
        "T_final"      T_val;
        "sigma"        σ_val;
        "chi_vac"      χ_vac;
        "tol_vac"      TOL_VAC;
        "x_K"          x_K;
        "x_AK"         x_AK;
        "E_init"       result.total_energy[1];
        "dE_frac"      abs(result.total_energy[end] - result.total_energy[1]) / abs(result.total_energy[1]);
        "dP"           abs(result.total_momentum[end] - result.total_momentum[1]);
    ];
    dir=RUN_DIR)

vprint("\nAll data saved to $RUN_DIR/")


# Summary

vprint()
vprint("═══════════════════════════════════════════════════════════")
vprint("  DONE")
vprint("═══════════════════════════════════════════════════════════")
vprint(@sprintf("  %d time snapshots saved\n", length(result.times)))
vprint(@sprintf("  E_tot(t=0)   = %.8f\n", result.total_energy[1]))
vprint(@sprintf("  E_tot(t=end) = %.8f\n", result.total_energy[end]))
vprint(@sprintf("  P_tot(t=0)   = %.8f\n", result.total_momentum[1]))
vprint(@sprintf("  P_tot(t=end) = %.8f\n", result.total_momentum[end]))
vprint(@sprintf("  ΔE/E₀        = %.2e\n",
        abs(result.total_energy[end] - result.total_energy[1]) /
          abs(result.total_energy[1])))

close_log()
