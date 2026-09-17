#!/usr/bin/env python3
"""Generate figures from the published sine-Gordon TSV data."""

import argparse
from pathlib import Path
import os
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PARSER = argparse.ArgumentParser(description=__doc__)
PARSER.add_argument("--data-root", type=Path, default=Path(__file__).resolve().parent.parent / "data", help="Directory containing continuum_scan, soliton_free, and soliton_scattering")
PARSER.add_argument("--figure-dir", type=Path, default=Path(__file__).resolve().parent / "figures" / "scattering_plots", help="Output directory for figures")
PARSER.add_argument("--run", default="20260807", help="Scattering run directory")
ARGS = PARSER.parse_args()
DATA_BASE = ARGS.data_root.resolve()
FIGURE_DIR = ARGS.figure_dir.resolve()
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLBACKEND", "Agg")

RUN_DIR = ARGS.run


import os
import re
import glob
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.special import gamma
import matplotlib.animation as animation
from scipy.integrate import quad

matplotlib.rcParams.update({
    "font.family":     "serif",
    "font.size":       14,
    "axes.labelsize":  12,
    "legend.fontsize": 9,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "axes.grid":       False,
    "figure.dpi":      130,
})

DATA_ROOT = os.fspath(DATA_BASE / "soliton_scattering")
run_path = os.path.join(DATA_ROOT, RUN_DIR)

# ── Discover files ────────────────────────────────────────────────────────────
def find_file(pattern):
    matches = glob.glob(os.path.join(run_path, pattern))
    if not matches:
        raise FileNotFoundError(f"No file matching '{pattern}' in {run_path}")
    return matches[0]

def read_tsv(pattern, **kwargs):
    """Read a TSV with a commented header line: '# col1\tcol2\t...'."""
    f = find_file(pattern)
    with open(f) as fh:
        col_names = [c.strip() for c in fh.readline().strip().lstrip("#").split("\t")]
    return pd.read_csv(f, sep="\t", comment="#", names=col_names, **kwargs)

def read_params(path):
    """Read a params_*.tsv (commented '# param\tvalue' header) into a dict."""
    with open(path) as fh:
        cols = [c.strip() for c in fh.readline().strip().lstrip("#").split("\t")]
    df = pd.read_csv(path, sep="\t", comment="#", names=cols)
    return dict(zip(df.iloc[:, 0], df.iloc[:, 1]))

# ── Load parameters ───────────────────────────────────────────────────────────
params = read_params(find_file("params_*.tsv"))
beta = float(params["beta"])
p0   = float(params["p0"])
N    = int(params["N"])
a    = float(params["a"])
m    = float(params["m"])
chi  = int(params["chi"])
dt   = float(params["dt"])
sigma = float(params["sigma"])          # Gaussian envelope width (lattice/site units)
x_K0 = int(float(params["x_K"]))
x_AK0 = int(float(params["x_AK"]))

# Centroid tracking window (half-width, in SITES).  The packet width in sites
# grows as a → 0 (σ_sites = σ_ref·a_ref/a), so a window fixed in sites would clip
# the lump and make the centroid wiggle.  Tie it to the actual envelope width:
# W = 1.5·σ reproduces the old W=12 at a=1 (σ=8) and auto-scales with a.
W_cent = int(round(1.5 * sigma))

print(f"Run: {RUN_DIR}")
print(f"  β={beta}  p0={p0}  m={m}  a={a}  χ={chi}  N={N}  dt={dt}")
print(f"  σ={sigma:.2f} sites  →  centroid window W={W_cent} sites")

# ── Free-soliton reference run: auto-select by matching parameters ────────────────
# The phase-shift analysis subtracts a single-soliton "free" run taken at the SAME
# physical parameters.  Every data file embeds the parameter tag
#     b{β}_p{p0}_m{m}_a{a}_chi{χ}_d{d}_N{N}_dt{dt}
# (scripts: @sprintf("b%.2f_p%.2f_m%.2f_a%.2f_chi%d_d%d_N%d_dt%.3f", …)),
# so the matching free run is found by comparing tags — no hard-coded folder, and
# the scattering / free run dates may differ.  Match is on the PHYSICAL params
# (β,p0,m,a,d,N,dt); χ is a convergence knob, only warned on if it differs.
FREE_RUN_FALLBACK = "20260618"     # forced folder used only if no param match is found

_TAG_RE = re.compile(
    r"b(?P<b>\d+\.\d+)_p(?P<p>\d+\.\d+)_m(?P<m>\d+\.\d+)_a(?P<a>\d+\.\d+)"
    r"_chi(?P<chi>\d+)_d(?P<d>\d+)_N(?P<N>\d+)_dt(?P<dt>\d+\.\d+)")
_phys = lambda mt: (mt["b"], mt["p"], mt["m"], mt["a"], mt["d"], mt["N"], mt["dt"])

_scatt_key = _phys(_TAG_RE.search(os.path.basename(find_file("charge_density_*.tsv"))))
FREE_ROOT  = os.path.join(DATA_ROOT, "..", "soliton_free")

_matches = []
for _f in glob.glob(os.path.join(FREE_ROOT, "*", "charge_density_*.tsv")):
    _mt = _TAG_RE.search(os.path.basename(_f))
    if _mt and _phys(_mt) == _scatt_key:
        _matches.append((os.path.basename(os.path.dirname(_f)), int(_mt["chi"])))

if _matches:
    _matches.sort(key=lambda r: r[0])          # dir names are dates YYYYMMDD[_k]
    FREE_RUN_DIR, _free_chi = _matches[-1]     # most recent
    _src = "auto-matched by params"
    if _free_chi != chi:
        print(f"  ⚠ free-run χ={_free_chi} ≠ scattering χ={chi} (convergence may differ)")
    if len(_matches) > 1:
        print(f"  free-run candidates (by date): {[r[0] for r in _matches]}")
else:
    FREE_RUN_DIR, _src = FREE_RUN_FALLBACK, "FALLBACK — no parameter match found"

free_path = os.path.join(FREE_ROOT, FREE_RUN_DIR)
if not glob.glob(os.path.join(free_path, "charge_density_*.tsv")):
    raise FileNotFoundError(
        f"No charge_density_*.tsv in free reference '{free_path}'. "
        f"Set FREE_RUN_FALLBACK to a valid data/soliton_free/<dir>.")
print(f"  free reference: {FREE_RUN_DIR}  [{_src}]")

# ── Soliton rest mass M_QP (from the selected free run) ──────────────────────────
M_QP = float(read_params(glob.glob(os.path.join(free_path, "params_*.tsv"))[0])["M_soliton"])

# ── Analytic Zamolodchikov soliton mass ──────────────────────────────────────────
C_RENORM = 4 * np.exp(np.euler_gamma)
def renormalized_mass(m_bare, a, beta):
    return m_bare * (a / C_RENORM)**(beta**2 / (8 * np.pi))
def M_zamolodchikov(m_ren, beta):
    alpha  = beta**2 / (8 * np.pi)
    xi_    = beta**2 / (8 * np.pi - beta**2)
    prefac = (2 / np.sqrt(np.pi)) * gamma(xi_ / 2) / gamma((1 + xi_) / 2)
    brak   = (np.pi * m_ren**2 / (2 * beta**2)) * (gamma(1 - alpha) / gamma(alpha))
    return prefac * brak**(1 / (2 - 2 * alpha))
m_ren = renormalized_mass(m, a, beta)
M_zam = M_zamolodchikov(m_ren, beta)
print(f"  M_QP  = {M_QP:.6f}  (measured soliton rest mass E(p=0), from free run M_soliton)")
print(f"  M_zam = {M_zam:.6f}  (analytic Zamolodchikov, m_ren={m_ren:.4f})   "
      f"Δ = {M_zam - M_QP:+.6f}  ({100*(M_zam - M_QP)/M_QP:+.2f}%)")

# ── Load data files ───────────────────────────────────────────────────────────
def load_grid(pattern):
    """Load a (sites × times) TSV: first column = j, remaining columns = t=..."""
    df = read_tsv(pattern)
    js = df.iloc[:, 0].values
    times = np.array([float(c.split("=")[1]) for c in df.columns[1:]])
    data  = df.iloc[:, 1:].values      # shape (n_sites, n_times)
    return js, times, data

scalars_df = read_tsv("scalars_*.tsv")
times_sc   = scalars_df["time"].values
E_total    = scalars_df["E_total"].values
P_total    = scalars_df["P_total"].values

js_en, times_en, energy_density  = load_grid("energy_density_*.tsv")
js_fp, times_fp, field_profile   = load_grid("field_profile_*.tsv")
js_cd, times_cd, charge_density  = load_grid("charge_density_*.tsv")

print(f"  snapshots: {len(times_en)}, sites: {N}")


# Vacuum-referenced energy density: subtract the static baseline ε_vac(j) to
# remove the uniform bulk offset AND the boundary spike, leaving the soliton
# excess energy + dynamics (incl. the boundary radiation, which is a real
# feature of the state and is NOT removed by a static subtraction).
e_vac = read_tsv("vac_energy_density_*.tsv")["e_vac"].values
Z = energy_density - e_vac[:, None]
elabel = r"$\varepsilon(j,t) - \varepsilon_{\mathrm{vac}}(j)$"

fig, ax = plt.subplots(figsize=(9, 5))
vmax = np.percentile(Z, 99.5)
im = ax.pcolormesh(times_en, js_en, Z, cmap="inferno", shading="auto",
                   vmin=0.0, vmax=vmax)
fig.colorbar(im, ax=ax, label=elabel)
ax.set_xlabel("time $t$")
ax.set_ylabel("site $j$")
ax.set_title(rf"Energy density  —  $\beta={beta}$,  $p_0={p0}$,  $\chi={chi}$,  $N={N}$")
plt.tight_layout()
fig.savefig(FIGURE_DIR / "energy_density.pdf")
plt.close()


fig, ax = plt.subplots(figsize=(9, 5))
im = ax.pcolormesh(times_cd, js_cd, charge_density, cmap="RdBu", shading="auto")
fig.colorbar(im, ax=ax, label=r"$\rho(j,t)$")
ax.set_xlabel("$t$")
ax.set_ylabel("position")
# ax.set_title(rf"Charge density")
plt.tight_layout()
plt.close()
fig.savefig(os.path.join(FIGURE_DIR, "charge_density.pdf"))


# Integrate over soliton/anti-soliton regions to get the (dimensionless) topological
# charge.  The saved ρ(j,j+1) = β/(2πa)⟨φ_{j+1}−φ_j⟩ is a charge DENSITY (per unit
# physical length), so the charge is the integral  Q = Σ_j ρ_j · a  with the lattice
# measure dx = a·dj.  Without the factor a the bare sum telescopes to
# (β/2π)(φ_N−φ_1)/a = 1/a per soliton (≈ 1.33 at a=0.75) — the spurious >1 charge.
# Split at the midpoint of the initial soliton / anti-soliton positions; after
# transmission the soliton sits on the far side of the split from where it started.
j_split = 0.5 * (x_K0 + x_AK0)
t_meas = 50.0
Q_soliton     = a * np.sum(charge_density[(js_cd >= j_split)], axis=0)
Q_antisoliton = a * np.sum(charge_density[(js_cd <= j_split)], axis=0)
ti = np.argmin(np.abs(times_cd - t_meas))
print(f"At t={times_cd[ti]:.1f}: Q_soliton={Q_soliton[ti]:.6f}, Q_antisoliton={Q_antisoliton[ti]:.6f}")


n_snap = min(8, len(times_fp))
snap_idx = np.unique(np.round(np.linspace(0, len(times_fp) - 1, n_snap)).astype(int))

cmap = plt.cm.viridis
colors = [cmap(i / max(len(snap_idx) - 1, 1)) for i in range(len(snap_idx))]

fig, ax = plt.subplots(figsize=(9, 4))
for color, idx in zip(colors, snap_idx):
    ax.plot(js_fp, field_profile[:, idx], color=color, lw=1.5,
            label=f"$t={times_fp[idx]:.1f}$")
ax.axhline(0,          color="gray", lw=0.8, ls="--")
ax.axhline(2 * np.pi / beta, color="gray", lw=0.8, ls="--", label=r"$2\pi/\beta$")
ax.set_xlabel("site $j$")
ax.set_ylabel(r"$\langle\phi(j)\rangle$")
ax.set_title(rf"Field profile snapshots  —  $\beta={beta}$,  $p_0={p0}$,  $\chi={chi}$,  $N={N}$")
ax.legend(fontsize=9, ncol=2, loc="best")
plt.tight_layout()
fig.savefig(FIGURE_DIR / "field_profile.pdf")
plt.close()


E0 = E_total[0]
dE_frac = (E_total - E0) / abs(E0)

fig, ax = plt.subplots(figsize=(9, 3))
ax.plot(times_sc, dE_frac, lw=1.5, color="steelblue")
ax.axhline(0, color="gray", lw=0.8, ls="--")
ax.set_xlabel("time $t$")
ax.set_ylabel(r"$(E(t)-E_0)/|E_0|$")
ax.set_title(rf"Relative energy drift  —  $\beta={beta}$,  $p_0={p0}$,  $\chi={chi}$,  $N={N}$")
plt.tight_layout()
fig.savefig(FIGURE_DIR / "energy_drift.pdf")
plt.close()
print(f"Max |dE/E0| = {np.max(np.abs(dE_frac)):.2e}")


fig, ax = plt.subplots(figsize=(9, 3))
ax.plot(times_sc, P_total, lw=1.5, color="firebrick")
ax.axhline(0, color="gray", lw=0.8, ls="--")
ax.set_xlabel("time $t$")
ax.set_ylabel(r"$\langle P\rangle$")
ax.set_title(rf"Total momentum  —  $\beta={beta}$,  $p_0={p0}$,  $\chi={chi}$,  $N={N}$")
plt.tight_layout()
fig.savefig(FIGURE_DIR / "total_momentum.pdf")
plt.close()
print(f"Max |ΔP| = {np.max(np.abs(P_total - P_total[0])):.2e}")


# ── User-tunable windows (plateau regions of D(t); with a real free-run
#    reference the result is only weakly sensitive to these) ──────────────────
# t_pre_lo,  t_pre_hi  = 5.0, 10.0    # after initial SGF-leakage transient, before collision
# t_post_lo, t_post_hi = 45.0, 50.0   # after collision, before boundary effects
t_pre_lo,  t_pre_hi  = 5.0, 8.0    # after initial SGF-leakage transient, before collision
t_post_lo, t_post_hi = 45.0, 50.0   # after collision, before boundary effects


def zamolodchikov_phase(theta, xi):
    """Phase Φ(θ) of the Zamolodchikov soliton–soliton amplitude,
        S_ss(θ) = −exp(−i Φ(θ)),
    standard integral representation (Zamolodchikov & Zamolodchikov 1979;
    Mussardo Eq. 18.10.12):

        Φ(θ) = ∫₀^∞ dt/t · sin(θt) · sinh(π(1−ξ)t/2) / [sinh(πξt/2) cosh(πt/2)]

    with ξ = β²/(8π−β²), θ the relative rapidity. NOTE: the earlier
    "typo fix" that swapped sinh↔cosh in the denominator was wrong — the
    correct denominator must carry sinh(πξt/2): in the classical limit ξ→0
    the phase has to diverge ∝ 1/ξ (phase = classical action / ℏ, ξ ∝ ℏ),
    which together with M ∝ 1/ξ leaves a finite classical displacement
    Δx = φ'/E. The swapped version stays finite at ξ→0 ⇒ vanishing
    classical displacement — unphysical. (The free-fermion check ξ=1,
    S=−1 passes for both versions and cannot discriminate.)
    """
    if xi <= 0 or theta <= 0:
        return np.nan
    def integrand(t):
        if t < 1e-10:
            return theta * (1.0 - xi) / xi   # t→0 limit of the integrand
        return (np.sin(theta * t) * np.sinh(np.pi * (1 - xi) * t / 2)
                / (np.sinh(np.pi * xi * t / 2) * np.cosh(np.pi * t / 2) * t))
    # integrand decays ∝ e^{−πξt}: cutoff at 60/(πξ) ⇒ residual ~e^{−60}
    val, _ = quad(integrand, 0, 60/(np.pi*xi), limit=500, epsabs=1e-12)
    return val

def S_T_amplitude(theta, xi):
    """Full KK̄ transmission amplitude
        S_T(θ) = sinh(θ/ξ)/sinh((iπ−θ)/ξ) · S_ss(θ).
    |S_T|² + |S_R|² = 1; β=2.5 is near the n=3 reflectionless point
    (ξ=1/3 ⇔ β=√(2π)≈2.5066) so |S_R|² ≈ 0 here."""
    pref = np.sinh(theta / xi) / np.sinh((complex(0, np.pi) - theta) / xi)
    return pref * (-np.exp(-1j * zamolodchikov_phase(theta, xi)))

def S_R_amplitude(theta, xi):
    """Full KK̄ reflection amplitude
        S_R(θ) = i sin(π/ξ)/sinh((iπ−θ)/ξ) · S_ss(θ)."""
    pref = 1j * np.sin(np.pi / xi) / np.sinh((complex(0, np.pi) - theta) / xi)
    return pref * (-np.exp(-1j * zamolodchikov_phase(theta, xi)))

def darg_S_T(theta, xi, h=1e-4):
    """d(arg S_T)/dθ — branch-safe central difference: the phase difference
    is taken via angle(S(θ+h)/S(θ−h)), immune to log branch cuts."""
    return np.angle(S_T_amplitude(theta + h, xi) / S_T_amplitude(theta - h, xi)) / (2 * h)

def wigner_dx(theta12, theta1, M, xi):
    """Wigner displacement of particle 1 (the soliton):
        Δx₁ = −∂(arg S_T)/∂p₁ = −(dargS_T/dθ)|_{θ₁₂} / (M cosh θ₁)
    Stationary phase for the transmitted packet |S_T|e^{iφ}e^{ipx−iEt}
    gives x(t) = vt − dφ/dp.  The S-matrix argument is the RELATIVE
    rapidity θ₁₂ = θ_K − θ_AK (≈2θ for symmetric beams); the energy in
    ∂θ₁₂/∂p₁ = 1/E₁ uses the SINGLE-particle rapidity θ₁."""
    E1 = M * np.cosh(theta1)
    return -darg_S_T(theta12, xi) / E1

# FREE_RUN_DIR / free_path and M_QP (renormalized soliton mass, from the free run's
# "M_soliton") are set in the initialization cell.
xi = beta**2 / (8 * np.pi - beta**2)

# ── Charge-weighted centroid tracking ─────────────────────────────────────────
# Centroid of the positive (soliton) / negative (antisoliton) part of the topological
# charge density within ±W sites of the global extremum.  Windowing suppresses
# vacuum noise and radiation far from the lump; the centroid is robust against
# peak hopping in a spreading packet (unlike argmax + parabola).  Charge density
# lives on bonds (j, j+1) ⇒ position j + 0.5.  The overall 1/a scale of ρ cancels
# in the weighted average, so the centroid is in pure SITE units.  W defaults to
# W_cent = 1.5·σ (set from the run's envelope width), so the physical window is
# held fixed as a → 0 — a window fixed in sites would clip the (wider in sites)
# packet and make the trajectory wiggle.
def centroid_track(rho, js, W=W_cent):
    x = np.full(rho.shape[1], np.nan)
    for ti in range(rho.shape[1]):
        r = np.clip(rho[:, ti], 0.0, None)
        if r.max() <= 0:
            continue
        i0 = int(np.argmax(r))
        lo, hi = max(0, i0 - W), min(len(r), i0 + W + 1)
        w, pos = r[lo:hi], js[lo:hi] + 0.5
        x[ti] = np.sum(pos * w) / np.sum(w)
    return x

x_K  = centroid_track( charge_density, js_cd)   # soliton:     centroid of ρ₊
x_AK = centroid_track(-charge_density, js_cd)   # antisoliton: centroid of ρ₋

def window_mask(times, t_lo, t_hi):
    return (times >= t_lo) & (times <= t_hi)

def linfit(ts, xs):
    A = np.column_stack([np.ones_like(ts), ts])
    c, _, _, _ = np.linalg.lstsq(A, xs, rcond=None)
    return c[0], c[1]   # intercept, velocity

pre_mask  = window_mask(times_cd, t_pre_lo,  t_pre_hi)
post_mask = window_mask(times_cd, t_post_lo, t_post_hi)

a_K_pre,  v_K_pre  = linfit(times_cd[pre_mask],  x_K[pre_mask])
a_K_post, v_K_post = linfit(times_cd[post_mask], x_K[post_mask])
a_AK_pre,  v_AK_pre  = linfit(times_cd[pre_mask],  x_AK[pre_mask])
a_AK_post, v_AK_post = linfit(times_cd[post_mask], x_AK[post_mask])

# ── Free-soliton reference trajectory ────────────────────────────────────────────
# Both runs nominally start at the same soliton site.  The visible gap at t = 0 (of
# order 3–5 sites) between the two centroid trajectories is due to the SGF gauge
# mismatch:
#   • scattering run: soliton B-tensor symmetrized against A_C = vac1.AL
#   • free run:       soliton B-tensor symmetrized against vac1.AR
# Different right-neighbour tensors in the _symmetrize_B cost function shift the
# physical soliton centre relative to the nominal B-tensor site by different
# amounts in each run.  This static gap is removed by the pre-window mean anchor
# Δx = ⟨D⟩_post − ⟨D⟩_pre below, which measures only the CHANGE in D.
#
# NOTE on linear detrending: in principle one could also subtract the slope of
# D(t) in the pre-window to remove a velocity mismatch between the two runs.
# In practice this is unreliable here: with only ~11 pre-window points and a
# centre-to-centre gap of ~35 t the propagated slope uncertainty (±0.56 sites
# for the soliton) exceeds the signal (~0.37 sites), making detrending worse than
# the plain mean anchor.  The velocity mismatch is small (|v_DK| < 0.02/t) and
# introduces a systematic of at most |v_DK| × gap ≈ 0.35 sites, which is the
# dominant remaining uncertainty on the measurement.  The remedy is a longer
# pre-window (if the simulation data allow) or a physically matched reference
# (free soliton built in the same A_C gauge environment as the scattering state).
#
# The antisoliton reference is the parity mirror x → (N+1) − x of the soliton
# reference; the resulting ~1-site offset at t=0 is absorbed by the anchor too.
# free_path is set in the initialization cell (parameter-matched free run).
f_cd = glob.glob(os.path.join(free_path, "charge_density_*.tsv"))[0]
with open(f_cd) as fh:
    names = [c.strip() for c in fh.readline().strip().lstrip("#").split("\t")]
df_free = pd.read_csv(f_cd, sep="\t", comment="#", names=names)
js_free = df_free.iloc[:, 0].values
times_free = np.array([float(c.split("=")[1]) for c in df_free.columns[1:]])
cd_free = df_free.iloc[:, 1:].values
x_free_raw = centroid_track(cd_free, js_free)
x_free_K   = np.interp(times_cd, times_free, x_free_raw, left=np.nan, right=np.nan)
x_free_AK  = (N + 1) - x_free_K        # parity mirror
FREE_SOURCE = f"free run {FREE_RUN_DIR}"

# ── Anchored displacement: Δx = ⟨D⟩_post − ⟨D⟩_pre ───────────────────────────
# D(t) = x_scatt(t) - x_free(t); the pre-window mean removes the static SGF
# offset so that Δx reflects only the change due to the collision.  All in SITE
# units (centroid index differences); multiply by the lattice spacing a for a
# physical length.
D_K  = x_K  - x_free_K
D_AK = x_AK - x_free_AK

Δx_K  = np.nanmean(D_K[post_mask])  - np.nanmean(D_K[pre_mask])
Δx_AK = np.nanmean(D_AK[post_mask]) - np.nanmean(D_AK[pre_mask])

# Antisymmetric (physical) part and common-mode artifact.
Δx_phys = 0.5 * (Δx_K - Δx_AK)
Δx_cm   = 0.5 * (Δx_K + Δx_AK)

# Velocity mismatch: slope of D in the pre-window.  Should be small; quantifies
# the systematic Δx_sys ≈ v_D × (t_post_centre − t_pre_centre).
_, v_DK_pre  = linfit(times_cd[pre_mask], D_K[pre_mask])
_, v_DAK_pre = linfit(times_cd[pre_mask], D_AK[pre_mask])
t_gap = times_cd[post_mask].mean() - times_cd[pre_mask].mean()

# Residual slope of D in the post-window (post-collision radiation / TDVP drift)
_, drift_K  = linfit(times_cd[post_mask], D_K[post_mask])
_, drift_AK = linfit(times_cd[post_mask], D_AK[post_mask])

# ── Kinematics (lattice ⇄ physical via the spacing a) ─────────────────────────
# The lattice spacing a relates the run's lattice (site) quantities to the
# continuum/physical ones that the S-matrix theory uses (renormalization.jl
# conventions):
#   • params["p0"] is the LATTICE momentum p_lat = k·a (lattice_momentum:
#     p(a) = p_ref·a/a_ref) ⇒ physical momentum  k = p_lat / a.
#   • centroid positions are site indices (x_phys = j·a) ⇒ a site-velocity
#     v_site = dj/dt is physical  v_phys = a·v_site  (c = 1, t in physical units).
#   • M_QP is the renormalized (continuum) soliton mass and is a-independent.
# At a = 1 every factor is unity and these reduce to the old lattice-unit forms.
k_phys = p0 / a                                       # physical (continuum) momentum
θ_K_kin = np.arcsinh(k_phys / M_QP)
θ_K_v  = np.arctanh(np.clip(abs(v_K_pre)  * a, 0.0, 0.9999))
θ_AK_v = np.arctanh(np.clip(abs(v_AK_pre) * a, 0.0, 0.9999))
θ12   = 2 * θ_K_kin
θ12_v = θ_K_v + θ_AK_v
p_meas = M_QP * np.sinh(θ_K_v)                        # physical momentum from measured v

# ── Theory: Wigner displacement from the corrected S_T ───────────────────────
# wigner_dx returns a PHYSICAL length (units 1/M_QP); divide by a to express it
# in site units for a direct comparison with the measured site displacements.
Δx_theory_len   = wigner_dx(θ12,   θ_K_kin, M_QP, xi)   # physical length
Δx_theory_v_len = wigner_dx(θ12_v, θ_K_v,   M_QP, xi)
Δx_theory   = Δx_theory_len   / a                       # site units
Δx_theory_v = Δx_theory_v_len / a
S_T_val = S_T_amplitude(θ12, xi)
S_R_val = S_R_amplitude(θ12, xi)

print(f"Free reference:   {FREE_SOURCE}")
print(f"Lattice spacing:  a = {a}   →   p0_lat = {p0:.4f}  maps to  k_phys = {k_phys:.4f}")
print(f"Soliton mass M_QP:   {M_QP:.6f}  (renormalized, from free run)")
print(f"Windows:          pre = [{t_pre_lo}, {t_pre_hi}]   post = [{t_post_lo}, {t_post_hi}]")
print(f"D(t) at t=0:      D_K ≈ {D_K[0]:+.2f}   D_AK ≈ {D_AK[0]:+.2f}  (SGF-gauge offset, removed by anchor)")
print(f"Pre-window slope: v_DK = {v_DK_pre:+.4f}/t   v_DAK = {v_DAK_pre:+.4f}/t  "
      f"→ velocity-mismatch syst ≈ {v_DK_pre*t_gap:+.3f} / {v_DAK_pre*t_gap:+.3f} sites")
print(f"Centroid velocities: v_K = {v_K_pre:+.4f} → {v_K_post:+.4f}   "
      f"v_AK = {v_AK_pre:+.4f} → {v_AK_post:+.4f}  (sites/t; ×a = {a} for physical)")
print(f"Rapidity:         θ_K = {θ_K_kin:.4f} (from k)   {θ_K_v:.4f} (from v)")
print(f"                  θ12 = {θ12:.4f} (from k)   {θ12_v:.4f} (from v)")
print(f"Momentum check:   k = {k_phys:.4f} (p0/a)   k = {p_meas:.4f} (M_QP·sinh θ_v)")
print(f"Unitarity:        |S_T|² = {abs(S_T_val)**2:.4f}   |S_R|² = {abs(S_R_val)**2:.4f}   "
      f"(β=2.5 ≈ reflectionless point ξ=1/3)")
print()
print(f"Displacements:    Δx_K = {Δx_K:+.4f}   Δx_AK = {Δx_AK:+.4f}  (sites)")
print(f"  → physical (antisym): Δx_phys = {Δx_phys:+.4f} sites = {Δx_phys*a:+.4f} (phys length)")
print(f"  → common mode (diag): Δx_cm   = {Δx_cm:+.4f} sites")
print(f"Post-window drift of D(t):  soliton {drift_K:+.4f}/t   antisoliton {drift_AK:+.4f}/t")
print()
print(f"Theory:           Δx_K = {Δx_theory:+.4f} sites = {Δx_theory_len:+.4f} (phys)  (θ12 from k)")
print(f"                  Δx_K = {Δx_theory_v:+.4f} sites = {Δx_theory_v_len:+.4f} (phys)  (θ12 from velocities)")
print(f"Full phase:       arg S_T(θ12) = {np.angle(S_T_val):+.4f} rad (mod 2π)")


D_K_pre_mean  = np.nanmean(D_K[pre_mask])
D_AK_pre_mean = np.nanmean(D_AK[pre_mask])

# Shift the free reference by the pre-window mean so it visually aligns with
# the scattering trajectory before the collision.  Purely cosmetic; same offset
# is removed in the Δx calculation above.
x_free_K_plot  = x_free_K  + D_K_pre_mean
x_free_AK_plot = x_free_AK + D_AK_pre_mean

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8), sharex=True,
                               gridspec_kw={"height_ratios": [2, 1]})

# ── Top: centroid trajectories + offset-corrected free reference ──────────────
ax1.plot(times_cd, x_K,            color="steelblue", lw=2,   label=r"soliton")
ax1.plot(times_cd, x_AK,           color="firebrick", lw=2,   label=r"antisoliton")
ax1.plot(times_cd, x_free_K_plot,  color="steelblue", lw=1.2, ls="--", label="free soliton")
ax1.plot(times_cd, x_free_AK_plot, color="firebrick", lw=1.2, ls="--", label="free antisoliton")

for ax in (ax1, ax2):
    ax.axvspan(t_pre_lo,  t_pre_hi,  alpha=0.10, color="steelblue")
    ax.axvspan(t_post_lo, t_post_hi, alpha=0.10, color="firebrick")

ax1.set_ylabel("position")
# ax1.set_title(rf"$\Delta x_{{\mathrm{{phys}}}}={Δx_phys:+.3f}$,  $\Delta x_{{\mathrm{{cm}}}}={Δx_cm:+.3f}$,  "
#               rf"$\Delta x^{{\mathrm{{th}}}}={Δx_theory:+.3f}$,  $\theta_{{12}}={θ12:.3f}$")
ax1.legend(fontsize=9, loc="best", ncol=2)
ax1.label_outer()

# ── Bottom: D(t) anchored to pre-window mean ──────────────────────────────────
D_K_anc  = D_K  - D_K_pre_mean
D_AK_anc = D_AK - D_AK_pre_mean

ax2.plot(times_cd, D_K_anc,  color="steelblue", lw=1.5, label=r"$D_K(t) - \langle D_K\rangle_\mathrm{pre}$")
ax2.plot(times_cd, D_AK_anc, color="firebrick", lw=1.5, label=r"$D_{AK}(t) - \langle D_{AK}\rangle_\mathrm{pre}$")
for D, col in ((D_K_anc, "steelblue"), (D_AK_anc, "firebrick")):
    ax2.hlines(np.nanmean(D[pre_mask]),  t_pre_lo,  t_pre_hi,  color=col, lw=2.5, alpha=0.6)
    ax2.hlines(np.nanmean(D[post_mask]), t_post_lo, t_post_hi, color=col, lw=2.5, alpha=0.6)
ax2.axhline(0, color="gray", lw=0.8, ls="--")
ax2.set_xlabel("$t$")
ax2.set_ylabel(r"$\Delta x(t)$")
ax2.legend(fontsize=9, loc="best")

plt.tight_layout()
plt.subplots_adjust(hspace=0)
plt.close()
fig.savefig(os.path.join(FIGURE_DIR, "soliton_antisoliton_trajectories.pdf"))


# ── Toggle overlays on/off here ────────────────────────────────────────────
SHOW_FILL             = True   # color +/- charge regions red/blue
SHOW_CENTROIDS        = True   # overlay soliton/antisoliton centroid markers
SHOW_ENERGY           = False   # show energy density (vacuum-subtracted, rescaled)
ENERGY_PANEL          = True   # True  -> energy density in its OWN panel directly
                               #          below charge (shared x-axis, no gap);
                               # False -> overlaid on a twin y-axis of the charge panel.
                               # (Ignored when SHOW_DISPLACEMENT is on: that panel owns
                               #  the bottom slot and uses a time x-axis.)
SHOW_GHOST            = False  # trailing low-alpha trace of recent frames
SHOW_DISPLACEMENT     = False  # bottom panel: anchored displacement D_K(t), D_AK(t)
SHOW_DURING_COLLISION = False  # if False, blank D(t) inside the collision window
GHOST_N = 5      # number of trailing frames to show
PAD = 0.05       # fractional y-axis padding applied to every panel


# ── Energy density: subtract the static vacuum baseline and rescale ──────────
# Raw e(j,t) is dominated by the bulk vacuum offset + boundary spike, so the
# dynamic soliton signal is a small, ragged ripple on a large pedestal.  We plot the
# vacuum-referenced excess e(j,t) - e_vac(j) (same baseline as the heatmap) and
# set the y-scale from THAT excess, not the raw offset-dominated min/max.
if SHOW_ENERGY:
    _e_vac   = read_tsv("vac_energy_density_*.tsv")["e_vac"].values
    Z_en     = energy_density - _e_vac[:, None]
    en_label = r"$\varepsilon(j,t)-\varepsilon_{\mathrm{vac}}(j)$"
    # Vacuum-subtracted ⇒ the bulk baseline is 0; anchor the bottom at 0.  Use the
    # TRUE max (not a percentile) so the peak is never clipped, padded on top by the
    # same fractional PAD as every other panel.
    en_lo = 0.0
    en_hi = float(Z_en.max())
    en_hi = en_hi + PAD * (en_hi - en_lo)

n_frames = len(times_cd)
if SHOW_ENERGY:
    n_frames = min(n_frames, Z_en.shape[1])

energy_in_panel = SHOW_ENERGY and ENERGY_PANEL and not SHOW_DISPLACEMENT

# ── Figure / panel layout ────────────────────────────────────────────────────
ax_eP = ax2 = None
if SHOW_DISPLACEMENT:
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(7, 7),
                                  gridspec_kw={"height_ratios": [2, 1]})
elif energy_in_panel:
    # charge density on top, energy density directly below: shared position
    # x-axis, zero vertical gap between the two panels.
    fig, (ax, ax_eP) = plt.subplots(2, 1, figsize=(7, 6), sharex=True,
                                    gridspec_kw={"height_ratios": [2, 1], "hspace": 0.0})
else:
    fig, ax = plt.subplots()

# ── Charge density (top panel) ───────────────────────────────────────────────
ax.set_ylabel(r"$\rho(j,t)$")
# Pad the data extremes by the same fraction PAD used for the energy panel.
cd_lo, cd_hi = float(charge_density.min()), float(charge_density.max())
cd_pad = PAD * (cd_hi - cd_lo)
ax.set_ylim(cd_lo - cd_pad, cd_hi + cd_pad)
plt_cd, = ax.plot(js_cd, charge_density[:, 0], color="black", lw=1.5, zorder=5, label="charge density")
ax.set_title(f"$t={times_cd[0]:.1f}$")

if energy_in_panel:
    ax.tick_params(labelbottom=False)          # bottom (energy) panel carries the x-axis
    ax_eP.set_xlabel("position")
else:
    ax.set_xlabel("position")

fill_pos = fill_neg = None  # PolyCollections from fill_between; re-created each frame

if SHOW_CENTROIDS:
    def _centroid(rho, js, W=W_cent):
        """Charge-weighted centroid within +/-W sites of the peak (same scheme as the
        phase-shift analysis); duplicated locally so this cell runs standalone.  W
        defaults to W_cent = 1.5·σ so the physical window tracks the packet width."""
        x = np.full(rho.shape[1], np.nan)
        for ti in range(rho.shape[1]):
            r = np.clip(rho[:, ti], 0.0, None)
            if r.max() <= 0:
                continue
            i0 = int(np.argmax(r))
            lo, hi = max(0, i0 - W), min(len(r), i0 + W + 1)
            w, pos = r[lo:hi], js[lo:hi] + 0.5
            x[ti] = np.sum(pos * w) / np.sum(w)
        return x

    x_K_anim  = _centroid(charge_density, js_cd)
    x_AK_anim = _centroid(-charge_density, js_cd)
    vline_K  = ax.axvline(x_K_anim[0],  color="firebrick", ls="--", lw=1.2, label="soliton centroid")
    vline_AK = ax.axvline(x_AK_anim[0], color="steelblue", ls="--", lw=1.2, label="antisoliton centroid")

# ── Energy density: own panel, or twin-axis overlay ──────────────────────────
plt_en = None
if SHOW_ENERGY:
    if energy_in_panel:
        ax_e = ax_eP
        ax_e.set_ylabel(en_label)
        ax_e.axhline(0.0, color="gray", lw=0.6, ls=":")
        plt_en, = ax_e.plot(js_en, Z_en[:, 0], color="darkorange", lw=1.2, label="energy density")
    else:
        ax_e = ax.twinx()
        ax_e.set_ylabel(en_label)
        plt_en, = ax_e.plot(js_en, Z_en[:, 0], color="darkorange", lw=1.0, alpha=0.8, label="energy density")
    ax_e.set_ylim(en_lo, en_hi)

if SHOW_GHOST:
    ghost_lines = [ax.plot([], [], color="black", lw=1.0, alpha=0.10 * (GHOST_N - i))[0]
                   for i in range(GHOST_N)]

if SHOW_CENTROIDS or SHOW_ENERGY:
    handles = [plt_cd] + ([vline_K, vline_AK] if SHOW_CENTROIDS else []) + ([plt_en] if SHOW_ENERGY else [])
    ax.legend(handles=handles, fontsize=9, loc="upper right")

if SHOW_DISPLACEMENT:
    # Anchored displacement D_K / D_AK from the phase-shift analysis above.
    D_K_plot, D_AK_plot = D_K_anc.copy(), D_AK_anc.copy()
    if not SHOW_DURING_COLLISION:
        collision = (times_cd > t_pre_hi) & (times_cd < t_post_lo)
        D_K_plot[collision]  = np.nan
        D_AK_plot[collision] = np.nan
    ax2.plot(times_cd, D_K_plot,  color="firebrick", lw=1.5, label=r"$D_K(t)$")
    ax2.plot(times_cd, D_AK_plot, color="steelblue", lw=1.5, label=r"$D_{AK}(t)$")
    ax2.axhline(0, color="gray", lw=0.8, ls="--")
    ax2.axvspan(t_pre_lo,  t_pre_hi,  alpha=0.10, color="steelblue")
    ax2.axvspan(t_post_lo, t_post_hi, alpha=0.10, color="firebrick")
    ax2.set_xlabel("$t$")
    ax2.set_ylabel(r"$\Delta x(t)$")
    ax2.margins(y=PAD)          # same fractional y-padding as the other panels
    ax2.legend(fontsize=9, loc="best")
    playhead = ax2.axvline(times_cd[0], color="black", lw=1.0, alpha=0.8)

if energy_in_panel:
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.0)   # re-assert zero gap (tight_layout reintroduces some)
else:
    plt.tight_layout()

def update(frame):
    global fill_pos, fill_neg
    artists = [plt_cd]

    y = charge_density[:, frame]
    plt_cd.set_ydata(y)
    ax.set_title(f"$t={times_cd[frame]:.1f}$")

    if SHOW_FILL:
        if fill_pos is not None:
            fill_pos.remove()
            fill_neg.remove()
        fill_pos = ax.fill_between(js_cd, 0, y, where=y >= 0, color="firebrick", alpha=0.3, zorder=1)
        fill_neg = ax.fill_between(js_cd, 0, y, where=y < 0,  color="steelblue", alpha=0.3, zorder=1)
        artists += [fill_pos, fill_neg]

    if SHOW_CENTROIDS:
        vline_K.set_xdata([x_K_anim[frame]] * 2)
        vline_AK.set_xdata([x_AK_anim[frame]] * 2)
        artists += [vline_K, vline_AK]

    if SHOW_ENERGY:
        plt_en.set_ydata(Z_en[:, frame])
        artists.append(plt_en)

    if SHOW_GHOST:
        for i, gl in enumerate(ghost_lines):
            gframe = frame - (i + 1)
            if gframe >= 0:
                gl.set_data(js_cd, charge_density[:, gframe])
            else:
                gl.set_data([], [])
        artists += ghost_lines

    if SHOW_DISPLACEMENT:
        playhead.set_xdata([times_cd[frame]] * 2)
        artists.append(playhead)

    return artists

anim = animation.FuncAnimation(fig=fig, func=update, frames=n_frames, interval=30)
anim.save(os.path.join(FIGURE_DIR, "charge_density.gif"), writer="pillow", fps=15)
plt.close()
