#!/usr/bin/env python3
"""Generate figures from the published sine-Gordon TSV data."""

import argparse
from pathlib import Path
import os
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PARSER = argparse.ArgumentParser(description=__doc__)
PARSER.add_argument("--data-root", type=Path, default=Path(__file__).resolve().parent.parent / "data", help="Directory containing continuum_scan, soliton_free, and soliton_scattering")
PARSER.add_argument("--figure-dir", type=Path, default=Path(__file__).resolve().parent / "figures" / "scattering_analysis", help="Output directory for figures")
PARSER.add_argument("--usetex", action="store_true", help="Use a local LaTeX installation for figure text")
ARGS = PARSER.parse_args()
DATA_BASE = ARGS.data_root.resolve()
FIGURE_DIR = ARGS.figure_dir.resolve()
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLBACKEND", "Agg")

import os
import re
import glob
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy.special import gamma
from scipy.integrate import quad

matplotlib.rcParams.update({
    "text.usetex": True,        # set True if TeX is available on this machine
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "legend.fontsize": 9,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "axes.grid": False,
    "figure.dpi": 130,
})
matplotlib.rcParams["text.usetex"] = ARGS.usetex

# ── Data paths ────────────────────────────────────────────────────────────────
DATA_ROOT = os.fspath(DATA_BASE / "soliton_scattering")
FREE_ROOT  = os.path.join(DATA_ROOT, "..", "soliton_free")

# ── Tag regex — matches the parameter tag embedded in every data filename ─────
# Format: b{β}_p{p0}_m{m}_a{a}_chi{χ}_d{d}_N{N}_dt{dt}
_TAG_RE = re.compile(
    r"b(?P<b>\d+\.\d+)_p(?P<p>\d+\.\d+)_m(?P<m>\d+\.\d+)_a(?P<a>\d+\.\d+)"
    r"_chi(?P<chi>\d+)_d(?P<d>\d+)_N(?P<N>\d+)_dt(?P<dt>\d+\.\d+)")
# Key for matching a scattering run to its free reference: physical params only.
_phys = lambda mt: (mt["b"], mt["p"], mt["m"], mt["a"], mt["d"], mt["N"], mt["dt"])

# ── Colour encoding (Okabe-Ito): colour = momentum, marker = lattice spacing ──
P_COLOR  = {0.60: "#0072B2", 0.80: "#D55E00"}
A_MARKER = {1.00: "o",       0.75: "s"}
INK, MUTED = "#1a1a19", "#6b6b68"
MARKER_SIZE = 6.5

def rstyle(r, ms=MARKER_SIZE):
    col = P_COLOR[round(r.p_phys, 2)]
    return dict(color=col, marker=A_MARKER[round(r.a, 2)], ms=ms,
                markerfacecolor=col, markeredgecolor="white", markeredgewidth=0.9)

def rlabel(r):
    return rf"$p = {r.p_phys:.1f}$, $a = {r.a:.2f}$"

# ── I/O utilities ─────────────────────────────────────────────────────────────
def find_file(directory, pattern):
    matches = glob.glob(os.path.join(directory, pattern))
    if not matches:
        raise FileNotFoundError(f"No file matching '{pattern}' in {directory}")
    return matches[0]

def read_tsv(path, **kwargs):
    """Read a TSV with a commented header '# col1\tcol2\t...'."""
    with open(path) as fh:
        col_names = [c.strip() for c in fh.readline().strip().lstrip("#").split("\t")]
    return pd.read_csv(path, sep="\t", comment="#", names=col_names, **kwargs)

def read_params(path):
    df = read_tsv(path)
    return dict(zip(df.iloc[:, 0], df.iloc[:, 1]))

def load_grid(directory, pattern):
    """Load a (sites × times) TSV; returns (js, times, data)."""
    df = read_tsv(find_file(directory, pattern))
    js    = df.iloc[:, 0].values.astype(float)
    times = np.array([float(c.split("=")[1]) for c in df.columns[1:]])
    return js, times, df.iloc[:, 1:].values

def linfit(ts, xs):
    m = np.isfinite(xs)
    A = np.column_stack([np.ones(m.sum()), ts[m]])
    c, _, _, _ = np.linalg.lstsq(A, xs[m], rcond=None)
    return c[0], c[1]

def window_mask(times, t_lo, t_hi):
    return (times >= t_lo) & (times <= t_hi)


# ── S-matrix theory (Zamolodchikov) ──────────────────────────────────────────
# Conventions: field φ, coupling β, ξ = β²/(8π−β²).
# KK̄ transmission amplitude: S_T(θ) = sinh(θ/ξ)/sinh((iπ−θ)/ξ) · S_ss(θ)
# where S_ss(θ) = −exp(−i Φ(θ)) with Φ from the integral representation.

C_RENORM = 4 * np.exp(np.euler_gamma)   # scheme-matching constant

def renormalized_mass(m_bare, a, beta):
    return m_bare * (a / C_RENORM) ** (beta**2 / (8 * np.pi))

def M_soliton_th(m_ren, beta):
    # Zamolodchikov formula for the physical soliton mass.
    alpha  = beta**2 / (8 * np.pi)
    xi_    = beta**2 / (8 * np.pi - beta**2)
    prefac = (2 / np.sqrt(np.pi)) * gamma(xi_ / 2) / gamma((1 + xi_) / 2)
    brak   = (np.pi * m_ren**2 / (2 * beta**2)) * (gamma(1 - alpha) / gamma(alpha))
    return prefac * brak ** (1 / (2 - 2 * alpha))

def soliton_phase_th(theta, xi):
    """Phase Φ(θ) of S_ss = −exp(−iΦ) via the Zamolodchikov integral representation."""
    if xi <= 0 or theta <= 0:
        return np.nan
    def integrand(t):
        if t < 1e-10:
            return theta * (1.0 - xi) / xi
        return (np.sin(theta * t) * np.sinh(np.pi * (1 - xi) * t / 2)
                / (np.sinh(np.pi * xi * t / 2) * np.cosh(np.pi * t / 2) * t))
    val, _ = quad(integrand, 0, 60 / (np.pi * xi), limit=500, epsabs=1e-12)
    return val

def S_T_amplitude(theta, xi):
    """KK̄ transmission amplitude."""
    pref = np.sinh(theta / xi) / np.sinh((complex(0, np.pi) - theta) / xi)
    return pref * (-np.exp(-1j * soliton_phase_th(theta, xi)))

def S_R_amplitude(theta, xi):
    """KK̄ reflection amplitude — used only to confirm the channel is negligible."""
    pref = np.sinh(complex(0, np.pi) / xi) / np.sinh((complex(0, np.pi) - theta) / xi)
    return pref * (-np.exp(-1j * soliton_phase_th(theta, xi)))

def darg_S_T(theta, xi, h=1e-4):
    """d(arg S_T)/dθ — branch-safe central difference."""
    return np.angle(S_T_amplitude(theta + h, xi) / S_T_amplitude(theta - h, xi)) / (2 * h)

def wigner_dx(theta12, theta1, M, xi):
    """Wigner displacement of the soliton (physical length units):
    Δx = −∂(arg S_T)/∂p_K = −(d arg S_T/dθ)|_{θ12} / (M cosh θ1)."""
    return -darg_S_T(theta12, xi) / (M * np.cosh(theta1))

def wigner_dx_p(p1, p2, M, xi):
    """Same, parametrised by the two physical momenta (both > 0, counter-propagating)."""
    th1, th2 = np.arcsinh(p1 / M), np.arcsinh(p2 / M)
    return -darg_S_T(th1 + th2, xi) / np.hypot(p1, M)

# ── Momentum spread of the initial wavepacket ────────────────────────────────
# wavepacket.jl builds the envelope   f_j ∝ exp(−(j−x₀)²/2σ²) · exp(i p₀ j),
# i.e. in physical variables (x = j·a, k = p/a) with σ_x ≡ σ·a:
#
#     f(x) ∝ exp(−(x−x₀)²/2σ_x²) e^{i k₀ x}
#     f̃(k) ∝ exp(−σ_x²(k−k₀)²/2)              (Fourier transform of a Gaussian)
#     |f̃(k)|² ∝ exp(−σ_x²(k−k₀)²)             ← the momentum PROBABILITY density
#
# so the momentum probability is Gaussian with standard deviation
#
#     σ_k = 1/(σ_x √2).
#
# Sanity check built into `momentum_spread` below: the position probability
# |f(x)|² has s.d. σ_x/√2, so σ_x_prob · σ_k = 1/2 exactly — a minimum-uncertainty
# packet.  σ_x is NOT fitted: it is `sigma` × `a` read from the run's params file.
def momentum_spread(sigma_x):
    """(σ_k, σ_x_prob, uncertainty product).  The product must be exactly 0.5."""
    sig_k = 1.0 / (sigma_x * np.sqrt(2))
    sig_x_prob = sigma_x / np.sqrt(2)
    return sig_k, sig_x_prob, sig_k * sig_x_prob

# ── Wavepacket-averaged displacement ─────────────────────────────────────────
# wigner_dx is derived for momentum EIGENSTATES.  The simulation launches
# wavepackets, and what is measured is the displacement of a packet CENTROID,
# which is the momentum average of the eigenstate result:
#
#     ⟨Δx⟩ = ∫dk₁∫dk₂ w(k₁) w(k₂) Δx(k₁, k₂),     w = normalised |f̃|²
#
# — one independent factor of w per particle.  No fit and no free parameter: the
# only input is σ_x.  Evaluated as a plain 2-D Gaussian quadrature on a uniform
# grid of `nq` nodes spanning ±`n_sig` σ_k (convergence is checked in section 3).
def wigner_dx_wavepacket(k0, M, xi, sigma_x, nq=41, n_sig=4.0):
    sig_k, _, _ = momentum_spread(sigma_x)
    u = np.linspace(-n_sig, n_sig, nq)
    w = np.exp(-u**2 / 2); w /= w.sum()          # normalised Gaussian weights
    k = k0 + u * sig_k
    if k[0] <= 0:
        raise ValueError(f"quadrature grid reaches k <= 0 (k_min={k[0]:.3f}); "
                         f"reduce n_sig")
    return sum(w[i] * w[j] * wigner_dx_p(k[i], k[j], M, xi)
               for i in range(nq) for j in range(nq))


# ── Position estimator ────────────────────────────────────────────────────────
# One method, settled: the centroid over the connected region around the peak
# where rho > frac*rho_max, with the pedestal subtracted.  The aperture is a
# level set of rho, so it follows the packet (no lattice sawtooth) and is bounded
# by the packet's own shape (no radiation leak).
def centroid_thresh(rho, js, frac=0.25):
    x = np.full(rho.shape[1], np.nan)
    pos = js + 0.5
    for ti in range(rho.shape[1]):
        r = np.clip(rho[:, ti], 0.0, None)
        if r.max() <= 0:
            continue
        i0 = int(np.argmax(r)); thr = frac * r[i0]
        lo = i0
        while lo > 0 and r[lo - 1] > thr:
            lo -= 1
        hi = i0
        while hi < len(r) - 1 and r[hi + 1] > thr:
            hi += 1
        w = r[lo:hi + 1] - thr
        x[ti] = np.sum(pos[lo:hi + 1] * w) / np.sum(w)
    return x

def xcorr_shift(prof, ref, js, n_polish=30):
    """Sub-lattice displacement minimising ||p(x) - q(x-d)||^2 between two
    unit-normalised poss.  Has no aperture parameter and is retained as an
    independent cross-check."""
    p = np.clip(prof, 0, None); q = np.clip(ref, 0, None)
    if p.sum() <= 0 or q.sum() <= 0:
        return np.nan
    n = len(p)
    lags = np.arange(-(n - 1), n)
    cc = np.correlate(p, q, mode="full")
    k = int(np.argmax(cc))
    d0 = float(lags[k])
    if 0 < k < len(cc) - 1:
        y0, y1, y2 = cc[k - 1], cc[k], cc[k + 1]
        den = y0 - 2 * y1 + y2
        if den != 0:
            d0 += 0.5 * (y0 - y2) / den
    x  = js.astype(float)
    pn = p / p.sum()
    def cost(d):
        qs = np.interp(x - d, x, q, left=0.0, right=0.0)
        s = qs.sum()
        return np.inf if s <= 0 else np.sum((pn - qs / s) ** 2)
    lo, hi = d0 - 1.5, d0 + 1.5
    for _ in range(n_polish):
        m1, m2 = lo + (hi - lo) / 3, hi - (hi - lo) / 3
        if cost(m1) < cost(m2):
            hi = m2
        else:
            lo = m1
    return 0.5 * (lo + hi)


# ── Run loading ───────────────────────────────────────────────────────────────
class Run:
    """One scattering run plus its auto-matched free single-soliton reference."""

def load_run(name):
    r = Run(); r.name = name
    d = os.path.join(DATA_ROOT, name)
    p = read_params(find_file(d, "params_*.tsv"))
    r.beta  = float(p["beta"]);  r.p0    = float(p["p0"]);   r.N     = int(p["N"])
    r.a     = float(p["a"]);     r.m     = float(p["m"]);    r.chi   = int(p["chi"])
    r.sigma = float(p["sigma"]); r.x_K0  = float(p["x_K"]);  r.x_AK0 = float(p["x_AK"])
    r.dE_frac = float(p["dE_frac"]); r.d = int(p["d"])
    r.xi     = r.beta**2 / (8 * np.pi - r.beta**2)
    r.p_phys = r.p0 / r.a                     # physical momentum k = p_lat / a
    r.sigma_phys = r.sigma * r.a

    # ── Auto-match the free reference (most recent by date among param matches) ─
    scat_key = _phys(_TAG_RE.search(os.path.basename(find_file(d, "charge_density_*.tsv"))))
    matches = []
    for f in glob.glob(os.path.join(FREE_ROOT, "*", "charge_density_*.tsv")):
        mt = _TAG_RE.search(os.path.basename(f))
        if mt and _phys(mt) == scat_key:
            matches.append((os.path.basename(os.path.dirname(f)), int(mt["chi"])))
    if not matches:
        raise FileNotFoundError(f"No free-run param match for '{name}'")
    matches.sort(key=lambda z: z[0])
    r.free_name, free_chi = matches[-1]
    if free_chi != r.chi:
        print(f"  Warning [{name}]: free χ={free_chi} ≠ scattering χ={r.chi}")
    fd = os.path.join(FREE_ROOT, r.free_name)
    r.M_QP = float(read_params(find_file(fd, "params_*.tsv"))["M_soliton"])

    r.js,   r.t,   r.cd   = load_grid(d,  "charge_density_*.tsv")
    r.js_f, r.t_f, r.cd_f = load_grid(fd, "charge_density_*.tsv")
    r.sc   = read_tsv(find_file(d,  "scalars_*.tsv"))
    r.sc_f = read_tsv(find_file(fd, "scalars_*.tsv"))

    r.M_th   = M_soliton_th(renormalized_mass(r.m, r.a, r.beta), r.beta)
    r.v_phys = r.p_phys / np.hypot(r.p_phys, r.M_QP)
    r.v_lat  = r.v_phys / r.a                            # sites per unit time
    r.t_coll = (r.x_AK0 - r.x_K0) / (2 * r.v_lat)
    # Charge in the positive pos: → 1 only once the two poss have separated.
    r.Q_pos = np.clip(r.cd, 0, None).sum(axis=0) * r.a
    r.Q_tot  = r.cd.sum(axis=0) * r.a
    r._tracks = {}
    return r


# ── Displacement tracks ───────────────────────────────────────────────────────
# D_K(t)  = x_K(t)  − x_free(t)                     (soliton vs the free reference)
# D_AK(t) = x_AK(t) − [(N+1) − x_free(t)]           (anti-soliton vs the parity mirror)
# The constant in the mirror is irrelevant: only the pre→post CHANGE is used.
# anti = ½(D_K − D_AK) is the physical, parity-odd combination.
def tracks(r, kind="thr25"):
    """(D_K, D_AK, anti, common) in SITE units, cached per run."""
    if kind in r._tracks:
        return r._tracks[kind]
    if kind == "xcorr":
        DK  = np.full(len(r.t), np.nan)
        DAK = np.full(len(r.t), np.nan)
        for ti, tt in enumerate(r.t):
            tf = int(np.argmin(abs(r.t_f - tt)))
            if abs(r.t_f[tf] - tt) > 1e-9:      # free run is shorter — leave NaN
                continue
            q = np.clip(r.cd_f[:, tf], 0, None)
            DK[ti] = xcorr_shift(np.clip(r.cd[:, ti], 0, None), q, r.js)
            # The AK pos is mirrored onto the soliton's orientation; the reversal
            # j → (n−1)−j flips the sign of the fitted shift, so negate it.
            DAK[ti] = -xcorr_shift(np.ascontiguousarray(np.clip(-r.cd[:, ti], 0, None)[::-1]), q, r.js)
    else:
        f = lambda rho, js: centroid_thresh(rho, js, 0.25)
        xK, xAK = f(r.cd, r.js), f(-r.cd, r.js)
        xf = np.interp(r.t, r.t_f, f(r.cd_f, r.js_f), left=np.nan, right=np.nan)
        DK, DAK = xK - xf, xAK - ((r.N + 1) - xf)
    res = (DK, DAK, 0.5 * (DK - DAK), 0.5 * (DK + DAK))
    r._tracks[kind] = res
    return res


def displacement(r, pre, post, kind="thr25"):
    """Δx in PHYSICAL length units for one pair of windows."""
    _, _, anti, _ = tracks(r, kind)
    a_pre  = np.nanmean(anti[window_mask(r.t, *pre)])
    a_post = np.nanmean(anti[window_mask(r.t, *post)])
    return (a_post - a_pre) * r.a


def displacement_thr(r, pre, post, frac):
    """Δx recomputed at a different threshold — only used for the e_thr term."""
    f = lambda rho, js: centroid_thresh(rho, js, frac)
    xK, xAK = f(r.cd, r.js), f(-r.cd, r.js)
    xf = np.interp(r.t, r.t_f, f(r.cd_f, r.js_f), left=np.nan, right=np.nan)
    anti = 0.5 * ((xK - xf) - (xAK - ((r.N + 1) - xf)))
    return (np.nanmean(anti[window_mask(r.t, *post)])
            - np.nanmean(anti[window_mask(r.t, *pre)])) * r.a


# ── Run catalog — one run per (a, p), at the largest bond dimension available ──
SCAT_RUNS = [
    "20260605_3",        # a=1.00, p_phys=0.80, chi=64
    "20260805",          # a=0.75, p_phys=0.80, chi=80
    "20260721",          # a=1.00, p_phys=0.60, chi=64
    "20260807",          # a=0.75, p_phys=0.60, chi=80
]

RUNS = {}
print(f"{'run':12s} {'a':>5s} {'p_phys':>7s} {'N':>4s} {'chi':>4s} {'d':>3s} "
      f"{'sigma_phys':>10s} {'t_coll':>7s} {'T':>4s}  free reference")
print("-" * 78)
for name in SCAT_RUNS:
    r = load_run(name)
    RUNS[name] = r
    tracks(r, "thr25"); tracks(r, "xcorr")
    print(f"{name:12s} {r.a:5.2f} {r.p_phys:7.2f} {r.N:4d} {r.chi:4d} {r.d:3d} "
          f"{r.sigma_phys:10.2f} {r.t_coll:7.1f} {r.t[-1]:4.0f}  {r.free_name}")

# ── Extraction windows ────────────────────────────────────────────────────────
PRE_W  = (2.0, 6.0)    # after the t=0 transient, pair still >= 4.2 sigma apart
POST_W = (52.0, 60.0)  # latest available; |Q_+ - 1| < 5e-3 for every run


# ── Wavepacket-averaged prediction ───────────────────────────────────────────
SIGMA_X = 8.0                    # = sigma * a, identical for all four runs
M_th_val, xi_th = RUNS[SCAT_RUNS[0]].M_th, RUNS[SCAT_RUNS[0]].xi

sig_k, sig_x_prob, uncert = momentum_spread(SIGMA_X)
for r in RUNS.values():
    assert abs(r.sigma_phys - SIGMA_X) < 1e-9, f"{r.name} has sigma*a = {r.sigma_phys}"
assert abs(uncert - 0.5) < 1e-12, "envelope is not minimum-uncertainty"

TH_WP    = {p: wigner_dx_wavepacket(p, M_th_val, xi_th, SIGMA_X) for p in (0.60, 0.80)}
TH_SHARP = {p: wigner_dx_p(p, p, M_th_val, xi_th)                for p in (0.60, 0.80)}

print(f"sigma_x = {SIGMA_X:.1f}   sigma_k = {sig_k:.5f}")
print(f"{'p':>5s} {'eigenstate':>11s} {'wavepacket':>11s} {'correction':>11s}")
for p in (0.60, 0.80):
    print(f"{p:5.2f} {TH_SHARP[p]:11.4f} {TH_WP[p]:11.4f} "
          f"{(TH_WP[p] - TH_SHARP[p]) / TH_SHARP[p]:+11.2%}")


# ── Separation diagnostic and Figure 1 ───────────────────────────────────────
def clean_from(r, tol=5e-3):
    """First post-collision time from which |Q_pos - 1| stays below `tol`."""
    ok = np.where((r.t > r.t_coll + 5) & (abs(r.Q_pos - 1) < tol))[0]
    for i in ok:
        if (abs(r.Q_pos[i:] - 1) < tol).all():
            return r.t[i]
    return np.nan

print(f"{'run':12s} {'a':>5s} {'p':>5s} "
      + " ".join(f"{f'Q(t={t:.0f})':>10s}" for t in [0, 45, 50, 55, 60])
      + f" {'clean from':>11s}")
for name in SCAT_RUNS:
    r = RUNS[name]
    vals = " ".join(f"{r.Q_pos[int(np.argmin(abs(r.t - t)))]:10.4f}"
                    for t in [0, 45, 50, 55, 60])
    print(f"{name:12s} {r.a:5.2f} {r.p_phys:5.2f} {vals} {clean_from(r):11.1f}")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 3.5), sharex=True,
                               gridspec_kw={"height_ratios": [1.25, 1],
                                             "hspace": 0})
for name in SCAT_RUNS:
    r = RUNS[name]
    _, _, anti, _ = tracks(r)
    # Blank the curve while the poss are merged: there the soliton position, and
    # hence the displacement, is not a defined quantity.
    y = (anti - np.nanmean(anti[window_mask(r.t, *PRE_W)])) * r.a
    ax1.plot(r.t, np.where(r.Q_pos > 0.5, y, np.nan), lw=1.6, alpha=0.9,
             markevery=6, label=rlabel(r), **rstyle(r))
    ax2.plot(r.t, r.Q_pos, lw=1.6, markevery=6, **rstyle(r))

for ax in (ax1, ax2):
    for w in (PRE_W, POST_W):
        ax.axvspan(*w, color="#dcdcd8", zorder=0, lw=0)
    ax.grid(alpha=0.25, lw=0.6); ax.set_axisbelow(True)

ax1.set_ylim(-1.0, 1.0)
ax1.set_yticks(np.arange(-1.0, 1.01, 0.5))
ax1.axhline(0, color=MUTED, lw=0.8, ls=":")
ax1.set_ylabel(r"$\frac{1}{2}(D_S - D_{\bar S})\,a$", color=INK)
ax1.text(PRE_W[0] + 0.4, 0.85, "pre", fontsize=9, color=MUTED)
ax1.text(POST_W[0] + 0.4, 0.85, "post", fontsize=9, color=MUTED)
ax1.legend(loc="lower right", ncol=2, framealpha=0.9)

ax2.axhline(1.0, color=MUTED, lw=0.8, ls=":")
ax2.set_ylim(0.0, 1.25)
ax2.set_yticks(np.arange(0.0, 1.01, 0.5))
ax2.set_ylabel(r"$Q_+$")
ax2.set_xlabel(r"$t$")
ax2.set_xlim(0, 62)

plt.tight_layout(h_pad=0)
fig.subplots_adjust(hspace=0)
plt.savefig(os.path.join(FIGURE_DIR, "dx_tracks.pdf"), bbox_inches="tight")
plt.close()


# ── Figure 2: Δx against the choice of post window ───────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.9))
starts = np.arange(30, 53, 1.0)

for ax, p in zip(axes, (0.60, 0.80)):
    for name in SCAT_RUNS:
        r = RUNS[name]
        if abs(r.p_phys - p) > 1e-6:
            continue
        vals = [displacement(r, PRE_W, (s, min(s + 8, r.t[-1]))) for s in starts]
        ax.plot(starts, vals, lw=1.6, markevery=4, label=rf"$a = {r.a:.2f}$",
                **rstyle(r))
        ax.axvline(clean_from(r), color=P_COLOR[p], lw=0.9, ls="--", alpha=0.55)
    ax.axhline(TH_WP[p], color=INK, lw=1.4, zorder=1)
    ax.set_xlabel(r"post-window start $t$")
    ax.set_ylim(TH_WP[p] * 0.5, TH_WP[p] * 2.6)
    ax.grid(alpha=0.25, lw=0.6); ax.set_axisbelow(True)
    ax.legend(loc="upper right")
axes[0].set_ylabel(r"$\Delta x$")
plt.tight_layout()
plt.savefig(os.path.join(FIGURE_DIR, "dx_window_convergence.pdf"), bbox_inches="tight")
plt.close()


# ── Result ────────────────────────────────────────────────────────────────────
THRESHOLDS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
PREWINS    = [(0, 4), (1, 5), (2, 6), (0, 8), (2, 10), (4, 12)]

records = []
for name in SCAT_RUNS:
    r = RUNS[name]
    dx = displacement(r, PRE_W, POST_W)
    _, _, anti, _ = tracks(r)
    mpost = window_mask(r.t, *POST_W)
    c, s = linfit(r.t[mpost], anti[mpost])
    resid = anti[mpost] - (c + s * r.t[mpost])

    e_thr   = 0.5 * np.ptp([displacement_thr(r, PRE_W, POST_W, q) for q in THRESHOLDS])
    e_pre   = 0.5 * np.ptp([displacement(r, w, POST_W) for w in PREWINS])
    e_scat  = np.nanstd(resid) / np.sqrt(np.isfinite(resid).sum()) * r.a
    err     = np.sqrt(e_thr**2 + e_pre**2 + e_scat**2)

    theta12 = 2 * np.arcsinh(r.p_phys / r.M_QP)
    records.append({
        # p_phys = p0/a carries float noise across runs; round it for grouping only.
        "run": name, "a": r.a, "p_phys": round(r.p_phys, 4), "theta12": theta12,
        "dx": dx, "err": err,
        "dx_th_eig": wigner_dx(theta12, theta12 / 2, r.M_QP, r.xi),
        "dx_th_wp":  wigner_dx_wavepacket(r.p_phys, r.M_QP, r.xi, r.sigma_phys),
    })

df = pd.DataFrame(records).sort_values(["p_phys", "a"]).reset_index(drop=True)
df["r_wp"] = df["dx"] / df["dx_th_wp"]

show = df[["run", "a", "p_phys", "theta12", "dx", "err",
           "dx_th_eig", "dx_th_wp", "r_wp"]].copy()
show.columns = ["run", "a", "p", "θ₁₂", "Δx", "±", "Δx eig", "Δx wp", "Δx/wp"]
for c in show.columns[3:]:
    show[c] = show[c].map(lambda v: f"{v:.4f}")
print(show.to_string(index=False))

for p, g in df.groupby("p_phys"):
    v = g["dx"].values
    print(f"  p = {p:.2f}:  Δx = {v.mean():.3f}  (a = 1.00: {v[1]:.4f}, a = 0.75: {v[0]:.4f}),"
          f"  spread {np.ptp(v):.3f},  mean error {np.sqrt((g['err'].values**2).mean()):.3f}")
    print(f"{'':12s}vs eigenstate {g['dx_th_eig'].iloc[0]:.3f} -> {v.mean()/g['dx_th_eig'].iloc[0]:.3f}"
          f"   |   vs wavepacket {g['dx_th_wp'].iloc[0]:.3f} -> {v.mean()/g['dx_th_wp'].iloc[0]:.3f}")


# ── Figure 3: Δx against relative rapidity ───────────────────────────────────
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9.8, 4.4),
                              gridspec_kw={"width_ratios": [1.55, 1]})

theta_arr = np.linspace(1.4, 3.4, 90)
k_arr     = M_th_val * np.sinh(theta_arr / 2)
ax.plot(theta_arr, [wigner_dx(th, th / 2, M_th_val, xi_th) for th in theta_arr],
        color=MUTED, lw=1.6, ls="--", zorder=1,
        label=r"$\Delta x(\theta_{12})$ (momentum eigenstate)")
ax.plot(theta_arr, [wigner_dx_wavepacket(k, M_th_val, xi_th, SIGMA_X) for k in k_arr],
        color=INK, lw=2.0, zorder=2,
        label=r"$\langle\Delta x\rangle$ (Gaussian wavepacket)")

# Points sit at their true θ₁₂.  At fixed p the two lattice spacings differ by
# Δθ₁₂ ~ 0.002, entirely through M_QP, so they overlap on this axis.  The
# a-dependence is the right panel, where a is an axis rather than a nudge.
for _, row in df.iterrows():
    r = RUNS[row["run"]]
    ax.errorbar(row["theta12"], row["dx"], yerr=row["err"],
                fmt="none", ecolor=P_COLOR[round(row["p_phys"], 2)],
                elinewidth=1.4, capsize=3, zorder=4)
    ax.plot(row["theta12"], row["dx"], ls="none", zorder=5,
            label=rlabel(r), **rstyle(r))

ax.set_xlabel(r"$\theta_{12}$")
ax.set_ylabel(r"$\Delta x$")
ax.set_xlim(1.85, 3.05)
ax.set_ylim(0.05, 0.92)
ax.grid(alpha=0.25, lw=0.6); ax.set_axisbelow(True)

for p, col in P_COLOR.items():
    g = df[np.isclose(df["p_phys"], p)].sort_values("a")
    ax2.plot(g["a"], g["r_wp"], color=col, lw=1.4, zorder=2)
    for _, row in g.iterrows():
        ax2.errorbar(row["a"], row["r_wp"], yerr=row["err"] / row["dx_th_wp"],
                     fmt="none", ecolor=col, elinewidth=1.4, capsize=3, zorder=4)
        ax2.plot(row["a"], row["r_wp"], ls="none", zorder=5,
                 **rstyle(RUNS[row["run"]]))
ax2.axhline(1.0, color=INK, lw=1.6)
ax2.set_xlabel(r"$a$")
ax2.set_ylabel(r"$\Delta x \,/\, \langle\Delta x\rangle_{\mathrm{theory}}$")
ax2.set_xlim(0.65, 1.10)
ax2.set_xticks([0.75, 1.0])
ax2.grid(alpha=0.25, lw=0.6); ax2.set_axisbelow(True)

h, l = ax.get_legend_handles_labels()
seen, hh, ll = set(), [], []
for handle, lab in zip(h, l):
    if lab not in seen:
        seen.add(lab); hh.append(handle); ll.append(lab)
fig.legend(hh, ll, loc="upper center", bbox_to_anchor=(0.5, 0.02), ncol=3,
           frameon=False, columnspacing=1.6, handletextpad=0.6, labelspacing=0.5)

plt.tight_layout()
plt.savefig(os.path.join(FIGURE_DIR, "dx_vs_theta12.pdf"), bbox_inches="tight")
plt.close()
