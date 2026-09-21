#!/usr/bin/env python3
"""Generate figures from the published sine-Gordon TSV data."""

import argparse
from pathlib import Path
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PARSER = argparse.ArgumentParser(description=__doc__)
PARSER.add_argument("--data-root", type=Path, default=Path(__file__).resolve().parent.parent / "data", help="Directory containing continuum_scan, soliton_free, and soliton_scattering")
PARSER.add_argument("--figure-dir", type=Path, default=Path(__file__).resolve().parent / "figures" / "continuum_scan", help="Output directory for figures")
PARSER.add_argument("--run", default=None, help="Continuum run directory")
PARSER.add_argument("--usetex", action=argparse.BooleanOptionalAction, default=True, help="Render figure text with LaTeX (use --no-usetex without TeX)")
ARGS = PARSER.parse_args()
DATA_BASE = ARGS.data_root.resolve()
FIGURE_DIR = ARGS.figure_dir.resolve()
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

matplotlib.rcParams.update({
    "text.usetex": ARGS.usetex,
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "legend.fontsize": 9,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "axes.grid": False,
    "figure.dpi": 130,
})

BASE_DIR = DATA_BASE / "continuum_scan"

# Set to a specific run folder name (e.g. "20260608_1") to pin a run;
# leave as None to fall back to the most recent run by folder name.
RUN_DIR = ARGS.run

if RUN_DIR is not None:
    DATA_DIR = BASE_DIR / RUN_DIR
else:
    run_dirs = sorted(d for d in BASE_DIR.iterdir() if d.is_dir())
    if not run_dirs:
        raise FileNotFoundError(f"No run folders found in {BASE_DIR}")
    DATA_DIR = run_dirs[-1]

print(f"Using run directory: {DATA_DIR}")


def read_tsv(path):
    return pd.read_csv(path, sep="\t")


# ── Load data grouped by β ───────────────────────────────────────────────────
import re

scalar_files = sorted(DATA_DIR.glob("scalars_*.tsv"))
disp_files   = sorted(DATA_DIR.glob("dispersion_*.tsv"))
print(f"Found {len(scalar_files)} scalar files, {len(disp_files)} dispersion files")

def _beta_from_stem(stem):
    m = re.search(r"_beta([0-9.]+)", stem)
    return float(m.group(1))

# Group scalars by β (all files have m_bare + m_ref columns from RENORM_MASS=true)
beta_groups = {}
for f in scalar_files:
    b = _beta_from_stem(f.stem)
    df = read_tsv(f)
    beta_groups.setdefault(b, []).append(df)

betas = sorted(beta_groups)
sc_by_beta = {b: pd.concat(beta_groups[b]).sort_values("a").reset_index(drop=True)
              for b in betas}

# Read Zamolodchikov exact soliton mass from data (column M_Zam, same for all a)
M_zam_by_beta = {b: float(sc_by_beta[b]["M_Zam"].iloc[0]) for b in betas}

# Group dispersion by (β, a)
disp_by_beta = {b: [] for b in betas}
for f in sorted(disp_files):
    b = _beta_from_stem(f.stem)
    a_v = float(re.search(r"_a([0-9.]+)", f.stem).group(1))
    disp_by_beta[b].append({"a": a_v, "df": read_tsv(f).sort_values("p_phys")})
for b in betas:
    disp_by_beta[b].sort(key=lambda x: x["a"])

# ── Summary per β ─────────────────────────────────────────────────────────────
print()
for b in betas:
    sc   = sc_by_beta[b]
    M_zam = M_zam_by_beta[b]
    print(f"β = {b}  (m_ref = {sc['m_ref'].iloc[0]:.4g},  M_Zam = {M_zam:.6g})")
    print(sc[["a", "m_bare", "M_QP", "M_fit", "delta_e"]].to_string(index=False))
    print()


# ── Load breather masses grouped by (β, a) ───────────────────────────────────
breather_files = sorted(DATA_DIR.glob("breathers_*.tsv"))
print(f"Found {len(breather_files)} breather files")

breather_by_beta = {b: [] for b in betas}
for f in breather_files:
    b_match = re.search(r"_beta([0-9.]+)", f.stem)
    a_match = re.search(r"_a([0-9.]+)", f.stem)
    if not b_match or not a_match:
        continue
    b = float(b_match.group(1))
    a_v = float(a_match.group(1))
    if b not in breather_by_beta:
        continue
    breather_by_beta[b].append({"a": a_v, "df": read_tsv(f)})

for b in betas:
    breather_by_beta[b].sort(key=lambda x: x["a"])

print()
for b in betas:
    n_files = len(breather_by_beta[b])
    xi = b**2 / (8 * np.pi - b**2)
    n_max = int(1.0 / xi) if xi > 0 else 0
    print(f"  β = {b:4.1f}  ξ = {xi:.4f}  floor(1/ξ) = {n_max}  breather files: {n_files}")


# ── Convergence thresholds ────────────────────────────────────────────────────
BAD_DE   = 1e-3   # vacuum splitting |Δε| above this → vacua not degenerate → bad
BAD_MDEV = 0.20   # |M_QP/M_Zam − 1| above this → flagged (but see note below)

# NOTE: with the correct tuning m_bare(a) = m_ref (a_ref/a)^{beta^2/(8pi)} and the
#   matching constant C0 = 4 e^gamma (see header), M_QP/M_Zam -> 1 as a -> 0.
#   Residual deviation at finite a is a discretization + finite-chi/d artifact and
#   shrinks as a -> 0.  Points are flagged only on vacuum non-degeneracy
#   (|delta_e| > BAD_DE) or a large mass deviation (|M_QP/M_Zam - 1| > BAD_MDEV).

BETAS_REN = betas   # all β values use RENORM_MASS=true

# ── Per-β bad mask using M_Zam as the physical target ────────────────────────
bad_by_beta = {}
for b in BETAS_REN:
    sc    = sc_by_beta[b]
    M_zam = M_zam_by_beta[b]
    bad_by_beta[b] = (
        (sc["delta_e"].abs() > BAD_DE) |
        ((sc["M_QP"] / M_zam - 1.0).abs() > BAD_MDEV)
    )

# ── Convergence table ─────────────────────────────────────────────────────────
rows = []
for b in BETAS_REN:
    sc    = sc_by_beta[b]
    M_zam = M_zam_by_beta[b]
    for _, row in sc.iterrows():
        reasons = []
        if abs(row["delta_e"]) > BAD_DE:
            reasons.append(f"|Δε|={row['delta_e']:.1e}")
        if abs(row["M_QP"] / M_zam - 1.0) > BAD_MDEV:
            reasons.append(f"M_QP/M_Zam={row['M_QP']/M_zam:.3f}")
        rows.append({
            "β":           b,
            "a":           row["a"],
            "m_bare":      row["m_bare"],
            "M_QP":        row["M_QP"],
            "M_Zam":       M_zam,
            "M_QP/M_Zam":  row["M_QP"] / M_zam,
            "|Δε|":        row["delta_e"],
            "converged":   "✓" if not reasons else "✗  " + ", ".join(reasons),
        })

tbl = pd.DataFrame(rows)
print("Convergence table  (BAD_DE={:.0e}, BAD_MDEV={:.0%})\n".format(BAD_DE, BAD_MDEV))
print(tbl.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

n_good = (tbl["converged"] == "✓").sum()
n_bad  = len(tbl) - n_good
print(f"\n  Converged: {n_good}/{len(tbl)}   Bad: {n_bad}/{len(tbl)}")
print(f"\n  Note: 'M_QP/M_Zam' failures are primarily finite-χ effects (χ=64).")
print(f"        Increase χ (see chi_scan.jl) to approach M_QP/M_Zam → 1.")


# ── Breather masses vs Zamolodchikov prediction ───────────────────────────────
# M_n^Zam = 2 M_sol sin(n π ξ / 2),  ξ = β² / (8π − β²)
print("Breather masses  M_n  vs  M_n^Zam = 2 M_sol sin(n π ξ / 2)\n")
for b in betas:
    xi = b**2 / (8 * np.pi - b**2)
    n_max = int(1.0 / xi) if xi > 0 else 0
    items = breather_by_beta[b]
    if not items:
        print(f"β = {b}  (ξ = {xi:.4f},  floor(1/ξ) = {n_max})  — no breather data\n")
        continue

    ns_present = sorted({int(r["n"]) for it in items for _, r in it["df"].iterrows()})
    print(f"β = {b}  (ξ = {xi:.4f},  floor(1/ξ) = {n_max},  measured n = {ns_present})")

    rows = []
    for it in items:
        a_v = it["a"]
        for _, row in it["df"].iterrows():
            rows.append({
                "a":     a_v,
                "n":     int(row["n"]),
                "M_B":   row["M_QP"],
                "M_Zam": row["M_Zam"],
                "ratio": row["ratio"],
            })
    tbl = pd.DataFrame(rows)
    print(tbl.to_string(index=False, float_format=lambda x: f"{x:.6g}"))
    print()


# # ── soliton mass vs a ───────────────────────────────────────────────────────────
# colors  = {1.25: "C0", 1.5: "C1", 2.0: "C2", 2.25: "C3", 2.5: "C4"}
# markers = {1.25: "o",  1.5: "s",  2.0: "^",  2.25: "D",  2.5: "v"}
# Use BETAS_REN instead of fixed values
colors = {b: f"C{i}" for i, b in enumerate(BETAS_REN)}
markers = {b: m for b, m in zip(BETAS_REN, ["o", "s", "^", "D", "v"])}

# ── Plot 1: M_QP vs a, with M_Zam horizontal reference ───────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
for b in BETAS_REN:
    sc    = sc_by_beta[b]
    M_zam = M_zam_by_beta[b]
    ax.plot(sc["a"], sc["M_QP"], markers[b] + "-",
            color=colors[b], label=rf"$\beta={b}$", ms=6.5)
    if np.isfinite(M_zam):
        ax.axhline(M_zam, ls="--", color=colors[b], lw=0.8, alpha=0.6)
ax.set_xlabel(r"$a$"); ax.set_ylabel(r"$M_\mathrm{QP}$")
ax.legend(fontsize=14, ncol=2)
plt.tight_layout()
fig.savefig(FIGURE_DIR / "soliton_mass_vs_a.pdf", bbox_inches="tight")
plt.close()

# ── Plot 2: M_QP / M_Zam vs a — convergence ratio ────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
for b in BETAS_REN:
    sc    = sc_by_beta[b]
    M_zam = M_zam_by_beta[b]
    if not np.isfinite(M_zam):
        continue
    ratio = sc["M_QP"] / M_zam
    ax.plot(sc["a"], ratio, markers[b] + "-",
            color=colors[b], label=rf"$\beta={b}$", ms=6.5)
ax.axhline(1.0, ls="--", color="gray", lw=1.2, label=r"$M_\mathrm{QP}/M_\mathrm{Zam} = 1$")
ax.set_xlabel(r"$a$",fontsize=14); ax.set_ylabel(r"$M_\mathrm{QP} / M_\mathrm{Zam}$",fontsize=14)
ax.legend(fontsize=14, ncol=2)
plt.tight_layout()
fig.savefig(FIGURE_DIR / "soliton_mass_ratio_vs_a.pdf", bbox_inches="tight")
plt.close()

# ── Zoom window for dispersion / Lorentz cells ────────────────────────────────
# Keep only a-values where M_QP/M_Zam ∈ [M_ZOOM_LO, M_ZOOM_HI]
M_ZOOM_LO = 0.5
M_ZOOM_HI = 2.0

zoom_a_sets = {}
for b in BETAS_REN:
    sc    = sc_by_beta[b]
    M_zam = M_zam_by_beta[b]
    if not np.isfinite(M_zam):
        zoom_a_sets[b] = set()
        continue
    ratio = sc["M_QP"] / M_zam
    zoom_a_sets[b] = set(sc.loc[(ratio >= M_ZOOM_LO) & (ratio <= M_ZOOM_HI), "a"].values)


def _grid_dims(n, max_cols=4):
    """Return (nrows, ncols) for a tight grid of n panels."""
    ncols = min(n, max_cols)
    nrows = (n + ncols - 1) // ncols
    return nrows, ncols

# ── Breather mass ratios M_n / M_n^Zam vs a — one panel per β ────────────────
br_n_colors  = {1: "C0", 2: "C1", 3: "C2", 4: "C3"}
br_n_markers = {1: "o",  2: "s",  3: "^",  4: "D"}
max_n = 2

n_betas_br = len(BETAS_REN)
nrows_br, ncols_br = _grid_dims(n_betas_br)
fig_br, axes_flat = plt.subplots(nrows_br, ncols_br,
                                  figsize=(5.5 * ncols_br, 4.5 * nrows_br),
                                  squeeze=False)
axes_br = list(axes_flat.flatten())
for ax in axes_br[n_betas_br:]:
    ax.set_visible(False)

for ax, b in zip(axes_br, BETAS_REN):
    items = breather_by_beta[b]
    if not items:
        ax.text(0.5, 0.5, "no breather data", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_title(rf"$\beta = {b}$")
        continue

    data_by_n = {}
    for it in sorted(items, key=lambda x: x["a"]):
        for _, row in it["df"].iterrows():
            n = int(row["n"])
            data_by_n.setdefault(n, {"a": [], "ratio": []})
            data_by_n[n]["a"].append(it["a"])
            data_by_n[n]["ratio"].append(row["ratio"])

    for n in sorted(n for n in data_by_n if n <= max_n):
        col = br_n_colors.get(n, f"C{n}")
        mrk = br_n_markers.get(n, "x")
        ax.plot(data_by_n[n]["a"], data_by_n[n]["ratio"],
                mrk + "-", color=col, ms=6.5, label=rf"$n={n}$")

    ax.axhline(1.0, ls="--", color="gray", lw=1.2)
    ax.set_xlabel(r"$a$")
    ax.set_ylabel(r"$M_n^\mathrm{B} / M_{n,\mathrm{Zam}}^\mathrm{B}$")
    ax.set_title(rf"$\beta = {b}$")
    ax.legend(fontsize=14)

fig_br.suptitle(
    r"Breather mass ratios $M_n^\mathrm{B} / M_{n,\mathrm{Zam}}^\mathrm{B} \to 1$ as $a \to 0$",
    fontsize=12)
plt.tight_layout()
fig_br.savefig(FIGURE_DIR / "breather_mass_ratio_vs_a.pdf", bbox_inches="tight")
plt.close()


cmap_disp = plt.cm.plasma

def _plot_disp_grid(a_sets_by_beta, suptitle=""):
    """Plot a grid of dispersion panels (one per β).

    a_sets_by_beta: dict {β: set of a-values to include}
    """
    # MODIFIED
    betas_to_plot = [b for b in BETAS_REN if a_sets_by_beta.get(b) and b != 1.0]
    if not betas_to_plot:
        print("No data to plot."); return None

    nrows, ncols = _grid_dims(len(betas_to_plot))
    fig, axes_flat = plt.subplots(nrows, ncols,
                                  figsize=(5.5 * ncols, 5 * nrows),
                                  sharey=False, squeeze=False)
    axes = list(axes_flat.flatten())
    for ax in axes[len(betas_to_plot):]:
        ax.set_visible(False)

    for ax, b in zip(axes, betas_to_plot):
        sc_b      = sc_by_beta[b]
        M_by_a_b  = dict(zip(sc_b["a"].values, sc_b["M_QP"].values))
        items     = [it for it in disp_by_beta[b] if it["a"] in a_sets_by_beta[b]]
        if not items:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes); ax.set_title(rf"$\beta = {b}$"); continue
        a_list = sorted([it["a"] for it in items])
        norm_b = plt.Normalize(min(a_list), max(a_list))

        for item in sorted(items, key=lambda x: x["a"]):
            a_v   = item["a"]
            df    = item["df"]
            color = cmap_disp(norm_b(a_v))
            pos   = df[df["p_phys"] >= 0]
            ax.plot(pos["p_phys"], pos["E_QP"],  "o", ms=6.5, color=color, zorder=3)
            ax.plot(pos["p_phys"], pos["E_fit"],  "-", lw=1.5, color=color,
                    label=f"$a={a_v:.2g}$")
            neg = df[df["p_phys"] <= 0]
            ax.plot(np.abs(neg["p_phys"]), neg["E_QP"],  "o", ms=6.5, color=color)
            ax.plot(np.abs(neg["p_phys"]), neg["E_fit"],  "-", lw=1.5, color=color)

        # relativistic reference using smallest-a M_QP in this panel
        M_ref = M_by_a_b[min(a_list)]
        k_max = max(df["p_phys"].abs().max()
                    for it in items for df in [it["df"]])
        k_ref = np.linspace(0, 1.05 * k_max, 300)
        ax.plot(k_ref, np.sqrt(M_ref**2 + k_ref**2), "k--", lw=1.5,
                label=rf"$\sqrt{{M^2+k^2}},\ M={M_ref:.3f}$")

        ax.set_xlabel(r"$|k_\mathrm{phys}|$")
        ax.set_ylabel(r"$E(k)$")
        ax.set_title(rf"$\beta = {b}$")
        ax.legend(fontsize=7, ncol=2)

    if suptitle:
        fig.suptitle(suptitle, y=1.02, fontsize=12)
    plt.tight_layout()
    return fig


# ── Grid 1: all a-values ──────────────────────────────────────────────────────
all_a_sets = {b: set(sc_by_beta[b]["a"].values) for b in BETAS_REN}
fig_all = _plot_disp_grid(all_a_sets, suptitle="")
fig_all.savefig(FIGURE_DIR / "dispersion_grid_all.pdf", bbox_inches="tight")
plt.close()

# ── Grid 2: only a-values with M_QP/M_Zam in zoom window ─────────────────────
# zoom_a_sets defined in cell 5 (based on M_QP/M_Zam ∈ [M_ZOOM_LO, M_ZOOM_HI])
fig_zoom = _plot_disp_grid(
    zoom_a_sets,
    suptitle=rf"soliton dispersion — $M_\mathrm{{QP}}/M_\mathrm{{Zam}} \in [{M_ZOOM_LO},\,{M_ZOOM_HI}]$ only"
)
fig_zoom.savefig(FIGURE_DIR / "dispersion_grid_zoom.pdf", bbox_inches="tight")
plt.close()

# -- Separate normalized dispersion figure: beta=1.5 and beta=2.5 -----------
normalized_betas = [b for b in (1.5, 2.5) if b in BETAS_REN and all_a_sets.get(b)]
items_by_beta = {b: [it for it in disp_by_beta[b]
                  if it["a"] in all_a_sets[b]] for b in normalized_betas}
a_values = np.array(sorted({it["a"] for items in items_by_beta.values() for it in items}))
norm_a = plt.Normalize(a_values.min(), a_values.max())

fig_disp_norm = plt.figure(figsize=(12, 5))
gs_norm = fig_disp_norm.add_gridspec(1, 3, width_ratios=[1, 1, 0.07],
                                     wspace=0.28)
axes_norm = [fig_disp_norm.add_subplot(gs_norm[0, 0]),
             fig_disp_norm.add_subplot(gs_norm[0, 1])]
cbar_ax = fig_disp_norm.add_subplot(gs_norm[0, 2])

for ax, b in zip(axes_norm, normalized_betas):
    sc_b     = sc_by_beta[b]
    M_by_a_b = dict(zip(sc_b["a"].values, sc_b["M_QP"].values))
    items    = items_by_beta[b]

    for item in sorted(items, key=lambda x: x["a"]):
        a_v   = item["a"]
        M     = M_by_a_b[a_v]
        df    = item["df"]
        color = cmap_disp(norm_a(a_v))
        pos   = df[df["p_phys"] >= 0]
        ax.plot(pos["p_phys"].values / M, pos["E_QP"].values / M,
                "-o", lw=1.5, ms=6.5, color=color, zorder=3)

    x_max = max(it["df"]["p_phys"].abs().max() / M_by_a_b[it["a"]]
                for it in items)
    x_ref = np.linspace(0, 1.05 * x_max, 300)
    ax.plot(x_ref, np.sqrt(1 + x_ref**2), "k--", lw=1.5)

    if ax is axes_norm[0]:
        ax.set_xlabel(r"$|k_\mathrm{phys}|/M_\mathrm{QP}$")
        ax.set_ylabel(r"$E(k)/M_\mathrm{QP}$")
    ax.set_title(rf"$\beta = {b}$")

sm = plt.cm.ScalarMappable(norm=norm_a, cmap=cmap_disp)
sm.set_array([])
cbar = fig_disp_norm.colorbar(sm, cax=cbar_ax)
cbar.set_label(r"$a$")
fig_disp_norm.savefig(FIGURE_DIR / "dispersion_normalized.pdf",
                      bbox_inches="tight")
plt.close()


# ── Lorentz invariance — collapsed dispersion E(k)/M vs k/M, one panel per β ─
# Only a-values with M_QP in [M_ZOOM_LO, M_ZOOM_HI] are shown (same filter as
# the zoom dispersion grid above).  Curves should collapse onto √(1+(k/M)²).

betas_to_plot = [b for b in BETAS_REN if zoom_a_sets.get(b) and b != 1.0]

nrows, ncols = _grid_dims(len(betas_to_plot))
fig_lor, axes_flat = plt.subplots(nrows, ncols,
                                  figsize=(5.5 * ncols, 5 * nrows),
                                  squeeze=False)
axes_lor = list(axes_flat.flatten())
for ax in axes_lor[len(betas_to_plot):]:
    ax.set_visible(False)

for ax, b in zip(axes_lor, betas_to_plot):
    sc_b   = sc_by_beta[b]
    M_by_a = dict(zip(sc_b["a"].values, sc_b["M_QP"].values))
    good_a = zoom_a_sets.get(b, set())          # M_QP ∈ [M_ZOOM_LO, M_ZOOM_HI]
    items  = [it for it in disp_by_beta[b] if it["a"] in good_a]
    # MODIFIED
    if not items or b == 1.0:  # skip β=1.0 (no converged points in zoom window)
        ax.text(0.5, 0.5, "no converged\npoints in window",
                ha="center", va="center", transform=ax.transAxes, fontsize=10)
        ax.set_title(rf"$\beta = {b}$")
        # remove unused axes for β=1.0 (no converged points in zoom window)
        if b == 1.0:
            ax.set_visible(False)
        continue
    

    a_list = sorted(it["a"] for it in items)
    norm_b = plt.Normalize(min(a_list), max(a_list))

    x_max = max(
        pos["p_phys"].max() / M_by_a[it["a"]]
        for it in items
        for pos in [it["df"][it["df"]["p_phys"] >= 0]]
    )
    x_ref = np.linspace(0, 1.05 * x_max, 400)
    ax.plot(x_ref, np.sqrt(1 + x_ref**2), "k--", lw=2,
            label=r"$\sqrt{1+(k/M)^2}$", zorder=10)

    for item in sorted(items, key=lambda x: x["a"]):
        a_v   = item["a"]
        M     = M_by_a[a_v]
        df    = item["df"]
        color = cmap_disp(norm_b(a_v))
        pos   = df[df["p_phys"] >= 0]
        ax.plot(pos["p_phys"].values / M, pos["E_QP"].values / M,
                "o", ms=6.5, color=color, alpha=0.6, zorder=3)
        ax.plot(pos["p_phys"].values / M, pos["E_fit"].values / M,
                "-", lw=1.5, color=color, label=f"$a={a_v:.2g}$")

    ax.set_xlabel(r"$k_\mathrm{phys}/M_\mathrm{QP}$")
    ax.set_ylabel(r"$E(k)/M_\mathrm{QP}$")
    ax.set_title(rf"$\beta = {b}$")
    ax.legend(fontsize=7, ncol=2)

fig_lor.suptitle(
    rf"Dispersion normalized to mass  —  $M_\mathrm{{QP}} \in [{M_ZOOM_LO},\,{M_ZOOM_HI}]$ only",
    fontsize=12)
plt.tight_layout()
fig_lor.savefig(FIGURE_DIR / "lorentz_collapse.pdf", bbox_inches="tight")
plt.close()


# ── Lorentz-invariance deviation vs a  +  dispersion on log-y axis ───────────
# Panel 0: deviation from relativistic dispersion vs a (all β overlaid)
# Panels 1…N: E(k) on log-y axis per β
# Same convergence filter: zoom_a_sets (M_QP ∈ [M_ZOOM_LO, M_ZOOM_HI])

n_panels = len(BETAS_REN) + 1
nrows, ncols = _grid_dims(n_panels)
fig_dev_log, axes_flat = plt.subplots(nrows, ncols,
                                      figsize=(5.5 * ncols, 5 * nrows),
                                      squeeze=False)
axes_dl = list(axes_flat.flatten())
for ax in axes_dl[n_panels:]:
    ax.set_visible(False)
ax_lor_dev = axes_dl[0]

fig_lor, ax_lor = plt.subplots(1,1, figsize=(6,5))
# ─── Panel 0: deviation from relativistic dispersion vs a ────────────────────
for b in BETAS_REN:
    sc_b   = sc_by_beta[b]
    M_by_a = dict(zip(sc_b["a"].values, sc_b["M_QP"].values))
    good_a = zoom_a_sets.get(b, set())

    a_vals_b, devs_b = [], []
    for item in disp_by_beta[b]:
        if item["a"] not in good_a:
            continue
        M     = M_by_a[item["a"]]
        df    = item["df"]
        k     = df["p_phys"].abs().values
        E_rel = np.sqrt(M**2 + k**2)
        nz    = E_rel > 0
        devs_b.append(np.max(np.abs(df["E_fit"].values[nz] / E_rel[nz] - 1.0)))
        a_vals_b.append(item["a"])
    if a_vals_b:
        idx = np.argsort(a_vals_b)
        ax_lor_dev.semilogy(np.array(a_vals_b)[idx], np.array(devs_b)[idx],
                            markers[b] + "-", color=colors[b],
                            label=rf"$\beta={b}$", ms=6.5)
        ax_lor.semilogy(np.array(a_vals_b)[idx], np.array(devs_b)[idx],
                        markers[b] + "-", color=colors[b],
                        label=rf"$\beta={b}$", ms=6.5)
ax_lor.set_xlabel(r"$a$")
ax_lor.set_ylabel(r"$\max_k\left|E_\mathrm{fit}/\sqrt{M^2+k^2}-1\right|$")
ax_lor.legend(fontsize=14)
# Show and save
fig_lor.savefig(FIGURE_DIR / "lorentz_deviation.pdf", bbox_inches="tight")
plt.close()

ax_lor_dev.set_xlabel(r"$a$")
ax_lor_dev.set_ylabel(r"$\max_k\left|E_\mathrm{fit}/\sqrt{M^2+k^2}-1\right|$")
ax_lor_dev.legend(fontsize=14)

# ─── Panels 1…N: dispersion E(k) on log-y, per β ─────────────────────────────
for ax, b in zip(axes_dl[1:], BETAS_REN):
    sc_b   = sc_by_beta[b]
    M_by_a = dict(zip(sc_b["a"].values, sc_b["M_QP"].values))
    good_a = zoom_a_sets.get(b, set())
    items  = [it for it in disp_by_beta[b] if it["a"] in good_a]
    if not items:
        ax.text(0.5, 0.5, "no converged\npoints in window",
                ha="center", va="center", transform=ax.transAxes, fontsize=10)
        ax.set_title(rf"$\beta = {b}$")
        continue

    a_list = sorted(it["a"] for it in items)
    norm_b = plt.Normalize(min(a_list), max(a_list))

    for item in sorted(items, key=lambda x: x["a"]):
        a_v   = item["a"]
        M     = M_by_a[a_v]
        df    = item["df"]
        color = cmap_disp(norm_b(a_v))
        pos   = df[df["p_phys"] >= 0]
        ax.semilogy(pos["p_phys"].values, pos["E_QP"].values,
                    "o", ms=6.5, color=color, alpha=0.6, zorder=3)
        ax.semilogy(pos["p_phys"].values, pos["E_fit"].values,
                    "-", lw=1.5, color=color, label=f"$a={a_v:.2g}$")

    M_ref = M_by_a[min(a_list)]
    k_max = max(it["df"]["p_phys"].abs().max() for it in items)
    k_ref = np.linspace(0, 1.05 * k_max, 300)
    ax.semilogy(k_ref, np.sqrt(M_ref**2 + k_ref**2), "k--", lw=1.5,
                label=rf"$\sqrt{{M^2+k^2}},\ M={M_ref:.3f}$")

    ax.set_xlabel(r"$k_\mathrm{phys}$")
    ax.set_ylabel(r"$E(k)$  [log scale]")
    ax.set_title(rf"$\beta = {b}$")
    ax.legend(fontsize=7, ncol=2)

fig_dev_log.suptitle(
    r"Lorentz deviation vs $a$  |  Dispersion $E(k)$ — log $y$-axis",
    y=1.02, fontsize=12)
plt.tight_layout()
fig_dev_log.savefig(FIGURE_DIR / "lorentz_deviation_and_logdisp.pdf", bbox_inches="tight")
plt.close()


# ── Effective lattice velocity and dispersion fit residuals — per β ───────────
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

for b in BETAS_REN:
    sc   = sc_by_beta[b]
    good = ~bad_by_beta[b]
    col, mrk = colors[b], markers[b]

    a_v    = sc.loc[good, "a"].values
    c2_lat = sc.loc[good, "C"].values * a_v**2 / 4.0
    axes[0].plot(a_v, c2_lat, mrk + "-", color=col, label=rf"$\beta={b}$", ms=6.5)

axes[0].axhline(1.0, ls="--", color="k", lw=1.2, label=r"$c=1$ (continuum)")
axes[0].set_xlabel(r"$a$")
axes[0].set_ylabel(r"$C\,a^2/4$")
axes[0].legend(fontsize=14)

for b in BETAS_REN:
    sc         = sc_by_beta[b]
    good_a_set = set(sc.loc[~bad_by_beta[b], "a"].values)
    col, mrk   = colors[b], markers[b]

    a_disp_b, max_rel_b = [], []
    for item in disp_by_beta[b]:
        if item["a"] not in good_a_set:
            continue
        df    = item["df"]
        resid = np.abs((df["E_QP"] - df["E_fit"]) / df["E_QP"])
        a_disp_b.append(item["a"])
        max_rel_b.append(resid.max())
    if a_disp_b:
        axes[1].semilogy(a_disp_b, max_rel_b, mrk + "-", color=col,
                         label=rf"$\beta={b}$", ms=6.5)

axes[1].set_xlabel(r"$a$")
axes[1].set_ylabel(r"$\max_k\,|E_\mathrm{QP} - E_\mathrm{fit}|/E_\mathrm{QP}$")
axes[1].legend(fontsize=14)

plt.tight_layout()
fig.savefig(FIGURE_DIR / "lattice_velocity.pdf", bbox_inches="tight")
plt.close()


# ── Vacuum diagnostics — per β ────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

for b in BETAS_REN:
    sc  = sc_by_beta[b]
    col = colors[b]
    mrk = markers[b]
    axes[0].semilogy(sc["a"].values, sc["delta_e"].abs().values,
                     mrk + "-", color=col, label=rf"$\beta={b}$", ms=6.5)

axes[0].axhline(BAD_DE, ls="--", color="k", lw=1.0, label=f"threshold {BAD_DE:.0e}")
axes[0].set_xlabel(r"$a$")
axes[0].set_ylabel(r"$|\varepsilon_0 - \varepsilon_1|$")
axes[0].legend(fontsize=14)

for b in BETAS_REN:
    sc         = sc_by_beta[b]
    col        = colors[b]
    mrk        = markers[b]
    phi_target = 2 * np.pi / b
    axes[1].plot(sc["a"].values, sc["phi_vac0"].values,
                 mrk + "-",  color=col, ms=6.5, label=rf"$\langle\phi\rangle_0,\ \beta={b}$")
    axes[1].plot(sc["a"].values, sc["phi_vac1"].values,
                 mrk + "--", color=col, ms=6.5,
                 label=rf"$\langle\phi\rangle_1,\ \beta={b}$ (tgt {phi_target:.2f})")

axes[1].set_xlabel(r"$a$")
axes[1].set_ylabel(r"$\langle\phi\rangle$")
axes[1].legend(fontsize=7, ncol=2)

plt.tight_layout()
fig.savefig(FIGURE_DIR / "vacuum_diagnostics.pdf", bbox_inches="tight")
plt.close()
