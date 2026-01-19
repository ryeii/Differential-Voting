#!/usr/bin/env python3
# Plotting script for Differential Voting experiment outputs (stitched PDFs)
#
# Usage:
#   !python plot.py --root results --out stitched_pdfs
#
# Outputs:
#   stitched_pdfs/
#     exp1_all_scenarios_8panel.pdf
#     exp2_stitched.pdf
#     exp3_geometry_stitched.pdf
#
# Dependencies: numpy, matplotlib

import os
import argparse
import glob
import numpy as np
import matplotlib.pyplot as plt


# ----------------------------
# Helpers
# ----------------------------

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def load_npz(path: str):
    return np.load(path, allow_pickle=True)

def save_pdf(fig, path: str):
    fig.tight_layout()
    fig.savefig(path, format="pdf")
    plt.close(fig)

def set_big_fonts(scale: float = 1.35):
    # Global Matplotlib font scaling
    base = 10.0
    plt.rcParams.update({
        "font.size": base * scale,
        "axes.titlesize": base * scale * 1.05,
        "axes.labelsize": base * scale * 1.05,
        "xtick.labelsize": base * scale * 1.05,
        "ytick.labelsize": base * scale * 1.05,
        "legend.fontsize": base * scale * 1.05,
        "figure.titlesize": base * scale * 1.05,
        "lines.linewidth": 2.0,
        "lines.markersize": 6.0,
    })


# ----------------------------
# Exp1: 8-panel (4x2) stitched figure
# ----------------------------

def exp1_stitch_8panel(root: str, out_dir: str):
    exp1_dir = os.path.join(root, "exp1")
    scenario_files = sorted(glob.glob(os.path.join(exp1_dir, "scenario_*.npz")))
    if not scenario_files:
        print(f"[Exp1] No scenario_*.npz files found in {exp1_dir}")
        return

    # Expecting 4 scenarios for an 8-panel 4x2 figure.
    # If more than 4 exist, take the first 4 (alphabetical); if fewer, stitch what exists.
    scenario_files = scenario_files[:4]

    # Layout: 2 rows x 4 cols = 8 panels.
    # We'll place (Copeland, Kemeny) per scenario as two vertical panels in each column:
    #   Row 0: Copeland for scenario i
    #   Row 1: Kemeny for scenario i
    ncols = len(scenario_files)
    nrows = 2

    fig = plt.figure(figsize=(4.6 * ncols, 3.3 * nrows))
    # fig.suptitle("Exp1: Recovery vs smoothing across scenarios", y=1.02)

    for col, f in enumerate(scenario_files):
        d = load_npz(f)
        scenario = str(d["scenario"])
        taus = d["taus"].astype(float)

        # Copeland panel data
        btl_hit_mean = float(d["btl_hit_mean"])
        best_cop_by_tau = d["best_cop_by_tau"].astype(float)

        # Kemeny panel data
        btl_kd_mean = float(d["btl_kd_mean"])
        kem_kd_mean = d["kem_kd_mean"].astype(float)
        kem_kd_std = d["kem_kd_std"].astype(float)

        # ---- Row 0: Copeland recovery ----
        ax1 = fig.add_subplot(nrows, ncols, 1 + col)
        ax1.plot(taus, [btl_hit_mean] * len(taus), marker="o", label="BTL (tau=0.2)")
        ax1.plot(taus, best_cop_by_tau, marker="o", label="Soft Copeland (best β,λ)")
        ax1.invert_xaxis()
        ax1.set_xlabel("tau")
        ax1.set_ylabel("Match Copeland winner")
        ax1.set_title(f"{scenario}\nCopeland")
        ax1.grid(True, alpha=0.25)

        # Put legends only on first column to reduce clutter
        if col == 0:
            ax1.legend(loc="best")

        # ---- Row 1: Kemeny recovery ----
        ax2 = fig.add_subplot(nrows, ncols, 1 + ncols + col)
        ax2.plot(taus, [btl_kd_mean] * len(taus), marker="o", label="BTL (tau=0.2)")
        ax2.errorbar(taus, kem_kd_mean, yerr=kem_kd_std, marker="o", capsize=3, label="Soft Kemeny")
        ax2.invert_xaxis()
        ax2.set_xlabel("tau")
        ax2.set_ylabel("Kendall dist to Kemeny")
        ax2.set_title(f"{scenario}\nKemeny")
        ax2.grid(True, alpha=0.25)
        if col == 0:
            ax2.legend(loc="best")

    out_path = os.path.join(out_dir, "exp1_all_scenarios_8panel.pdf")
    save_pdf(fig, out_path)
    print(f"[Exp1] Wrote {out_path}")

    out_path = os.path.join(out_dir, "exp1_all_scenarios_8panel.png")
    fig.savefig(out_path, format="png", dpi=150)
    print(f"[Exp1] Wrote {out_path} (PNG copy)")


# ----------------------------
# Exp2: stitched 1x3 (single shared legend)
# ----------------------------

def exp2_stitch(root: str, out_dir: str):
    exp2_path = os.path.join(root, "exp2", "exp2_mid_raw.npz")
    if not os.path.exists(exp2_path):
        print(f"[Exp2] Missing {exp2_path}")
        return

    d = load_npz(exp2_path)
    pi1_grid = d["pi1_grid"].astype(float)
    method_names = list(d["method_names"])
    cw_exists_rate = d["cw_exists_rate"].astype(float)
    cond_hit_mean = d["cond_hit_mean"].astype(float)
    kdist_mean = d["kdist_mean"].astype(float)
    cop_agree_mean = d["cop_agree_mean"].astype(float)

    fig = plt.figure(figsize=(18, 4))

    # --- Panel 1: Condorcet criterion ---
    ax1 = fig.add_subplot(1, 3, 1)
    line_handles = []
    line_labels = []

    for mi, mn in enumerate(method_names):
        h, = ax1.plot(pi1_grid, cond_hit_mean[mi], marker="o", label=str(mn))
        line_handles.append(h)
        line_labels.append(str(mn))

    h_cw, = ax1.plot(
        pi1_grid, cw_exists_rate,
        marker="x", linewidth=3, linestyle="--",
        label="Condorcet exists rate"
    )
    line_handles.append(h_cw)
    line_labels.append("Condorcet exists rate")

    ax1.set_xlabel("pi1 (context 0 weight)")
    ax1.set_ylabel("Condorcet hit-rate")
    ax1.set_title("Condorcet criterion")
    ax1.grid(True, alpha=0.25)

    # --- Panel 2: Kemeny distance ---
    ax2 = fig.add_subplot(1, 3, 2)
    for mi, mn in enumerate(method_names):
        ax2.plot(pi1_grid, kdist_mean[mi], marker="o")
    ax2.set_xlabel("pi1 (context 0 weight)")
    ax2.set_ylabel("Kendall distance to Kemeny")
    ax2.set_title("Distance to Kemeny optimum")
    ax2.grid(True, alpha=0.25)

    # --- Panel 3: Copeland agreement ---
    ax3 = fig.add_subplot(1, 3, 3)
    for mi, mn in enumerate(method_names):
        ax3.plot(pi1_grid, cop_agree_mean[mi], marker="o")
    ax3.set_xlabel("pi1 (context 0 weight)")
    ax3.set_ylabel("Match Copeland winner")
    ax3.set_title("Copeland-winner agreement")
    ax3.grid(True, alpha=0.25)

    # --- Single shared legend (bottom center) ---
    fig.legend(
        line_handles,
        line_labels,
        loc="lower center",
        ncol=min(len(line_labels), 4),
        frameon=False,
        bbox_to_anchor=(0.5, -0.05)
    )

    # Leave room for legend
    fig.tight_layout(rect=[0, 0.08, 1, 1])

    out_path = os.path.join(out_dir, "exp2_stitched.pdf")
    save_pdf(fig, out_path)
    print(f"[Exp2] Wrote {out_path}")

    out_path = os.path.join(out_dir, "exp2_stitched.png")
    fig.savefig(out_path, format="png", dpi=150)
    print(f"[Exp2] Wrote {out_path} (PNG copy)")



# ----------------------------
# Exp3: geometry only (discard noise)
# ----------------------------

def exp3_geometry_stitch(root: str, out_dir: str):
    exp3_dir = os.path.join(root, "exp3")
    geo_files = sorted(glob.glob(os.path.join(exp3_dir, "geo_*.npz")))
    if not geo_files:
        print(f"[Exp3-geo] No geo_*.npz files found in {exp3_dir}")
        return

    # Prefer known set if present, else take first 4.
    preferred = [
        "geo_BTL_tau0.2.npz",
        "geo_Kem_tau0.1.npz",
        "geo_Cop_tau0.1_b2.npz",
        "geo_Cop_tau0.1_b10.npz",
    ]
    pref_paths = [os.path.join(exp3_dir, p) for p in preferred if os.path.exists(os.path.join(exp3_dir, p))]
    if len(pref_paths) == 4:
        geo_files = pref_paths
    else:
        geo_files = geo_files[:4]

    fig = plt.figure(figsize=(10, 8))
    # fig.suptitle("Exp3: Loss geometry (gradient vs margin)", y=1.02)

    for i, f in enumerate(geo_files):
        d = load_npz(f)
        setup_name = str(d["setup_name"])
        bins = d["bins"].astype(float)
        delta = d["delta"].astype(float)
        gradabs = d["gradabs"].astype(float)

        centers = 0.5 * (bins[:-1] + bins[1:])
        inds = np.digitize(delta, bins) - 1
        vals = np.zeros_like(centers)
        counts = np.zeros_like(centers, dtype=int)
        for j in range(len(delta)):
            k = inds[j]
            if 0 <= k < len(centers):
                vals[k] += gradabs[j]
                counts[k] += 1
        vals = vals / np.maximum(counts, 1)

        ax = fig.add_subplot(2, 2, i + 1)
        ax.plot(centers, vals, marker="o")
        ax.set_xlabel("Margin Δ")
        ax.set_ylabel("Avg |∂ℓ/∂Δ|")
        ax.set_title(setup_name)
        ax.grid(True, alpha=0.25)

    out_path = os.path.join(out_dir, "exp3_geometry_stitched.pdf")
    save_pdf(fig, out_path)
    print(f"[Exp3-geo] Wrote {out_path}")

    out_path = os.path.join(out_dir, "exp3_geometry_stitched.png")
    fig.savefig(out_path, format="png", dpi=150)
    print(f"[Exp3-geo] Wrote {out_path} (PNG copy)")


# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="results_mid_raw", help="Root results directory produced by the experiment script")
    ap.add_argument("--out", type=str, default="stitched_pdfs", help="Output directory for stitched PDFs")
    ap.add_argument("--font_scale", type=float, default=1.45, help="Global font scale multiplier")
    args = ap.parse_args()

    set_big_fonts(scale=args.font_scale)

    root = args.root
    out_dir = args.out
    ensure_dir(out_dir)

    exp1_stitch_8panel(root, out_dir)
    exp2_stitch(root, out_dir)
    exp3_geometry_stitch(root, out_dir)

    print(f"\nAll done. PDFs in: {os.path.abspath(out_dir)}")

if __name__ == "__main__":
    main()
