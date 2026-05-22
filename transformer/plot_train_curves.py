"""
Plot training loss curves for the five scaling runs, either SP (Part 2) or muP
(Part 3).

Reads `train_losses` arrays from per-model JSON files in the chosen runs
directory and produces a single overlay figure with one line per model size.

Usage:
    # SP (Part 2) — default
    python transformer/plot_train_curves.py
    # muP (Part 3)
    python transformer/plot_train_curves.py --mup

    # Override paths explicitly
    python transformer/plot_train_curves.py \
        --mup --runs_dir transformer/runs/mup \
        --out train_curves_mup.png
"""
import json
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


SIZES  = ["tiny", "small", "medium", "large", "xl"]
COLORS = {
    "tiny":   "#4477AA",
    "small":  "#66CCEE",
    "medium": "#228833",
    "large":  "#EE6677",
    "xl":     "#AA3377",
}


def main():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--mup", action="store_true",
                   help="Plot muP runs (Part 3) instead of SP runs (Part 2)")
    p.add_argument("--runs_dir", default=None,
                   help="Override directory holding the *_results.json files")
    p.add_argument("--out", default=None,
                   help="Override output PNG path")
    p.add_argument("--smooth", type=int, default=5,
                   help="Boxcar smoothing window in samples (0 = none)")
    args = p.parse_args()

    # Defaults that flip together based on --mup
    if args.mup:
        runs_dir      = Path(args.runs_dir or "transformer/runs/mup")
        file_pattern  = "{size}_lr1e-02_mup_results.json"
        out_path      = Path(args.out or "train_curves_mup.png")
        title         = (
            r"$\mu$P training loss curves "
            r"(1 epoch, $\eta_{\max}=10^{-2}$)"
        )
    else:
        runs_dir      = Path(args.runs_dir or "transformer/runs")
        file_pattern  = "{size}_lr1e-02_results.json"
        out_path      = Path(args.out or "train_curves_sp.png")
        title         = (
            r"SP training loss curves "
            r"(1 epoch, $\eta_{\max}=10^{-2}$)"
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4.5))

    for size in SIZES:
        path = runs_dir / file_pattern.format(size=size)
        if not path.exists():
            print(f"  missing: {path}")
            continue
        d      = json.load(open(path))
        steps  = np.array([t["step"] for t in d["train_losses"]])
        losses = np.array([t["loss"] for t in d["train_losses"]])

        # Light smoothing to reduce per-step noise
        if args.smooth > 1 and len(losses) > args.smooth:
            kernel = np.ones(args.smooth) / args.smooth
            losses = np.convolve(losses, kernel, mode="same")

        ax.plot(
            steps, losses,
            color=COLORS[size], linewidth=1.6,
            label=f"{size.capitalize()} ({d['n_params']/1e6:.1f}M params)",
        )

    ax.set_xlabel("Training step", fontsize=11)
    ax.set_ylabel("Training loss (nats / token)", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_yscale("log")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, which="both", alpha=0.25, linestyle="--")
    ax.grid(True, which="major", alpha=0.45)
    ax.set_ylim(bottom=0.5)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()