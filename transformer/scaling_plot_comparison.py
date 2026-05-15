"""
Scaling law comparison: Standard Parameterization (SP) vs muP (Part 3).

Reads SP results from transformer/runs/ and muP results from transformer/runs/mup/,
fits power laws L = a * N^(-alpha) + c to both, and produces:
  1. scaling_comparison.png  — three-panel figure (SP alone, muP alone, overlay)
  2. lr_sweep_comparison.png — LR sweep comparison for the Tiny model

Usage:
    python transformer/scaling_plot_comparison.py
    python transformer/scaling_plot_comparison.py \\
        --sp_runs_dir transformer/runs \\
        --mup_runs_dir transformer/runs/mup \\
        --out transformer/scaling_comparison.png \\
        --lr_sweep_out transformer/lr_sweep_comparison.png
"""

import sys
import json
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from scaling_plot import power_law, fit_power_law, extrapolate


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_results(runs_dir: Path, glob_pattern: str = "*_results.json") -> dict[int, dict]:
    """Load best val_loss result per unique n_params from a directory."""
    best: dict[int, dict] = {}
    for f in sorted(runs_dir.glob(glob_pattern)):
        with open(f) as fp:
            r = json.load(fp)
        if r.get("val_loss") is None:
            continue
        n = r["n_params"]
        if n not in best or r["val_loss"] < best[n]["val_loss"]:
            best[n] = r
    return best


def load_lr_sweep(path: Path) -> tuple[list[float], list[float]]:
    """Load lr and val_loss lists from a lr_sweep*.json file."""
    with open(path) as f:
        d = json.load(f)
    entries = d["sweep"]
    lrs    = [e["lr"]       for e in entries]
    losses = [e["val_loss"] for e in entries]
    return lrs, losses


# ---------------------------------------------------------------------------
# Single-curve panel helper
# ---------------------------------------------------------------------------

def _plot_curve(
    ax,
    params: np.ndarray,
    losses: np.ndarray,
    labels: list[str],
    color: str,
    title: str,
    popt=None,
    pcov=None,
    show_extrap: bool = True,
) -> None:
    """Draw data points + fitted power law + extrapolation on a single axes."""
    if popt is not None:
        a, alpha, c = popt
        N_lo  = params[0] * 0.6
        N_hi  = params[-1] * 10 * 1.5
        N_fit = np.logspace(np.log10(N_lo), np.log10(N_hi), 400)
        L_fit = power_law(N_fit, *popt)
        ax.plot(
            N_fit, L_fit,
            color=color, linewidth=2.0, zorder=2,
            label=rf"Fit: $L={a:.2f}\cdot N^{{-{alpha:.3f}}}+{c:.2f}$  ($\alpha={alpha:.3f}$)",
        )
        ax.axvspan(params[0] * 0.7, params[-1] * 1.3, alpha=0.06, color=color)

        if show_extrap and pcov is not None:
            N_extrap       = params[-1] * 10
            L_extrap, ci95 = extrapolate(popt, pcov, N_extrap)
            ax.errorbar(
                [N_extrap], [L_extrap], yerr=[[ci95], [ci95]],
                fmt="*", markersize=14, color="darkorange",
                ecolor="darkorange", capsize=6, zorder=6,
                label=f"10x extrap: {L_extrap:.3f} (±{ci95:.3f})",
            )

    ax.scatter(params, losses, s=90, zorder=5,
               color=color, edgecolors="white", linewidths=1.5,
               label="Trained models")
    for i, lbl in enumerate(labels):
        ax.annotate(lbl, (params[i], losses[i]),
                    textcoords="offset points", xytext=(7, 3),
                    fontsize=9, color="#333333")

    ax.set_xscale("log")
    ax.set_xlabel("Number of Parameters (N)", fontsize=11)
    ax.set_ylabel("Validation Loss (1 epoch)", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, which="both", alpha=0.25, linestyle="--")
    ax.grid(True, which="major", alpha=0.45)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="SVG Transformer SP vs muP scaling comparison (Part 3)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--sp_runs_dir",  default="transformer/runs",
                   help="Directory with SP *_results.json files (top-level only)")
    p.add_argument("--mup_runs_dir", default="transformer/runs/mup",
                   help="Directory with muP *_mup_results.json files")
    p.add_argument("--out",          default="transformer/scaling_comparison.png")
    p.add_argument("--lr_sweep_out", default="transformer/lr_sweep_comparison.png")
    args = p.parse_args()

    sp_dir  = Path(args.sp_runs_dir)
    mup_dir = Path(args.mup_runs_dir)

    # -- load SP results (top-level only; exclude mup/ subdirectory) --------
    sp_best  = load_results(sp_dir, "*_results.json")
    mup_best = load_results(mup_dir, "*_mup_results.json")

    def to_sorted_arrays(best_dict):
        data   = sorted(best_dict.values(), key=lambda r: r["n_params"])
        params = np.array([r["n_params"] for r in data], dtype=float)
        losses = np.array([r["val_loss"] for r in data], dtype=float)
        labels = [r["model_size"].capitalize() for r in data]
        return data, params, losses, labels

    sp_data,  sp_params,  sp_losses,  sp_labels  = to_sorted_arrays(sp_best)
    mup_data, mup_params, mup_losses, mup_labels = to_sorted_arrays(mup_best)

    # -- print tables --------------------------------------------------------
    for tag, data in [("SP", sp_data), ("muP", mup_data)]:
        if not data:
            continue
        print(f"\n{tag} results:")
        print(f"  {'Model':<10} {'Params':>12}  {'Val Loss':>10}  {'LR':>8}")
        print(f"  {'-'*10} {'-'*12}  {'-'*10}  {'-'*8}")
        for r in data:
            print(f"  {r['model_size']:<10} {r['n_params']:>12,}  {r['val_loss']:>10.4f}  {r['lr']:>8.2e}")

    # -- fit power laws -------------------------------------------------------
    def try_fit(params, losses, tag):
        if len(params) < 3:
            print(f"\n{tag}: need >=3 points for fit (have {len(params)})")
            return None, None
        try:
            popt, pcov = fit_power_law(params, losses)
            return popt, pcov
        except RuntimeError as e:
            print(f"\n{tag}: power law fit failed ({e})")
            return None, None

    sp_popt,  sp_pcov  = try_fit(sp_params,  sp_losses,  "SP")
    mup_popt, mup_pcov = try_fit(mup_params, mup_losses, "muP")

    # -- comparison table ----------------------------------------------------
    print("\n" + "=" * 62)
    print(f"  {'Parameterization':<18} {'alpha':>7}  {'c':>7}  {'10x extrap (±95% CI)'}")
    print(f"  {'-'*18} {'-'*7}  {'-'*7}  {'-'*26}")
    for tag, params, popt, pcov in [
        ("SP",  sp_params,  sp_popt,  sp_pcov),
        ("muP", mup_params, mup_popt, mup_pcov),
    ]:
        if popt is None or not len(params):
            print(f"  {tag:<18} {'N/A':>7}  {'N/A':>7}  N/A")
            continue
        a, alpha, c = popt
        N_extrap        = params[-1] * 10
        L_extrap, ci95  = extrapolate(popt, pcov, N_extrap)
        print(
            f"  {tag:<18} {alpha:>7.4f}  {c:>7.4f}  "
            f"{L_extrap:.4f} ± {ci95:.4f}  "
            f"(N={N_extrap:,.0f})"
        )
    print("=" * 62)

    if sp_popt is not None and mup_popt is not None:
        sp_alpha  = sp_popt[1]
        mup_alpha = mup_popt[1]
        better    = "muP" if mup_alpha > sp_alpha else "SP"
        print(f"\n  Scaling exponent: SP alpha={sp_alpha:.4f}, muP alpha={mup_alpha:.4f}")
        print(f"  -> {better} has steeper scaling (larger alpha = faster loss decrease with scale)")

    # ── Figure 1: three-panel scaling comparison ───────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("SVG Transformer: SP vs muP Scaling Laws (Part 3)", fontsize=13, fontweight="bold")

    if len(sp_params):
        _plot_curve(axes[0], sp_params, sp_losses, sp_labels,
                    color="steelblue", title="Standard Parameterization (SP)",
                    popt=sp_popt, pcov=sp_pcov)
    else:
        axes[0].text(0.5, 0.5, "No SP results found", ha="center", va="center",
                     transform=axes[0].transAxes)

    if len(mup_params):
        _plot_curve(axes[1], mup_params, mup_losses, mup_labels,
                    color="darkorange", title="muP",
                    popt=mup_popt, pcov=mup_pcov)
    else:
        axes[1].text(0.5, 0.5, "No muP results found", ha="center", va="center",
                     transform=axes[1].transAxes)

    # Panel 3: overlay
    ax = axes[2]
    all_params = np.concatenate([sp_params, mup_params]) if (len(sp_params) and len(mup_params)) else (sp_params if len(sp_params) else mup_params)

    for tag, params, losses, labels, color, marker, popt in [
        ("SP",  sp_params,  sp_losses,  sp_labels,  "steelblue",  "o", sp_popt),
        ("muP", mup_params, mup_losses, mup_labels, "darkorange", "s", mup_popt),
    ]:
        if not len(params):
            continue
        alpha_str = f", α={popt[1]:.3f}" if popt is not None else ""
        ax.scatter(params, losses, s=90, zorder=5, marker=marker,
                   color=color, edgecolors="white", linewidths=1.5,
                   label=f"{tag} models{alpha_str}")
        for i, lbl in enumerate(labels):
            ax.annotate(lbl, (params[i], losses[i]),
                        textcoords="offset points", xytext=(7, 3),
                        fontsize=8, color="#333333")
        if popt is not None:
            N_lo  = all_params.min() * 0.6 if len(all_params) else params[0] * 0.6
            N_hi  = all_params.max() * 10 * 1.5 if len(all_params) else params[-1] * 15
            N_fit = np.logspace(np.log10(N_lo), np.log10(N_hi), 400)
            ax.plot(N_fit, power_law(N_fit, *popt),
                    color=color, linewidth=2.0, linestyle="--", zorder=2, alpha=0.8)

    # Comparison annotation
    if sp_popt is not None and mup_popt is not None:
        sp_a  = sp_popt[1]
        mu_a  = mup_popt[1]
        txt   = (
            f"SP α = {sp_a:.3f}\n"
            f"muP α = {mu_a:.3f}\n"
            f"{'muP scales better' if mu_a > sp_a else 'SP scales better'}"
        )
        ax.text(0.03, 0.97, txt, transform=ax.transAxes,
                fontsize=9, va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.8))

    ax.set_xscale("log")
    ax.set_xlabel("Number of Parameters (N)", fontsize=11)
    ax.set_ylabel("Validation Loss (1 epoch)", fontsize=11)
    ax.set_title("SP vs muP: Overlay Comparison", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, which="both", alpha=0.25, linestyle="--")
    ax.grid(True, which="major", alpha=0.45)

    fig.tight_layout()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nScaling comparison plot -> {out_path}")

    # ── Figure 2: LR sweep comparison ─────────────────────────────────────
    sp_sweep_path  = sp_dir  / "lr_sweep" / "lr_sweep.json"
    mup_sweep_path = mup_dir / "lr_sweep" / "lr_sweep_mup.json"

    has_sp_sweep  = sp_sweep_path.exists()
    has_mup_sweep = mup_sweep_path.exists()

    if has_sp_sweep or has_mup_sweep:
        fig2, ax2 = plt.subplots(figsize=(8, 5))

        for path, label, color, marker in [
            (sp_sweep_path,  "SP (Tiny)",  "steelblue",  "o"),
            (mup_sweep_path, "muP (Tiny)", "darkorange", "s"),
        ]:
            if not path.exists():
                continue
            lrs, losses = load_lr_sweep(path)
            best_idx    = int(np.argmin(losses))
            ax2.plot(lrs, losses, color=color, linewidth=1.8,
                     marker=marker, markersize=7, label=label)
            ax2.axvline(lrs[best_idx], color=color, linestyle=":",
                        linewidth=1.5, alpha=0.7,
                        label=f"{label} best LR = {lrs[best_idx]:.2e}")

        ax2.set_xscale("log")
        ax2.set_xlabel("Learning Rate", fontsize=12)
        ax2.set_ylabel("Validation Loss", fontsize=12)
        ax2.set_title("LR Sweep Comparison: SP vs muP (Tiny model)", fontsize=13, fontweight="bold")
        ax2.legend(fontsize=9)
        ax2.grid(True, which="both", alpha=0.25, linestyle="--")
        ax2.grid(True, which="major", alpha=0.45)
        fig2.tight_layout()

        lr_out = Path(args.lr_sweep_out)
        lr_out.parent.mkdir(parents=True, exist_ok=True)
        fig2.savefig(lr_out, dpi=150, bbox_inches="tight")
        print(f"LR sweep comparison plot -> {lr_out}")
    else:
        print("\nNo LR sweep files found — skipping lr_sweep_comparison.png")
        print(f"  Expected: {sp_sweep_path}")
        print(f"  Expected: {mup_sweep_path}")

    plt.close("all")


if __name__ == "__main__":
    main()
