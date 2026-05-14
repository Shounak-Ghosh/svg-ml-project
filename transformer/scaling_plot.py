"""
Scaling law analysis for SVG Transformer (Part 2).

Reads *_results.json files produced by train.py, fits the power law:
    L = a * N^(-alpha) + c
and produces a scaling plot with the fitted curve and a 10x extrapolation.

Usage:
    python transformer/scaling_plot.py
    python transformer/scaling_plot.py --runs_dir transformer/runs --out transformer/scaling_plot.png
"""

import sys
import json
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


# ---------------------------------------------------------------------------
# Power law
# ---------------------------------------------------------------------------

def power_law(N: np.ndarray, a: float, alpha: float, c: float) -> np.ndarray:
    """L = a * N^(-alpha) + c"""
    return a * np.power(N, -alpha) + c


def fit_power_law(
    params: np.ndarray,
    losses: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit L = a * N^(-alpha) + c via nonlinear least squares.

    Initial guess: use a log-log linear fit (ignoring the offset c) to estimate
    a and alpha, then seed c from the minimum observed loss.
    """
    # Bootstrap initial guess from log-log linear regression (c=0 approximation)
    log_N = np.log(params)
    log_L = np.log(losses)
    slope, intercept = np.polyfit(log_N, log_L, 1)
    alpha0 = max(-slope, 0.01)          # power law exponent (should be positive)
    a0     = max(np.exp(intercept), 1e-6)
    c0     = max(losses.min() * 0.5, 1e-6)

    p0     = [a0, alpha0, c0]
    bounds = ([0.0, 0.0, 0.0], [np.inf, 10.0, np.inf])

    popt, pcov = curve_fit(
        power_law, params, losses,
        p0=p0, bounds=bounds,
        maxfev=20_000,
    )
    return popt, pcov


def extrapolate(
    popt: np.ndarray,
    pcov: np.ndarray,
    N_extrap: float,
) -> tuple[float, float]:
    """
    Predict L at N_extrap and return (predicted_loss, 95% half-width CI).
    Uncertainty is propagated from the covariance matrix via the Jacobian.
    """
    a, alpha, c = popt
    L_pred = power_law(np.array([N_extrap]), *popt)[0]

    # Gradient of power_law w.r.t. (a, alpha, c)
    J = np.array([
        N_extrap ** (-alpha),                          # dL/da
        -a * N_extrap ** (-alpha) * np.log(N_extrap), # dL/d_alpha
        1.0,                                           # dL/dc
    ])
    var_L = float(J @ pcov @ J)
    std_L = np.sqrt(max(var_L, 0.0))

    return L_pred, 1.96 * std_L   # 95% CI half-width


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="SVG Transformer scaling law fit and plot (Part 2)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--runs_dir", default="transformer/runs",
                   help="Directory containing *_results.json files from train.py")
    p.add_argument("--out", default="transformer/scaling_plot.png",
                   help="Output path for the saved plot")
    args = p.parse_args()

    # -- load results --------------------------------------------------------
    runs_dir     = Path(args.runs_dir)
    result_files = sorted(runs_dir.glob("*_results.json"))

    if not result_files:
        print(f"No *_results.json files found in {runs_dir}")
        print("Run train.py for each model size first.")
        sys.exit(1)

    # For each unique n_params, keep the run with the lowest val_loss
    # (handles multiple LR sweep runs for the same model size)
    best: dict[int, dict] = {}
    for f in result_files:
        with open(f) as fp:
            r = json.load(fp)
        n = r["n_params"]
        if r["val_loss"] is None:
            continue
        if n not in best or r["val_loss"] < best[n]["val_loss"]:
            best[n] = r

    data   = sorted(best.values(), key=lambda r: r["n_params"])
    params = np.array([r["n_params"] for r in data], dtype=float)
    losses = np.array([r["val_loss"] for r in data], dtype=float)
    labels = [r["model_size"].capitalize() for r in data]

    print("Loaded results:")
    print(f"  {'Model':<10} {'Params':>12}  {'Val Loss':>10}  {'LR':>8}")
    print(f"  {'-'*10} {'-'*12}  {'-'*10}  {'-'*8}")
    for r in data:
        print(f"  {r['model_size']:<10} {r['n_params']:>12,}  {r['val_loss']:>10.4f}  {r['lr']:>8.2e}")

    # -- fit -----------------------------------------------------------------
    can_fit = len(data) >= 3   # need at least 3 points for 3 free params

    if can_fit:
        try:
            popt, pcov = fit_power_law(params, losses)
        except RuntimeError as e:
            print(f"\nWarning: power law fit failed ({e}). Plotting data only.")
            can_fit = False

    if can_fit:
        a, alpha, c = popt
        perr = np.sqrt(np.diag(pcov))

        print("\nFitted power law:  L = a * N^(-alpha) + c")
        print(f"  a     = {a:.4f}  +/-  {perr[0]:.4f}")
        print(f"  alpha  = {alpha:.4f}  +/-  {perr[1]:.4f}   <-- scaling exponent")
        print(f"  c     = {c:.4f}  +/-  {perr[2]:.4f}   (irreducible loss floor)")
        print()
        print(f"  Scaling exponent alpha = {alpha:.4f}")
        print("  Reference: natural language approx. 0.07-0.10 (Kaplan et al., 2020)")
        if alpha < 0.07:
            print("  Interpretation: shallower than NLP - SVG structure may be harder to exploit at scale.")
        elif alpha > 0.10:
            print("  Interpretation: steeper than NLP - SVG's strict syntax may help models scale more efficiently.")
        else:
            print("  Interpretation: similar to NLP scaling laws.")

        # 10x extrapolation
        N_extrap       = params[-1] * 10
        L_extrap, ci95 = extrapolate(popt, pcov, N_extrap)
        print(f"\n10x extrapolation (N = {N_extrap:,.0f} params):")
        print(f"  Predicted val_loss = {L_extrap:.4f}  (+/-)  {ci95:.4f}  (95% CI)")
        print(f"  Range: [{L_extrap - ci95:.4f}, {L_extrap + ci95:.4f}]")

    # -- plot ----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))

    if can_fit:
        # Smooth fitted curve spanning from smallest model to 10x largest
        N_lo  = params[0] * 0.6
        N_hi  = params[-1] * 10 * 1.5
        N_fit = np.logspace(np.log10(N_lo), np.log10(N_hi), 400)
        L_fit = power_law(N_fit, *popt)

        ax.plot(
            N_fit, L_fit,
            color="steelblue", linewidth=2.0, zorder=2,
            label=rf"Fit: $L = {a:.2f} \cdot N^{{-{alpha:.3f}}} + {c:.2f}$  ($\alpha={alpha:.3f}$)",
        )

        # Light shading over the trained range
        ax.axvspan(params[0] * 0.7, params[-1] * 1.3, alpha=0.06, color="steelblue")

        # 10x extrapolation point with error bar
        ax.errorbar(
            [N_extrap], [L_extrap], yerr=[[ci95], [ci95]],
            fmt="*", markersize=14, color="darkorange",
            ecolor="darkorange", capsize=6, zorder=6,
            label=f"10x extrap: {L_extrap:.3f} (+/-) {ci95:.3f}",
        )

    # Trained model data points
    ax.scatter(
        params, losses,
        s=90, zorder=5,
        color="steelblue", edgecolors="white", linewidths=1.5,
        label="Trained models",
    )
    for i, lbl in enumerate(labels):
        ax.annotate(
            lbl, (params[i], losses[i]),
            textcoords="offset points", xytext=(7, 3),
            fontsize=9, color="#333333",
        )

    ax.set_xscale("log")
    ax.set_xlabel("Number of Parameters (N)", fontsize=12)
    ax.set_ylabel("Validation Loss (1 epoch)", fontsize=12)
    ax.set_title("SVG Transformer Scaling Laws - Part 2", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, which="both", alpha=0.25, linestyle="--")
    ax.grid(True, which="major", alpha=0.45)

    fig.tight_layout()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to {out_path}")


if __name__ == "__main__":
    main()
