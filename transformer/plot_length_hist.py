"""
Plot BPE-token sequence-length histogram for the training split.

Re-encodes each line of data/processed/train.txt with the trained tokenizer
and saves a log-y histogram of per-sequence token lengths.

Usage:
    python transformer/plot_length_hist.py
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from tokenizers import Tokenizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train_path",     default="data/processed/train.txt")
    p.add_argument("--tokenizer_path", default="data/tokenizer/tokenizer.json")
    p.add_argument("--out",            default="length_hist.png")
    p.add_argument("--bins",  type=int, default=80)
    p.add_argument("--xcap",  type=int, default=2200,
                   help="X-axis upper limit (tokens); long tail truncated visually")
    args = p.parse_args()

    tok = Tokenizer.from_file(args.tokenizer_path)

    print(f"encoding {args.train_path} ...")
    lengths = []
    with open(args.train_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            lengths.append(len(tok.encode(line).ids))
            if (i + 1) % 50_000 == 0:
                print(f"  {i+1:,} sequences encoded")
    lengths = np.array(lengths)

    print(f"  total: {len(lengths):,} sequences")
    print(f"  mean={lengths.mean():.1f}  median={np.median(lengths):.1f}  "
          f"p95={np.percentile(lengths, 95):.0f}  max={lengths.max()}")

    fig, ax = plt.subplots(figsize=(8, 4.5))

    clipped = np.clip(lengths, 0, args.xcap)
    n_above = (lengths > args.xcap).sum()
    frac_above = n_above / len(lengths)

    ax.hist(clipped, bins=args.bins, color="steelblue", edgecolor="white",
            linewidth=0.4, alpha=0.85)

    med   = float(np.median(lengths))
    mean  = float(lengths.mean())
    p95   = float(np.percentile(lengths, 95))
    for value, label, color, style in [
        (med,  f"median {med:.0f}",  "#222222", ":"),
        (mean, f"mean {mean:.0f}",   "#444444", "--"),
        (p95,  f"p95 {p95:.0f}",     "#888888", "-."),
    ]:
        ax.axvline(value, color=color, linewidth=1.4, linestyle=style, label=label)

    ax.set_yscale("log")
    ax.set_xlim(0, args.xcap)
    ax.set_xlabel("BPE tokens per SVG", fontsize=11)
    ax.set_ylabel("Sequence count (log scale)", fontsize=11)
    ax.set_title(
        f"Training-split sequence-length distribution  "
        f"(N = {len(lengths):,};  {frac_above:.1%} beyond {args.xcap} tokens)",
        fontsize=11.5, fontweight="bold",
    )
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, which="major", axis="y", alpha=0.35, linestyle="--")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()