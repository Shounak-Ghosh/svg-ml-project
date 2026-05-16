"""
Sample SVGs from the training set at five complexity levels, render each to PNG.

Complexity is measured by SVG character length, bucketed at actual dataset
percentiles (p25=574, p50=968, p75=1817, p90=3208, computed from train.txt).

Output layout:
    data/svg/
        simple/           # < p25  chars
        medium_simple/    # p25-p50
        medium/           # p50-p75
        complex/          # p75-p90
        very_complex/     # > p90
    Each subfolder contains N .svg files and corresponding .png renders.
"""

import random
from pathlib import Path
import os

# cairocffi (used by cairosvg) calls dlopen() at import time. On macOS with
# Homebrew on Apple Silicon the library lives in /opt/homebrew/lib, which is
# not in the default dyld search path, so we add it before the import.
_brew_lib = "/opt/homebrew/lib"
if os.path.isdir(_brew_lib):
    os.environ["DYLD_LIBRARY_PATH"] = (
        _brew_lib + ":" + os.environ.get("DYLD_LIBRARY_PATH", "")
    ).rstrip(":")

import cairosvg

DATA_FILE = Path("data/processed/train.txt")
OUTPUT_DIR = Path("data/svg")

# Bin edges derived from train.txt character-length percentiles.
BINS = {
    "simple":        (0,    574),
    "medium_simple": (574,  968),
    "medium":        (968,  1817),
    "complex":       (1817, 3208),
    "very_complex":  (3208, float("inf")),
}

SAMPLES_PER_BIN = 5
RANDOM_SEED = 42
PNG_SIZE = 512  # px, passed to cairosvg as output_width / output_height


def classify(svg: str) -> str | None:
    n = len(svg)
    for name, (lo, hi) in BINS.items():
        if lo <= n < hi:
            return name
    return None


def sample_svgs(path: Path, n: int, seed: int) -> dict[str, list[str]]:
    """Reservoir-sample n SVGs per complexity bin from path."""
    rng = random.Random(seed)
    buckets: dict[str, list[str]] = {name: [] for name in BINS}
    seen:    dict[str, int]       = {name: 0  for name in BINS}

    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            svg = line.rstrip("\n")
            if not svg:
                continue
            name = classify(svg)
            if name is None:
                continue
            seen[name] += 1
            bucket = buckets[name]
            if len(bucket) < n:
                bucket.append(svg)
            else:
                j = rng.randint(0, seen[name] - 1)
                if j < n:
                    bucket[j] = svg

    return buckets


def render(buckets: dict[str, list[str]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    for bin_name, svgs in buckets.items():
        bin_dir = out_dir / bin_name
        bin_dir.mkdir(exist_ok=True)
        lo, hi = BINS[bin_name]
        hi_str = f"{hi:,}" if hi != float("inf") else "inf"
        print(f"\n{bin_name}  ({lo:,}-{hi_str} chars)  — {len(svgs)} samples")

        for i, svg in enumerate(svgs, 1):
            stem = f"{bin_name}_{i:02d}"
            svg_path = bin_dir / f"{stem}.svg"
            png_path = bin_dir / f"{stem}.png"

            svg_path.write_text(svg, encoding="utf-8")
            cairosvg.svg2png(
                bytestring=svg.encode(),
                write_to=str(png_path),
                output_width=PNG_SIZE,
                output_height=PNG_SIZE,
            )
            print(f"  {stem}  {len(svg):>6,} chars  --> {png_path}")


def main() -> None:
    print(f"Sampling from {DATA_FILE} …")
    buckets = sample_svgs(DATA_FILE, SAMPLES_PER_BIN, RANDOM_SEED)
    render(buckets, OUTPUT_DIR)
    print(f"\nDone. Output in {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
