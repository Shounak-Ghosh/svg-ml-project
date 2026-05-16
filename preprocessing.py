#!/usr/bin/env python3
"""
SVG Data Preprocessing Pipeline — CS-GY 6923 Optional Project, Part 1.

Steps:
  1. Download SVG datasets from HuggingFace
  2. Clean / normalize each SVG
  3. Filter by character length
  4. Validate XML with lxml
  5. Create 98 / 1 / 1 train / val / test splits
  6. Save splits and report statistics
"""

import argparse
import json
import random
import re
import statistics as stats_lib
from pathlib import Path
from typing import Optional

import os

# cairocffi (used by cairosvg) calls dlopen() at import time. On macOS with
# Homebrew on Apple Silicon the library lives in /opt/homebrew/lib, which is
# not in the default dyld search path, so we add it before the import.
_brew_lib = "/opt/homebrew/lib"
if os.path.isdir(_brew_lib):
    os.environ["DYLD_LIBRARY_PATH"] = (
        _brew_lib + ":" + os.environ.get("DYLD_LIBRARY_PATH", "")
    ).rstrip(":")

import cairosvg  # noqa: E402
from datasets import load_dataset  # noqa: E402
from lxml import etree  # noqa: E402


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
MIN_CHARS = 50
MAX_CHARS = 8192
TRAIN_RATIO = 0.98
VAL_RATIO = 0.01
TEST_RATIO = 0.01
RANDOM_SEED = 42

# Tags whose content is pure metadata — safe to drop entirely
_META_TAGS = {"metadata", "title", "desc"}


# ---------------------------------------------------------------------------
# Number normalisation
# ---------------------------------------------------------------------------

def _round_match(m: re.Match) -> str:
    """Round a matched float to 1 decimal place; drop trailing .0."""
    val = float(m.group())
    rounded = round(val, 1)
    if rounded == int(rounded):
        return str(int(rounded))
    return f"{rounded:.1f}"


def normalize_numbers(text: str) -> str:
    """Round every floating-point number in *text* to 1 decimal place."""
    return re.sub(r"\d+\.\d+", _round_match, text)


# ---------------------------------------------------------------------------
# Single-SVG cleaning
# ---------------------------------------------------------------------------

def clean_svg(svg_text: str) -> Optional[str]:
    """
    Parse, clean, and normalise one SVG string.

    Actions:
    - Strip XML comments
    - Remove <metadata>, <title>, <desc> elements
    - Round all floating-point coordinates to 1 decimal place
    - Sort element attributes alphabetically (canonicalise ordering)
    - Collapse whitespace to single spaces

    Returns the cleaned string, or None if the input is unparseable.
    """
    if not svg_text or not svg_text.strip():
        return None

    # --- parse (comments removed by the parser) ---
    try:
        parser = etree.XMLParser(remove_comments=True, recover=False)
        root = etree.fromstring(svg_text.strip().encode("utf-8"), parser)
    except etree.XMLSyntaxError:
        return None

    # --- collect metadata elements to remove (avoid mutating during iter) ---
    to_remove = []
    for elem in root.iter():
        if callable(elem.tag):  # skip comments, processing instructions, etc.
            continue
        local = etree.QName(elem.tag).localname
        if local in _META_TAGS:
            to_remove.append(elem)

    for elem in to_remove:
        parent = elem.getparent()
        if parent is not None:
            parent.remove(elem)

    # --- normalise numbers & sort attributes ---
    for elem in root.iter():
        if callable(elem.tag):  # skip comments, processing instructions, etc.
            continue

        # Normalise attribute values
        for key in list(elem.attrib.keys()):
            elem.attrib[key] = normalize_numbers(elem.attrib[key])

        # Normalise text nodes (e.g. inline style values, animate values)
        if elem.text:
            elem.text = normalize_numbers(elem.text)

        # Sort attributes alphabetically
        sorted_attribs = sorted(elem.attrib.items())
        elem.attrib.clear()
        for k, v in sorted_attribs:
            elem.attrib[k] = v

    # --- serialise ---
    result = etree.tostring(root, encoding="unicode", xml_declaration=False)

    # Collapse all whitespace runs to a single space
    result = re.sub(r"\s+", " ", result).strip()

    return result


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def is_valid_xml(svg_text: str) -> bool:
    """Return True if *svg_text* parses as valid XML."""
    try:
        etree.fromstring(svg_text.encode("utf-8"))
        return True
    except etree.XMLSyntaxError:
        return False


# ---------------------------------------------------------------------------
# Render validation
# ---------------------------------------------------------------------------

def render_validate(svgs: list[str]) -> tuple[list[str], int]:
    """
    Filter out SVGs that CairoSVG cannot render.

    Returns
    -------
    valid : list[str]
        SVGs that rendered without error.
    n_failed : int
        Number of SVGs that failed to render.
    """
    valid: list[str] = []
    n_failed = 0
    for svg in svgs:
        try:
            cairosvg.svg2png(bytestring=svg.encode("utf-8"))
            valid.append(svg)
        except Exception:
            n_failed += 1
    return valid, n_failed


# ---------------------------------------------------------------------------
# Dataset-level preprocessing
# ---------------------------------------------------------------------------

def preprocess_dataset(
    svgs: list[str],
    min_chars: int = MIN_CHARS,
    max_chars: int = MAX_CHARS,
) -> tuple[list[str], dict]:
    """
    Apply cleaning and filtering to a list of raw SVG strings.

    Returns
    -------
    cleaned : list[str]
        SVGs that passed all filters.
    stats : dict
        Counts for each filter stage.
    """
    stats = {
        "total_input": len(svgs),
        "removed_empty": 0,
        "removed_parse_error": 0,
        "removed_too_short": 0,
        "removed_too_long": 0,
        "total_output": 0,
    }

    cleaned: list[str] = []

    for svg in svgs:
        if not svg or not svg.strip():
            stats["removed_empty"] += 1
            continue

        result = clean_svg(svg)
        if result is None:
            stats["removed_parse_error"] += 1
            continue

        if len(result) < min_chars:
            stats["removed_too_short"] += 1
            continue

        if len(result) > max_chars:
            stats["removed_too_long"] += 1
            continue

        cleaned.append(result)

    stats["total_output"] = len(cleaned)
    return cleaned, stats


# ---------------------------------------------------------------------------
# Train / val / test splitting
# ---------------------------------------------------------------------------

def create_splits(
    svgs: list[str],
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    seed: int = RANDOM_SEED,
) -> dict[str, list[str]]:
    """
    Randomly shuffle and split SVGs into train / val / test.

    Splitting is done by *file* (not by token position) to prevent data leakage.
    """
    rng = random.Random(seed)
    indices = list(range(len(svgs)))
    rng.shuffle(indices)

    n = len(indices)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]

    return {
        "train": [svgs[i] for i in train_idx],
        "val": [svgs[i] for i in val_idx],
        "test": [svgs[i] for i in test_idx],
    }


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def compute_and_print_statistics(splits: dict[str, list[str]]) -> dict:
    """Compute per-split character-level statistics and print a summary."""
    all_stats: dict[str, dict] = {}

    for split_name, svgs in splits.items():
        lengths = [len(s) for s in svgs]
        if not lengths:
            continue
        s = {
            "count": len(svgs),
            "total_chars": sum(lengths),
            "mean_chars": round(stats_lib.mean(lengths), 1),
            "median_chars": round(stats_lib.median(lengths), 1),
            "min_chars": min(lengths),
            "max_chars": max(lengths),
            "stdev_chars": round(stats_lib.stdev(lengths), 1) if len(lengths) > 1 else 0,
        }
        all_stats[split_name] = s

        print(f"\n=== {split_name.upper()} split ===")
        print(f"  Files:        {s['count']:>10,}")
        print(f"  Total chars:  {s['total_chars']:>10,}")
        print(f"  Mean chars:   {s['mean_chars']:>10,.1f}")
        print(f"  Median chars: {s['median_chars']:>10,.1f}")
        print(f"  Std chars:    {s['stdev_chars']:>10,.1f}")
        print(f"  Min chars:    {s['min_chars']:>10,}")
        print(f"  Max chars:    {s['max_chars']:>10,}")

        # Rough token estimate (assuming ~4 chars / token on average)
        est_tokens = s["total_chars"] // 4
        print(f"  Est. tokens:  {est_tokens:>10,}  (@ 4 chars/token)")

    return all_stats


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

def save_splits(splits: dict[str, list[str]], output_dir: str) -> None:
    """
    Save each split as a plain-text file (one SVG per line).

    Intra-SVG whitespace has already been collapsed to spaces by clean_svg(),
    so each line is self-contained.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for split_name, svgs in splits.items():
        out_path = out / f"{split_name}.txt"
        with open(out_path, "w", encoding="utf-8") as f:
            for svg in svgs:
                f.write(svg + "\n")
        print(f"Saved {len(svgs):,} SVGs:  {out_path}")


# ---------------------------------------------------------------------------
# HuggingFace helpers
# ---------------------------------------------------------------------------

_SVG_COLUMN_CANDIDATES = ["svg", "Svg", "SVG", "svg_code", "text", "code", "output"]


def detect_svg_column(dataset) -> str:
    """Return the column name that holds the SVG text."""
    for col in _SVG_COLUMN_CANDIDATES:
        if col in dataset.column_names:
            return col
    raise ValueError(
        f"Cannot find SVG column. Available columns: {dataset.column_names}"
    )


def load_svgs_from_hf(repo_id: str) -> list[str]:
    """Download a HuggingFace dataset and return its SVG strings."""
    from datasets import DatasetDict

    print(f"  Downloading {repo_id} …")
    ds = load_dataset(repo_id)

    if isinstance(ds, DatasetDict):
        all_svgs: list[str] = []
        for split_name, split_ds in ds.items():
            col = detect_svg_column(split_ds)
            svgs = [row for row in split_ds[col] if row]
            print(f"  Loaded {len(svgs):,} SVGs from split '{split_name}', column '{col}'")
            all_svgs.extend(svgs)
        return all_svgs

    col = detect_svg_column(ds)
    svgs = [row for row in ds[col] if row]
    print(f"  Loaded {len(svgs):,} SVGs from column '{col}'")
    return svgs


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="SVG preprocessing pipeline for CS-GY 6923 Optional Project"
    )
    parser.add_argument(
        "--output-dir", default="data/processed",
        help="Directory to write splits and statistics (default: data/processed)"
    )
    parser.add_argument(
        "--max-chars", type=int, default=MAX_CHARS,
        help=f"Maximum SVG character length after cleaning (default: {MAX_CHARS})"
    )
    parser.add_argument(
        "--min-chars", type=int, default=MIN_CHARS,
        help=f"Minimum SVG character length after cleaning (default: {MIN_CHARS})"
    )
    parser.add_argument(
        "--no-emoji", action="store_true",
        help="Skip starvector/svg-emoji-simple dataset"
    )
    parser.add_argument(
        "--no-fonts", action="store_true",
        help="Skip starvector/svg-fonts-simple dataset"
    )
    parser.add_argument(
        "--max-fonts", type=int, default=None,
        metavar="N",
        help="Randomly subsample at most N SVGs from svg-fonts-simple (default: use all). "
             "The dataset has grown to ~1.8M SVGs on HuggingFace; use ~270000 to match "
             "the scale of the reference implementation."
    )
    parser.add_argument(
        "--no-svgen", action="store_true",
        help="Skip umuthopeyildirim/svgen-500k dataset"
    )
    parser.add_argument(
        "--validate-render", action="store_true",
        help="Filter out SVGs that CairoSVG cannot render (slow)"
    )
    parser.add_argument(
        "--seed", type=int, default=RANDOM_SEED,
        help=f"Random seed for shuffling (default: {RANDOM_SEED})"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  SVG Preprocessing Pipeline")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Download
    # ------------------------------------------------------------------
    print("\n[Step 1] Downloading datasets …")
    all_svgs = load_svgs_from_hf("starvector/svg-icons-simple")

    if not args.no_emoji:
        emoji_svgs = load_svgs_from_hf("starvector/svg-emoji-simple")
        all_svgs.extend(emoji_svgs)

    if not args.no_fonts:
        fonts_svgs = load_svgs_from_hf("starvector/svg-fonts-simple")
        if args.max_fonts is not None and len(fonts_svgs) > args.max_fonts:
            rng = random.Random(args.seed)
            fonts_svgs = rng.sample(fonts_svgs, args.max_fonts)
            print(f"  Subsampled svg-fonts-simple to {args.max_fonts:,} SVGs")
        all_svgs.extend(fonts_svgs)

    if not args.no_svgen:
        svgen_svgs = load_svgs_from_hf("umuthopeyildirim/svgen-500k")
        all_svgs.extend(svgen_svgs)

    print(f"\nTotal raw SVGs: {len(all_svgs):,}")

    # ------------------------------------------------------------------
    # 2. Clean & filter
    # ------------------------------------------------------------------
    print("\n[Step 2] Cleaning and filtering …")
    cleaned_svgs, filter_stats = preprocess_dataset(
        all_svgs,
        min_chars=args.min_chars,
        max_chars=args.max_chars,
    )

    print("\n--- Filtering summary ---")
    for k, v in filter_stats.items():
        print(f"  {k:<30} {v:>8,}")

    if not cleaned_svgs:
        raise RuntimeError("No SVGs remained after filtering — check your thresholds.")

    # ------------------------------------------------------------------
    # 2.5 Render validation (opt-in)
    # ------------------------------------------------------------------
    if args.validate_render:
        print("\n[Step 2.5] Render validation with CairoSVG …")
        cleaned_svgs, render_failed = render_validate(cleaned_svgs)
        filter_stats["removed_render_error"] = render_failed
        print(f"  Removed (render error): {render_failed:,}")
        print(f"  Remaining:              {len(cleaned_svgs):,}")

        if not cleaned_svgs:
            raise RuntimeError("No SVGs remained after render validation.")
    else:
        print("\n[Step 2.5] Skipping render validation (pass --validate-render to enable)")

    # ------------------------------------------------------------------
    # 3. Split
    # ------------------------------------------------------------------
    print(f"\n[Step 3] Creating 98/1/1 splits (seed={args.seed}) …")
    splits = create_splits(cleaned_svgs, seed=args.seed)

    # ------------------------------------------------------------------
    # 4. Statistics
    # ------------------------------------------------------------------
    print("\n[Step 4] Dataset statistics")
    split_stats = compute_and_print_statistics(splits)

    # ------------------------------------------------------------------
    # 5. Save
    # ------------------------------------------------------------------
    print(f"\n[Step 5] Saving to '{args.output_dir}' …")
    save_splits(splits, args.output_dir)

    stats_path = Path(args.output_dir) / "stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(
            {"filter_stats": filter_stats, "split_stats": split_stats},
            f,
            indent=2,
        )
    print(f"Saved statistics: {stats_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
