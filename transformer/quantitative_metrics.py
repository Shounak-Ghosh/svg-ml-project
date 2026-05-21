"""
Quantitative metrics for SVG Transformer (Part 4).

Computes:
  1. Test-set perplexity
  2. XML validity rate
  3. SVG render rate 
  4. Structural validity rate

Usage:
    # Load pre-generated samples from generate.py output:
    python transformer/quantitative_metrics.py \\
        --ckpt transformer/runs/mup/small_lr1e-02_mup_ckpt.pt \\
        --samples transformer/runs/generated/small_test/generation_results.json

    # Generate samples inline:
    python transformer/quantitative_metrics.py \\
        --ckpt transformer/runs/mup/small_lr1e-02_mup_ckpt.pt \\
        --generate --n_samples 50

    # Perplexity only (skip generation metrics):
    python transformer/quantitative_metrics.py \\
        --ckpt transformer/runs/mup/small_lr1e-02_mup_ckpt.pt \\
        --no_render
"""

import os
import sys
import json
import math
import argparse
import logging
from pathlib import Path
from typing import Optional

import torch
from tokenizers import Tokenizer

# Platform library paths — must be set before lxml / cairosvg are imported.
# Mirrors the same pattern used in generate.py and preprocessing.py.
_brew_lib = "/opt/homebrew/lib"
if os.path.isdir(_brew_lib):
    os.environ["DYLD_LIBRARY_PATH"] = (
        _brew_lib + ":" + os.environ.get("DYLD_LIBRARY_PATH", "")
    ).rstrip(":")

_gtk_bin = r"C:\Program Files\GTK3-Runtime Win64\bin"
if os.path.isdir(_gtk_bin) and _gtk_bin not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _gtk_bin + os.pathsep + os.environ.get("PATH", "")

from lxml import etree  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from generate import load_model, generate_unconditional  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Perplexity
# ---------------------------------------------------------------------------

def compute_perplexity(
    model,
    tokenizer: Tokenizer,
    test_path: str,
    device: str,
    block_size: int,
    max_seqs: Optional[int] = None,
) -> dict:
    """
    Compute test-set perplexity using non-overlapping windows of block_size.

    Each SVG is tokenized with BOS/EOS (added automatically by the tokenizer
    post-processor). Long sequences are split into consecutive chunks so every
    token is counted exactly once.
    """
    test_file = Path(test_path)
    if not test_file.exists():
        log.warning(f"Test file not found: {test_path} — skipping perplexity")
        return {"perplexity": None, "avg_loss": None, "n_seqs": 0, "n_tokens": 0}

    lines = [ln for ln in test_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if max_seqs is not None:
        lines = lines[:max_seqs]

    total_nll = 0.0
    total_tokens = 0
    n_seqs = 0

    model.eval()
    with torch.no_grad():
        for i, svg in enumerate(lines):
            if (i + 1) % 500 == 0:
                log.info(f"  {i+1}/{len(lines)} sequences processed ...")

            ids = tokenizer.encode(svg).ids
            if len(ids) < 2:
                continue

            seq = torch.tensor(ids, dtype=torch.long)
            seq_len = len(seq)

            for begin in range(0, seq_len - 1, block_size):
                end = min(begin + block_size + 1, seq_len)
                chunk = seq[begin:end]
                if len(chunk) < 2:
                    break
                x = chunk[:-1].unsqueeze(0).to(device)
                y = chunk[1:].unsqueeze(0).to(device)
                _, loss = model(x, y)
                n_tok = len(chunk) - 1
                total_nll += loss.item() * n_tok
                total_tokens += n_tok

            n_seqs += 1

    if total_tokens == 0:
        return {"perplexity": None, "avg_loss": None, "n_seqs": n_seqs, "n_tokens": 0}

    avg_loss = total_nll / total_tokens
    return {
        "perplexity": round(math.exp(avg_loss), 4),
        "avg_loss":   round(avg_loss, 6),
        "n_seqs":     n_seqs,
        "n_tokens":   total_tokens,
    }


# ---------------------------------------------------------------------------
# 2. XML validity
# ---------------------------------------------------------------------------

def _is_valid_xml(svg_text: str) -> bool:
    try:
        etree.fromstring(svg_text.strip().encode("utf-8"))
        return True
    except etree.XMLSyntaxError:
        return False


def xml_validity_rate(svgs: list[str]) -> dict:
    n_valid = sum(1 for s in svgs if _is_valid_xml(s))
    return {
        "xml_valid":      n_valid,
        "xml_total":      len(svgs),
        "xml_valid_rate": round(n_valid / len(svgs), 4) if svgs else 0.0,
    }


# ---------------------------------------------------------------------------
# 3. SVG render rate
# ---------------------------------------------------------------------------

def _try_render(svg_text: str) -> bool:
    try:
        import cairosvg
        cairosvg.svg2png(bytestring=svg_text.encode("utf-8"))
        return True
    except Exception:
        return False


def render_rate(svgs: list[str]) -> dict:
    try:
        import cairosvg  # noqa: F401
    except ImportError:
        log.warning("cairosvg not installed — skipping render rate. Run: pip install cairosvg")
        return {"render_valid": None, "render_total": len(svgs), "render_rate": None}

    n_rendered = 0
    for i, s in enumerate(svgs):
        if (i + 1) % 25 == 0:
            log.info(f"  {i+1}/{len(svgs)} renders attempted ...")
        if _try_render(s):
            n_rendered += 1

    return {
        "render_valid": n_rendered,
        "render_total": len(svgs),
        "render_rate":  round(n_rendered / len(svgs), 4) if svgs else 0.0,
    }


# ---------------------------------------------------------------------------
# 4. Structural validity
# ---------------------------------------------------------------------------

def _is_structurally_valid(svg_text: str) -> bool:
    """
    Checks three conditions:
    - Root element is <svg> (any namespace)
    - viewBox attribute, if present, has exactly 4 numeric values
    - width / height attributes, if present, are non-empty strings
    """
    try:
        root = etree.fromstring(svg_text.strip().encode("utf-8"))
    except etree.XMLSyntaxError:
        return False

    if etree.QName(root.tag).localname != "svg":
        return False

    vb = root.get("viewBox") or root.get("viewbox")
    if vb is not None:
        parts = vb.strip().split()
        if len(parts) != 4:
            return False
        try:
            list(map(float, parts))
        except ValueError:
            return False

    for attr in ("width", "height"):
        val = root.get(attr)
        if val is not None and not val.strip():
            return False

    return True


def structural_validity_rate(svgs: list[str]) -> dict:
    n_valid = sum(1 for s in svgs if _is_structurally_valid(s))
    return {
        "struct_valid":      n_valid,
        "struct_total":      len(svgs),
        "struct_valid_rate": round(n_valid / len(svgs), 4) if svgs else 0.0,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SVG Transformer — quantitative metrics",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--ckpt", required=True, help="Path to .pt checkpoint")
    p.add_argument("--tokenizer_path", default="data/tokenizer/tokenizer.json")
    p.add_argument("--test_data", default="data/processed/test.txt",
                   help="Test split (one SVG per line) used for perplexity")
    p.add_argument("--samples", default=None,
                   help="generation_results.json from generate.py. "
                        "Omit (with no --generate) to skip generation-based metrics.")
    p.add_argument("--generate", action="store_true",
                   help="Generate samples inline instead of loading from --samples")
    p.add_argument("--n_samples", type=int, default=50,
                   help="Samples to generate when --generate is used")
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--temperatures", type=float, nargs="+", default=[0.5, 0.8, 1.0])
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--no_perplexity", action="store_true",
                   help="Skip test-set perplexity computation")
    p.add_argument("--no_render", action="store_true",
                   help="Skip CairoSVG render-rate evaluation")
    p.add_argument("--max_test_seqs", type=int, default=None,
                   help="Cap number of test sequences for perplexity (quick sanity check)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    ))
    p.add_argument("--out", default=None,
                   help="Output JSON path (default: <ckpt_dir>/metrics.json)")
    return p.parse_args(argv)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    tokenizer = Tokenizer.from_file(args.tokenizer_path)
    log.info(f"Tokenizer loaded (vocab_size={tokenizer.get_vocab_size()})")

    model, cfg = load_model(args.ckpt, args.device)

    results: dict = {}

    # ── 1. Perplexity ─────────────────────────────────────────────────────────
    if args.no_perplexity:
        log.info("Skipping perplexity (--no_perplexity)")
        results["perplexity"] = {"perplexity": None, "avg_loss": None, "n_seqs": 0, "n_tokens": 0}
    else:
        log.info(f"\n[1/4] Test-set perplexity ({args.test_data}) ...")
        results["perplexity"] = compute_perplexity(
            model, tokenizer, args.test_data, args.device,
            block_size=cfg.block_size,
            max_seqs=args.max_test_seqs,
        )

    # ── Obtain generated SVG strings ──────────────────────────────────────────
    svgs: list[str] = []

    if args.generate:
        log.info(f"\nGenerating {args.n_samples} samples ...")
        gen_results = generate_unconditional(
            model, tokenizer, args.device,
            n_samples=args.n_samples,
            max_new_tokens=args.max_new_tokens,
            temperatures=args.temperatures,
            top_k=args.top_k,
            top_p=args.top_p,
        )
        svgs = [r["svg"] for r in gen_results]

    elif args.samples:
        log.info(f"\nLoading samples from {args.samples} ...")
        with open(args.samples) as f:
            gen_data = json.load(f)
        svgs = [s["svg"] for s in gen_data.get("samples", [])]
        log.info(f"  Loaded {len(svgs)} samples")

    else:
        log.info("\nNo samples provided — pass --samples or --generate to evaluate generation metrics")

    # ── 2–4. Generation-based metrics ─────────────────────────────────────────
    if svgs:
        log.info(f"\n[2/4] XML validity rate ({len(svgs)} samples) ...")
        results["xml_validity"] = xml_validity_rate(svgs)

        if args.no_render:
            log.info("[3/4] Skipping render rate (--no_render)")
            results["render_rate"] = {
                "render_valid": None, "render_total": len(svgs), "render_rate": None
            }
        else:
            log.info(f"\n[3/4] SVG render rate ({len(svgs)} samples) ...")
            results["render_rate"] = render_rate(svgs)

        log.info(f"\n[4/4] Structural validity rate ({len(svgs)} samples) ...")
        results["structural_validity"] = structural_validity_rate(svgs)

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = Path(args.out) if args.out else Path(args.ckpt).parent / "metrics.json"
    with open(out_path, "w") as f:
        json.dump({"ckpt": args.ckpt, "metrics": results}, f, indent=2)
    log.info(f"\nResults saved --> {out_path}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'═'*58}")
    print("  Quantitative Metrics")
    print(f"{'═'*58}")

    ppl = results.get("perplexity", {})
    if ppl.get("perplexity") is not None:
        print(f"  Perplexity        : {ppl['perplexity']:>10.2f}"
              f"  (loss={ppl['avg_loss']:.4f}, {ppl['n_seqs']:,} seqs, {ppl['n_tokens']:,} tokens)")
    else:
        print(f"  Perplexity        : {'N/A':>10}")

    if "xml_validity" in results:
        xv = results["xml_validity"]
        print(f"  XML validity      : {xv['xml_valid_rate']:>10.1%}"
              f"  ({xv['xml_valid']}/{xv['xml_total']})")

    if "render_rate" in results:
        rv = results["render_rate"]
        if rv["render_rate"] is not None:
            print(f"  SVG render rate   : {rv['render_rate']:>10.1%}"
                  f"  ({rv['render_valid']}/{rv['render_total']})")
        else:
            print(f"  SVG render rate   : {'N/A':>10}  (cairosvg unavailable or --no_render)")

    if "structural_validity" in results:
        sv = results["structural_validity"]
        print(f"  Structural valid  : {sv['struct_valid_rate']:>10.1%}"
              f"  ({sv['struct_valid']}/{sv['struct_total']})")

    print(f"{'═'*58}\n")


if __name__ == "__main__":
    main()
