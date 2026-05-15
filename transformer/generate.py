"""
SVG Transformer generation script (Part 4).

Loads a checkpoint (SP or muP auto-detected), generates unconditional and
prefix-conditioned SVG samples, and saves them to an output directory.

Usage:
    python transformer/generate.py --ckpt transformer/runs/mup/xl_lr1e-02_2ep_mup_ckpt.pt
    python transformer/generate.py --ckpt transformer/runs/mup/small_lr1e-02_mup_ckpt.pt \\
        --out_dir transformer/runs/generated/small_test/
"""

import sys
import json
import time
import argparse
import logging
from pathlib import Path
from typing import Optional

import torch
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).parent))
from model import SVGTransformer, ModelConfig
from model_mup import SVGTransformerMuP

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

BOS_ID = 2
EOS_ID = 3

# 5 interesting SVG prefixes for conditioned generation
PREFIXES = [
    (
        "face_partial",
        '<svg viewBox="0 0 100 100">'
        '<circle cx="50" cy="50" r="40"/>'
        '<circle cx="35" cy="45" r="5" fill="black"/>',
    ),
    (
        "open_path",
        '<svg viewBox="0 0 100 100">'
        '<path d="M 10 50 C 20 20,',
    ),
    (
        "group_rect",
        '<svg viewBox="0 0 100 100">'
        '<g><rect x="20" y="20" width="60" height="60" fill="blue"/>',
    ),
    (
        "partial_polygon",
        '<svg viewBox="0 0 100 100">'
        '<polygon points="50,10 61,35',
    ),
    (
        "text_element",
        '<svg viewBox="0 0 200 100">'
        '<text x="10" y="40">Hello',
    ),
]


def load_model(
    ckpt_path: str,
    device: str,
) -> tuple:
    """
    Load a checkpoint and reconstruct the model.

    Auto-detects model type by presence of 'base_shapes_path' key:
      present  → SVGTransformerMuP
      absent   → SVGTransformer (SP)
    """
    log.info(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    cfg = ModelConfig(**ckpt["config"])
    is_mup = "base_shapes_path" in ckpt

    if is_mup:
        base_shapes_path = ckpt["base_shapes_path"]
        if not Path(base_shapes_path).exists():
            base_shapes_path = str(Path(ckpt_path).parent / Path(base_shapes_path).name)
        log.info(f"muP model detected; base_shapes: {base_shapes_path}")
        model = SVGTransformerMuP(cfg, base_shapes=base_shapes_path)
    else:
        log.info("SP model detected")
        model = SVGTransformer(cfg)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    model.to(device)

    n_params = ckpt.get("n_params", sum(p.numel() for p in model.parameters()))
    val_loss = ckpt.get("val_loss", "unknown")
    log.info(f"Model loaded: {n_params:,} params | val_loss={val_loss}")
    return model, cfg


def encode_no_special(text: str, tokenizer: Tokenizer) -> list:
    """Encode text without adding <bos>/<eos>."""
    return tokenizer.encode(text, add_special_tokens=False).ids


def decode_tokens(ids: list, tokenizer: Tokenizer) -> str:
    """Decode token IDs to text, stripping all special tokens."""
    clean_ids = [i for i in ids if i not in (0, 1, 2, 3)]
    return tokenizer.decode(clean_ids)


def generate_unconditional(
    model,
    tokenizer: Tokenizer,
    device: str,
    n_samples: int = 10,
    max_new_tokens: int = 512,
    temperatures: Optional[list] = None,
    top_k: int = 50,
    top_p: float = 0.9,
) -> list:
    """
    Generate n_samples unconditional SVGs starting from <bos>.

    Distributes samples evenly across the given temperatures.
    """
    if temperatures is None:
        temperatures = [0.5, 0.8, 1.0]

    results = []
    for i in range(n_samples):
        temp = temperatures[i % len(temperatures)]
        seed = torch.tensor([[BOS_ID]], dtype=torch.long, device=device)

        t0 = time.perf_counter()
        generated = model.generate(
            seed,
            max_new_tokens=max_new_tokens,
            temperature=temp,
            top_k=top_k,
            top_p=top_p,
            eos_id=EOS_ID,
        )
        elapsed = time.perf_counter() - t0

        token_ids = generated[0].tolist()
        svg_text  = decode_tokens(token_ids, tokenizer)
        n_generated = len(token_ids) - 1  # subtract seed bos

        results.append({
            "name":             f"unconditional_{i:02d}",
            "svg":              svg_text,
            "temperature":      temp,
            "top_k":            top_k,
            "top_p":            top_p,
            "tokens_generated": n_generated,
            "elapsed_s":        round(elapsed, 2),
        })
        log.info(f"  [{i+1:02d}/{n_samples}] temp={temp} | {n_generated} tokens | {elapsed:.1f}s")

    return results


def generate_from_prefix(
    model,
    tokenizer: Tokenizer,
    device: str,
    prefix_name: str,
    prefix_text: str,
    max_new_tokens: int = 512,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.9,
) -> dict:
    """Complete an SVG from a given prefix string."""
    prefix_ids = encode_no_special(prefix_text, tokenizer)
    seed_ids   = [BOS_ID] + prefix_ids
    seed       = torch.tensor([seed_ids], dtype=torch.long, device=device)

    log.info(f"  '{prefix_name}': {len(prefix_ids)} prefix tokens")

    t0 = time.perf_counter()
    generated = model.generate(
        seed,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        eos_id=EOS_ID,
    )
    elapsed = time.perf_counter() - t0

    token_ids   = generated[0].tolist()
    svg_text    = decode_tokens(token_ids, tokenizer)
    n_generated = len(token_ids) - len(seed_ids)

    log.info(f"    → {n_generated} new tokens | {elapsed:.1f}s")

    return {
        "name":             f"prefix_{prefix_name}",
        "prefix_text":      prefix_text,
        "svg":              svg_text,
        "temperature":      temperature,
        "top_k":            top_k,
        "top_p":            top_p,
        "prefix_tokens":    len(prefix_ids),
        "tokens_generated": n_generated,
        "elapsed_s":        round(elapsed, 2),
    }


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SVG Transformer — Part 4 Generation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--ckpt", required=True,
                   help="Path to .pt checkpoint")
    p.add_argument("--tokenizer_path", default="data/tokenizer/tokenizer.json")
    p.add_argument("--out_dir", default="transformer/runs/generated/")
    p.add_argument("--n_unconditional", type=int, default=10,
                   help="Number of unconditional samples to generate")
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    ))
    return p.parse_args(argv)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = Tokenizer.from_file(args.tokenizer_path)
    log.info(f"Tokenizer loaded (vocab_size={tokenizer.get_vocab_size()})")

    model, _ = load_model(args.ckpt, args.device)

    all_results = []

    # ── unconditional generation ───────────────────────────────────────────
    log.info(f"\nGenerating {args.n_unconditional} unconditional samples ...")
    unconditional = generate_unconditional(
        model, tokenizer, args.device,
        n_samples=args.n_unconditional,
        max_new_tokens=args.max_new_tokens,
        temperatures=[0.5, 0.8, 1.0],
        top_k=50,
        top_p=0.9,
    )
    all_results.extend(unconditional)

    # ── prefix-conditioned generation ─────────────────────────────────────
    log.info(f"\nGenerating {len(PREFIXES)} prefix-conditioned samples ...")
    for name, prefix_text in PREFIXES:
        result = generate_from_prefix(
            model, tokenizer, args.device,
            prefix_name=name,
            prefix_text=prefix_text,
            max_new_tokens=args.max_new_tokens,
            temperature=0.8,
            top_k=50,
            top_p=0.9,
        )
        all_results.append(result)

    # ── save outputs ───────────────────────────────────────────────────────
    log.info(f"\nSaving {len(all_results)} samples to {out_dir} ...")
    for r in all_results:
        (out_dir / f"{r['name']}.svg").write_text(r["svg"], encoding="utf-8")

    metadata = {
        "ckpt":           args.ckpt,
        "seed":           args.seed,
        "device":         args.device,
        "max_new_tokens": args.max_new_tokens,
        "samples":        all_results,
    }
    metadata_path = out_dir / "generation_results.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    log.info(f"Metadata -> {metadata_path}")

    # ── summary table ──────────────────────────────────────────────────────
    print(f"\n{'─'*80}")
    print(f"{'Name':<30} {'Temp':>5} {'top_k':>6} {'top_p':>6} {'Tokens':>7} {'Elapsed':>8}")
    print(f"{'─'*80}")
    for r in all_results:
        print(
            f"{r['name']:<30} {r['temperature']:>5.1f} {r['top_k']:>6} "
            f"{r['top_p']:>6.1f} {r['tokens_generated']:>7} {r['elapsed_s']:>7.1f}s"
        )
    print(f"{'─'*80}\n")


if __name__ == "__main__":
    main()
