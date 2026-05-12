#!/usr/bin/env python3
"""
BPE Tokenizer Training for SVG Corpus — CS-GY 6923 Optional Project, Part 1.

Trains a Byte-Pair Encoding (BPE) tokenizer on the preprocessed SVG training
split using the HuggingFace `tokenizers` library and saves the result.

Vocabulary size: 4096
"""

import argparse
import json
import statistics as stats_lib
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.trainers import BpeTrainer
from tokenizers.processors import TemplateProcessing


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
VOCAB_SIZE = 4096
SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"


# ---------------------------------------------------------------------------
# Build and train tokenizer
# ---------------------------------------------------------------------------

def build_tokenizer(vocab_size: int = VOCAB_SIZE) -> tuple[Tokenizer, BpeTrainer]:
    """Construct a fresh BPE tokenizer and its trainer."""
    tokenizer = Tokenizer(BPE(unk_token=UNK_TOKEN))

    # ByteLevel pre-tokenizer: maps every raw byte to a printable character,
    # so the tokenizer never emits <unk> for arbitrary Unicode/binary content.
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        min_frequency=2,          # ignore token pairs seen only once
        show_progress=True,
    )
    return tokenizer, trainer


def train_tokenizer(
    train_file: Path,
    vocab_size: int = VOCAB_SIZE,
) -> Tokenizer:
    """Train BPE on *train_file* and return the fitted tokenizer."""
    tokenizer, trainer = build_tokenizer(vocab_size)
    tokenizer.train(files=[str(train_file)], trainer=trainer)

    # Add BOS/EOS post-processing so every encoded sequence is wrapped
    bos_id = tokenizer.token_to_id(BOS_TOKEN)
    eos_id = tokenizer.token_to_id(EOS_TOKEN)
    tokenizer.post_processor = TemplateProcessing(
        single=f"{BOS_TOKEN}:0 $A:0 {EOS_TOKEN}:0",
        pair=f"{BOS_TOKEN}:0 $A:0 {EOS_TOKEN}:0 $B:1 {EOS_TOKEN}:1",
        special_tokens=[
            (BOS_TOKEN, bos_id),
            (EOS_TOKEN, eos_id),
        ],
    )
    return tokenizer


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def compute_token_stats(tokenizer: Tokenizer, split_file: Path) -> dict:
    """
    Encode every line in *split_file* and return summary token statistics.

    The post-processor adds BOS/EOS, so lengths include those two tokens.
    """
    lengths: list[int] = []
    with open(split_file, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            encoded = tokenizer.encode(line)
            lengths.append(len(encoded.ids))

    if not lengths:
        return {}

    return {
        "n_sequences": len(lengths),
        "total_tokens": sum(lengths),
        "mean_tokens": round(stats_lib.mean(lengths), 1),
        "median_tokens": round(stats_lib.median(lengths), 1),
        "stdev_tokens": round(stats_lib.stdev(lengths), 1) if len(lengths) > 1 else 0,
        "min_tokens": min(lengths),
        "max_tokens": max(lengths),
        "p95_tokens": sorted(lengths)[int(len(lengths) * 0.95)],
    }


def print_token_stats(split_name: str, s: dict) -> None:
    print(f"\n=== {split_name.upper()} split token statistics ===")
    print(f"  Sequences:      {s['n_sequences']:>10,}")
    print(f"  Total tokens:   {s['total_tokens']:>10,}")
    print(f"  Mean tokens:    {s['mean_tokens']:>10,.1f}")
    print(f"  Median tokens:  {s['median_tokens']:>10,.1f}")
    print(f"  Std tokens:     {s['stdev_tokens']:>10,.1f}")
    print(f"  Min tokens:     {s['min_tokens']:>10,}")
    print(f"  Max tokens:     {s['max_tokens']:>10,}")
    print(f"  P95 tokens:     {s['p95_tokens']:>10,}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a BPE tokenizer on the SVG corpus"
    )
    parser.add_argument(
        "--data-dir", default="data/processed",
        help="Directory containing train.txt / val.txt / test.txt (default: data/processed)"
    )
    parser.add_argument(
        "--output-dir", default="data/tokenizer",
        help="Directory to save the tokenizer (default: data/tokenizer)"
    )
    parser.add_argument(
        "--vocab-size", type=int, default=VOCAB_SIZE,
        help=f"BPE vocabulary size (default: {VOCAB_SIZE})"
    )
    parser.add_argument(
        "--no-stats", action="store_true",
        help="Skip per-split token statistics (faster)"
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_file = data_dir / "train.txt"
    if not train_file.exists():
        raise FileNotFoundError(
            f"Training file not found: {train_file}\n"
            "Run preprocessing.py first to generate the splits."
        )

    print("=" * 60)
    print("  BPE Tokenizer Training")
    print("=" * 60)
    print(f"  Training file : {train_file}")
    print(f"  Vocabulary size: {args.vocab_size:,}")
    print(f"  Special tokens: {SPECIAL_TOKENS}")

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    print("\n[Step 1] Training BPE tokenizer …")
    tokenizer = train_tokenizer(train_file, vocab_size=args.vocab_size)

    actual_vocab_size = tokenizer.get_vocab_size()
    print(f"\nActual vocabulary size (may be < requested): {actual_vocab_size:,}")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    tokenizer_path = output_dir / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))
    print(f"Saved tokenizer: {tokenizer_path}")

    # ------------------------------------------------------------------
    # Quick sanity check
    # ------------------------------------------------------------------
    print("\n[Step 2] Sanity check — encoding a sample SVG …")
    sample = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><path d="M12 2L2 22h20L12 2z"/></svg>'
    encoded = tokenizer.encode(sample)
    decoded = tokenizer.decode(encoded.ids)
    print(f"  Input  : {sample}")
    print(f"  Tokens : {encoded.tokens}")
    print(f"  IDs    : {encoded.ids}")
    print(f"  Decoded: {decoded}")

    # ------------------------------------------------------------------
    # Token statistics per split
    # ------------------------------------------------------------------
    all_stats: dict = {"vocab_size": actual_vocab_size}

    if not args.no_stats:
        print("\n[Step 3] Computing token statistics per split …")
        for split in ("train", "val", "test"):
            split_file = data_dir / f"{split}.txt"
            if not split_file.exists():
                print(f"  Skipping {split} (file not found)")
                continue
            s = compute_token_stats(tokenizer, split_file)
            print_token_stats(split, s)
            all_stats[split] = s

    # ------------------------------------------------------------------
    # Save statistics
    # ------------------------------------------------------------------
    stats_path = output_dir / "tokenizer_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(all_stats, f, indent=2)
    print(f"\nSaved statistics: {stats_path}")
    print("Done.")


if __name__ == "__main__":
    main()
