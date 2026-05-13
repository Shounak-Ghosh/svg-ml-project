"""
Training script for SVG Transformer scaling study (Part 2).

Based on nanoGPT (Karpathy, 2022): https://github.com/karpathy/nanoGPT

--- What is borrowed from nanoGPT ---
  - Cosine LR schedule with linear warmup (get_lr())
  - AMP training pattern: torch.amp.autocast + GradScaler
  - Gradient clipping before optimizer step
  - AdamW setup (delegated to model.configure_optimizers)

--- What is modified / added for this project ---
  - SVGDataset: packed-sequence dataset from flat token array       
  - tokenize_file(): loads one-SVG-per-line .txt files using the
    HuggingFace tokenizer (which auto-adds <bos>/<eos> via its
    TemplateProcessing post-processor) and caches as .npy           
  - train(): full training loop with throughput / GPU memory stats   
  - run_lr_sweep(): sweeps LRs on the Tiny model (Part 2.1)        
  - JSON results logging for downstream scaling-law analysis         

Usage examples
--------------
# LR sweep on Tiny (use --max_steps to limit per-run cost):
  python transformer/train.py --mode lr_sweep --max_steps 3000

# Train a single model size for 1 full epoch:
  python transformer/train.py --model_size tiny  --lr 3e-4 --save_checkpoint
  python transformer/train.py --model_size small --lr 3e-4 --save_checkpoint
  python transformer/train.py --model_size medium --lr 3e-4 --save_checkpoint
  python transformer/train.py --model_size large  --lr 3e-4 --save_checkpoint
  python transformer/train.py --model_size xl     --lr 3e-4 --save_checkpoint
"""

import sys
import time
import json
import math
import logging
import argparse
from pathlib import Path
from contextlib import nullcontext
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer
from tqdm import tqdm

# Allow running from the repo root or from within transformer/
sys.path.insert(0, str(Path(__file__).parent))
from model import SVGTransformer, MODEL_CONFIGS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def tokenize_file(
    txt_path: str,
    tokenizer: Tokenizer,
    cache_path: Optional[str] = None,
) -> np.ndarray:
    """
    Tokenize an SVG .txt file (one SVG per line) into a flat uint16 numpy array.

    The tokenizer's TemplateProcessing post-processor already wraps each
    sequence with <bos> ... <eos>, so we do NOT add them manually.

    Caches the result as a .npy file to avoid re-tokenization on repeated runs.
    Delete the cache file if you change the tokenizer or source data.
    [Original implementation for this project]
    """
    if cache_path and Path(cache_path).exists():
        log.info(f"Loading cached tokens from {cache_path}")
        return np.load(cache_path)

    log.info(f"Tokenizing {txt_path} ...")
    all_tokens: list[int] = []

    with open(txt_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in tqdm(lines, desc=Path(txt_path).name, unit="svg"):
        line = line.strip()
        if not line:
            continue
        enc = tokenizer.encode(line)   # <bos> + subword ids + <eos> via post-processor
        all_tokens.extend(enc.ids)

    arr = np.array(all_tokens, dtype=np.uint16)
    log.info(f"  -> {len(arr):,} tokens total")

    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, arr)
        log.info(f"  Cached to {cache_path}")

    return arr


class SVGDataset(Dataset):
    """
    Packed-sequence dataset.

    Slices a flat token array into non-overlapping (input, target) pairs of
    length block_size, with target = input shifted right by one position.
    No padding is used — all tokens are real SVG content.
    [Original implementation for this project]
    """

    def __init__(self, token_array: np.ndarray, block_size: int):
        self.data       = token_array
        self.block_size = block_size
        # Number of complete pairs we can extract without running off the end
        self.n_blocks   = (len(token_array) - 1) // block_size

    def __len__(self) -> int:
        return self.n_blocks

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.block_size
        chunk = self.data[start : start + self.block_size + 1].astype(np.int64)
        x = torch.from_numpy(chunk[:-1])  # input tokens
        y = torch.from_numpy(chunk[1:])   # target tokens (shifted by 1)
        return x, y


# ---------------------------------------------------------------------------
# Learning rate schedule
# ---------------------------------------------------------------------------

def get_lr(
    step: int,
    warmup_steps: int,
    total_steps: int,
    lr_max: float,
    lr_min: float,
) -> float:
    """
    Linear warmup then cosine decay from lr_max down to lr_min.
    [Adapted from nanoGPT]
    """
    if step < warmup_steps:
        return lr_max * (step + 1) / warmup_steps
    if step >= total_steps:
        return lr_min
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    coeff    = 0.5 * (1.0 + math.cos(math.pi * progress))
    return lr_min + coeff * (lr_max - lr_min)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model: SVGTransformer,
    loader: DataLoader,
    device: str,
    ctx,
    max_batches: Optional[int] = None,
) -> float:
    """
    Compute mean cross-entropy loss on a DataLoader.
    Optionally limited to max_batches for speed.
    [Original]
    """
    model.eval()
    losses: list[float] = []
    for i, (x, y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        with ctx:
            _, loss = model(x.to(device), y.to(device))
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses)) if losses else float("nan")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> dict:
    """
    Train one model for one epoch (or --max_steps steps) and return a results dict.

    Results dict contains: model_size, n_params, lr, val_loss,
    train_losses (list of step/loss/lr), wall_clock_s, tokens_per_sec, gpu_mem_gb.
    [Original implementation for this project]
    """
    # ── device / precision ─────────────────────────────────────────────────
    device = args.device
    if "cuda" in device:
        device_type = "cuda"
    elif "mps" in device:
        device_type = "mps"
    else:
        device_type = "cpu"

    # CUDA: bfloat16 (preferred) or float16 with GradScaler.
    # MPS (Apple Silicon): float32 — MPS autocast has limited op coverage.
    # CPU: float32.
    if device_type == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype = torch.float32

    ctx = (
        torch.amp.autocast(device_type="cuda", dtype=dtype)
        if device_type == "cuda"
        else nullcontext()
    )

    torch.manual_seed(args.seed)
    if device_type == "cuda":
        torch.cuda.manual_seed(args.seed)
        torch.cuda.reset_peak_memory_stats(device)

    # ── tokenizer ──────────────────────────────────────────────────────────
    tokenizer  = Tokenizer.from_file(args.tokenizer_path)
    vocab_size = tokenizer.get_vocab_size()
    log.info(f"Tokenizer loaded: vocab_size={vocab_size}")

    # ── model ──────────────────────────────────────────────────────────────
    cfg            = MODEL_CONFIGS[args.model_size]
    cfg.vocab_size = vocab_size
    cfg.block_size = args.block_size
    cfg.dropout    = args.dropout

    model   = SVGTransformer(cfg).to(device)
    n_params = model.count_parameters()
    log.info(f"Model '{args.model_size}': {n_params:,} parameters")

    if args.compile and hasattr(torch, "compile"):
        log.info("Compiling model with torch.compile() ...")
        model = torch.compile(model)   # type: ignore[assignment]

    # ── data ───────────────────────────────────────────────────────────────
    cache_dir = Path(args.cache_dir) if args.cache_dir else None

    def cache_path(split: str) -> Optional[str]:
        if cache_dir is None:
            return None
        return str(cache_dir / f"{split}_bs{args.block_size}_tokens.npy")

    train_tokens = tokenize_file(args.train_path, tokenizer, cache_path("train"))
    val_tokens   = tokenize_file(args.val_path,   tokenizer, cache_path("val"))

    train_ds = SVGDataset(train_tokens, args.block_size)
    val_ds   = SVGDataset(val_tokens,   args.block_size)

    log.info(
        f"Dataset: train={len(train_ds):,} blocks, val={len(val_ds):,} blocks "
        f"(block_size={args.block_size})"
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device_type == "cuda"),   # pin_memory unsupported on MPS
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device_type == "cuda"),   # pin_memory unsupported on MPS
    )

    # Limit steps if requested (useful for LR sweep to reduce compute)
    total_steps  = len(train_loader)
    if args.max_steps is not None:
        total_steps = min(args.max_steps, total_steps)

    warmup_steps = max(1, round(args.warmup_ratio * total_steps))
    tokens_per_step = args.batch_size * args.block_size
    log.info(
        f"Training: {total_steps} steps | warmup={warmup_steps} | "
        f"{tokens_per_step * total_steps / 1e6:.1f}M tokens"
    )

    # ── optimizer & scaler ─────────────────────────────────────────────────
    optimizer = model.configure_optimizers(   # type: ignore[union-attr]
        weight_decay=args.weight_decay,
        lr=args.lr,
        betas=(0.9, 0.95),
        device_type=device_type,
    )
    # GradScaler is only meaningful on CUDA with float16; disabled on MPS / CPU
    scaler = torch.amp.GradScaler(enabled=(device_type == "cuda" and dtype == torch.float16))

    # ── training loop ──────────────────────────────────────────────────────
    results: dict = {
        "model_size":   args.model_size,
        "n_params":     n_params,
        "lr":           args.lr,
        "train_losses": [],          # list of {step, loss, lr}
        "val_loss":     None,
        "wall_clock_s": None,
        "tokens_per_sec": None,
        "gpu_mem_gb":   None,
    }

    model.train()
    t0          = time.perf_counter()
    tokens_seen = 0
    log_every   = max(1, total_steps // 200)  # log ~200 checkpoints per epoch

    train_iter = iter(train_loader)
    for step in range(total_steps):
        try:
            x, y = next(train_iter)
        except StopIteration:
            break

        x, y = x.to(device), y.to(device)

        # Set learning rate for this step (cosine schedule with warmup)
        lr = get_lr(step, warmup_steps, total_steps, args.lr, args.lr * args.lr_decay_ratio)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # Forward + backward
        with ctx:
            _, loss = model(x, y)

        scaler.scale(loss).backward()
        if args.grad_clip > 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        tokens_seen += x.numel()

        if step % log_every == 0 or step == total_steps - 1:
            elapsed = time.perf_counter() - t0
            tps = tokens_seen / elapsed if elapsed > 0 else 0.0
            results["train_losses"].append(
                {"step": step, "loss": round(loss.item(), 5), "lr": lr}
            )
            log.info(
                f"  step {step:5d}/{total_steps} | "
                f"loss {loss.item():.4f} | lr {lr:.2e} | {tps:,.0f} tok/s"
            )

    # ── end-of-epoch metrics ───────────────────────────────────────────────
    wall_clock = time.perf_counter() - t0
    results["wall_clock_s"]   = round(wall_clock, 2)
    results["tokens_per_sec"] = round(tokens_seen / max(wall_clock, 1e-9))

    if device_type == "cuda":
        results["gpu_mem_gb"] = round(
            torch.cuda.max_memory_allocated(device) / 1e9, 3
        )
    elif device_type == "mps":
        results["gpu_mem_gb"] = round(
            torch.mps.current_allocated_memory() / 1e9, 3
        )

    # Full validation pass
    val_loss = evaluate(model, val_loader, device, ctx)
    results["val_loss"] = round(val_loss, 6)

    log.info(
        f"\n{'─' * 60}\n"
        f"  {args.model_size.upper()} | {n_params:,} params | LR={args.lr:.2e}\n"
        f"  val_loss  = {val_loss:.4f}\n"
        f"  wall time = {wall_clock / 60:.1f} min\n"
        f"  throughput= {results['tokens_per_sec']:,} tok/s\n"
        f"  GPU mem   = {results.get('gpu_mem_gb', 'N/A')} GB\n"
        f"{'─' * 60}"
    )

    # ── persist results ────────────────────────────────────────────────────
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_tag      = f"{args.model_size}_lr{args.lr:.0e}"
    results_path = out_dir / f"{run_tag}_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"Results -> {results_path}")

    if args.save_checkpoint:
        # Handle torch.compile() wrapping
        try:
            state_dict = model._orig_mod.state_dict()   # type: ignore[union-attr]
        except AttributeError:
            state_dict = model.state_dict()

        ckpt = {
            "model_state_dict": state_dict,
            "config":           cfg.__dict__,
            "val_loss":         val_loss,
            "n_params":         n_params,
            "args":             vars(args),
        }
        ckpt_path = out_dir / f"{run_tag}_ckpt.pt"
        torch.save(ckpt, ckpt_path)
        log.info(f"Checkpoint -> {ckpt_path}")

    return results


# ---------------------------------------------------------------------------
# LR sweep (Part 2, Requirement 1)
# ---------------------------------------------------------------------------

def run_lr_sweep(args: argparse.Namespace) -> float:
    """
    Sweep learning rates on the Tiny model to find the best LR.

    Tests --n_lrs values log-spaced in [--lr_sweep_min, --lr_sweep_max].
    Each run trains for --max_steps steps (defaults to the full epoch if not set;
    recommend --max_steps 3000 for a practical sweep on a single GPU).

    Returns the best learning rate (lowest val_loss).
    [Original implementation for this project]
    """
    lrs: list[float] = np.logspace(
        np.log10(args.lr_sweep_min),
        np.log10(args.lr_sweep_max),
        args.n_lrs,
    ).tolist()

    log.info(f"LR sweep on 'tiny' over {args.n_lrs} values: {[f'{lr:.2e}' for lr in lrs]}")

    args.model_size = "tiny"
    sweep_results: list[dict] = []

    for lr in lrs:
        log.info(f"\n{'=' * 60}\nSweeping LR = {lr:.2e}\n{'=' * 60}")
        args.lr = lr
        result  = train(args)
        entry   = {"lr": lr, "val_loss": result["val_loss"]}
        sweep_results.append(entry)
        log.info(f"  LR {lr:.2e}  ->  val_loss {result['val_loss']:.4f}")

    # Save sweep summary
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sweep_path = out_dir / "lr_sweep.json"
    with open(sweep_path, "w") as f:
        json.dump({"sweep": sweep_results}, f, indent=2)
    log.info(f"LR sweep results -> {sweep_path}")

    best = min(sweep_results, key=lambda r: r["val_loss"])
    log.info(
        f"\nBest LR: {best['lr']:.2e}  (val_loss={best['val_loss']:.4f})\n"
        f"Use:  --lr {best['lr']:.2e}  for all model sizes in Part 2."
    )
    return float(best["lr"])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SVG Transformer training — Part 2 Scaling Study",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Mode
    p.add_argument(
        "--mode", choices=["train", "lr_sweep"], default="train",
        help="'train': single training run; 'lr_sweep': LR sweep on tiny model",
    )

    # Paths
    p.add_argument("--train_path",     default="data/processed/train.txt")
    p.add_argument("--val_path",       default="data/processed/val.txt")
    p.add_argument("--tokenizer_path", default="data/tokenizer/tokenizer.json")
    p.add_argument("--out_dir",        default="transformer/runs",
                   help="Directory for results JSON and checkpoints")
    p.add_argument("--cache_dir",      default="data/cache",
                   help="Directory for cached tokenized .npy arrays (delete to re-tokenize)")

    # Model
    p.add_argument("--model_size", choices=list(MODEL_CONFIGS.keys()), default="tiny",
                   help="One of: tiny (~1M), small (~3M), medium (~10M), large (~30M), xl (~88M)")
    p.add_argument("--block_size", type=int, default=1024,
                   help="Context window length in tokens")

    # Training hyperparameters
    p.add_argument("--batch_size",      type=int,   default=16,
                   help="Sequences per gradient step (tokens/step = batch_size * block_size)")
    p.add_argument("--lr",              type=float, default=3e-4,
                   help="Peak learning rate")
    p.add_argument("--lr_decay_ratio",  type=float, default=0.1,
                   help="LR_min = lr * lr_decay_ratio (cosine decay floor)")
    p.add_argument("--weight_decay",    type=float, default=0.1)
    p.add_argument("--dropout",         type=float, default=0.0)
    p.add_argument("--grad_clip",       type=float, default=1.0,
                   help="Max gradient norm (0 = disabled)")
    p.add_argument("--warmup_ratio",    type=float, default=0.05,
                   help="Fraction of total_steps used for linear LR warmup")
    p.add_argument("--max_steps",       type=int,   default=None,
                   help="Limit training to N steps (default: full epoch). "
                        "Recommended: --max_steps 3000 for LR sweep runs.")
    p.add_argument("--num_workers",     type=int,   default=0,
                   help="DataLoader worker processes (0 = main process; safe on all OSes)")
    p.add_argument("--seed",            type=int,   default=42)
    p.add_argument("--device",          default=(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    ))
    p.add_argument("--compile",         action="store_true",
                   help="Wrap model with torch.compile() (PyTorch >= 2.0, CUDA recommended)")
    p.add_argument("--save_checkpoint", action="store_true",
                   help="Save model checkpoint after training")

    # LR sweep options
    p.add_argument("--lr_sweep_min", type=float, default=1e-5,
                   help="Lower bound of LR sweep (log scale)")
    p.add_argument("--lr_sweep_max", type=float, default=1e-2,
                   help="Upper bound of LR sweep (log scale)")
    p.add_argument("--n_lrs",        type=int,   default=7,
                   help="Number of LR values to test in the sweep")

    return p.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    if args.mode == "lr_sweep":
        best_lr = run_lr_sweep(args)
        log.info(f"Recommended LR for all model sizes: {best_lr:.2e}")
    else:
        train(args)
