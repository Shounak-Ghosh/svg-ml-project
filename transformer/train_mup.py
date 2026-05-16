"""
muP training script for SVG Transformer scaling study (Part 3).

Mirrors train.py but uses SVGTransformerMuP and MuAdamW so that the optimal
learning rate found on the Tiny model transfers zero-shot to all larger widths.

Shared utilities (SVGSequenceDataset, tokenize_file_to_sequences, pad_collate,
get_lr, evaluate) are imported directly from train.py to avoid duplication.

Usage
-----
# muP LR sweep on Tiny (generates base_shapes.bsh on first run):
  python transformer/train_mup.py --mode lr_sweep --max_steps 3000

# Train each model size for 1 full epoch with best muP LR:
  python transformer/train_mup.py --model_size tiny   --lr <best_lr> --save_checkpoint
  python transformer/train_mup.py --model_size small  --lr <best_lr> --save_checkpoint
  python transformer/train_mup.py --model_size medium --lr <best_lr> --save_checkpoint
  python transformer/train_mup.py --model_size large  --lr <best_lr> --save_checkpoint
  python transformer/train_mup.py --model_size xl     --lr <best_lr> --save_checkpoint
"""

import sys
import time
import json
import logging
import argparse
from pathlib import Path
from contextlib import nullcontext
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from tokenizers import Tokenizer

# Allow running from repo root or from within transformer/
sys.path.insert(0, str(Path(__file__).parent))
from model import MODEL_CONFIGS
from model_mup import SVGTransformerMuP, create_mup_base_shapes
# Reuse data / schedule / eval helpers — no duplication
from train import SVGSequenceDataset, tokenize_file_to_sequences, pad_collate, get_lr, evaluate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_mup(args: argparse.Namespace) -> dict:
    """
    Train one muP model for one epoch (or --max_steps steps).

    Differs from train() in train.py in four ways:
      1. Model is SVGTransformerMuP (muP attention scale, MuReadout, muP init)
      2. Optimizer is MuAdamW — per-layer LR scaled by 1/width_mult
      3. LR schedule update preserves width_mult ratios via initial_lrs snapshot
      4. Results saved to out_dir/mup/ with _mup suffix
    No GradScaler — MPS backend is always float32.
    """
    # ── device / precision ─────────────────────────────────────────────────
    device = args.device
    if "cuda" in device:
        device_type = "cuda"
    elif "mps" in device:
        device_type = "mps"
    else:
        device_type = "cpu"

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

    # ── model config (needed before base shapes) ───────────────────────────
    cfg            = MODEL_CONFIGS[args.model_size]
    cfg.vocab_size = vocab_size
    cfg.block_size = args.block_size
    cfg.dropout    = args.dropout

    # ── base shapes (per model size: parameter names depend on n_layers/n_heads)
    mup_out_dir        = Path(args.out_dir) / "mup"
    mup_out_dir.mkdir(parents=True, exist_ok=True)
    # Each unique (n_layers, n_heads) topology needs its own base shapes file
    # because set_base_shapes does exact parameter-name matching
    base_shapes_path   = mup_out_dir / f"base_shapes_{args.model_size}.bsh"
    if not base_shapes_path.exists():
        log.info(f"Generating muP base shapes for '{args.model_size}' -> {base_shapes_path}")
        create_mup_base_shapes(
            savefile=str(base_shapes_path),
            config=cfg,
        )

    model    = SVGTransformerMuP(cfg, base_shapes=str(base_shapes_path)).to(device)
    n_params = model.count_parameters()
    log.info(f"muP model '{args.model_size}': {n_params:,} parameters")

    if args.resume_ckpt:
        log.info(f"Resuming weights from: {args.resume_ckpt}")
        resume_data = torch.load(args.resume_ckpt, map_location="cpu", weights_only=False)
        raw_for_load = model._orig_mod if hasattr(model, "_orig_mod") else model
        raw_for_load.load_state_dict(resume_data["model_state_dict"])
        log.info(f"  Resumed from val_loss={resume_data.get('val_loss', 'unknown')}")

    if args.compile and hasattr(torch, "compile"):
        log.info("Compiling model with torch.compile() ...")
        model = torch.compile(model)   # type: ignore[assignment]

    # ── data ───────────────────────────────────────────────────────────────
    cache_dir = Path(args.cache_dir) if args.cache_dir else None

    def flat_cache(split: str) -> Optional[str]:
        if cache_dir is None:
            return None
        return str(cache_dir / f"{split}_bs{args.block_size}_tokens.npy")

    def offsets_cache(split: str) -> Optional[str]:
        if cache_dir is None:
            return None
        return str(cache_dir / f"{split}_seq_offsets.npy")

    train_tokens, train_offsets = tokenize_file_to_sequences(
        args.train_path, tokenizer, flat_cache("train"), offsets_cache("train"))
    val_tokens, val_offsets = tokenize_file_to_sequences(
        args.val_path, tokenizer, flat_cache("val"), offsets_cache("val"))

    filter_long = not args.no_filter_long
    train_ds = SVGSequenceDataset(train_tokens, train_offsets, args.block_size, filter_long=filter_long)
    val_ds   = SVGSequenceDataset(val_tokens,   val_offsets,   args.block_size, filter_long=filter_long)

    log.info(
        f"Dataset: train={len(train_ds):,} sequences, val={len(val_ds):,} sequences "
        f"(block_size={args.block_size})"
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device_type == "cuda"),
        drop_last=True,
        collate_fn=pad_collate,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device_type == "cuda"),
        collate_fn=pad_collate,
    )

    steps_per_epoch = len(train_loader)
    total_steps     = args.n_epochs * steps_per_epoch
    if args.max_steps is not None:
        total_steps = min(args.max_steps, total_steps)

    warmup_steps = max(1, round(args.warmup_ratio * total_steps))
    avg_seq_len  = len(train_tokens) / max(1, len(train_ds))
    log.info(
        f"Training: {args.n_epochs} epoch(s) × {steps_per_epoch} steps = "
        f"{total_steps} total steps | warmup={warmup_steps} | "
        f"{len(train_ds):,} sequences (avg {avg_seq_len:.0f} tokens each)"
    )

    # ── optimizer ──────────────────────────────────────────────────────────
    # MuAdamW applies per-layer LR scaling; initial_lrs captures the width-scaled
    # per-group LRs so the cosine schedule can update them proportionally
    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    optimizer, initial_lrs = raw_model.configure_optimizers_mup(   # type: ignore[union-attr]
        weight_decay=args.weight_decay,
        lr=args.lr,
        betas=(0.9, 0.95),
    )
    # No GradScaler — float32 only on MPS; CUDA path keeps same simplicity here

    # ── training loop ──────────────────────────────────────────────────────
    results: dict = {
        "model_size":     args.model_size,
        "n_params":       n_params,
        "lr":             args.lr,
        "n_epochs":       args.n_epochs,
        "resumed_from":   args.resume_ckpt,
        "train_losses":   [],
        "val_loss":       None,
        "wall_clock_s":   None,
        "tokens_per_sec": None,
        "gpu_mem_gb":     None,
    }

    model.train()
    t0          = time.perf_counter()
    tokens_seen = 0
    log_every   = max(1, total_steps // 200)
    step        = 0

    for epoch in range(args.n_epochs):
        train_iter = iter(train_loader)
        for x, y in train_iter:
            if step >= total_steps:
                break

            x, y = x.to(device), y.to(device)

            # Cosine LR schedule — preserve per-group width_mult ratios
            # initial_lrs[i] = args.lr / width_mult_i  (set by MuAdamW at creation)
            # g['lr'] = initial_lrs[i] * (lr_t / args.lr) = lr_t / width_mult_i
            lr_t = get_lr(step, warmup_steps, total_steps, args.lr, args.lr * args.lr_decay_ratio)
            for g, init_lr in zip(optimizer.param_groups, initial_lrs):
                g["lr"] = init_lr * (lr_t / args.lr)

            with ctx:
                _, loss = model(x, y)

            loss.backward()
            if args.grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            tokens_seen += x.numel()
            step        += 1

            if step % log_every == 0 or step == total_steps:
                elapsed = time.perf_counter() - t0
                tps = tokens_seen / elapsed if elapsed > 0 else 0.0
                results["train_losses"].append(
                    {"step": step, "epoch": epoch, "loss": round(loss.item(), 5), "lr": lr_t}
                )
                log.info(
                    f"  epoch {epoch+1}/{args.n_epochs} step {step:5d}/{total_steps} | "
                    f"loss {loss.item():.4f} | lr {lr_t:.2e} | {tps:,.0f} tok/s"
                )

        if step >= total_steps:
            break

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

    val_loss = evaluate(model, val_loader, device, ctx)
    results["val_loss"] = round(val_loss, 6)

    log.info(
        f"\n{'─' * 60}\n"
        f"  muP {args.model_size.upper()} | {n_params:,} params | LR={args.lr:.2e}\n"
        f"  val_loss  = {val_loss:.4f}\n"
        f"  wall time = {wall_clock / 60:.1f} min\n"
        f"  throughput= {results['tokens_per_sec']:,} tok/s\n"
        f"  GPU mem   = {results.get('gpu_mem_gb', 'N/A')} GB\n"
        f"{'─' * 60}"
    )

    # ── persist results ────────────────────────────────────────────────────
    run_tag      = f"{args.model_size}_lr{args.lr:.0e}_{args.n_epochs}ep_mup"
    results_path = mup_out_dir / f"{run_tag}_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"Results -> {results_path}")

    if args.save_checkpoint:
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
            "base_shapes_path": str(base_shapes_path),
        }
        ckpt_path = mup_out_dir / f"{run_tag}_ckpt.pt"
        torch.save(ckpt, ckpt_path)
        log.info(f"Checkpoint -> {ckpt_path}")

    return results


# ---------------------------------------------------------------------------
# LR sweep
# ---------------------------------------------------------------------------

def run_mup_lr_sweep(args: argparse.Namespace) -> float:
    """
    Sweep learning rates on the Tiny muP model.

    The best LR found here should transfer zero-shot to all larger model sizes
    under muP (that is the key claim of the muP paper).
    """
    lrs: list[float] = np.logspace(
        np.log10(args.lr_sweep_min),
        np.log10(args.lr_sweep_max),
        args.n_lrs,
    ).tolist()

    log.info(f"muP LR sweep on 'tiny' over {args.n_lrs} values: {[f'{lr:.2e}' for lr in lrs]}")

    args.model_size = "tiny"
    sweep_results: list[dict] = []

    for lr in lrs:
        log.info(f"\n{'=' * 60}\nSweeping LR = {lr:.2e}\n{'=' * 60}")
        args.lr = lr
        result  = train_mup(args)
        entry   = {"lr": lr, "val_loss": result["val_loss"]}
        sweep_results.append(entry)
        log.info(f"  LR {lr:.2e}  ->  val_loss {result['val_loss']:.4f}")

    mup_out_dir = Path(args.out_dir) / "mup"
    mup_out_dir.mkdir(parents=True, exist_ok=True)
    sweep_path = mup_out_dir / "lr_sweep_mup.json"
    with open(sweep_path, "w") as f:
        json.dump({"sweep": sweep_results}, f, indent=2)
    log.info(f"muP LR sweep results -> {sweep_path}")

    best = min(sweep_results, key=lambda r: r["val_loss"])
    log.info(
        f"\nBest muP LR: {best['lr']:.2e}  (val_loss={best['val_loss']:.4f})\n"
        f"Use:  --lr {best['lr']:.2e}  for all model sizes (muP transfers zero-shot)."
    )
    return float(best["lr"])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args_mup(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SVG Transformer training — Part 3 muP Scaling Study",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument(
        "--mode", choices=["train", "lr_sweep"], default="train",
        help="'train': single training run; 'lr_sweep': LR sweep on tiny muP model",
    )

    # Paths
    p.add_argument("--train_path",     default="data/processed/train.txt")
    p.add_argument("--val_path",       default="data/processed/val.txt")
    p.add_argument("--tokenizer_path", default="data/tokenizer/tokenizer.json")
    p.add_argument("--out_dir",        default="transformer/runs",
                   help="Parent directory; muP results go into out_dir/mup/")
    p.add_argument("--cache_dir",      default="data/cache")

    # Model
    p.add_argument("--model_size", choices=list(MODEL_CONFIGS.keys()), default="tiny")
    p.add_argument("--block_size", type=int, default=1024)

    # Training hyperparameters
    p.add_argument("--batch_size",      type=int,   default=16)
    p.add_argument("--lr",              type=float, default=3e-4)
    p.add_argument("--lr_decay_ratio",  type=float, default=0.1)
    p.add_argument("--weight_decay",    type=float, default=0.1)
    p.add_argument("--dropout",         type=float, default=0.0)
    p.add_argument("--grad_clip",       type=float, default=1.0)
    p.add_argument("--warmup_ratio",    type=float, default=0.05)
    p.add_argument("--max_steps",       type=int,   default=None)
    p.add_argument("--n_epochs",        type=int,   default=1,
                   help="Number of full training epochs (cosine LR spans all epochs)")
    p.add_argument("--resume_ckpt",     type=str,   default=None,
                   help="Path to a .pt checkpoint to resume model weights from")
    p.add_argument("--num_workers",     type=int,   default=0)
    p.add_argument("--seed",            type=int,   default=42)
    p.add_argument("--device",          default=(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    ))
    p.add_argument("--compile",         action="store_true")
    p.add_argument("--save_checkpoint", action="store_true")
    p.add_argument("--no_filter_long",  action="store_true",
                   help="Disable filtering of sequences longer than block_size")

    # LR sweep options
    p.add_argument("--lr_sweep_min", type=float, default=1e-5)
    p.add_argument("--lr_sweep_max", type=float, default=1e-2)
    p.add_argument("--n_lrs",        type=int,   default=7)

    return p.parse_args(argv)


if __name__ == "__main__":
    args = parse_args_mup()
    if args.mode == "lr_sweep":
        best_lr = run_mup_lr_sweep(args)
        log.info(f"muP recommended LR for all model sizes: {best_lr:.2e}")
    else:
        train_mup(args)
