# svg-ml-project
Spring 2026 Machine Learning Project with Professor Pavel Izmailov

## Setup

```bash
uv init
uv venv
source ./venv/bin/activate 
```

## Part 1: Data Preprocessing

`preprocessing.py` downloads SVG datasets from HuggingFace, cleans and normalizes them, filters by length, validates XML, and writes 98/1/1 train/val/test splits to disk.

**Basic usage** (downloads `svg-icons-simple` + `svg-emoji-simple` + `svgen-500k`):

```bash
python preprocessing.py
```

Output is saved to `data/processed/` by default:
- `train.txt`, `val.txt`, `test.txt` — one SVG per line
- `stats.json` — filtering counts and per-split statistics

**Options:**

| Flag | Default | Description |
|---|---|---|
| `--output-dir PATH` | `data/processed` | Where to write output files |
| `--max-chars N` | `2048` | Drop SVGs longer than N characters after cleaning |
| `--min-chars N` | `50` | Drop SVGs shorter than N characters after cleaning |
| `--no-emoji` | off | Skip `starvector/svg-emoji-simple` |
| `--validate-render` | off | Render-validate each SVG with CairoSVG (slow) |
| `--seed N` | `42` | Random seed for the train/val/test shuffle |

**Example with custom options:**

```bash
python preprocessing.py --output-dir data/processed --max-chars 4096 --validate-render
```

**What the cleaning pipeline does:**
1. Strips XML comments and `<metadata>` / `<title>` / `<desc>` elements
2. Rounds all floating-point coordinates to 1 decimal place (reduces vocabulary size)
3. Sorts element attributes alphabetically (canonical ordering)
4. Collapses whitespace to a single space per SVG
5. Validates each result as well-formed XML via `lxml`
6. Optionally render-validates via CairoSVG (`--validate-render`)

## BPE Tokenizer

`tokenizer.py` trains a Byte-Pair Encoding (BPE) tokenizer on the preprocessed SVG training split using the HuggingFace `tokenizers` library.

**Basic usage** (reads from `data/processed/`, writes to `data/tokenizer/`):

```bash
python tokenizer.py
```

Output is saved to `data/tokenizer/` by default:
- `tokenizer.json` — the trained tokenizer (load with `Tokenizer.from_file(...)`)
- `tokenizer_stats.json` — vocabulary size and per-split token count statistics

**Options:**

| Flag | Default | Description |
|---|---|---|
| `--data-dir PATH` | `data/processed` | Directory containing `train.txt` / `val.txt` / `test.txt` |
| `--output-dir PATH` | `data/tokenizer` | Where to write the tokenizer and stats |
| `--vocab-size N` | `4096` | BPE vocabulary size |
| `--no-stats` | off | Skip per-split token statistics (faster) |

**Example command:**

```bash
python tokenizer.py --vocab-size 4096
```

**Design decisions:**
- **Algorithm:** Byte-Pair Encoding (BPE) with a ByteLevel pre-tokenizer, so every raw byte maps to a printable character and `<unk>` is never emitted for valid UTF-8 input.
- **Vocabulary size: 4096** — SVG is a constrained XML language. After preprocessing, recurring patterns (tag names, attribute names, path commands) dominate the corpus. 4096 tokens captures these efficiently without over-segmenting structure (too small) or overfitting rare coordinate strings (too large).
- **Special tokens:** `<pad>`, `<unk>`, `<bos>`, `<eos>` — sequences are automatically wrapped with `<bos>`/`<eos>` at encode time.
- **Min frequency:** 2 — token pairs seen only once are excluded, keeping the vocabulary robust.

---

## Part 2: Transformer Training & Scaling Study

### Key files

| File | Description |
|---|---|
| `transformer/model.py` | Decoder-only transformer (`SVGTransformer`) and five named model configs |
| `transformer/train.py` | Training loop, dataset classes, LR schedule, LR sweep, results logging |
| `transformer/scaling_plot.py` | Fits power law `L = a·N^(-α) + c` and produces a scaling-law plot |

### Model configs (`transformer/model.py`)

| Name | ~Params | d_model | n_layers | n_heads | d_ff |
|---|---|---|---|---|---|
| `tiny` | ~1M | 128 | 4 | 4 | 512 |
| `small` | ~3M | 192 | 6 | 6 | 768 |
| `medium` | ~10M | 384 | 6 | 6 | 1536 |
| `large` | ~30M | 512 | 10 | 8 | 2048 |
| `xl` | ~88M | 768 | 12 | 12 | 3072 |

### Training usage (`transformer/train.py`)

```bash
# LR sweep on Tiny model (use --max_steps to limit compute):
python transformer/train.py --mode lr_sweep --max_steps 3000

# Train each model size for 1 full epoch with the best LR:
python transformer/train.py --model_size tiny   --lr 1e-2 --save_checkpoint
python transformer/train.py --model_size small  --lr 1e-2 --save_checkpoint
python transformer/train.py --model_size medium --lr 1e-2 --save_checkpoint
python transformer/train.py --model_size large  --lr 1e-2 --save_checkpoint
python transformer/train.py --model_size xl     --lr 1e-2 --save_checkpoint
```

**Key options:**

| Flag | Default | Description |
|---|---|---|
| `--model_size` | `tiny` | One of: tiny, small, medium, large, xl |
| `--lr` | `1e-2` | Peak learning rate |
| `--batch_size` | `16` | Sequences per gradient step |
| `--block_size` | `1024` | Context window length in tokens |
| `--max_steps` | full epoch | Limit training to N optimizer steps |
| `--grad_clip` | `1.0` | Max gradient norm (0 = disabled) |
| `--save_checkpoint` | off | Save `.pt` checkpoint after training |
| `--compile` | off | Wrap model with `torch.compile()` (PyTorch ≥ 2.0) |
| `--device` | auto | `cuda` > `mps` > `cpu` |

Results are written to `transformer/runs/<size>_lr<lr>_results.json`. Checkpoints go to `transformer/runs/<size>_lr<lr>_ckpt.pt`.

### Scaling plot (`transformer/scaling_plot.py`)

Reads all `*_results.json` files in `transformer/runs/`, fits a power law, and saves a plot with a 10× parameter extrapolation.

```bash
python transformer/scaling_plot.py
# Custom paths:
python transformer/scaling_plot.py --runs_dir transformer/runs --out transformer/scaling_plot.png
```

---

## Part 3: muP (Maximal Update Parameterization)

### Key files

| File | Description |
|---|---|
| `transformer/model_mup.py` | muP transformer (`SVGTransformerMuP`), muP attention scale, `MuReadout`, `create_mup_base_shapes()` |
| `transformer/train_mup.py` | muP training loop using `MuAdamW`, gradient accumulation, per-epoch checkpointing |
| `transformer/scaling_plot_comparison.py` | SP vs muP scaling comparison plot and LR sweep overlay |

### muP training usage (`transformer/train_mup.py`)

muP allows the optimal learning rate found on the Tiny model to transfer zero-shot to all larger widths.

```bash
# muP LR sweep on Tiny (base shapes are generated automatically):
python transformer/train_mup.py --mode lr_sweep --max_steps 3000

# Train each model size with the best muP LR:
python transformer/train_mup.py --model_size tiny   --lr <best_lr> --save_checkpoint
python transformer/train_mup.py --model_size small  --lr <best_lr> --save_checkpoint
python transformer/train_mup.py --model_size medium --lr <best_lr> --save_checkpoint
python transformer/train_mup.py --model_size large  --lr <best_lr> --save_checkpoint
python transformer/train_mup.py --model_size xl     --lr <best_lr> --save_checkpoint

# Multi-epoch training with gradient accumulation:
python transformer/train_mup.py --model_size large --lr <best_lr> \
    --n_epochs 2 --grad_accum 4 --save_checkpoint
```

**Additional options vs `train.py`:**

| Flag | Default | Description |
|---|---|---|
| `--n_epochs` | `1` | Number of full training epochs |
| `--grad_accum` | `1` | Gradient accumulation steps (effective batch = batch_size × grad_accum) |
| `--resume_ckpt` | — | Path to a `.pt` checkpoint to resume weights from |

muP results go to `transformer/runs/mup/`. Base shape files (`.bsh`) are generated once per model topology and cached there.

### Comparison plot (`transformer/scaling_plot_comparison.py`)

Reads SP results from `transformer/runs/` and muP results from `transformer/runs/mup/`, fits both power laws, and produces a three-panel comparison figure plus an LR sweep overlay.

```bash
python transformer/scaling_plot_comparison.py
# Outputs: transformer/scaling_comparison.png, transformer/lr_sweep_comparison.png
```

---

## Part 4: Generation

### Key file

| File | Description |
|---|---|
| `transformer/generate.py` | Loads an SP or muP checkpoint, runs unconditional and prefix-conditioned generation, optionally renders to PNG |

### Usage (`transformer/generate.py`)

```bash
# Generate from a muP checkpoint (auto-detected):
python transformer/generate.py --ckpt transformer/runs/mup/xl_lr1e-02_2ep_mup_ckpt.pt

# Custom output dir, more samples, render to PNG:
python transformer/generate.py \
    --ckpt transformer/runs/mup/large_lr1e-02_mup_ckpt.pt \
    --out_dir transformer/runs/generated/large/ \
    --n_unconditional 20 \
    --render

# Beam search decoding (muP models only):
python transformer/generate.py \
    --ckpt transformer/runs/mup/xl_lr1e-02_2ep_mup_ckpt.pt \
    --beam_size 5

# Greedy decoding:
python transformer/generate.py --ckpt <ckpt_path> --greedy
```

**Key options:**

| Flag | Default | Description |
|---|---|---|
| `--ckpt` | required | Path to `.pt` checkpoint (SP or muP auto-detected) |
| `--n_unconditional` | `10` | Number of unconditional samples |
| `--max_new_tokens` | `1024` | Max tokens to generate per sample |
| `--temperatures` | `0.5 0.8 1.0` | Temperatures cycled across unconditional samples |
| `--top_k` | `50` | Top-k filtering |
| `--top_p` | `0.9` | Nucleus (top-p) filtering |
| `--beam_size` | `1` | Beam search width (1 = sampling/greedy; muP only) |
| `--greedy` | off | Argmax decoding at every step |
| `--render` | off | Render each SVG to PNG via `cairosvg` |

Each run produces individual `.svg` files and a `generation_results.json` metadata file in `--out_dir`. The script generates 10 unconditional samples and 5 prefix-conditioned completions (partial face, open path, group+rect, partial polygon, icon path).

