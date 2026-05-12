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

**Example with custom options:**

```bash
python tokenizer.py --vocab-size 8192 --no-stats
```

**Design decisions:**
- **Algorithm:** Byte-Pair Encoding (BPE) with a ByteLevel pre-tokenizer, so every raw byte maps to a printable character and `<unk>` is never emitted for valid UTF-8 input.
- **Vocabulary size: 4096** — SVG is a constrained XML language. After preprocessing, recurring patterns (tag names, attribute names, path commands) dominate the corpus. 4096 tokens captures these efficiently without over-segmenting structure (too small) or overfitting rare coordinate strings (too large).
- **Special tokens:** `<pad>`, `<unk>`, `<bos>`, `<eos>` — sequences are automatically wrapped with `<bos>`/`<eos>` at encode time.
- **Min frequency:** 2 — token pairs seen only once are excluded, keeping the vocabulary robust.

