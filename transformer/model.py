"""
Decoder-only Transformer for SVG language modeling (Part 2: Scaling Study).

Based on nanoGPT (Karpathy, 2022): https://github.com/karpathy/nanoGPT

--- What is borrowed from nanoGPT (largely unchanged) ---
  - Overall GPT-style skeleton: token/position embeddings, stacked blocks, LM head
  - CausalSelfAttention: QKV fused projection, multi-head split, causal mask, flash attention
  - LayerNorm with optional bias
  - Weight initialization: N(0, 0.02) + scaled-down residual projections (GPT-2 §2.3)
  - Weight tying between token embedding and LM head
  - configure_optimizers(): AdamW with separate decay / no-decay param groups
  - generate(): temperature + top-k autoregressive sampling skeleton

--- What is modified / added for this project ---
  - ModelConfig.d_ff: configurable feedforward hidden dim (not fixed at 4×d_model)
  - MODEL_CONFIGS dict: 5 named sizes matching the Part 2 scaling table (Tiny–XL)
  - SVGTransformer.count_parameters(): deduplicates weight-tied params via id()
  - generate(): added top-p (nucleus) sampling and eos_id early stopping
"""

import math
import inspect
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch.nn import functional as F


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    vocab_size: int = 4096
    block_size: int = 1024        # context window (tokens)
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    d_ff: int = 512               # MLP hidden dim; not constrained to 4×d_model [ours]
    dropout: float = 0.0
    bias: bool = False            # bias in Linear/LayerNorm layers


# Part 2 scaling study — 5 model sizes (project spec Table 1)          [ours]
MODEL_CONFIGS: dict[str, ModelConfig] = {
    "tiny":   ModelConfig(d_model=128,  n_layers=4,  n_heads=4,  d_ff=512),
    "small":  ModelConfig(d_model=192,  n_layers=6,  n_heads=6,  d_ff=768),
    "medium": ModelConfig(d_model=384,  n_layers=6,  n_heads=6,  d_ff=1536),
    "large":  ModelConfig(d_model=512,  n_layers=10, n_heads=8,  d_ff=2048),
    "xl":     ModelConfig(d_model=768,  n_layers=12, n_heads=12, d_ff=3072),
}


# ---------------------------------------------------------------------------
# Submodules
# ---------------------------------------------------------------------------

class LayerNorm(nn.Module):
    """LayerNorm with optional bias. [Borrowed from nanoGPT]"""

    def __init__(self, ndim: int, bias: bool):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias   = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention with optional flash attention.
    [Borrowed from nanoGPT — no substantive changes]
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        assert config.d_model % config.n_heads == 0, (
            f"d_model ({config.d_model}) must be divisible by n_heads ({config.n_heads})"
        )
        self.n_heads  = config.n_heads
        self.d_model  = config.d_model
        self.head_dim = config.d_model // config.n_heads
        self.dropout  = config.dropout

        # Fused QKV projection
        self.c_attn     = nn.Linear(config.d_model, 3 * config.d_model, bias=config.bias)
        self.c_proj     = nn.Linear(config.d_model, config.d_model,     bias=config.bias)
        self.attn_drop  = nn.Dropout(config.dropout)
        self.resid_drop = nn.Dropout(config.dropout)

        # Use flash attention when available (PyTorch >= 2.0)
        self.flash = hasattr(F, "scaled_dot_product_attention")
        if not self.flash:
            # Fallback: pre-computed causal mask
            self.register_buffer(
                "causal_mask",
                torch.tril(torch.ones(config.block_size, config.block_size))
                     .view(1, 1, config.block_size, config.block_size),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.d_model, dim=2)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        if self.flash:
            y = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            scale = 1.0 / math.sqrt(self.head_dim)
            att = (q @ k.transpose(-2, -1)) * scale
            att = att.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_drop(att)
            y   = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.c_proj(y))


class MLP(nn.Module):
    """
    Feed-forward network with GELU activation.
    [Modified from nanoGPT: uses config.d_ff instead of hardcoded 4×d_model]
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.c_fc   = nn.Linear(config.d_model, config.d_ff,   bias=config.bias)
        self.gelu   = nn.GELU()
        self.c_proj = nn.Linear(config.d_ff,   config.d_model, bias=config.bias)
        self.drop   = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.c_proj(self.gelu(self.c_fc(x))))


class Block(nn.Module):
    """Transformer block: pre-LN attention residual + pre-LN MLP residual. [Borrowed from nanoGPT]"""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ln_1 = LayerNorm(config.d_model, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.d_model, bias=config.bias)
        self.mlp  = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class SVGTransformer(nn.Module):
    """
    Decoder-only transformer language model for SVG code generation.

    Architecture based on nanoGPT (Karpathy, 2022).
    Key differences vs nanoGPT:
      - Configurable d_ff (via ModelConfig.d_ff)                  [ours]
      - Named model presets in MODEL_CONFIGS                       [ours]
      - count_parameters() with weight-tie deduplication           [ours]
      - generate() extended with top-p and eos_id early stop       [ours]
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte  = nn.Embedding(config.vocab_size, config.d_model),   # token embeddings
            wpe  = nn.Embedding(config.block_size, config.d_model),   # position embeddings
            drop = nn.Dropout(config.dropout),
            h    = nn.ModuleList([Block(config) for _ in range(config.n_layers)]),
            ln_f = LayerNorm(config.d_model, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying: token embedding ↔ LM head share the same tensor
        # [from GPT-2 paper, via nanoGPT]
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)
        # Scale down residual projection weights by 1/sqrt(2*n_layers)
        # [GPT-2 paper §2.3, via nanoGPT]
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layers))

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        _, T = idx.shape
        assert T <= self.config.block_size, (
            f"Sequence length {T} exceeds block_size {self.config.block_size}"
        )

        pos = torch.arange(T, dtype=torch.long, device=idx.device)
        x = self.transformer.drop(
            self.transformer.wte(idx) + self.transformer.wpe(pos)
        )
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )
        else:
            # Inference: only compute logits for the last position
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss

    def count_parameters(self) -> int:
        """
        Count unique trainable parameters.
        Uses id() to deduplicate weight-tied tensors (wte.weight == lm_head.weight).
        [Original]
        """
        seen: set[int] = set()
        total = 0
        for p in self.parameters():
            if p.requires_grad and id(p) not in seen:
                seen.add(id(p))
                total += p.numel()
        return total

    def configure_optimizers(
        self,
        weight_decay: float,
        lr: float,
        betas: tuple[float, float],
        device_type: str,
    ) -> torch.optim.AdamW:
        """
        AdamW with weight decay applied only to 2D+ tensors (no biases or LayerNorm weights).
        Deduplicates weight-tied parameters so each tensor appears in exactly one group.
        [Pattern borrowed from nanoGPT; deduplication logic is ours]
        """
        # Collect unique parameters by id
        unique: dict[int, tuple[str, torch.nn.Parameter]] = {}
        for name, param in self.named_parameters():
            if param.requires_grad and id(param) not in unique:
                unique[id(param)] = (name, param)

        decay    = [p for _, (_, p) in unique.items() if p.dim() >= 2]
        no_decay = [p for _, (_, p) in unique.items() if p.dim() < 2]

        optim_groups = [
            {"params": decay,    "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]

        # Use fused AdamW on CUDA if available (faster)
        extra_kwargs: dict = {}
        if device_type == "cuda" and "fused" in inspect.signature(torch.optim.AdamW).parameters:
            extra_kwargs["fused"] = True

        return torch.optim.AdamW(optim_groups, lr=lr, betas=betas, **extra_kwargs)

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        eos_id: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Autoregressive generation.
        [Base structure borrowed from nanoGPT; top-p and eos_id added for this project]

        Args:
            idx:            (B, T) seed token tensor
            max_new_tokens: maximum tokens to generate
            temperature:    softmax temperature (< 1 = sharper, > 1 = flatter)
            top_k:          keep only the top-k logits before sampling
            top_p:          nucleus (top-p) filtering — cumulative probability threshold
            eos_id:         stop generation when all sequences emit this token
        """
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature   # (B, vocab_size)

            # Top-k filtering [nanoGPT]
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            # Top-p (nucleus) filtering [added for this project]
            if top_p is not None:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                # Remove tokens once cumulative prob exceeds top_p
                remove = cum_probs - F.softmax(sorted_logits, dim=-1) > top_p
                sorted_logits[remove] = float("-inf")
                logits = torch.zeros_like(logits).scatter_(1, sorted_idx, sorted_logits)

            probs     = F.softmax(logits, dim=-1)
            next_tok  = torch.multinomial(probs, num_samples=1)
            idx       = torch.cat([idx, next_tok], dim=1)

            if eos_id is not None and (next_tok == eos_id).all():
                break

        return idx
