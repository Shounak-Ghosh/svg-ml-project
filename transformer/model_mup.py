"""
muP (Maximal Update Parameterization) Transformer for SVG language modeling (Part 3).

Reparameterizes SVGTransformer under muP so that the optimal learning rate found on the
smallest model (Tiny) transfers zero-shot to all larger widths.

Key changes vs the SP model in model.py:
  - Attention scale: 1/head_dim  (was 1/sqrt(head_dim))
  - LM head: MuReadout           (no weight tying with wte)
  - Embedding init: N(0, 1.0)    (was N(0, 0.02))
  - Hidden linear init: mup.init.kaiming_normal_(fan_in, linear) (scales with width)
  - Readout init: zeros           (was tied to embedding)
  - Optimizer: MuAdamW            (applies per-layer LR scaling automatically)

Reference: Yang et al. (2022), "Tensor Programs V: Tuning Large Neural Networks via
Zero-Shot Hyperparameter Transfer" (https://arxiv.org/abs/2203.09789)
"""

import math
from typing import Optional, Union

import torch
import torch.nn as nn
from torch.nn import functional as F

from mup import make_base_shapes, set_base_shapes, MuReadout
import mup.init as mu_init

# Reuse unchanged components from the SP model
from model import ModelConfig, LayerNorm, MLP


# ---------------------------------------------------------------------------
# Attention (muP version — only the scale changes)
# ---------------------------------------------------------------------------

class CausalSelfAttentionMuP(nn.Module):
    """
    Multi-head causal self-attention with muP attention scale.

    The only difference from CausalSelfAttention in model.py is the attention
    scale: 1/head_dim instead of 1/sqrt(head_dim). This is required by muP so
    that attention logit magnitude stays O(1) as width grows.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        assert config.d_model % config.n_heads == 0
        self.n_heads  = config.n_heads
        self.d_model  = config.d_model
        self.head_dim = config.d_model // config.n_heads
        self.dropout  = config.dropout

        self.c_attn     = nn.Linear(config.d_model, 3 * config.d_model, bias=config.bias)
        self.c_proj     = nn.Linear(config.d_model, config.d_model,     bias=config.bias)
        self.attn_drop  = nn.Dropout(config.dropout)
        self.resid_drop = nn.Dropout(config.dropout)

        self.flash = hasattr(F, "scaled_dot_product_attention")
        if not self.flash:
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
            # muP scale: 1/head_dim (PyTorch >= 2.1 supports scale kwarg)
            y = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
                scale=1.0 / self.head_dim,
            )
        else:
            # muP scale: 1/head_dim instead of 1/sqrt(head_dim)
            scale = 1.0 / self.head_dim
            att = (q @ k.transpose(-2, -1)) * scale
            att = att.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_drop(att)
            y   = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.c_proj(y))


class BlockMuP(nn.Module):
    """Transformer block using muP attention. MLP is unchanged from SP."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ln_1 = LayerNorm(config.d_model, bias=config.bias)
        self.attn = CausalSelfAttentionMuP(config)
        self.ln_2 = LayerNorm(config.d_model, bias=config.bias)
        self.mlp  = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


# ---------------------------------------------------------------------------
# Full muP model
# ---------------------------------------------------------------------------

class SVGTransformerMuP(nn.Module):
    """
    Decoder-only transformer with muP reparameterization.

    Architectural differences vs SVGTransformer (model.py):
      - lm_head is MuReadout (not nn.Linear); no weight tying with wte
      - Attention scale = 1/head_dim
      - Weight init uses mup.init functions (width-aware) after set_base_shapes
      - Optimizer should be MuAdamW (via configure_optimizers_mup)

    muP models count vocab_size * d_model more parameters than their SP
    counterparts because the lm_head weight is no longer shared with wte.
    """

    def __init__(self, config: ModelConfig, base_shapes: Optional[Union[str, object]] = None):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte  = nn.Embedding(config.vocab_size, config.d_model),
            wpe  = nn.Embedding(config.block_size, config.d_model),
            drop = nn.Dropout(config.dropout),
            h    = nn.ModuleList([BlockMuP(config) for _ in range(config.n_layers)]),
            ln_f = LayerNorm(config.d_model, bias=config.bias),
        ))
        # No weight tying: MuReadout has different init (zeros) and LR scaling
        self.lm_head = MuReadout(config.d_model, config.vocab_size, bias=False)

        if base_shapes is not None:
            # Set infshape attributes on all parameters (required for mup.init and MuAdamW)
            # rescale_params=False: we handle init ourselves in _mu_init_weights below
            set_base_shapes(self, base_shapes, rescale_params=False)

        self.apply(self._mu_init_weights)

        # Scale down residual projection weights by 1/sqrt(2*n_layers) [GPT-2 §2.3]
        # Applied after _mu_init_weights so it post-scales the kaiming result in-place
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                p.data.mul_(1.0 / math.sqrt(2 * config.n_layers))

    def _mu_init_weights(self, module: nn.Module) -> None:
        # MuReadout check must come before nn.Linear (MuReadout IS a Linear subclass)
        if isinstance(module, MuReadout):
            nn.init.zeros_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Linear):
            if hasattr(module.weight, "infshape"):
                # infshape set → training model: use width-aware muP init
                # kaiming_normal_ with fan_in + linear gives std = 1/sqrt(fan_in),
                # then mup scales by sqrt(base_fan_in / fan_in) for "infinite" dims
                mu_init.kaiming_normal_(
                    module.weight, a=0, mode="fan_in", nonlinearity="linear"
                )
            else:
                # base/delta construction: no infshape yet, use SP fallback
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            # Embeddings are input layers in muP — init at O(1) std, no width scaling
            nn.init.normal_(module.weight, mean=0.0, std=1.0)

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        _, T = idx.shape
        assert T <= self.config.block_size

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
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss

    def count_parameters(self) -> int:
        """Count unique trainable parameters. No weight tying in muP model."""
        seen: set[int] = set()
        total = 0
        for p in self.parameters():
            if p.requires_grad and id(p) not in seen:
                seen.add(id(p))
                total += p.numel()
        return total

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
        Autoregressive generation. Mirrors SVGTransformer.generate() exactly.

        When targets=None, the muP forward pass returns logits of shape
        (B, 1, vocab_size) — only the last token position. The slice
        logits[:, -1, :] still gives (B, vocab_size) correctly.
        """
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature   # (B, vocab_size)

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            if top_p is not None:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                remove = cum_probs - F.softmax(sorted_logits, dim=-1) > top_p
                sorted_logits[remove] = float("-inf")
                logits = torch.zeros_like(logits).scatter_(1, sorted_idx, sorted_logits)

            probs    = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            idx      = torch.cat([idx, next_tok], dim=1)

            if eos_id is not None and (next_tok == eos_id).all():
                break

        return idx

    def configure_optimizers_mup(
        self,
        weight_decay: float,
        lr: float,
        betas: tuple[float, float],
    ) -> tuple[torch.optim.AdamW, list[float]]:
        """
        MuAdamW with weight decay on 2D+ params only.

        MuAdamW splits param groups by width_mult and adjusts lr per group so
        that hidden-layer params get lr / width_mult (the muP LR transfer rule).
        Returns (optimizer, initial_lrs) where initial_lrs is the per-group lr
        snapshot needed to apply the cosine schedule correctly:

            g['lr'] = initial_lrs[i] * (lr_t / lr_max)

        This preserves the width_mult ratios across all schedule steps.
        """
        from mup.optim import MuAdamW

        unique: dict[int, tuple[str, nn.Parameter]] = {}
        for name, param in self.named_parameters():
            if param.requires_grad and id(param) not in unique:
                unique[id(param)] = (name, param)

        decay    = [p for _, (_, p) in unique.items() if p.dim() >= 2]
        no_decay = [p for _, (_, p) in unique.items() if p.dim() < 2]

        param_groups = [
            {"params": decay,    "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]

        optimizer   = MuAdamW(param_groups, lr=lr, betas=betas)
        initial_lrs = [g["lr"] for g in optimizer.param_groups]
        return optimizer, initial_lrs


# ---------------------------------------------------------------------------
# Base-shapes factory (run once before training)
# ---------------------------------------------------------------------------

def create_mup_base_shapes(
    savefile: str,
    config: ModelConfig,
) -> object:
    """
    Build and save muP base shapes for the given model config topology.

    Creates base (narrow) and delta (2x wider) models with the SAME n_layers,
    n_heads, vocab_size, and block_size as `config` but with smaller d_model.
    This ensures parameter names match when set_base_shapes is later called on
    the actual training model.

    Only d_model and d_ff differ between base and delta, so only those
    dimensions are flagged as "infinite" (width) dimensions by mup.

    Must be called once per unique (n_layers, n_heads) topology before training.
    """
    # Smallest valid d_model for this topology: n_heads * 8 (head_dim = 8)
    base_d_model  = config.n_heads * 8
    delta_d_model = base_d_model * 2

    base_cfg = ModelConfig(
        vocab_size=config.vocab_size,
        block_size=config.block_size,
        d_model=base_d_model,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        d_ff=base_d_model * 4,
    )
    delta_cfg = ModelConfig(
        vocab_size=config.vocab_size,
        block_size=config.block_size,
        d_model=delta_d_model,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        d_ff=delta_d_model * 4,
    )
    # Build CPU models without base_shapes — just for shape annotation
    base_model  = SVGTransformerMuP(base_cfg,  base_shapes=None)
    delta_model = SVGTransformerMuP(delta_cfg, base_shapes=None)
    bsh = make_base_shapes(base_model, delta_model, savefile=savefile)
    return bsh
