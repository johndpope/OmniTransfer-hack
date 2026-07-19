#!/usr/bin/env python3
"""V2: MetaQuery/TMA semantic-control layer on top of the Ingredients pipeline.

Ingredients carries appearance from the reference image (in-context latents).
This adds a *semantic* channel: an MLLM (Qwen-VL) reads the reference, a bank of
learnable MetaQueries aggregate its features via cross-attention, a connector maps
them to LTX-2.3's cross-attention dim, and the result is PREPENDED to the pipeline's
`video_context` (castlehill distilled.py:101). That gives learnable control over
*which* semantics carry — the "transcend" step beyond raw Ingredients.

Design notes:
- The injection is `video_context = cat([meta_context, video_context], dim=1)` —
  exactly the mechanism the OmniTransfer strategy used to prepend TMA context.
- The conditioner has a **zero-init output gate**, so before training it emits ~0
  and is a no-op: injecting it never degrades the working Ingredients output. Only
  training moves the gate off zero.

Status: inference plumbing + trainable module (this file). Training loop:
`train_connector()` below is the flow-matching connector-only objective (DiT +
Ingredients LoRA frozen), analogous to OmniTransfer Stage 2. Wiring the training
data (portrait, prompt, GT reel clip) is the remaining step.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

MLLM_DIM = 3584   # Qwen2.5-VL-7B hidden
CTX_DIM = 4096    # LTX-2.3 cross-attention / video_context dim


class MetaQueryConditioner(nn.Module):
    """MLLM features [B, S, MLLM_DIM] -> semantic context [B, K, CTX_DIM].

    Learnable queries cross-attend to the MLLM features (task-adaptive aggregation),
    a connector MLP maps to the DiT context dim, and a zero-init scalar gate keeps
    the whole thing a no-op until trained.
    """

    def __init__(self, num_queries: int = 8, num_tasks: int = 1, heads: int = 8):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(num_tasks, num_queries, CTX_DIM) * 0.02)
        self.kv_proj = nn.Linear(MLLM_DIM, CTX_DIM)
        self.in_norm = nn.LayerNorm(MLLM_DIM)
        self.attn = nn.MultiheadAttention(CTX_DIM, heads, batch_first=True)
        self.connector = nn.Sequential(
            nn.Linear(CTX_DIM, CTX_DIM), nn.GELU(),
            nn.Linear(CTX_DIM, CTX_DIM), nn.GELU(),
            nn.Linear(CTX_DIM, CTX_DIM),
        )
        self.out_norm = nn.LayerNorm(CTX_DIM)
        # Zero-init gate -> untrained conditioner emits 0 (no-op injection).
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, mllm_features: torch.Tensor, task: int = 0) -> torch.Tensor:
        b = mllm_features.shape[0]
        kv = self.kv_proj(self.in_norm(mllm_features))          # [B, S, CTX]
        q = self.queries[task].unsqueeze(0).expand(b, -1, -1)   # [B, K, CTX]
        agg, _ = self.attn(q, kv, kv)                           # [B, K, CTX]
        ctx = self.out_norm(self.connector(agg))
        return torch.tanh(self.gate) * ctx                      # gated (0 until trained)


def inject_into_distilled(pipeline, conditioner: MetaQueryConditioner, mllm_features: torch.Tensor):
    """Monkeypatch a castlehill DistilledPipeline so its next __call__ prepends the
    MetaQuery context to `video_context`. Call once before pipeline(...).

    `mllm_features`: [1, S, MLLM_DIM] Qwen-VL features for the reference image
    (produce with ltx-trainer/scripts/compute_qwen_vl_features.py-style extraction).
    """
    import types
    from ltx_pipelines.utils.helpers import encode_prompts as _encode_prompts  # noqa: PLC0415

    dev = pipeline.device
    meta_ctx = conditioner(mllm_features.to(dev))  # [1, K, CTX]

    orig_call = pipeline.__call__

    def patched_call(self, *a, **k):
        # Patch encode_prompts result via a thin wrapper on video_context.
        import ltx_pipelines.distilled as _d  # noqa: PLC0415
        real = _d.encode_prompts

        def wrapped(*pa, **pk):
            (ctx_p,) = real(*pa, **pk)
            ctx_p.video_encoding = torch.cat(
                [meta_ctx.to(ctx_p.video_encoding.dtype), ctx_p.video_encoding], dim=1
            )
            return (ctx_p,)

        _d.encode_prompts = wrapped
        try:
            return orig_call(*a, **k)
        finally:
            _d.encode_prompts = real

    pipeline.__call__ = types.MethodType(patched_call, pipeline)
    return pipeline


def train_connector(*args, **kwargs):  # noqa: D401
    """Connector-only training (flow-matching, DiT + Ingredients LoRA frozen).

    TODO(v2): for each (portrait, prompt, GT reel clip):
      1. Qwen-VL features from the portrait -> conditioner -> meta_ctx.
      2. Prepend to video_context; run the frozen DiT forward at a sampled sigma.
      3. Flow-matching MSE vs the GT clip's velocity; step ONLY conditioner params.
    Reuses the gender/composition portrait library as (character, GT) pairs.
    """
    raise NotImplementedError("v2 training loop — next milestone (see docstring).")


if __name__ == "__main__":
    # Shape self-test: conditioner is a no-op at init (gate=0), correct out shape.
    m = MetaQueryConditioner()
    feats = torch.randn(1, 546, MLLM_DIM)
    out = m(feats)
    print("out:", tuple(out.shape), "max|ctx| at init (should be ~0):", out.abs().max().item())
    assert out.shape == (1, 8, CTX_DIM)
    assert out.abs().max().item() < 1e-6, "gate should zero the output before training"
    print("OK: [1,8,4096], zero-gated (safe no-op until trained)")
