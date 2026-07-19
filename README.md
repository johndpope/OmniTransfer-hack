# OmniTransfer‑hack → Ingredients: identity video from one image (LTX‑2.3)

This repo started as a from‑scratch recreation of **OmniTransfer**
([arXiv:2601.14250](https://arxiv.org/abs/2601.14250)) on LTX‑2.3. After building and
training it, we **pivoted for identity** to the official
**[LTX‑2.3‑22B IC‑LoRA‑Ingredients](https://huggingface.co/Lightricks/LTX-2.3-22b-IC-LoRA-Ingredients)** —
a proven, rank‑128 in‑context reference adapter that carries a character's
appearance from **a single image** into any prompted scene. It beats a
from‑scratch OmniTransfer Stage‑1 for identity and needs **zero training** to use.

The OmniTransfer code (and a set of real bug fixes) remains in the repo as the
research track and as the **in‑repo trainer** for making/adapting IC‑LoRAs.

---

## TL;DR

| Goal | Path | Training? |
|------|------|-----------|
| **Character identity → new scene from 1 image** | Ingredients IC‑LoRA (inference) | ❌ none |
| Adapt/finetune identity to *your* characters | `video_to_video` IC‑LoRA trainer (this repo), warm‑start from Ingredients | ✅ in‑repo |
| Semantic control over *which* attributes carry | MetaQuery/TMA layer (V2, scaffolded) | ✅ in‑repo |

```bash
# One portrait + a prompt → identity‑preserving clip (no training)
python tools/ingredients_generate.py \
    --image path/to/portrait.png \
    --prompt "A bald older man with a grey goatee in a tan trench coat walks down a rainy neon street at night, cinematic" \
    --output out.mp4
```

Result: the man's face, goatee and coat carry into a brand‑new prompted scene.
(Woman portrait → "sits at a sunlit cafe sipping coffee" works the same way.)

---

## How it works

**Ingredients is an in‑context reference adapter.** You give the LTX‑2.3 distilled
pipeline one (or up to four) **reference image(s)** as frame‑0 conditioning plus the
Ingredients LoRA; the model generates a video that keeps the reference's
appearance while following the text prompt. It **generates** (it does not edit
existing footage).

- **LoRA:** rank 128, targets `attn1`/`attn2` (q/k/v/out) **and** the FFN across all
  48 DiT blocks — same key naming as LTX‑2.3, so it loads cleanly.
- **Same family as our trainer:** that target‑module set is exactly what this repo's
  **`video_to_video` (IC‑LoRA)** training strategy trains — clean reference latents
  concatenated with a noised target, loss on the target only. So you can train or
  finetune Ingredients‑style LoRAs here.

---

## Pipeline

### 1. Build a portrait library (references)
```bash
python tools/extract_portraits.py \
    --clips-dir  /media/2TB/omnitransfer/data/mashup_v2/clips \
    --gender-json /media/2TB/omnitransfer/data/mashup_v2/scene_gender.json \
    --out-dir    /media/2TB/omnitransfer/data/mashup_v2/portraits
```
Face‑detects + crops one clean front‑facing portrait (face + outfit) per
single‑person scene. Use a **clean cropped portrait**, not a full scene frame, or
the background bleeds into early frames.

### 2. Generate (inference, no training)
```bash
python tools/ingredients_generate.py --image <portrait.png> --prompt "<scene>" --output out.mp4
```
Reproducible wrapper around castlehill's `ltx_pipelines.distilled` with the config
that works on this box (see **Gotchas** — full gemma, fp8‑cast, the LoRA).

### 3. (Optional) Train an identity IC‑LoRA on your characters
Warm‑start from the Ingredients LoRA and adapt to your data with the in‑repo trainer:
```yaml
# ltx-trainer/configs/ltx2_v2v_ic_lora.yaml (edit)
model:
  training_mode: lora
  load_checkpoint: /media/2TB/ltx-models/LTX-2.3-22b-IC-LoRA-Ingredients/ltx-2.3-22b-ic-lora-ingredients-0.9.safetensors
training_strategy:
  name: video_to_video           # IC‑LoRA: clean ref latents + noised target, loss on target
  reference_latents_dir: reference_latents
```
```bash
uv run python ltx-trainer/scripts/train.py ltx-trainer/configs/ltx2_v2v_ic_lora.yaml
```
Data layout is the same one `tools/rebuild_mashup_v2.py` produces:
`latents/` (target), `reference_latents/` (clean ref), `conditions/` (text).

### 4. (V2) MetaQuery semantic control — scaffolded
`tools/metaquery_ingredients.py` adds a learnable **MetaQuery** channel: an MLLM
(Qwen‑VL) reads the reference, learnable queries aggregate it, a connector maps to
the DiT context dim, and the result is **prepended to `video_context`** (the
injection point in castlehill `distilled.py:101`). The conditioner is **zero‑init
gated** (a no‑op until trained, so it can't degrade output). Remaining milestone:
train the connector (flow‑matching, DiT + Ingredients LoRA frozen) with the
in‑repo trainer.

---

## Data tooling (facebook_reels mashups)

Built for the movie‑mashup reels (a base scene with famous characters/outfits
swapped in). Reference↔target pairs must match the subject or the model ignores the
reference — verified from W&B reconstructions.

- `tools/classify_scene_gender.py` — Qwen2.5‑VL labels each scene by **composition**
  (`one_woman`/`one_man`/`two_women`/`two_men`/`man_and_woman`/`group`/`none`).
- `tools/rebuild_mashup_v2.py` — builds multi‑frame latents + **composition‑matched**
  reference/target pairs (a two‑men target only pairs with two‑men references),
  neutral prompts, symlinked dataset, config. (`prepare` / `finalize`.)
- `ltx-trainer/scripts/compute_qwen_vl_features.py` — precompute MLLM features for
  the MetaQuery/TMA path (`--dummy` for wiring tests).

---

## Models & paths (this machine)

| Thing | Path |
|------|------|
| LTX‑2.3 distilled | `/media/2TB/ltx-models/ltx2.3/ltx-2.3-22b-distilled.safetensors` |
| Spatial upscaler | `/media/2TB/ltx-models/ltx2.3/ltx-2.3-spatial-upscaler-x2-1.1.safetensors` |
| **Gemma (FULL, multi‑shard)** | `/media/2TB/ltx-models/gemma` |
| Ingredients IC‑LoRA | `/media/2TB/ltx-models/LTX-2.3-22b-IC-LoRA-Ingredients/ltx-2.3-22b-ic-lora-ingredients-0.9.safetensors` |
| Qwen2.5‑VL‑7B (MetaQuery) | `/media/2TB/ltx-models/qwen2.5-vl-7b` |
| Castlehill inference pipeline | `~/Documents/GitHub/ltx2-castlehill` (`LTX2_PATH`) |

---

## Gotchas (hard‑won)

- **Use the FULL multi‑shard gemma for inference.** The `gemma-3-12b-fp4` variant is
  text‑only → its vision tower can't init → `Cannot copy out of meta tensor` crash.
- **Clean cropped portraits**, not full scene frames (background bleed).
- **Ingredients inference is FlashAttention2‑free here** — the wrapper falls back to
  SDPA (`use_flash_attention=False`).
- **Training stability (OmniTransfer path):** a single non‑finite step corrupts every
  checkpoint; the trainer now **skips non‑finite‑grad steps** (guard). `AdamW lr 3e‑5
  + warmup` (not `1e‑4`, which diverged). Muon rejects the 3D ConceptEmbedding param.
- **Monitor by `output_dir/debug_info.txt`**, not the lazily‑buffered local W&B files.
- **Launch background training with `setsid`** (bare `nohup` dies with the shell).

---

## OmniTransfer (origin / research track)

The OmniTransfer implementation is under
`ltx-trainer/src/ltx_trainer/omnitransfer/` (TPB, RCL, TMA/MetaQuery,
ConceptEmbedding). Notable fixes made while training it (all on the open PR):

- **RCL was a no‑op** — `rcl_split_point` was consumed nowhere in ltx‑core, so the
  reference was never decoupled. Now enforced via a self‑attention keep‑mask
  (`Modality.attention_mask`).
- **Non‑finite‑gradient guard** in the trainer.
- **Muon param routing** (ndim==2), **torchaudio** lazy import, **TMA connector dim**
  4096 (was 3840, mismatched LTX‑2.3).

Faithful 3‑stage recreation (Stage 1 DiT+TPB+RCL → Stage 2 TMA → Stage 3 joint) is
possible with the in‑repo trainer, but for **identity specifically the Ingredients
approach wins** — hence the pivot.

---

## Repo layout

```
tools/
  ingredients_generate.py     # inference wrapper (Ingredients LoRA + 1 image)
  extract_portraits.py        # face‑crop portrait library
  classify_scene_gender.py    # Qwen‑VL composition labels
  rebuild_mashup_v2.py        # composition‑matched dataset builder
  metaquery_ingredients.py    # V2 MetaQuery semantic‑control scaffold
ltx-trainer/
  src/ltx_trainer/training_strategies/video_to_video.py   # IC‑LoRA trainer
  src/ltx_trainer/omnitransfer/                            # OmniTransfer (TPB/RCL/TMA)
  scripts/{train,process_videos,compute_embeddings_ltx23,compute_qwen_vl_features}.py
  configs/ltx2_v2v_ic_lora.yaml
```

---

## Support this project

Training video models needs real GPU compute. If this is useful, consider donating
[Vast.ai](https://vast.ai) credits.

**Send credits to:** `jp@bellgeorge.com` · `vastai transfer credit jp@bellgeorge.com <AMOUNT>`

| Tier | Amount | Helps with |
|------|--------|-----------|
| Coffee | $5–10 | experiments, fixes |
| Mates rates | $25–50 | a few A100 hours |
| Supporter | $100–250 | a full training run |
| Enterprise | $500+ | multi‑stage training, new features |

---

## References
- OmniTransfer — [arXiv:2601.14250](https://arxiv.org/abs/2601.14250)
- LTX‑2.3‑22B IC‑LoRA‑Ingredients — [HuggingFace](https://huggingface.co/Lightricks/LTX-2.3-22b-IC-LoRA-Ingredients)
- Uses the LTX‑2 Community License (commercial entities >$10M rev need a commercial license; derivatives inherit the license).
