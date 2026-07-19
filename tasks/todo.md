# OmniTransfer — Facebook Reels: paper alignment + training

Paper: OmniTransfer (arXiv:2601.14250v1). Base: LTX 2.3 (22B distilled).
Goal: align training data with the paper's **video-reference** design, fix prompt
style-leakage, and train a style-transfer LoRA on the facebook_reels mashup data.

## Review findings (paper vs. current setup)
- Code (TPB/RCL/TMA/ConceptEmbedding in `omnitransfer/components.py`,`strategy.py`) is
  faithful to the paper. Misalignments are at DATA + ENV level:
  1. [CRITICAL] Trained on single-frame latents (F=1) → TPB Δ=(f,0,0) & RCL inert. FIX: re-encode multi-frame.
  2. [HIGH] Prompts leak target style ("warm film grain, cinematic lighting"). FIX: neutral content prompts.
  3. [BLOCKER] stable-diffusion-webui-forge holds ~24GB on 5090 → run died at first forward. (User frees GPU.)
  4. [MED] Config used AdamW; repo mandates Muon (lr 0.02).
  5. [MED] No output_dir / checkpoint interval; validation off.

## Decisions (from user)
- Frames: **re-encode at 49 frames (F=7)**. 85/92 scenes have ≥49 frames.
- Prompts: **rewrite neutral + re-encode with Gemma (LTX 2.3)**.
- GPU: user frees 5090; both GPUs (4000 + 5090) available.
- Model: LTX 2.3 `/media/2TB/ltx-models/ltx2.3/ltx-2.3-22b-distilled.safetensors`.
- Output root: `/media/2TB/omnitransfer/data/mashup_v2` + `/media/2TB/omnitransfer/output/mashup_v2`.

## Plan
### Phase A — Re-encode scenes multi-frame (GPU 0 available now)
- [ ] A1. Enumerate scene clips with ≥49 frames (~85). Transcode → 768×1152, 49f, 24fps.
- [ ] A2. Build scenes.csv (media_path, neutral caption).
- [ ] A3. VAE-encode via process_videos.py → scene-latent pool (128,7,36,24).

### Phase B — Neutral prompts + Gemma embeddings (GPU after 5090 free)
- [ ] B1. Generate neutral, style-free prompts per pair (content/scene only).
- [ ] B2. Gemma-encode via compute_embeddings_ltx23.py → conditions pool.

### Phase C — Materialize training set
- [ ] C1. Cross-pair scenes within each reel (ref≠tgt). Build metadata_v2.json.
- [ ] C2. Symlink indexed latents/ (tgt), reference_latents/ (ref), conditions/ per pair.

### Phase D — Config + launch
- [ ] D1. Write train_config.yaml: omnitransfer/style_transfer, Muon lr 0.02, int8-quanto,
        grad-checkpoint, output_dir, checkpoint interval, num_frames_to_visualize>1, W&B on.
- [ ] D2. (User frees 5090) Launch on both GPUs; inspect first 5–10 steps for NaN.
- [ ] D3. Verify checkpoints written; monitor loss trend.

## Review (done)
Training RUNNING at 512×768 F=4 (~3 s/step), step 100+ in ~6 min, no NaN, 20.8 GB.
Paper-aligned dataset: 468 pairs, multi-frame (F=4) video refs, ref≠tgt, neutral prompts.

Bugs found & fixed along the way (all in `ltx-trainer`):
1. Data: single-frame (F=1) latents → re-encoded MULTI-FRAME (F=4, 25 frames). Core paper fix.
2. Data: style-leaking prompts → neutral "A scene from {movie}." (compute_embeddings_ltx23).
3. process_videos.py: unconditional `import torchaudio` → made lazy (video path needs no audio).
   Installed torchaudio==2.10.0 (matches torch 2.10.0+cu128).
4. Optimizer: Muon rejects the 3D ConceptEmbedding param even in its AdamW-fallback group
   (torch.optim.Muon validates ALL params) → switched to AdamW lr 1e-4 (user OK'd dropping Muon).
5. Conditions key: compute_embeddings saves `prompt_embeds`; OmniTransfer strategy reads
   `video_prompt_embeds` (no legacy fallback) → renamed key in condition files.
6. **Style-loss hang**: `use_decoded_pixels_for_style` decodes latents via the VAE decoder,
   which the trainer offloads to CPU between uses → CPU conv hangs the step forever.
   Disabled pixel/VGG style loss (not in the paper; loss is flow-matching MSE). trainer.py:576.
7. **Process death at ~2 min**: `nohup … &` stays in the tool's process group; my Bash-tool
   2-min timeout killed the whole tree. Fix: launch with `setsid` (own session) + poll from
   separate tool calls.
8. Resolution: 768×1152 F=4 = 6912 tokens is too slow (O(N²) attention). Dropped to 512×768
   (3072 tokens) per user; ~3 s/step.

KEY MONITORING LESSON: W&B local summary/debug files buffer heavily and lag reality by minutes.
The real-time progress signal is `output_dir/debug_info.txt` (Step/Sigma) — check THAT, not
the local wandb files, to judge whether a run is progressing.

Config: /media/2TB/omnitransfer/data/mashup_v2/train_config.yaml
Data:   /media/2TB/omnitransfer/data/mashup_v2/ (latents, reference_latents, conditions)
Output: /media/2TB/omnitransfer/output/mashup_v2/ (checkpoints every 250 steps)
Rebuild driver: tools/rebuild_mashup_v2.py (prepare|finalize)
