# Lessons

## Monitoring long training runs
- **`output_dir/debug_info.txt` is the real-time progress signal** (Step, Sigma), written
  every optimization step. W&B's LOCAL files (`wandb/run-*/files/wandb-summary.json`,
  `logs/debug-internal.log`) buffer heavily and lag by minutes — an empty summary does NOT
  mean the run is stuck. Always check `debug_info.txt` mtime + Step before concluding a hang.
  (Cost me a lot of time and possibly a good 768×1152 run I killed prematurely.)
- The first training step is slow (CUDA kernel autotuning under int8-quanto) — several minutes
  before the first artifact appears; subsequent steps are ~3 s. Don't judge speed on step 1.

## Launching background training in this harness
- `nohup … &` alone stays in the Bash tool's process group; when the tool call hits its 2-min
  wall the harness kills the whole tree, killing training right around the first step.
  **Launch with `setsid` (+ `disown`, stdin `< /dev/null`)** so it's a session leader, then poll
  from SEPARATE short tool calls. Verify with `ps -o stat` showing `s` (session leader).

## Diagnosing silent hangs/deaths (no Python traceback)
- Async CUDA faults / CPU spins produce no traceback. Use a **faulthandler watchdog**:
  a tiny launcher that calls `faulthandler.dump_traceback_later(N, repeat=True)` then
  `runpy.run_path("scripts/train.py")`. It auto-dumps ALL thread stacks every N s — reveals the
  exact stuck line (py-spy needs root here; `kill -ABRT` was ignored while in native code).

## OmniTransfer trainer gotchas (ltx-trainer)
- Pixel-space VGG style loss (`use_decoded_pixels_for_style`) decodes latents through the VAE
  decoder, which the trainer offloads to **CPU** between uses → a CPU video-VAE conv that never
  finishes. It's a Grok add-on, NOT in the paper (loss is flow-matching MSE). Disable it.
- `torch.optim.Muon` validates EVERY param in its constructor and rejects non-2D — so a 3D param
  (ConceptEmbedding `[tasks, concepts, dim]`) breaks it even when placed in the `muon:False`
  fallback group. Use AdamW when the strategy has non-2D trainable params.
- OmniTransfer strategy reads `conditions["video_prompt_embeds"]` with no `prompt_embeds`
  fallback; `compute_embeddings_ltx23.py` saves `prompt_embeds`. Rename the key.
- `process_videos.py` / ltx-core `audio_vae` hard-import `torchaudio`; install a matching
  version (`torchaudio==2.10.0` for torch 2.10.0+cu128).

## Token budget = resolution × frames (O(N²) attention)
- Tokens/stream = F_latent × H_latent × W_latent; RCL doubles it (ref+tgt). Fast diorama runs
  were ~364 tokens (3.5 s/step). 768×1152 F=4 = 6912 tokens is too slow on a 32 GB card with
  int8-quanto 22B. 512×768 F=4 = 3072 tokens ≈ 3 s/step is the tractable sweet spot here.

## CUDA device ordering
- Default CUDA order ≠ `nvidia-smi` index. Here default order: dev 0 = RTX 5090 (32 GB),
  dev 1 = PRO 4000 (24 GB) — REVERSED vs nvidia-smi. Use `CUDA_DEVICE_ORDER=PCI_BUS_ID` to align,
  or verify with `torch.cuda.get_device_name(i)` before setting `CUDA_VISIBLE_DEVICES`.

## NaN divergence (added after first mashup_v2 run)
- First mashup_v2 run (AdamW lr 1e-4, bf16, int8-quanto, effective batch 1) diverged to NaN
  (~step 400); ALL saved checkpoints had ~47% NaN weights — unusable. `debug_info.txt` does NOT
  show loss, so the NaN was invisible there; must check loss (W&B) or checkpoint weights.
- Fix: lr 1e-4 → **3e-5** + `warmup_steps: 100`. New run's step-250 checkpoint = 0 NaN. 
- **Verify a run's health by loading a checkpoint and checking `torch.isnan` over its tensors**
  (safetensors load_file) — the definitive, reliable signal (W&B local files parse unreliably live).
- Data was clean (no NaN/Inf in latents/conditions), so this was pure optimizer instability.
- ROOT CAUSE + real fix: the NaN was an OCCASIONAL non-finite gradient (one bad step, e.g. step
  256) in the VIDEO-attention branch (attn1/attn2 all blocks; audio_attn* stayed clean because
  audio is disconnected in T2V). A single NaN step's grad → clip(NaN)=NaN → NaN weights → cascade.
  Fix: added a non-finite-grad guard in trainer.py — use the norm from clip_grad_norm_ and SKIP
  the optimizer step when it's non-finite. Guard fired ~once per 250 steps; checkpoints stay clean.
  (lr 3e-5 + warmup alone only DELAYED the NaN; the guard is what actually fixes it.)
