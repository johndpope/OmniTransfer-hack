# AMB3R 3D Reconstruction Integration — Training Data Report

**Date**: March 2026
**Context**: Integration of AMB3R (arXiv:2511.20343) 3D losses + geometric token encoder into LTX-2 OmniTransfer
**Branch**: `feat/amb3r`

---

## 1. Executive Summary

The AMB3R integration adds geometric consistency enforcement to OmniTransfer's video generation pipeline. It requires **pre-computed 3D pseudo-ground-truth** (depth maps, 3D point clouds, confidence scores, surface normals) generated offline by a frozen AMB3R or DepthAnything3 model.

This report identifies which training datasets provide the strongest 3D signal, how to generate the pseudo-GT, the exact `.pt` file format required, and a prioritized data pipeline strategy.

**Key recommendation**: Run frozen AMB3R on your existing OmniTransfer training videos first (zero domain gap), then supplement with public 3D benchmarks for validation.

---

## 2. Pseudo-GT Format Specification

### 2.1 File Structure

```
preprocessed_data_root/
├── latents/              # Existing: VAE-encoded video latents
│   ├── 000.pt
│   └── ...
├── conditions/           # Existing: Gemma text embeddings
│   ├── 000.pt
│   └── ...
├── reference_latents/    # Existing: Reference video latents
│   ├── 000.pt
│   └── ...
├── depth_3d/             # NEW: AMB3R 3D pseudo-GT
│   ├── 000.pt            # Must match filenames in latents/
│   ├── 001.pt
│   └── ...
└── metadata.json
```

### 2.2 Per-Sample `.pt` Format

```python
# depth_3d/000.pt
{
    "depth":          Tensor[F, H, W],        # Per-frame metric depth maps (meters)
    "points":         Tensor[F, H, W, 3],     # Per-frame 3D world points (x, y, z)
    "confidence":     Tensor[F, H, W],        # AMB3R confidence: 1 + exp(log_conf)
    "normals":        Tensor[F, H, W, 3],     # Per-pixel surface normals (derived from points)
    "pose":           Tensor[F, 4, 4],        # Camera-to-world (c2w) poses
    "frame_features": Tensor[F, D],           # Per-frame summary features for Geometric3DEncoder
}
```

Where:
- `F` = number of video frames (typically 25 for OmniTransfer clips)
- `H, W` = spatial resolution of AMB3R output (default 392 x 518)
- `D` = feature dimension (default 7: mean_depth + std_depth + mean_conf + mean_normal_xyz + ...)

### 2.3 Confidence Format

AMB3R uses `conf = 1 + exp(log_conf)` where:
- `conf > 1.0` always (minimum value)
- Normalized as `w = (conf - 1) / conf` which maps to `[0, 1)`
- High confidence → `w ≈ 1.0` (strong 3D signal)
- Low confidence → `w ≈ 0.0` (uncertain, down-weighted)

### 2.4 Frame Features Format

The `frame_features` tensor provides a compact per-frame summary for the `Geometric3DEncoder`:

```python
# Per-frame feature vector (D=7 by default):
frame_features[f] = [
    mean_depth,       # Mean valid depth for frame f
    std_depth,        # Std dev of valid depth
    mean_confidence,  # Mean normalized confidence (conf-1)/conf
    mean_normal_x,    # Mean surface normal x-component
    mean_normal_y,    # Mean surface normal y-component
    mean_normal_z,    # Mean surface normal z-component
    depth_coverage,   # Fraction of pixels with valid depth
]
```

---

## 3. How to Generate Pseudo-GT

### 3.1 AMB3R Direct Inference (Recommended)

```python
import torch
from amb3r.model_zoo import load_model

# Load frozen AMB3R
model = load_model('amb3r', ckpt_path='./checkpoints/amb3r.pt')
model.prepare(data_type='bf16')
model = model.cuda().eval()

# Process video frames: (1, T, 3, H, W) in [-1, 1]
frames = {'images': video_tensor}

with torch.inference_mode():
    results = model.forward(frames, iters=1)
    res = results[-1]  # Use backend-refined output

# Extract outputs
depth       = res['depth_metric'][:, :, :, :, 0]  # (1, T, H, W)
points      = res['world_points'][:, :, :, :, :]   # (1, T, H, W, 3)
confidence  = res['world_points_conf'][:, :, :, :, 0]  # (1, T, H, W)
poses       = res['pose']                           # (1, T, 4, 4)
```

### 3.2 DepthAnything3 Alternative

```python
model = load_model('da3', ckpt_path='depth-anything/DA3NESTED-GIANT-LARGE')

# Same interface
with torch.inference_mode():
    res = model.forward(frames)
    # Same output keys: world_points, world_points_conf, depth, pose
```

### 3.3 Surface Normals from Points

```python
def compute_normals_from_points(points):
    """Derive surface normals from 3D point grid via cross-products."""
    # points: (F, H, W, 3)
    dx = points[:, 1:, :, :] - points[:, :-1, :, :]  # (F, H-1, W, 3)
    dy = points[:, :, 1:, :] - points[:, :, :-1, :]  # (F, H, W-1, 3)
    # Pad to original size
    dx = F.pad(dx, (0,0, 0,0, 0,1), mode='replicate')
    dy = F.pad(dy, (0,0, 0,1, 0,0), mode='replicate')
    normals = torch.cross(dx, dy, dim=-1)
    normals = F.normalize(normals, dim=-1)
    return normals
```

### 3.4 Frame Feature Extraction

```python
def compute_frame_features(depth, confidence, normals):
    """Compute per-frame summary features for Geometric3DEncoder."""
    F, H, W = depth.shape
    features = []
    for f in range(F):
        d = depth[f]
        c = confidence[f]
        n = normals[f]

        valid = d > 0
        conf_norm = (c - 1) / c.clamp(min=1.0)

        features.append(torch.stack([
            d[valid].mean() if valid.any() else torch.tensor(0.0),
            d[valid].std() if valid.sum() > 1 else torch.tensor(0.0),
            conf_norm[valid].mean() if valid.any() else torch.tensor(0.0),
            n[valid, 0].mean() if valid.any() else torch.tensor(0.0),
            n[valid, 1].mean() if valid.any() else torch.tensor(0.0),
            n[valid, 2].mean() if valid.any() else torch.tensor(0.0),
            valid.float().mean(),
        ]))
    return torch.stack(features)  # (F, 7)
```

### 3.5 AMB3R Dependencies for Preprocessing

```
conda create -n amb3r python=3.9 cmake=3.14.0 -y
pip install torch==2.5.0 torchvision==0.20.0 --index-url https://download.pytorch.org/whl/cu118
pip install torch-scatter==2.1.2 -f https://data.pyg.org/whl/torch-2.5.0+cu118.html
pip install "git+https://github.com/facebookresearch/pytorch3d.git@V0.7.8" --no-build-isolation
pip install flash-attn==2.7.3 --no-build-isolation
pip install -r requirements.txt   # amb3r/requirements.txt
```

**Checkpoint**: https://drive.google.com/file/d/14x0WW2rUE_he2hUEouP6ywSRnlJDeLel
**Location**: `./checkpoints/amb3r.pt`

---

## 4. Dataset Recommendations

### 4.1 Priority Tiers

#### Tier 0: Your Existing Training Videos (Highest Priority)

| Source | Action | Why |
|--------|--------|-----|
| Existing OmniTransfer training set | Run frozen AMB3R on the same videos | Zero domain gap. 3D losses regularize the exact distribution you train on. |

**This is the single most impactful step.** The 3D losses anneal to zero over 15K steps — they're a training signal, not a permanent constraint. Running AMB3R on the training distribution ensures the geometric guidance aligns with the visual content the model already learns.

#### Tier 1: High 3D Signal, Directly Useful for Training

| Dataset | Videos/Scenes | Resolution | 3D Signal Quality | Best For |
|---------|---------------|------------|-------------------|----------|
| **RealEstate10K** (AMB3R split) | 1,721 sequences × 10 frames | ~518×294 | Excellent — real camera motion through 3D spaces | Motion transfer, camera motion |
| **Tanks & Temples** | ~20 scenes (training split) | Variable (high) | Excellent — architectural outdoor with strong parallax | Scene composition, camera motion |
| **DTU** | 80 scenes | 1152×864 | Cleanest pseudo-GT (AMB3R Rel=0.81, Acc=0.22cm) | Identity preservation, product shots |

#### Tier 2: Good Supplementary Data

| Dataset | Videos/Scenes | Resolution | 3D Signal Quality | Best For |
|---------|---------------|------------|-------------------|----------|
| **ETH3D** | Multiple scenes | Up to 6048×4032 | Strong — high-res indoor/outdoor | Validation benchmark |
| **7-Scenes** | 7 indoor scenes × ~500 frames | 640×480 | Good — indoor SLAM sequences | Indoor walkthrough generation |
| **KITTI** | 13 driving sequences | 1242×375 | Good — strong forward-facing depth | Driving/outdoor motion transfer |
| **ScanNet** | Indoor scenes | Variable | Good (AMB3R Rel=2.7) | Indoor scene generation |

#### Tier 3: Validation Only (Not for Training)

| Dataset | Purpose |
|---------|---------|
| **Sintel** | Synthetic — validate temporal depth consistency |
| **Bonn** | Indoor video — validate SLAM consistency |
| **IMC Phototourism** | Internet photos — validate SfM quality |
| **TUM RGB-D** | Indoor SLAM — validate VO metrics |

### 4.2 Dataset Download Commands

```bash
cd /path/to/amb3r

# Tier 1: RealEstate10K (AMB3R split) — 1,721 sequences
bash scripts/download_pose.sh
# Downloads: data/pose/re10k_amb3r_split/
# Source: HuggingFace HengyiWang/re10k_amb3r_split

# Tier 1: Tanks & Temples
bash scripts/download_sfm.sh
# Downloads: data/sfm/tnt/ (IBR3D subset)

# Tier 1: DTU
bash scripts/download_rmvd.sh
# Downloads: data/rmvd/dtu/ (requires conversion)

# Tier 2: 7-Scenes
bash scripts/download_3d.sh
# Downloads: data/rmvd/7scenes/ (7 indoor scenes)

# Tier 2: KITTI + Video Depth datasets
bash scripts/download_videodepth.sh
# Downloads: data/dynamic/kitti/, sintel/, bonn/

# Tier 2: ETH3D (included in rmvd download)
# Already in data/rmvd/eth3d/

# Tier 2: SLAM evaluation
bash scripts/download_slam.sh
# Downloads: data/slam/eth_slam/ (8 monocular scenes)
```

---

## 5. AMB3R Benchmark Performance

The following tables justify dataset selection based on where AMB3R produces the most reliable pseudo-GT.

### 5.1 Monocular Depth (Rel ↓ is better)

| Dataset | VGGT | DA3 | **AMB3R** |
|---------|------|-----|-----------|
| NYUv2 | 3.6 | 4.4 | **3.0** |
| KITTI | 8.8 | 7.9 | **7.3** |
| ETH3D | 3.8 | 4.6 | **3.2** |
| ScanNet | 2.7 | 4.2 | **2.7** |
| DIODE | 26.9 | 30.3 | **24.7** |

### 5.2 Multi-View Depth (Rel ↓ / δ₁.₀₃ ↑)

| Dataset | VGGT Rel | DA3 Rel | **AMB3R Rel** | **AMB3R δ₁.₀₃** |
|---------|----------|---------|---------------|-----------------|
| KITTI | 4.5 | 3.9 | **2.8** | **74.4** |
| ScanNet | 2.3 | 2.7 | **1.9** | **85.8** |
| ETH3D | 1.8 | 2.2 | **1.4** | **90.9** |
| DTU | 0.9 | 1.5 | **0.9** | 95.1 |
| T&T | 2.4 | 2.5 | **1.7** | **90.2** |
| **Average** | 2.4 | 2.6 | **1.7** | **87.3** |

### 5.3 3D Reconstruction (Rel ↓ / Acc ↓ / Completeness ↓)

| Dataset | VGGT Rel | DA3 Rel | **AMB3R Rel** | **AMB3R Acc (cm)** |
|---------|----------|---------|---------------|--------------------|
| ETH3D | 6.02 | 9.24 | **4.64** | **9.98** |
| DTU | 0.83 | 2.06 | **0.81** | **0.22** |
| 7-Scenes | 5.51 | 5.26 | **4.70** | **1.75** |

### 5.4 Visual Odometry / SLAM (ATE in cm ↓)

| Method | Calib | Optim | Avg ATE |
|--------|-------|-------|---------|
| MASt3R-SLAM | Yes | Yes | 3.0 |
| MASt3R-SLAM | No | Yes | 6.0 |
| VGGT-SLAM | No | Yes | 5.3 |
| **AMB3R-VO** | **No** | **No** | **2.7** |
| **AMB3R-VO (DA3)** | **No** | **No** | **2.6** |

**Key insight**: AMB3R achieves best SLAM accuracy without calibration or optimization — meaning pseudo-GT poses from AMB3R are reliable enough for training supervision.

### 5.5 Pose Estimation (RealEstate10K, mAA@30° ↑)

| Method | mAA@30° |
|--------|---------|
| VGGT | 81.8 |
| **AMB3R** | **86.3** |
| DA3 | **87.5** |

---

## 6. Recommended Pipeline Strategy

### Phase 1: Baseline (Immediate)

```
Existing OmniTransfer training videos
    → Frozen AMB3R inference
    → depth_3d/*.pt pseudo-GT
    → Enable depth_3d_loss only (no encoder yet)
    → Train with: enable_depth_3d_loss: true
```

**Estimated compute**: ~2 seconds per 25-frame clip on A100
**Config**:
```yaml
training_strategy:
  enable_depth_3d_loss: true
  depth_3d_loss_weight: 0.05
  normal_3d_loss_weight: 0.03
  edge_3d_loss_weight: 0.02
  depth_3d_loss_anneal_steps: 15000
  depth_3d_pseudo_gt_dir: "depth_3d"
```

### Phase 2: Geometric Encoder (After Phase 1 Validates)

```
Same depth_3d/*.pt + frame_features
    → Enable Geometric3DEncoder (8 tokens → cross-attention)
    → Enable Geometric3DDecoder (auxiliary depth/normal loss)
    → Train with: enable_geometric_3d_encoder: true
```

**Config addition**:
```yaml
training_strategy:
  enable_geometric_3d_encoder: true
  geometric_3d_num_tokens: 8
  geometric_3d_hidden_dim: 512
  geometric_3d_num_layers: 4
  geometric_3d_feature_dim: 7
  geometric_3d_decoder_loss_weight: 0.05
  geometric_3d_decoder_anneal_steps: 15000
```

### Phase 3: Expand Data (After Architecture Validates)

```
Add RealEstate10K (1,721 sequences) for camera motion tasks
Add DTU (80 scenes) for object-centric tasks
Add T&T for outdoor scenes
→ Larger, more diverse 3D supervision
```

### Phase 4: Confidence-Weighted MSE (Optional)

```
Enable confidence_weighted_mse to modulate the main flow-matching loss
→ Focuses training on high-confidence 3D regions
→ De-emphasizes sky, reflective surfaces, thin structures
```

```yaml
training_strategy:
  use_confidence_weighted_mse: true
```

---

## 7. Dataset Characteristics vs. OmniTransfer Tasks

| OmniTransfer Task | Best Training Datasets | Why |
|-------------------|----------------------|-----|
| **Motion Transfer** | RE10K, your videos | Camera motion through 3D spaces → depth consistency crucial |
| **Camera Motion** | RE10K, T&T, KITTI | Explicit camera trajectories with parallax |
| **Style Transfer** | DTU, your videos | Object-centric with clean geometry → normals matter |
| **Identity Preservation** | DTU, 7-Scenes | Consistent 3D structure across viewpoints |
| **Scene Composition** | T&T, ETH3D | Large-scale 3D structure enforcement |

---

## 8. Estimated Compute Requirements

| Dataset | Samples | AMB3R Time (A100) | Storage (depth_3d/) |
|---------|---------|-------------------|---------------------|
| Your training set | ~10K videos | ~6 hours | ~50 GB |
| RealEstate10K | 1,721 × 10 frames | ~1 hour | ~8 GB |
| DTU | 80 scenes × ~49 frames | ~10 min | ~2 GB |
| Tanks & Temples | ~20 scenes × ~300 frames | ~20 min | ~4 GB |
| 7-Scenes | 7 × ~500 frames | ~15 min | ~3 GB |
| **Total (all)** | | **~8 hours** | **~67 GB** |

Estimates assume:
- A100 80GB GPU
- 25 frames per forward pass
- bf16 precision
- ~2s per 25-frame batch (AMB3R with backend)

---

## 9. Data Quality Considerations

### What Makes Good 3D Pseudo-GT

| Property | Good Signal | Poor Signal |
|----------|-------------|-------------|
| Camera motion | Translational (parallax) | Pure rotation (no depth cue) |
| Scene content | Rigid, textured surfaces | Sky, water, reflections, smoke |
| Depth range | 0.5m – 50m | Very close (<0.1m) or very far (>100m) |
| Confidence | conf_norm > 0.5 for >70% pixels | Low confidence everywhere |
| Frame rate | Moderate (5-15 fps effective) | Too fast (blur) or too slow (no overlap) |

### Filtering Heuristics

```python
# Skip samples with poor 3D signal
conf_norm = (confidence - 1) / confidence.clamp(min=1.0)
valid_ratio = (conf_norm > 0.3).float().mean()

if valid_ratio < 0.3:
    # Less than 30% of pixels have decent confidence
    # Skip this sample — 3D pseudo-GT is unreliable
    continue
```

### Known AMB3R Failure Modes

1. **Sky regions** — infinite depth, zero confidence (handled by confidence weighting)
2. **Reflective surfaces** — inconsistent depth (mirrors, glass, water)
3. **Thin structures** — fence wires, tree branches (edge loss helps here)
4. **Textureless surfaces** — large white walls (low confidence, auto-downweighted)
5. **Dynamic objects** — moving people/cars (temporal inconsistency)

---

## 10. Preprocessing Script Outline

```python
#!/usr/bin/env python3
"""Generate AMB3R depth_3d pseudo-GT for OmniTransfer training."""

import torch
from pathlib import Path
from amb3r.model_zoo import load_model

def generate_pseudo_gt(
    video_dir: Path,
    output_dir: Path,
    model_name: str = "amb3r",
    ckpt_path: str = "./checkpoints/amb3r.pt",
    resolution: tuple = (518, 392),
    batch_frames: int = 25,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model once
    model = load_model(model_name, ckpt_path=ckpt_path)
    model.prepare(data_type='bf16')
    model = model.cuda().eval()

    for video_path in sorted(video_dir.glob("*.pt")):
        stem = video_path.stem
        out_path = output_dir / f"{stem}.pt"

        if out_path.exists():
            continue  # Skip already processed

        # Load video latents → decode to pixels → run AMB3R
        # (Or load raw video frames directly)
        video_data = torch.load(video_path)
        images = load_and_preprocess(video_data, resolution)
        # images: (1, F, 3, H, W) in [-1, 1]

        with torch.inference_mode():
            results = model.forward({'images': images}, iters=1)
            res = results[-1]

        depth = res['depth_metric'][0, :, :, :, 0].cpu()     # (F, H, W)
        points = res['world_points'][0].cpu()                  # (F, H, W, 3)
        confidence = res['world_points_conf'][0, :, :, :, 0].cpu()  # (F, H, W)
        poses = res['pose'][0].cpu()                           # (F, 4, 4)

        normals = compute_normals_from_points(points)          # (F, H, W, 3)
        features = compute_frame_features(depth, confidence, normals)  # (F, 7)

        torch.save({
            "depth": depth,
            "points": points,
            "confidence": confidence,
            "normals": normals,
            "pose": poses,
            "frame_features": features,
        }, out_path)

        print(f"Saved {out_path} | conf_coverage={...:.1%}")
```

---

## 11. Validation Strategy

### A. Quantitative — 3D Metrics on Held-Out Data

Generate videos with OmniTransfer (with and without 3D losses), then run AMB3R on the generated output to measure:

| Metric | What It Measures |
|--------|-----------------|
| Depth consistency (σ) | Std of depth at corresponding pixels across frames |
| Normal consistency (°) | Mean angular error of normals between adjacent frames |
| Pose error (ATE cm) | Trajectory accuracy of estimated poses |
| Confidence coverage (%) | Fraction of high-confidence pixels |

### B. Qualitative — Visual Inspection

- Side-by-side point cloud renders (with vs. without 3D losses)
- Temporal depth map videos (should be smoother with 3D losses)
- Surface normal consistency overlays

### C. FID/FVD — Generation Quality

- Verify that 3D losses don't degrade visual quality (they anneal to zero, so shouldn't)
- Compare FID/FVD with and without 3D losses enabled

---

## Appendix A: AMB3R Model Variants

| Model | Checkpoint | Front-End | Back-End | Best For |
|-------|-----------|-----------|----------|----------|
| `amb3r` | `checkpoints/amb3r.pt` | VGGT | PTV3 | Highest accuracy (Rel=1.7 avg) |
| `da3` | `depth-anything/DA3NESTED-GIANT-LARGE` | DepthAnything3 | None | Faster, slightly better poses (mAA=87.5) |

**Recommendation**: Use `amb3r` for pseudo-GT generation (backend refinement improves quality significantly). Use `da3` only if compute budget is limited.

## Appendix B: Configuration Reference

```yaml
# Full AMB3R integration config for OmniTransfer
training_strategy:
  # 3D Reconstruction Losses
  enable_depth_3d_loss: true
  depth_3d_loss_weight: 0.05          # Scale-invariant depth
  normal_3d_loss_weight: 0.03         # Surface normal consistency
  edge_3d_loss_weight: 0.02           # Edge direction consistency
  depth_3d_loss_anneal_steps: 15000   # Linear anneal to zero
  depth_3d_pseudo_gt_dir: "depth_3d"  # Relative to preprocessed_data_root
  use_confidence_weighted_mse: false  # Optional: weight main MSE by confidence

  # 3D Geometric Token Encoder
  enable_geometric_3d_encoder: true
  geometric_3d_num_tokens: 8          # Cross-attention tokens
  geometric_3d_hidden_dim: 512        # Internal dimension
  geometric_3d_num_layers: 4          # Transformer layers
  geometric_3d_feature_dim: 7         # Per-frame features
  geometric_3d_decoder_loss_weight: 0.05  # Auxiliary decoder loss
  geometric_3d_decoder_anneal_steps: 15000
```

## Appendix C: Download URLs

| Dataset | URL | Size |
|---------|-----|------|
| AMB3R checkpoint | https://drive.google.com/file/d/14x0WW2rUE_he2hUEouP6ywSRnlJDeLel | ~2 GB |
| RE10K (AMB3R split) | https://huggingface.co/datasets/HengyiWang/re10k_amb3r_split | ~5 GB |
| DTU (raw) | Via `scripts/download_rmvd.sh` | ~10 GB |
| Tanks & Temples (IBR3D) | https://storage.googleapis.com/isl-datasets/FreeViewSynthesis/ibr3d_tat.tar.gz | ~8 GB |
| 7-Scenes | http://download.microsoft.com/download/2/8/5/28564B23-... | ~15 GB |
| KITTI (depth + raw) | https://s3.eu-central-1.amazonaws.com/avg-kitti/ | ~25 GB |
| ETH3D | Via `scripts/download_rmvd.sh` | ~5 GB |
| Sintel (depth) | http://files.is.tue.mpg.de/sintel/ | ~6 GB |
| Bonn RGBD | https://www.ipb.uni-bonn.de/html/projects/rgbd_dynamic2019/ | ~3 GB |
| ETH3D SLAM | https://www.eth3d.net/data/slam/datasets/ | ~4 GB |
| IMC Phototourism | https://www.cs.ubc.ca/research/kmyi_data/imw2020/TestData/ | ~10 GB |
