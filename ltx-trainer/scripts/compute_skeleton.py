#!/usr/bin/env python3
"""Compute SMPL/MANO skeleton pseudo-GT from videos for 3DiMo geometric supervision.

Extracts body pose (SMPL) and hand joints (MANO) from driving videos using
4DHumans (HMR2.0) + HaMeR. Falls back to MediaPipe if those aren't installed.
Output matches the format expected by geometric_decoder.py for auxiliary
supervision during 3DiMo training.

Output format per sample (.pt file):
    body_pose:       [T, 23, 3, 3]   SMPL rotation matrices (no global orient)
    hand_joints_3d:  [T, 21, 3]      MANO 3D hand joints
    body_joints_3d:  [T, 44, 3]      Full SMPL 3D body joints
    body_confidence: [T, 44]          Per-joint confidence scores
    fps:             float            Video frame rate
    num_frames:      int              Number of frames processed

Usage:
    # Using 4DHumans + HaMeR (best quality)
    python scripts/compute_skeleton.py \\
        --video-dir /path/to/raw_videos \\
        --output-dir /path/to/dataset/skeleton

    # Using MediaPipe fallback (no special install needed)
    python scripts/compute_skeleton.py \\
        --video-dir /path/to/raw_videos \\
        --output-dir /path/to/dataset/skeleton \\
        --backend mediapipe

    # Match output filenames to dataset latent names
    python scripts/compute_skeleton.py \\
        --video-dir /path/to/raw_videos \\
        --output-dir /media/2TB/omnitransfer_effect_motion/skeleton \\
        --match-latents /media/2TB/omnitransfer_effect_motion/latents

Installation (for 4DHumans + HaMeR backend):
    git clone https://github.com/shubham-goel/4D-Humans.git
    cd 4D-Humans && pip install -e .
    cd ..
    git clone https://github.com/geopavlakos/hamer.git
    cd hamer && pip install -e . && bash fetch_demo_data.sh

References:
    - 3DiMo (arXiv:2602.03796v2) Section 3.3
    - 4DHumans: https://github.com/shubham-goel/4D-Humans
    - HaMeR: https://github.com/geopavlakos/hamer
    - DreamActorExtractor pattern: /media/2TB/IMTalker/dreamactor_extractor.py
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm


# ─── Rotation utilities ─────────────────────────────────────────────────────

def axis_angle_to_rotation_matrix(axis_angle: np.ndarray) -> np.ndarray:
    """Convert axis-angle [3] to rotation matrix [3, 3] via Rodrigues formula."""
    angle = np.linalg.norm(axis_angle)
    if angle < 1e-6:
        return np.eye(3, dtype=np.float32)
    axis = axis_angle / angle
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0],
    ], dtype=np.float32)
    return np.eye(3, dtype=np.float32) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def batch_axis_angle_to_rotmat(body_pose: np.ndarray) -> np.ndarray:
    """Convert body_pose [T, 23, 3] axis-angle to [T, 23, 3, 3] rotation matrices."""
    T, J, _ = body_pose.shape
    rotmats = np.zeros((T, J, 3, 3), dtype=np.float32)
    for t in range(T):
        for j in range(J):
            rotmats[t, j] = axis_angle_to_rotation_matrix(body_pose[t, j])
    return rotmats


# ─── 4DHumans + HaMeR Backend ───────────────────────────────────────────────

class HMR2Backend:
    """Skeleton extraction using 4DHumans (HMR2.0) + HaMeR.

    Based on the DreamActorExtractor pattern from IMTalker:
    - 4DHumans extracts SMPL body pose (23 joints) + 3D body joints (44 joints)
    - HaMeR extracts MANO hand joints (21 joints per hand)
    - Uses weak-perspective camera for 2D projection & confidence estimation
    """

    def __init__(self, device: str = "cuda", hmr2_path: str | None = None, hamer_path: str | None = None):
        self.device = device
        self.hmr2_model = None
        self.hmr2_cfg = None
        self.hamer_model = None
        self.hamer_cfg = None

        # Add repos to path if provided
        if hmr2_path:
            sys.path.insert(0, str(hmr2_path))
        else:
            # Search common locations
            for p in [Path.home() / "4D-Humans", Path("/media/12TB/4D-Humans")]:
                if p.exists():
                    sys.path.insert(0, str(p))
                    break

        if hamer_path:
            sys.path.insert(0, str(hamer_path))
        else:
            for p in [Path.home() / "hamer", Path("/media/12TB/hamer")]:
                if p.exists():
                    sys.path.insert(0, str(p))
                    break

        self._load_hmr2()
        self._load_hamer()

    def _load_hmr2(self) -> None:
        """Load 4DHumans (HMR2.0) model for body pose estimation."""
        try:
            from hmr2.models import load_hmr2, DEFAULT_CHECKPOINT
            print(f"Loading 4DHumans from {DEFAULT_CHECKPOINT}")
            self.hmr2_model, self.hmr2_cfg = load_hmr2(DEFAULT_CHECKPOINT)
            self.hmr2_model = self.hmr2_model.to(self.device)
            self.hmr2_model.eval()
            print("4DHumans loaded successfully")
        except Exception as e:
            print(f"Warning: Could not load 4DHumans: {e}")
            print("Install: git clone https://github.com/shubham-goel/4D-Humans.git && cd 4D-Humans && pip install -e .")

    def _load_hamer(self) -> None:
        """Load HaMeR model for hand mesh recovery."""
        try:
            from hamer.models import load_hamer, DEFAULT_CHECKPOINT as HAMER_CKPT
            ckpt = Path(HAMER_CKPT)
            # Search common checkpoint locations
            if not ckpt.exists():
                for candidate in [
                    Path.home() / "hamer" / "_DATA" / "hamer_ckpts" / "checkpoints" / "hamer.ckpt",
                    Path("/media/12TB/hamer/_DATA/hamer_ckpts/checkpoints/hamer.ckpt"),
                ]:
                    if candidate.exists():
                        ckpt = candidate
                        break
            if ckpt.exists():
                print(f"Loading HaMeR from {ckpt}")
                self.hamer_model, self.hamer_cfg = load_hamer(str(ckpt))
                self.hamer_model = self.hamer_model.to(self.device)
                self.hamer_model.eval()
                print("HaMeR loaded successfully")
            else:
                print(f"HaMeR checkpoint not found. Run: cd hamer && bash fetch_demo_data.sh")
        except Exception as e:
            print(f"Warning: Could not load HaMeR: {e}")

    @torch.inference_mode()
    def _extract_body_frame(self, frame: np.ndarray) -> dict | None:
        """Extract SMPL body from a single BGR frame (DreamActorExtractor pattern)."""
        if self.hmr2_model is None:
            return None

        from hmr2.utils import recursive_to
        from hmr2.datasets.vitdet_dataset import ViTDetDataset

        img_h, img_w = frame.shape[:2]
        bbox = np.array([[0, 0, img_w, img_h]])

        dataset = ViTDetDataset(self.hmr2_cfg, frame, bbox)
        loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

        for batch in loader:
            batch = recursive_to(batch, self.device)
            out = self.hmr2_model(batch)

            joints_3d = out["pred_keypoints_3d"][0].cpu().numpy()  # [44, 3]
            body_pose = out["pred_smpl_params"]["body_pose"][0].cpu().numpy()  # [23, 3]
            global_orient = out["pred_smpl_params"]["global_orient"][0].cpu().numpy()  # [1, 3]
            pred_cam = out["pred_cam"][0].cpu().numpy()

            # Weak-perspective 2D projection for confidence estimation
            kp_2d = joints_3d[:, :2].copy()
            kp_2d = kp_2d * pred_cam[0]
            kp_2d[:, 0] += pred_cam[1]
            kp_2d[:, 1] += pred_cam[2]
            box_center = batch["box_center"][0].cpu().numpy()
            box_size = batch["box_size"][0].cpu().numpy()
            kp_2d = kp_2d * box_size / 2 + box_center

            # Confidence: in-frame joints get 0.8, out-of-frame get 0
            in_frame = (kp_2d[:, 0] >= 0) & (kp_2d[:, 0] < img_w) & \
                       (kp_2d[:, 1] >= 0) & (kp_2d[:, 1] < img_h)
            confidence = np.where(in_frame, 0.8, 0.0).astype(np.float32)

            return {
                "joints_3d": joints_3d,
                "body_pose": body_pose,       # [23, 3] axis-angle
                "global_orient": global_orient,
                "confidence": confidence,     # [44]
            }
        return None

    @torch.inference_mode()
    def _extract_hands_frame(self, frame: np.ndarray) -> dict | None:
        """Extract MANO hand joints from a single BGR frame."""
        if self.hamer_model is None:
            return None

        from hamer.datasets.vitdet_dataset import DEFAULT_MEAN, DEFAULT_STD

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_h, img_w = img_rgb.shape[:2]

        img_tensor = torch.from_numpy(img_rgb.astype(np.float32)).permute(2, 0, 1) / 255.0
        img_tensor = torch.nn.functional.interpolate(
            img_tensor.unsqueeze(0), size=(256, 256), mode="bilinear", align_corners=False,
        )
        mean = torch.tensor(DEFAULT_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor(DEFAULT_STD, dtype=torch.float32).view(1, 3, 1, 1)
        img_tensor = (img_tensor - mean) / std

        batch = {
            "img": img_tensor.to(self.device),
            "img_size": torch.tensor([[img_w, img_h]], dtype=torch.float32).to(self.device),
            "center": torch.tensor([[img_w / 2, img_h / 2]], dtype=torch.float32).to(self.device),
            "scale": torch.tensor([[max(img_h, img_w) / 200.0]], dtype=torch.float32).to(self.device),
        }

        out = self.hamer_model(batch)
        hand_joints_3d = out["pred_keypoints_3d"][0].cpu().numpy()  # [21, 3]

        return {"joints_3d": hand_joints_3d}

    def process_video(self, video_path: str, max_frames: int | None = None) -> dict:
        """Process a video and return skeleton data tensors."""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if max_frames:
            total = min(total, max_frames)

        body_joints_3d_list = []
        body_pose_list = []
        body_confidence_list = []
        hand_joints_3d_list = []

        video_name = Path(video_path).stem

        for i in tqdm(range(total), desc=f"  {video_name}", leave=False):
            ret, frame = cap.read()
            if not ret:
                break

            body = self._extract_body_frame(frame)
            hands = self._extract_hands_frame(frame)

            if body is not None:
                body_joints_3d_list.append(body["joints_3d"])
                body_pose_list.append(body["body_pose"])
                body_confidence_list.append(body["confidence"])
            else:
                body_joints_3d_list.append(np.zeros((44, 3), dtype=np.float32))
                body_pose_list.append(np.zeros((23, 3), dtype=np.float32))
                body_confidence_list.append(np.zeros(44, dtype=np.float32))

            if hands is not None:
                hand_joints_3d_list.append(hands["joints_3d"])
            else:
                hand_joints_3d_list.append(np.zeros((21, 3), dtype=np.float32))

        cap.release()

        # Convert body_pose from axis-angle [T, 23, 3] to rotation matrices [T, 23, 3, 3]
        body_pose_aa = np.stack(body_pose_list)
        body_pose_rm = batch_axis_angle_to_rotmat(body_pose_aa)

        return {
            "body_pose": torch.from_numpy(body_pose_rm).float(),                    # [T, 23, 3, 3]
            "hand_joints_3d": torch.from_numpy(np.stack(hand_joints_3d_list)).float(),  # [T, 21, 3]
            "body_joints_3d": torch.from_numpy(np.stack(body_joints_3d_list)).float(),  # [T, 44, 3]
            "body_confidence": torch.from_numpy(np.stack(body_confidence_list)).float(),  # [T, 44]
            "fps": fps,
            "num_frames": len(body_pose_list),
            "video_path": str(video_path),
        }


# ─── MediaPipe Fallback Backend ─────────────────────────────────────────────

class MediaPipeBackend:
    """Lightweight skeleton extraction using MediaPipe Pose + Hands.

    Produces approximate SMPL-format output. Less accurate than 4DHumans+HaMeR
    but requires only: pip install mediapipe opencv-python
    """

    def __init__(self, device: str = "cpu"):
        import mediapipe as mp
        self.mp_pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.mp_hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def process_video(self, video_path: str, max_frames: int | None = None) -> dict:
        """Process video with MediaPipe and convert to SMPL-compatible format."""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if max_frames:
            total = min(total, max_frames)

        body_joints_3d_list = []
        body_confidence_list = []
        hand_joints_3d_list = []

        video_name = Path(video_path).stem

        for i in tqdm(range(total), desc=f"  {video_name}", leave=False):
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Body pose (33 MediaPipe landmarks -> pad to 44 SMPL joints)
            pose_result = self.mp_pose.process(rgb)
            if pose_result.pose_world_landmarks:
                landmarks = pose_result.pose_world_landmarks.landmark
                joints = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
                conf = np.array([lm.visibility for lm in landmarks], dtype=np.float32)
                joints_44 = np.zeros((44, 3), dtype=np.float32)
                conf_44 = np.zeros(44, dtype=np.float32)
                n = min(33, 44)
                joints_44[:n] = joints[:n]
                conf_44[:n] = conf[:n]
                body_joints_3d_list.append(joints_44)
                body_confidence_list.append(conf_44)
            else:
                body_joints_3d_list.append(np.zeros((44, 3), dtype=np.float32))
                body_confidence_list.append(np.zeros(44, dtype=np.float32))

            # Hand pose (21 landmarks per hand)
            hand_result = self.mp_hands.process(rgb)
            if hand_result.multi_hand_world_landmarks:
                hand_lm = hand_result.multi_hand_world_landmarks[0].landmark
                joints = np.array([[lm.x, lm.y, lm.z] for lm in hand_lm], dtype=np.float32)
                hand_joints_3d_list.append(joints)
            else:
                hand_joints_3d_list.append(np.zeros((21, 3), dtype=np.float32))

        cap.release()

        # MediaPipe doesn't provide SMPL rotation matrices - use identity
        T = len(body_joints_3d_list)
        body_pose_rm = np.tile(np.eye(3, dtype=np.float32), (T, 23, 1, 1))

        return {
            "body_pose": torch.from_numpy(body_pose_rm).float(),
            "hand_joints_3d": torch.from_numpy(np.stack(hand_joints_3d_list)).float(),
            "body_joints_3d": torch.from_numpy(np.stack(body_joints_3d_list)).float(),
            "body_confidence": torch.from_numpy(np.stack(body_confidence_list)).float(),
            "fps": fps,
            "num_frames": T,
            "video_path": str(video_path),
        }


# ─── Processing logic ───────────────────────────────────────────────────────

def process_directory(
    video_dir: Path,
    output_dir: Path,
    backend,
    max_frames: int | None = None,
    match_latents: Path | None = None,
) -> None:
    """Process all videos in a directory."""
    video_exts = {".mp4", ".avi", ".mov", ".webm", ".mkv"}
    output_dir.mkdir(parents=True, exist_ok=True)

    video_files = sorted([p for p in video_dir.iterdir() if p.suffix.lower() in video_exts])

    if not video_files:
        print(f"No videos found in {video_dir}")
        return

    # If matching to latents, use latent filenames for output
    if match_latents:
        latent_files = sorted(match_latents.glob("*.pt"))
        print(f"Matching {len(video_files)} videos to {len(latent_files)} latent samples")
        n = min(len(video_files), len(latent_files))
        video_files = video_files[:n]
        output_names = [lf.stem for lf in latent_files[:n]]
    else:
        output_names = [vf.stem for vf in video_files]

    print(f"Processing {len(video_files)} videos -> {output_dir}")

    success = 0
    for video_path, out_name in tqdm(list(zip(video_files, output_names)), desc="Videos"):
        output_path = output_dir / f"{out_name}.pt"
        if output_path.exists():
            print(f"  Skipping {video_path.name}: already processed")
            success += 1
            continue

        try:
            result = backend.process_video(str(video_path), max_frames=max_frames)
            torch.save(result, output_path)

            T = result["num_frames"]
            conf = result["body_confidence"]
            avg_valid = (conf > 0.5).sum(dim=1).float().mean().item()
            print(f"  {video_path.name} -> {out_name}.pt: {T} frames, avg valid: {avg_valid:.1f}/44")
            success += 1
        except Exception as e:
            print(f"  Error: {video_path.name}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\nDone: {success}/{len(video_files)} processed. Output: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Compute SMPL/MANO skeleton pseudo-GT for 3DiMo geometric supervision"
    )
    parser.add_argument("--video-dir", type=str, required=True, help="Directory with source videos")
    parser.add_argument("--output-dir", type=str, required=True, help="Output dir for skeleton .pt files")
    parser.add_argument("--backend", type=str, default="hmr2", choices=["hmr2", "mediapipe"],
                        help="Extraction backend: hmr2 (4DHumans+HaMeR) or mediapipe (lightweight)")
    parser.add_argument("--device", type=str, default="cuda", help="Device for inference")
    parser.add_argument("--max-frames", type=int, default=None, help="Max frames per video")
    parser.add_argument("--hmr2-path", type=str, default=None, help="Path to 4D-Humans repo")
    parser.add_argument("--hamer-path", type=str, default=None, help="Path to hamer repo")
    parser.add_argument("--match-latents", type=str, default=None,
                        help="Match output filenames to latent files in this directory")

    args = parser.parse_args()
    video_dir = Path(args.video_dir)

    if not video_dir.exists():
        print(f"Error: Video directory not found: {video_dir}")
        sys.exit(1)

    # Create backend
    if args.backend == "hmr2":
        print("Using 4DHumans (HMR2.0) + HaMeR backend")
        backend = HMR2Backend(
            device=args.device, hmr2_path=args.hmr2_path, hamer_path=args.hamer_path,
        )
    else:
        print("Using MediaPipe fallback backend (pip install mediapipe)")
        backend = MediaPipeBackend(device=args.device)

    process_directory(
        video_dir=video_dir,
        output_dir=Path(args.output_dir),
        backend=backend,
        max_frames=args.max_frames,
        match_latents=Path(args.match_latents) if args.match_latents else None,
    )


if __name__ == "__main__":
    main()
