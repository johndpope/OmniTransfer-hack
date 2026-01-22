#!/usr/bin/env python3
"""Prepare Multi-Task Dataset for Unified OmniTransfer Training.

This script helps create and organize training data for all 6 OmniTransfer task types,
enabling training of a single unified model that handles:

APPEARANCE REFERENCE TASKS (reference provides spatial/appearance cues):
  - identity_preservation: Maintain subject identity across scenes
  - style_transfer: Apply artistic styles while preserving content
  - scene_composition: Integrate reference elements into target environment

TEMPORAL REFERENCE TASKS (reference provides motion/temporal cues):
  - motion_transfer: Transfer motion patterns between videos
  - pose_reenactment: Drive target with reference poses
  - action_customization: Adapt actions to target context

Usage:
    # Organize existing data into task directories
    python scripts/prepare_multitask_dataset.py organize \
        --input-dir /path/to/raw_videos \
        --output-dir /path/to/multitask_dataset

    # Generate synthetic data for all task types
    python scripts/prepare_multitask_dataset.py generate \
        --output-dir /path/to/multitask_dataset \
        --task-types identity_preservation style_transfer motion_transfer \
        --num-pairs 50

    # Validate dataset structure
    python scripts/prepare_multitask_dataset.py validate \
        --dataset-dir /path/to/multitask_dataset

    # Show prompts for a task type (dry run)
    python scripts/prepare_multitask_dataset.py prompts \
        --task-type motion_transfer \
        --num-prompts 10

Dataset structure for multi-task training:
    preprocessed_data_root/
    ├── identity_preservation/
    │   ├── latents/
    │   ├── conditions/
    │   └── reference_latents/
    ├── style_transfer/
    │   ├── latents/
    │   ├── conditions/
    │   └── reference_latents/
    ├── motion_transfer/
    │   ├── latents/
    │   ├── conditions/
    │   └── reference_latents/
    └── ... (other task types)

Reference: OmniTransfer paper (arXiv:2601.14250v1)
"""

import argparse
import json
import random
import shutil
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

# Task type definitions matching OmniTransfer components
TASK_TYPES = [
    "identity_preservation",
    "style_transfer",
    "scene_composition",
    "motion_transfer",
    "pose_reenactment",
    "action_customization",
]

APPEARANCE_TASKS = ["identity_preservation", "style_transfer", "scene_composition"]
TEMPORAL_TASKS = ["motion_transfer", "pose_reenactment", "action_customization"]


# =============================================================================
# PROMPT TEMPLATES FOR EACH TASK TYPE
# =============================================================================

@dataclass
class TaskPromptTemplate:
    """Prompt templates for generating training data for a specific task."""

    task_type: str
    reference_templates: list[str]
    target_templates: list[str]
    description: str
    category: str  # "appearance" or "temporal"


# Identity Preservation - same person, different context
IDENTITY_PRESERVATION_PROMPTS = TaskPromptTemplate(
    task_type="identity_preservation",
    category="appearance",
    description="Transfer subject identity to different scenes/poses while preserving facial features",
    reference_templates=[
        "A {gender} with {hair_desc} and {face_desc}, {expression}, wearing {outfit}, {setting}, {lighting}",
        "Portrait of a {gender}, {hair_desc}, {face_desc}, {expression}, {outfit}, studio lighting",
        "{gender} with {hair_desc}, {face_desc}, close-up portrait, natural lighting",
    ],
    target_templates=[
        "Same person walking in {location}, {weather}, {time_of_day}",
        "Same person sitting at {place}, {activity}, natural lighting",
        "Same person in {environment}, {pose}, cinematic lighting",
        "Same person dancing in {location}, energetic movement, dynamic lighting",
    ],
)

# Style Transfer - apply artistic style
STYLE_TRANSFER_PROMPTS = TaskPromptTemplate(
    task_type="style_transfer",
    category="appearance",
    description="Apply artistic styles from reference while preserving content",
    reference_templates=[
        "{style_name} style painting, {color_palette} colors, {brush_technique}",
        "Artwork in the style of {artist}, {medium}, dramatic lighting",
        "{art_movement} art style, {texture} texture, vibrant colors",
    ],
    target_templates=[
        "A person walking through a city, realistic video",
        "Someone sitting in a cafe, natural lighting, realistic",
        "Person in a garden, daytime, documentary style",
        "Someone cooking in a kitchen, overhead shot, realistic",
    ],
)

# Motion Transfer - transfer motion patterns
MOTION_TRANSFER_PROMPTS = TaskPromptTemplate(
    task_type="motion_transfer",
    category="temporal",
    description="Transfer motion patterns from reference to target subject",
    reference_templates=[
        "Person doing {dance_style} dance, full body shot, studio background",
        "Someone performing {action}, side view, neutral background",
        "A dancer doing {move_type}, professional lighting",
        "Person doing {exercise}, gym setting, clear movements",
    ],
    target_templates=[
        "Different person standing still, same camera angle, waiting to move",
        "Another individual in neutral pose, ready position",
        "New subject in {outfit}, standing centered, neutral pose",
    ],
)

# Pose Reenactment - drive target with reference poses
POSE_REENACTMENT_PROMPTS = TaskPromptTemplate(
    task_type="pose_reenactment",
    category="temporal",
    description="Drive target subject with poses from reference",
    reference_templates=[
        "Person in {pose_desc}, clear body outline, neutral background",
        "Someone gesturing {gesture_type}, full body visible",
        "A person doing yoga pose {pose_name}, side view",
        "Individual in {athletic_pose}, sports setting",
    ],
    target_templates=[
        "A {gender} in neutral standing pose, ready to move, similar outfit",
        "Person waiting in relaxed stance, same camera angle",
        "Subject in base position, centered frame, clean background",
    ],
)

# Action Customization - adapt actions to context
ACTION_CUSTOMIZATION_PROMPTS = TaskPromptTemplate(
    task_type="action_customization",
    category="temporal",
    description="Customize actions from reference to fit target context",
    reference_templates=[
        "Person {action_verb} in {reference_setting}, clear action",
        "Someone demonstrating {skill}, instructional angle",
        "Individual performing {activity}, focused shot",
    ],
    target_templates=[
        "Same type of action but in {new_setting}, different subject",
        "Similar movement adapted to {new_context}",
        "Action performed by different person in {environment}",
    ],
)

# Scene Composition - integrate elements
SCENE_COMPOSITION_PROMPTS = TaskPromptTemplate(
    task_type="scene_composition",
    category="appearance",
    description="Integrate reference elements into target scene",
    reference_templates=[
        "A {object} with {details}, isolated on clean background",
        "{element_type} with {characteristics}, product shot style",
        "Close-up of {subject} with {visual_features}, studio lighting",
    ],
    target_templates=[
        "Empty {scene_type} scene, waiting for element placement",
        "{environment} with space for new element, natural lighting",
        "Background scene of {location}, ready for composition",
    ],
)

TASK_PROMPTS = {
    "identity_preservation": IDENTITY_PRESERVATION_PROMPTS,
    "style_transfer": STYLE_TRANSFER_PROMPTS,
    "motion_transfer": MOTION_TRANSFER_PROMPTS,
    "pose_reenactment": POSE_REENACTMENT_PROMPTS,
    "action_customization": ACTION_CUSTOMIZATION_PROMPTS,
    "scene_composition": SCENE_COMPOSITION_PROMPTS,
}


# =============================================================================
# FILL-IN VALUES FOR PROMPT TEMPLATES
# =============================================================================

FILL_VALUES = {
    # Identity
    "gender": ["woman", "man", "person"],
    "hair_desc": [
        "shoulder-length black hair", "short brown hair", "long blonde hair",
        "curly red hair", "straight dark hair with highlights", "silver-gray hair"
    ],
    "face_desc": [
        "sharp features and bright eyes", "soft features and warm smile",
        "angular jawline and deep-set eyes", "round face with dimples"
    ],
    "expression": [
        "neutral expression", "slight smile", "thoughtful look", "confident gaze"
    ],
    "outfit": [
        "white turtleneck sweater", "black leather jacket", "blue denim shirt",
        "gray business suit", "casual t-shirt and jeans"
    ],
    "setting": [
        "modern living room", "outdoor park", "coffee shop", "office space"
    ],
    "lighting": [
        "soft natural light", "golden hour lighting", "studio lighting", "dramatic side lighting"
    ],

    # Locations
    "location": [
        "city street", "beach", "forest trail", "shopping mall", "train station"
    ],
    "place": ["cafe table", "park bench", "office desk", "kitchen counter"],
    "environment": ["urban rooftop", "mountain vista", "seaside dock", "garden"],
    "weather": ["sunny day", "overcast sky", "light rain", "misty morning"],
    "time_of_day": ["morning", "golden hour", "midday", "dusk"],
    "pose": ["relaxed stance", "dynamic pose", "seated position", "walking"],
    "activity": ["reading a book", "drinking coffee", "working on laptop", "chatting"],

    # Style
    "style_name": ["Van Gogh", "Monet", "Picasso", "Hokusai", "Art Deco", "Pop Art"],
    "artist": ["Van Gogh", "Monet", "Klimt", "Warhol", "Banksy"],
    "color_palette": ["warm sunset", "cool blue", "vibrant neon", "muted earth"],
    "brush_technique": ["thick impasto strokes", "soft watercolor washes", "bold geometric shapes"],
    "art_movement": ["Impressionist", "Cubist", "Surrealist", "Minimalist"],
    "medium": ["oil painting", "watercolor", "digital art", "pastel"],
    "texture": ["rough canvas", "smooth gradient", "grainy film"],

    # Motion/Dance
    "dance_style": ["hip hop", "ballet", "salsa", "contemporary", "breakdance"],
    "action": ["jumping", "spinning", "reaching up", "crouching", "stretching"],
    "move_type": ["pirouette", "moonwalk", "body wave", "jump split"],
    "exercise": ["yoga stretches", "martial arts kata", "aerobics routine"],

    # Poses
    "pose_desc": ["T-pose", "warrior stance", "seated meditation", "jumping pose"],
    "gesture_type": ["pointing", "waving", "clapping", "conducting"],
    "pose_name": ["warrior II", "tree pose", "downward dog", "cobra"],
    "athletic_pose": ["sprinter start", "basketball jump shot", "tennis serve"],

    # Actions
    "action_verb": ["cooking", "painting", "typing", "playing guitar", "gardening"],
    "skill": ["origami folding", "card shuffling", "juggling", "calligraphy"],
    "reference_setting": ["kitchen", "art studio", "office", "workshop"],
    "new_setting": ["outdoor patio", "different room", "public space"],
    "new_context": ["professional environment", "casual setting", "performance stage"],

    # Scene composition
    "object": ["vintage camera", "potted plant", "coffee mug", "antique clock"],
    "element_type": ["furniture piece", "decorative item", "electronic device"],
    "characteristics": ["intricate details", "unique texture", "vibrant colors"],
    "details": ["brass accents", "weathered finish", "modern design"],
    "visual_features": ["reflective surface", "organic shapes", "geometric patterns"],
    "subject": ["jewelry piece", "sculpture", "food item", "flower arrangement"],
    "scene_type": ["living room", "office", "outdoor", "studio"],
}


def fill_template(template: str) -> str:
    """Fill a template string with random values."""
    result = template
    for key, values in FILL_VALUES.items():
        placeholder = "{" + key + "}"
        while placeholder in result:
            result = result.replace(placeholder, random.choice(values), 1)
    return result


def generate_prompt_pair(task_type: str) -> tuple[str, str]:
    """Generate a reference/target prompt pair for a task type."""
    prompts = TASK_PROMPTS[task_type]
    ref_template = random.choice(prompts.reference_templates)
    tgt_template = random.choice(prompts.target_templates)
    return fill_template(ref_template), fill_template(tgt_template)


# =============================================================================
# DATASET ORGANIZATION
# =============================================================================

def organize_dataset(input_dir: Path, output_dir: Path, task_mapping: dict[str, str] | None = None) -> None:
    """Organize raw videos into task-specific directories.

    Args:
        input_dir: Directory containing raw video pairs
        output_dir: Output directory for organized dataset
        task_mapping: Optional mapping of video prefixes to task types
    """
    print(f"Organizing dataset from {input_dir} to {output_dir}")

    # Default mapping based on filename patterns
    if task_mapping is None:
        task_mapping = {
            "id_": "identity_preservation",
            "identity_": "identity_preservation",
            "style_": "style_transfer",
            "motion_": "motion_transfer",
            "pose_": "pose_reenactment",
            "action_": "action_customization",
            "scene_": "scene_composition",
        }

    # Create task directories
    for task_type in TASK_TYPES:
        task_dir = output_dir / task_type
        (task_dir / "videos").mkdir(parents=True, exist_ok=True)
        (task_dir / "reference_videos").mkdir(parents=True, exist_ok=True)

    # Scan input directory and organize
    video_files = list(input_dir.glob("*.mp4")) + list(input_dir.glob("*.webm"))
    organized_count = {t: 0 for t in TASK_TYPES}

    for video_file in video_files:
        # Determine task type from filename
        task_type = None
        for prefix, task in task_mapping.items():
            if video_file.name.lower().startswith(prefix):
                task_type = task
                break

        if task_type is None:
            print(f"  Warning: Could not determine task type for {video_file.name}")
            continue

        # Determine if reference or target
        is_reference = "_ref" in video_file.name.lower() or "reference" in video_file.name.lower()

        # Copy to appropriate directory
        dest_subdir = "reference_videos" if is_reference else "videos"
        dest_path = output_dir / task_type / dest_subdir / video_file.name

        if not dest_path.exists():
            shutil.copy2(video_file, dest_path)
            organized_count[task_type] += 1
            print(f"  {video_file.name} -> {task_type}/{dest_subdir}/")

    print("\nOrganization summary:")
    for task_type, count in organized_count.items():
        if count > 0:
            print(f"  {task_type}: {count} videos")


def validate_dataset(dataset_dir: Path) -> dict[str, Any]:
    """Validate multi-task dataset structure.

    Returns dict with validation results for each task type.
    """
    results = {}

    for task_type in TASK_TYPES:
        task_dir = dataset_dir / task_type
        task_result = {
            "exists": task_dir.exists(),
            "latents": 0,
            "conditions": 0,
            "reference_latents": 0,
            "valid": False,
        }

        if task_dir.exists():
            latents_dir = task_dir / "latents"
            conditions_dir = task_dir / "conditions"
            ref_latents_dir = task_dir / "reference_latents"

            if latents_dir.exists():
                task_result["latents"] = len(list(latents_dir.glob("*.safetensors")))
            if conditions_dir.exists():
                task_result["conditions"] = len(list(conditions_dir.glob("*.safetensors")))
            if ref_latents_dir.exists():
                task_result["reference_latents"] = len(list(ref_latents_dir.glob("*.safetensors")))

            # Valid if all three directories have matching counts
            task_result["valid"] = (
                task_result["latents"] > 0 and
                task_result["latents"] == task_result["conditions"] == task_result["reference_latents"]
            )

        results[task_type] = task_result

    return results


def print_validation_results(results: dict[str, Any]) -> None:
    """Print validation results in a nice format."""
    print("\nDataset Validation Results:")
    print("=" * 70)

    total_samples = 0
    valid_tasks = []

    for task_type, result in results.items():
        status = "✓" if result["valid"] else "✗"
        category = "appearance" if task_type in APPEARANCE_TASKS else "temporal"

        if result["exists"]:
            print(f"{status} {task_type} ({category})")
            print(f"    Latents: {result['latents']}, Conditions: {result['conditions']}, "
                  f"Refs: {result['reference_latents']}")
            if result["valid"]:
                total_samples += result["latents"]
                valid_tasks.append(task_type)
        else:
            print(f"✗ {task_type} - directory not found")

    print("=" * 70)
    print(f"Total valid samples: {total_samples}")
    print(f"Valid task types: {', '.join(valid_tasks) if valid_tasks else 'None'}")

    if valid_tasks:
        print("\nTo use multi-task training, add to your config:")
        print("```yaml")
        print("training_strategy:")
        print("  multi_task_mode: true")
        print(f"  task_types: {valid_tasks}")
        print("  task_sampling: uniform")
        print("```")


def show_prompts(task_type: str, num_prompts: int = 5) -> None:
    """Show example prompts for a task type."""
    if task_type not in TASK_PROMPTS:
        print(f"Unknown task type: {task_type}")
        print(f"Available: {list(TASK_PROMPTS.keys())}")
        return

    prompts = TASK_PROMPTS[task_type]
    print(f"\n{task_type.upper()} - {prompts.description}")
    print(f"Category: {prompts.category}")
    print("=" * 70)

    for i in range(num_prompts):
        ref, tgt = generate_prompt_pair(task_type)
        print(f"\nPair {i+1}:")
        print(f"  Reference: {ref}")
        print(f"  Target:    {tgt}")


def create_multitask_config(output_path: Path, task_types: list[str], dataset_dir: str) -> None:
    """Create a config YAML for multi-task training."""
    config = f"""# Multi-Task OmniTransfer Configuration
# Generated for unified model training across {len(task_types)} task types

model:
  model_path: "/media/2TB/ltx-models/ltx2/ltx-2-19b-dev.safetensors"
  text_encoder_path: "/media/2TB/ltx-models/gemma"
  training_mode: "lora"

lora:
  rank: 32
  alpha: 32
  dropout: 0.0
  target_modules:
    - "to_k"
    - "to_q"
    - "to_v"
    - "to_out.0"

training_strategy:
  name: "omnitransfer"

  # Multi-task configuration
  multi_task_mode: true
  task_types: {task_types}
  task_sampling: "uniform"  # Options: uniform, weighted, round_robin
  task_weights: {{}}  # Optional: {{"identity_preservation": 2.0}} to weight tasks

  # OmniTransfer components
  enable_tpb: true
  enable_rcl: true
  enable_tma: true

  # First frame conditioning
  first_frame_conditioning_p: 0.1

  # Reference latents directory
  reference_latents_dir: "reference_latents"

  # Loss configuration
  target_loss_weight: 1.0
  min_snr_gamma: 5.0

  # W&B Visualization
  log_reconstructions: true
  reconstruction_log_interval: 500

optimization:
  learning_rate: 1e-5
  steps: 20000  # More steps for multi-task
  batch_size: 1
  gradient_accumulation_steps: 16
  enable_gradient_checkpointing: true

acceleration:
  mixed_precision_mode: "bf16"
  quantization: "int8-quanto"

data:
  preprocessed_data_root: "{dataset_dir}"
  num_dataloader_workers: 2

wandb:
  enabled: true
  project: "omnitransfer-multitask"
  tags: ["omnitransfer", "multi-task", "unified"]

seed: 42
output_dir: "outputs/omnitransfer_multitask"
"""

    output_path.write_text(config)
    print(f"Created config: {output_path}")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Prepare multi-task dataset for unified OmniTransfer training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Organize command
    org_parser = subparsers.add_parser("organize", help="Organize raw videos into task directories")
    org_parser.add_argument("--input-dir", type=Path, required=True, help="Input directory with raw videos")
    org_parser.add_argument("--output-dir", type=Path, required=True, help="Output directory for organized dataset")

    # Validate command
    val_parser = subparsers.add_parser("validate", help="Validate dataset structure")
    val_parser.add_argument("--dataset-dir", type=Path, required=True, help="Dataset directory to validate")

    # Prompts command
    prompt_parser = subparsers.add_parser("prompts", help="Show example prompts for a task type")
    prompt_parser.add_argument("--task-type", type=str, required=True, choices=TASK_TYPES)
    prompt_parser.add_argument("--num-prompts", type=int, default=5, help="Number of prompt pairs to show")

    # Config command
    cfg_parser = subparsers.add_parser("config", help="Generate multi-task training config")
    cfg_parser.add_argument("--output", type=Path, default=Path("configs/ltx2_omnitransfer_multitask.yaml"))
    cfg_parser.add_argument("--task-types", type=str, nargs="+", default=TASK_TYPES)
    cfg_parser.add_argument("--dataset-dir", type=str, required=True)

    # Info command
    info_parser = subparsers.add_parser("info", help="Show information about task types")

    args = parser.parse_args()

    if args.command == "organize":
        organize_dataset(args.input_dir, args.output_dir)

    elif args.command == "validate":
        results = validate_dataset(args.dataset_dir)
        print_validation_results(results)

    elif args.command == "prompts":
        show_prompts(args.task_type, args.num_prompts)

    elif args.command == "config":
        create_multitask_config(args.output, args.task_types, args.dataset_dir)

    elif args.command == "info":
        print("\nOmniTransfer Task Types:")
        print("=" * 70)
        print("\nAPPEARANCE REFERENCE TASKS (reference provides spatial/appearance cues):")
        for task in APPEARANCE_TASKS:
            desc = TASK_PROMPTS[task].description
            print(f"  • {task}: {desc}")

        print("\nTEMPORAL REFERENCE TASKS (reference provides motion/temporal cues):")
        for task in TEMPORAL_TASKS:
            desc = TASK_PROMPTS[task].description
            print(f"  • {task}: {desc}")

        print("\nTask-aware Positional Bias (TPB) applies different RoPE offsets:")
        print("  • Temporal tasks: Δ = (0, w_tgt, 0) - spatial offset")
        print("  • Appearance tasks: Δ = (f, 0, 0) - temporal offset")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
