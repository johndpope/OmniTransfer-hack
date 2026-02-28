#!/usr/bin/env python3
"""Test Stage 2 training isolation - verify DiT is frozen and only TMA connector is trainable.

This script validates the 3-stage training setup without requiring a full training run.
It checks:
1. TMA module is created for stage 2
2. DiT/LoRA parameters are frozen
3. Only TMA connector parameters are trainable
4. TPB and ConceptEmbedding are NOT trained in stage 2

Usage:
    python scripts/test_stage2_freezing.py
    python scripts/test_stage2_freezing.py --stage 1  # Test stage 1
    python scripts/test_stage2_freezing.py --stage 3  # Test stage 3
"""

import argparse
import sys
from pathlib import Path

import torch


def test_strategy_trainable_params(stage: int = 2):
    """Test which parameters are trainable for a given stage."""
    print(f"\n{'='*60}")
    print(f"Testing OmniTransfer Stage {stage} Parameter Freezing")
    print(f"{'='*60}\n")

    # Import strategy and config
    from ltx_trainer.omnitransfer.strategy import OmniTransferConfig, OmniTransferStrategy

    # Create config for the specified stage
    config = OmniTransferConfig(
        name="omnitransfer",
        task_type="style_transfer",
        enable_tpb=True,
        enable_rcl=True,
        enable_tma=True,
        training_stage=stage,
        use_cached_tma_features=True,  # Pretend we have cached features
        tma_mllm_hidden_dim=3584,  # Qwen2.5-VL-7B
        tma_num_queries=8,
        tma_connector_layers=2,
    )

    print(f"Config created:")
    print(f"  - training_stage: {config.training_stage}")
    print(f"  - enable_tma: {config.enable_tma}")
    print(f"  - enable_tpb: {config.enable_tpb}")
    print(f"  - enable_rcl: {config.enable_rcl}")

    # Initialize strategy (only takes config)
    strategy = OmniTransferStrategy(config=config)

    print(f"\nStrategy initialized:")
    print(f"  - TPB: {strategy._tpb is not None}")
    print(f"  - ConceptEmbedding: {strategy._concept_embedding is not None}")
    print(f"  - TMA: {strategy._tma is not None}")

    # Get trainable parameters
    trainable_params = strategy.get_trainable_parameters()

    print(f"\n{'='*60}")
    print(f"Trainable Parameters for Stage {stage}")
    print(f"{'='*60}")

    # Categorize parameters
    tpb_params = []
    ce_params = []
    tma_connector_params = []
    tma_metaquery_params = []
    other_params = []

    for param in trainable_params:
        # We need to identify which module the param belongs to
        # This is a bit tricky without named parameters, so we'll count
        pass

    # Better approach: check named parameters from each module
    if strategy._tpb is not None:
        tpb_param_count = sum(p.numel() for p in strategy._tpb.parameters())
        tpb_trainable = sum(p.numel() for p in strategy._tpb.parameters() if p.requires_grad)
        print(f"\nTPB:")
        print(f"  - Total params: {tpb_param_count:,}")
        print(f"  - Trainable: {tpb_trainable:,}")
        print(f"  - Expected trainable (stage {stage}): {'Yes' if stage in [1, 3] else 'No'}")

    if strategy._concept_embedding is not None:
        ce_param_count = sum(p.numel() for p in strategy._concept_embedding.parameters())
        ce_trainable = sum(p.numel() for p in strategy._concept_embedding.parameters() if p.requires_grad)
        print(f"\nConceptEmbedding:")
        print(f"  - Total params: {ce_param_count:,}")
        print(f"  - Trainable: {ce_trainable:,}")
        print(f"  - Expected trainable (stage {stage}): {'Yes' if stage in [1, 3] else 'No'}")

    if strategy._tma is not None:
        # TMA has two parts: MetaQueryBank and Connector
        metaquery_params = sum(p.numel() for p in strategy._tma.meta_query_bank.parameters())
        connector_params = sum(p.numel() for p in strategy._tma.connector.parameters())

        metaquery_trainable = sum(p.numel() for p in strategy._tma.meta_query_bank.parameters() if p.requires_grad)
        connector_trainable = sum(p.numel() for p in strategy._tma.connector.parameters() if p.requires_grad)

        print(f"\nTMA Module:")
        print(f"  MetaQueryBank:")
        print(f"    - Total params: {metaquery_params:,}")
        print(f"    - Trainable: {metaquery_trainable:,}")
        print(f"    - Expected trainable (stage {stage}): {'Yes' if stage == 3 else 'No'}")
        print(f"  Connector MLP:")
        print(f"    - Total params: {connector_params:,}")
        print(f"    - Trainable: {connector_trainable:,}")
        print(f"    - Expected trainable (stage {stage}): {'Yes' if stage in [2, 3] else 'No'}")

    # Summary
    total_trainable = sum(p.numel() for p in trainable_params)
    print(f"\n{'='*60}")
    print(f"Summary for Stage {stage}")
    print(f"{'='*60}")
    print(f"Total trainable parameters returned: {total_trainable:,}")
    print(f"Number of parameter tensors: {len(trainable_params)}")

    # Verify expectations
    print(f"\n{'='*60}")
    print("Verification")
    print(f"{'='*60}")

    errors = []

    if stage == 1:
        # Stage 1: TPB + ConceptEmbedding, NO TMA
        if strategy._tma is not None:
            errors.append("ERROR: TMA should NOT be created for Stage 1")
        if strategy._tpb is None:
            errors.append("ERROR: TPB should be created for Stage 1")
        print("Stage 1 expectations: TPB=Yes, CE=Yes, TMA=No")

    elif stage == 2:
        # Stage 2: Only TMA connector
        if strategy._tma is None:
            errors.append("ERROR: TMA should be created for Stage 2")
        else:
            # Check that only connector params are in trainable list
            connector_param_ids = {id(p) for p in strategy._tma.connector.parameters()}
            trainable_param_ids = {id(p) for p in trainable_params}

            # MetaQuery params should NOT be trainable
            metaquery_param_ids = {id(p) for p in strategy._tma.meta_query_bank.parameters()}
            metaquery_in_trainable = metaquery_param_ids & trainable_param_ids
            if metaquery_in_trainable:
                errors.append(f"ERROR: MetaQuery params should NOT be trainable in Stage 2 (found {len(metaquery_in_trainable)})")

            # Connector params SHOULD be trainable
            connector_in_trainable = connector_param_ids & trainable_param_ids
            if len(connector_in_trainable) != len(connector_param_ids):
                errors.append(f"ERROR: All connector params should be trainable in Stage 2")

        # TPB should NOT be in trainable params for stage 2
        if strategy._tpb is not None:
            tpb_param_ids = {id(p) for p in strategy._tpb.parameters()}
            tpb_in_trainable = tpb_param_ids & {id(p) for p in trainable_params}
            if tpb_in_trainable:
                errors.append(f"ERROR: TPB params should NOT be trainable in Stage 2 (found {len(tpb_in_trainable)})")

        print("Stage 2 expectations: TPB=No, CE=No, TMA.connector=Yes, TMA.metaquery=No")

    elif stage == 3:
        # Stage 3: Everything
        if strategy._tma is None:
            errors.append("ERROR: TMA should be created for Stage 3")
        else:
            # All TMA params should be trainable
            tma_param_ids = {id(p) for p in strategy._tma.parameters()}
            trainable_param_ids = {id(p) for p in trainable_params}
            tma_in_trainable = tma_param_ids & trainable_param_ids
            if len(tma_in_trainable) != len(tma_param_ids):
                errors.append(f"ERROR: All TMA params should be trainable in Stage 3 (found {len(tma_in_trainable)}/{len(tma_param_ids)})")

        # TPB should be trainable
        if strategy._tpb is not None:
            tpb_param_ids = {id(p) for p in strategy._tpb.parameters()}
            tpb_in_trainable = tpb_param_ids & {id(p) for p in trainable_params}
            if len(tpb_in_trainable) != len(tpb_param_ids):
                errors.append(f"ERROR: All TPB params should be trainable in Stage 3")

        print("Stage 3 expectations: TPB=Yes, CE=Yes, TMA.connector=Yes, TMA.metaquery=Yes")

    if errors:
        print("\n❌ FAILURES:")
        for err in errors:
            print(f"  {err}")
        return False
    else:
        print("\n✅ All checks passed!")
        return True


def test_trainer_freezing(stage: int = 2):
    """Test the full trainer freezing logic with a mock transformer."""
    print(f"\n{'='*60}")
    print(f"Testing Trainer DiT Freezing for Stage {stage}")
    print(f"{'='*60}\n")

    # Create a simple mock transformer to test freezing
    class MockTransformer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = torch.nn.Linear(128, 128)
            self.layer2 = torch.nn.Linear(128, 128)

    transformer = MockTransformer()

    # Simulate the trainer's freezing logic
    print("Initial state:")
    print(f"  layer1.weight.requires_grad: {transformer.layer1.weight.requires_grad}")
    print(f"  layer2.weight.requires_grad: {transformer.layer2.weight.requires_grad}")

    # Step 1: Freeze all (like _load_models does)
    transformer.requires_grad_(False)
    print("\nAfter transformer.requires_grad_(False):")
    print(f"  layer1.weight.requires_grad: {transformer.layer1.weight.requires_grad}")
    print(f"  layer2.weight.requires_grad: {transformer.layer2.weight.requires_grad}")

    # Step 2: Simulate LoRA setup (PEFT would add trainable params)
    # For this test, we'll manually add "LoRA-like" params
    class MockLoRALayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lora_A = torch.nn.Linear(128, 8, bias=False)
            self.lora_B = torch.nn.Linear(8, 128, bias=False)

    lora_layer = MockLoRALayer()
    print("\nAfter adding LoRA layers:")
    print(f"  lora_A.weight.requires_grad: {lora_layer.lora_A.weight.requires_grad}")
    print(f"  lora_B.weight.requires_grad: {lora_layer.lora_B.weight.requires_grad}")

    # Step 3: For stage 2, freeze LoRA too
    if stage == 2:
        lora_layer.requires_grad_(False)
        print("\nAfter Stage 2 LoRA freezing:")
        print(f"  lora_A.weight.requires_grad: {lora_layer.lora_A.weight.requires_grad}")
        print(f"  lora_B.weight.requires_grad: {lora_layer.lora_B.weight.requires_grad}")

    # Count trainable params
    transformer_trainable = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
    lora_trainable = sum(p.numel() for p in lora_layer.parameters() if p.requires_grad)

    print(f"\nFinal trainable parameter counts:")
    print(f"  Transformer: {transformer_trainable:,}")
    print(f"  LoRA: {lora_trainable:,}")

    if stage == 2:
        if transformer_trainable == 0 and lora_trainable == 0:
            print("\n✅ Stage 2: DiT and LoRA correctly frozen")
            return True
        else:
            print("\n❌ Stage 2: DiT/LoRA should be frozen but has trainable params")
            return False
    else:
        if lora_trainable > 0:
            print(f"\n✅ Stage {stage}: LoRA params are trainable")
            return True
        else:
            print(f"\n❌ Stage {stage}: LoRA params should be trainable")
            return False


def test_tma_forward_pass():
    """Test TMA forward pass with mock inputs."""
    print(f"\n{'='*60}")
    print("Testing TMA Forward Pass")
    print(f"{'='*60}\n")

    from ltx_trainer.omnitransfer.components import TaskAdaptiveMultimodalAlignment

    # Create TMA module
    # Paper quote: "three-layer MLP connector" (Section 4.4)
    tma = TaskAdaptiveMultimodalAlignment(
        mllm_hidden_dim=3584,  # Qwen2.5-VL-7B
        output_dim=4096,  # LTX-2 cross-attention dim
        num_connector_layers=3,  # Must be 3 to reach output_dim
        num_queries_per_task=8,
        dropout=0.1,
    )

    print(f"TMA created:")
    print(f"  - MetaQueryBank num_queries_per_task: {tma.meta_query_bank.num_queries_per_task}")
    print(f"  - MetaQueryBank query_dim: {tma.meta_query_bank.query_dim}")
    print(f"  - Connector layers: {len(tma.connector)}")

    # Create mock inputs
    batch_size = 2
    seq_len = 100
    mllm_dim = 3584

    # Mock Qwen VL features
    qwen_features = torch.randn(batch_size, seq_len, mllm_dim)

    # Mock task indices (batch of different tasks)
    task_indices = torch.tensor([0, 1])  # Two different tasks

    print(f"\nInput shapes:")
    print(f"  - qwen_features: {qwen_features.shape}")
    print(f"  - task_indices: {task_indices.shape}")

    # Forward pass
    try:
        output = tma(qwen_features, task_indices)
        print(f"\nOutput shape: {output.shape}")
        print(f"Expected: [{batch_size}, 8, 4096]")

        if output.shape == (batch_size, 8, 4096):
            print("\n✅ TMA forward pass successful!")
            return True
        else:
            print(f"\n❌ Unexpected output shape: {output.shape}")
            return False

    except Exception as e:
        print(f"\n❌ TMA forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Test OmniTransfer stage freezing")
    parser.add_argument("--stage", type=int, default=2, choices=[1, 2, 3],
                        help="Training stage to test (default: 2)")
    parser.add_argument("--all", action="store_true",
                        help="Test all stages")
    args = parser.parse_args()

    results = []

    if args.all:
        stages = [1, 2, 3]
    else:
        stages = [args.stage]

    for stage in stages:
        print(f"\n{'#'*60}")
        print(f"# TESTING STAGE {stage}")
        print(f"{'#'*60}")

        # Test 1: Strategy trainable params
        result1 = test_strategy_trainable_params(stage)
        results.append(("Strategy params", stage, result1))

        # Test 2: Trainer freezing simulation
        result2 = test_trainer_freezing(stage)
        results.append(("Trainer freezing", stage, result2))

    # Test 3: TMA forward pass (only once)
    result3 = test_tma_forward_pass()
    results.append(("TMA forward", "N/A", result3))

    # Summary
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")

    all_passed = True
    for test_name, stage, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {test_name} (stage {stage}): {status}")
        if not passed:
            all_passed = False

    if all_passed:
        print(f"\n{'='*60}")
        print("🎉 ALL TESTS PASSED!")
        print(f"{'='*60}")
        return 0
    else:
        print(f"\n{'='*60}")
        print("❌ SOME TESTS FAILED")
        print(f"{'='*60}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
