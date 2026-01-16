"""
Direction Selection for Yi-6B-Chat-Uncensored

This script runs the full direction selection procedure from the paper:
1. Generate candidate directions from multiple positions/layers
2. Evaluate each direction with bypass_score, induce_score, kl_score
3. Filter directions based on criteria (induce_score > 0, kl_score < 0.1, layer < 0.8L)
4. Select the direction with minimum bypass_score
"""

import torch
import random
import json
import os

from dataset.load_dataset import load_dataset_split
from pipeline.model_utils.model_factory import construct_model_base
from pipeline.submodules.generate_directions import generate_directions
from pipeline.submodules.select_direction import select_direction

# Configuration
MODEL_PATH = "spkgyk/Yi-6B-Chat-uncensored"
N_TRAIN = 260  # Number of training examples
N_VAL = 64     # Number of validation examples
BATCH_SIZE = 8


def main():
    print(f"=" * 60)
    print(f"Direction Selection for {MODEL_PATH}")
    print(f"=" * 60)
    
    # Set up paths
    model_name = os.path.basename(MODEL_PATH)
    output_dir = os.path.join("pipeline", "runs", model_name)
    generate_dir = os.path.join(output_dir, "generate_directions")
    select_dir = os.path.join(output_dir, "select_direction")
    
    os.makedirs(generate_dir, exist_ok=True)
    os.makedirs(select_dir, exist_ok=True)
    
    # Load model
    print(f"\nLoading model: {MODEL_PATH}")
    model_base = construct_model_base(MODEL_PATH)
    
    # Load and sample datasets
    print("\nLoading datasets...")
    random.seed(42)
    
    # Load full datasets
    harmful_train_full = load_dataset_split(harmtype='harmful', split='train', instructions_only=True)
    harmless_train_full = load_dataset_split(harmtype='harmless', split='train', instructions_only=True)
    harmful_val_full = load_dataset_split(harmtype='harmful', split='val', instructions_only=True)
    harmless_val_full = load_dataset_split(harmtype='harmless', split='val', instructions_only=True)
    
    # Sample, using min() to avoid sampling more than available
    harmful_train = random.sample(harmful_train_full, min(N_TRAIN, len(harmful_train_full)))
    harmless_train = random.sample(harmless_train_full, min(N_TRAIN, len(harmless_train_full)))
    harmful_val = random.sample(harmful_val_full, min(N_VAL, len(harmful_val_full)))
    harmless_val = random.sample(harmless_val_full, min(N_VAL, len(harmless_val_full)))
    
    print(f"  Harmful train: {len(harmful_train)}")
    print(f"  Harmless train: {len(harmless_train)}")
    print(f"  Harmful val: {len(harmful_val)}")
    print(f"  Harmless val: {len(harmless_val)}")
    
    # Step 1: Generate candidate directions
    print("\n" + "=" * 60)
    print("Step 1: Generating candidate directions...")
    print("=" * 60)
    
    mean_diffs_path = os.path.join(generate_dir, "mean_diffs.pt")
    
    if os.path.exists(mean_diffs_path):
        print(f"Loading existing mean_diffs from {mean_diffs_path}")
        mean_diffs = torch.load(mean_diffs_path, map_location=model_base.model.device)
    else:
        mean_diffs = generate_directions(
            model_base=model_base,
            harmful_instructions=harmful_train,
            harmless_instructions=harmless_train,
            artifact_dir=generate_dir
        )
        torch.save(mean_diffs, mean_diffs_path)
        print(f"Saved mean_diffs to {mean_diffs_path}")
    
    print(f"Candidate directions shape: {mean_diffs.shape}")
    print(f"  n_positions: {mean_diffs.shape[0]}")
    print(f"  n_layers: {mean_diffs.shape[1]}")
    print(f"  d_model: {mean_diffs.shape[2]}")
    
    # Step 2: Select the best direction
    print("\n" + "=" * 60)
    print("Step 2: Selecting best direction...")
    print("=" * 60)
    print("This will evaluate all candidate directions with:")
    print("  - bypass_score (ablation on harmful prompts)")
    print("  - induce_score (activation addition on harmless prompts)")
    print("  - kl_score (KL divergence when ablating on harmless prompts)")
    print("\nFiltering criteria (relaxed for uncensored model):")
    print("  - induce_score: DISABLED (uncensored model can't induce refusal)")
    print("  - kl_score < 0.5 (relaxed from 0.1)")
    print("  - layer < 0.8 * n_layers")
    print()
    
    pos, layer, direction = select_direction(
        model_base=model_base,
        harmful_instructions=harmful_val,
        harmless_instructions=harmless_val,
        candidate_directions=mean_diffs,
        artifact_dir=select_dir,
        kl_threshold=0.5,  # Relaxed from 0.1
        induce_refusal_threshold=None,  # Disabled - uncensored model can't induce refusal
        prune_layer_percentage=0.2,
        batch_size=BATCH_SIZE
    )
    
    # Normalize the direction
    direction_normalized = direction / direction.norm()
    
    # Save results
    print("\n" + "=" * 60)
    print("Saving results...")
    print("=" * 60)
    
    direction_path = os.path.join(output_dir, "direction.pt")
    metadata_path = os.path.join(output_dir, "direction_metadata.json")
    
    torch.save(direction_normalized, direction_path)
    print(f"Saved direction to {direction_path}")
    
    with open(metadata_path, 'w') as f:
        json.dump({"pos": pos, "layer": layer}, f, indent=4)
    print(f"Saved metadata to {metadata_path}")
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)
    print(f"\nSelected direction: position={pos}, layer={layer}")
    print(f"Direction norm (before normalization): {direction.norm().item():.4f}")


if __name__ == "__main__":
    main()
