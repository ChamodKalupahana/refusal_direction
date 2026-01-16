"""
Extract refusal direction from uncensored model by comparing:
- harmful prompts (without assistant response)
- harmless prompts (without assistant response)

The refusal direction is: mean(harmful_activations) + mean(harmless_activations)

This matches the base model's methodology from pipeline/submodules/generate_directions.py
"""
import torch
import os
import json
import sys
import random
import gc
from tqdm import tqdm

# Add project root to sys.path to allow importing from pipeline
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from dataset.load_dataset import load_dataset_split
from pipeline.model_utils.model_factory import construct_model_base


def get_mean_activations(model, tokenizer, instructions, tokenize_instructions_fn, block_modules, target_layer, positions, batch_size=16):
    """
    Compute mean activations at specified positions for a given layer.
    Matches the methodology from pipeline/submodules/generate_directions.py
    """
    torch.cuda.empty_cache()
    
    n_positions = len(positions)
    n_samples = len(instructions)
    d_model = model.config.hidden_size
    
    # Store mean activations in high-precision to avoid numerical issues
    mean_activations = torch.zeros((n_positions, d_model), dtype=torch.float64, device=model.device)
    
    for i in tqdm(range(0, len(instructions), batch_size)):
        batch_instructions = instructions[i:i+batch_size]
        inputs = tokenize_instructions_fn(instructions=batch_instructions)
        
        with torch.inference_mode():
            outputs = model(
                input_ids=inputs.input_ids.to(model.device),
                attention_mask=inputs.attention_mask.to(model.device),
                output_hidden_states=True,
                use_cache=False,
            )
        
        # hidden_states[0] is embeddings, hidden_states[layer+1] is output of layer
        hidden_states = outputs.hidden_states[target_layer + 1]  # [batch, seq, d_model]
        
        # Extract activations at specified positions and accumulate
        for batch_idx in range(hidden_states.shape[0]):
            for pos_idx, pos in enumerate(positions):
                activation = hidden_states[batch_idx, pos, :].to(torch.float64)
                mean_activations[pos_idx] += activation / n_samples
        
        # Clear cache periodically
        if (i // batch_size) % 5 == 0:
            gc.collect()
            torch.cuda.empty_cache()
    
    return mean_activations


def main():
    # Configuration
    model_path = "spkgyk/Yi-6B-Chat-uncensored"
    N_TRAIN = 256  # Number of prompts to use (matching base model's n_train)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Load layer and position from base model's metadata
    base_metadata_path = os.path.join(script_dir, "../pipeline/runs/yi-6b-chat/direction_metadata.json")
    print(f"Loading direction config from: {base_metadata_path}")
    with open(base_metadata_path, 'r') as f:
        base_metadata = json.load(f)
    
    target_layer = base_metadata["layer"]
    target_pos = base_metadata["pos"]  # Should be -5
    
    # Use same positions as base model (multiple positions around EOI)
    # Base model uses: positions=list(range(-len(model_base.eoi_toks), 0))
    # For Yi, eoi_toks are the tokens after the instruction, typically around 5 tokens
    positions = list(range(-5, 0))  # [-5, -4, -3, -2, -1]
    
    print(f"Using layer={target_layer}, positions={positions}")
    
    # Output files
    output_dir = os.path.join(script_dir, "refusal_direction")
    os.makedirs(output_dir, exist_ok=True)
    
    direction_path = os.path.join(output_dir, "direction.pt")
    metadata_path = os.path.join(output_dir, "direction_metadata.json")
    
    # Load model using the model factory (same as base model pipeline)
    print(f"Loading model: {model_path}...")
    model_base = construct_model_base(model_path)
    model = model_base.model
    tokenizer = model_base.tokenizer
    tokenize_instructions_fn = model_base.tokenize_instructions_fn
    block_modules = model_base.model_block_modules
    
    print("Model loaded successfully.")
    
    # Load and sample datasets (matching base model methodology)
    random.seed(42)
    print(f"\nLoading harmful train prompts...")
    harmful_train = random.sample(
        load_dataset_split(harmtype='harmful', split='train', instructions_only=True), 
        N_TRAIN
    )
    print(f"Loaded {len(harmful_train)} harmful prompts")
    
    print(f"Loading harmless train prompts...")
    harmless_train = random.sample(
        load_dataset_split(harmtype='harmless', split='train', instructions_only=True), 
        N_TRAIN
    )
    print(f"Loaded {len(harmless_train)} harmless prompts")
    
    # Compute mean activations for harmful and harmless prompts
    print(f"\nExtracting mean activations for harmful prompts at layer {target_layer}...")
    mean_activations_harmful = get_mean_activations(
        model, tokenizer, harmful_train, tokenize_instructions_fn, 
        block_modules, target_layer, positions
    )
    
    print(f"\nExtracting mean activations for harmless prompts at layer {target_layer}...")
    mean_activations_harmless = get_mean_activations(
        model, tokenizer, harmless_train, tokenize_instructions_fn, 
        block_modules, target_layer, positions
    )
    
    # Compute mean sum: harmful + harmless
    mean_sum = mean_activations_harmful + mean_activations_harmless
    
    # Select direction at the target position (pos=-5 corresponds to index 0 in our positions list)
    # positions = [-5, -4, -3, -2, -1], so pos=-5 is at index 0
    pos_index = positions.index(target_pos)
    refusal_dir = mean_sum[pos_index]
    
    refusal_dir_normalized = refusal_dir / refusal_dir.norm()
    
    print(f"\nRefusal direction computed at layer {target_layer}, pos={target_pos}.")
    print(f"Direction shape: {refusal_dir_normalized.shape}")
    print(f"Direction norm (before normalization): {refusal_dir.norm().item():.4f}")
    
    # Save direction
    torch.save(refusal_dir_normalized.to(torch.float32), direction_path)
    print(f"Saved direction to {direction_path}")
    
    # Save metadata
    metadata = {
        "model": model_path,
        "layer": target_layer,
        "pos": target_pos,
        "positions_computed": positions,
        "n_harmful": len(harmful_train),
        "n_harmless": len(harmless_train),
        "method": "mean(harmful_activations) + mean(harmless_activations) at EOI positions",
        "note": "Matches base model methodology from pipeline/submodules/generate_directions.py"
    }
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    print(f"Saved metadata to {metadata_path}")


if __name__ == "__main__":
    main()
