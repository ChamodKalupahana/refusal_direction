"""
Extract refusal direction from uncensored model by comparing:
- actadd prompts (with refusal behavior) 
- baseline prompts (without refusal behavior)

The refusal direction is: mean(actadd activations) - mean(baseline activations)
"""
import torch
import os
import json
from pipeline.model_utils.model_factory import construct_model_base
from pipeline.submodules.generate_directions import get_mean_activations
from tqdm import tqdm


def load_prompts(json_file):
    """Load prompts from a chat-format JSON file."""
    with open(json_file, 'r') as f:
        data = json.load(f)
    return [item['prompt'] for item in data]


def main():
    # Configuration
    model_path = "spkgyk/Yi-6B-Chat-uncensored"
    target_layer = 14
    target_pos_idx = 0  # Index into positions list (we use positions=[-1])
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Input files
    actadd_file = os.path.join(script_dir, "add_chat_format/llama3_jailbreakbench_actadd_chat_format.json")
    baseline_file = os.path.join(script_dir, "add_chat_format/llama3_jailbreakbench_baseline_chat_format.json")
    
    # Output files
    output_dir = os.path.join(script_dir, "refusal_direction")
    os.makedirs(output_dir, exist_ok=True)
    
    direction_path = os.path.join(output_dir, "direction.pt")
    metadata_path = os.path.join(output_dir, "direction_metadata.json")
    
    print(f"Loading model: {model_path}")
    model_base = construct_model_base(model_path)
    model = model_base.model
    tokenizer = model_base.tokenizer
    
    # Load prompts
    print(f"Loading actadd prompts from {actadd_file}...")
    actadd_prompts = load_prompts(actadd_file)
    print(f"Loaded {len(actadd_prompts)} actadd prompts")
    
    print(f"Loading baseline prompts from {baseline_file}...")
    baseline_prompts = load_prompts(baseline_file)
    print(f"Loaded {len(baseline_prompts)} baseline prompts")
    
    # Extract mean activations for actadd prompts (with refusal)
    print("\nExtracting mean activations for actadd prompts (with refusal)...")
    mean_activations_actadd = get_mean_activations(
        model=model,
        tokenizer=tokenizer,
        instructions=actadd_prompts,
        tokenize_instructions_fn=model_base.tokenize_instructions_fn,
        block_modules=model_base.model_block_modules,
        positions=[-1],  # Last token position
        batch_size=8
    )
    
    # Extract mean activations for baseline prompts (without refusal)
    print("\nExtracting mean activations for baseline prompts (without refusal)...")
    mean_activations_baseline = get_mean_activations(
        model=model,
        tokenizer=tokenizer,
        instructions=baseline_prompts,
        tokenize_instructions_fn=model_base.tokenize_instructions_fn,
        block_modules=model_base.model_block_modules,
        positions=[-1],  # Last token position
        batch_size=8
    )
    
    # Compute refusal direction: actadd - baseline
    # actadd has refusal behavior, baseline does not
    # So the difference captures what makes the model refuse
    mean_diff = mean_activations_actadd - mean_activations_baseline
    
    # mean_diff shape: [n_positions, n_layers, d_model]
    # n_positions=1 because we passed positions=[-1]
    refusal_dir = mean_diff[target_pos_idx, target_layer, :]
    refusal_dir_normalized = refusal_dir / refusal_dir.norm()
    
    print(f"\nRefusal direction computed at layer {target_layer}.")
    print(f"Direction shape: {refusal_dir_normalized.shape}")
    print(f"Direction norm (before normalization): {refusal_dir.norm().item():.4f}")
    
    # Save direction
    torch.save(refusal_dir_normalized, direction_path)
    print(f"Saved direction to {direction_path}")
    
    # Save metadata
    metadata = {
        "model": model_path,
        "layer": target_layer,
        "pos": -1,
        "actadd_file": os.path.basename(actadd_file),
        "baseline_file": os.path.basename(baseline_file),
        "num_actadd_prompts": len(actadd_prompts),
        "num_baseline_prompts": len(baseline_prompts),
        "method": "mean(actadd) - mean(baseline)"
    }
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    print(f"Saved metadata to {metadata_path}")


if __name__ == "__main__":
    main()
