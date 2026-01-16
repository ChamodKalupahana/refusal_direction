"""
Extract refusal direction from uncensored model by comparing:
- actadd prompts (with refusal behavior) 
- baseline prompts (without refusal behavior)

The refusal direction is: mean(actadd activations) - mean(baseline activations)

Note: Prompts already contain chat format, so we load model directly without construct_model_base.
"""
import torch
import os
import json
import sys
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add project root to sys.path to allow importing from pipeline
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from pipeline.submodules.generate_directions import get_mean_activations


def load_prompts(json_file):
    """Load prompts from a chat-format JSON file."""
    with open(json_file, 'r') as f:
        data = json.load(f)
    return [item['prompt'] for item in data]


def tokenize_fn(tokenizer):
    """Create a tokenize function that doesn't add chat formatting."""
    def _tokenize(instructions):
        return tokenizer(
            instructions,
            padding=True,
            truncation=True,
            max_length=2048,  # Limit length to avoid OOM
            return_tensors="pt"
        )
    return _tokenize


def main():
    # Configuration
    model_path = "spkgyk/Yi-6B-Chat-uncensored"
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Load layer and position from base model's metadata
    base_metadata_path = os.path.join(script_dir, "../pipeline/runs/yi-6b-chat/direction_metadata.json")
    print(f"Loading direction config from: {base_metadata_path}")
    with open(base_metadata_path, 'r') as f:
        base_metadata = json.load(f)
    
    target_layer = base_metadata["layer"]
    target_pos = base_metadata["pos"]  # e.g., -5
    target_pos_idx = 0  # Index into the positions list (we pass a single position)
    
    print(f"Using layer={target_layer}, pos={target_pos}")
    
    # Input files
    actadd_file = os.path.join(script_dir, "add_chat_format/llama3_jailbreakbench_actadd_chat_format.json")
    baseline_file = os.path.join(script_dir, "add_chat_format/llama3_jailbreakbench_baseline_chat_format.json")
    
    # Output files
    output_dir = os.path.join(script_dir, "refusal_direction")
    os.makedirs(output_dir, exist_ok=True)

    
    direction_path = os.path.join(output_dir, "direction.pt")
    metadata_path = os.path.join(output_dir, "direction_metadata.json")
    
    # Load model and tokenizer directly (prompts already have chat format)
    print(f"Loading model: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained("01-ai/Yi-6B-Chat")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    # Get block modules (transformer layers)
    block_modules = model.model.layers
    
    # Load prompts
    print(f"Loading actadd prompts from {actadd_file}...")
    actadd_prompts = load_prompts(actadd_file)
    print(f"Loaded {len(actadd_prompts)} actadd prompts")
    
    print(f"Loading baseline prompts from {baseline_file}...")
    baseline_prompts = load_prompts(baseline_file)
    print(f"Loaded {len(baseline_prompts)} baseline prompts")
    
    # Create tokenize function (no chat formatting needed)
    tokenize_instructions_fn = tokenize_fn(tokenizer)
    
    # Extract mean activations for actadd prompts (with refusal)
    print("\nExtracting mean activations for actadd prompts (with refusal)...")
    model.eval()
    with torch.no_grad():
        mean_activations_actadd = get_mean_activations(
            model=model,
            tokenizer=tokenizer,
            instructions=actadd_prompts,
            tokenize_instructions_fn=tokenize_instructions_fn,
            block_modules=block_modules,
            positions=[target_pos],  # Position from metadata
            batch_size=1  # Smallest batch size
        )
    
    # Clear CUDA cache between extractions
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    
    # Extract mean activations for baseline prompts (without refusal)
    print("\nExtracting mean activations for baseline prompts (without refusal)...")
    with torch.no_grad():
        mean_activations_baseline = get_mean_activations(
            model=model,
            tokenizer=tokenizer,
            instructions=baseline_prompts,
            tokenize_instructions_fn=tokenize_instructions_fn,
            block_modules=block_modules,
            positions=[target_pos],  # Position from metadata
            batch_size=1  # Smallest batch size
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
        "pos": target_pos,
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
