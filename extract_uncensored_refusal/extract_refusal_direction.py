"""
Extract refusal direction from uncensored model by comparing:
- actadd prompts (with refusal behavior) 
- baseline prompts (without refusal behavior)

The refusal direction is: mean(actadd[i] - baseline[i]) for each paired prompt.

Note: Prompts already contain chat format, so we load model directly without construct_model_base.
"""
import torch
import os
import json
import sys
import gc
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add project root to sys.path to allow importing from pipeline
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

# Hook logic is implemented locally in get_activation_at_position


def load_prompts(json_file):
    """Load prompts from a chat-format JSON file."""
    with open(json_file, 'r') as f:
        data = json.load(f)
    return [item['prompt'] for item in data]


def get_activation_at_position(model, tokenizer, prompt, block_modules, target_layer, n_tokens=15):
    """
    Extract mean activation at a specific layer for the first n_tokens after <|im_start|>assistant.
    """
    n_layers = model.config.num_hidden_layers
    d_model = model.config.hidden_size
    
    # Tokenize prompt
    inputs = tokenizer(
        prompt,
        padding=False,
        truncation=True,
        max_length=2048,
        return_tensors="pt"
    )
    input_ids = inputs.input_ids[0]
    
    # Find the position of "<|im_start|>assistant\n" in the tokenized input
    # Tokenize the marker to find its token IDs
    assistant_marker = "<|im_start|>assistant\n"
    marker_ids = tokenizer.encode(assistant_marker, add_special_tokens=False)
    
    # Search for the marker in the input
    marker_start = -1
    for i in range(len(input_ids) - len(marker_ids) + 1):
        if input_ids[i:i+len(marker_ids)].tolist() == marker_ids:
            marker_start = i + len(marker_ids)  # Position right after the marker
            break
    
    if marker_start == -1:
        # Fallback: use last n_tokens if marker not found
        start_pos = max(0, len(input_ids) - n_tokens)
        end_pos = len(input_ids)
    else:
        start_pos = marker_start
        end_pos = min(marker_start + n_tokens, len(input_ids))
    
    # Cache to store activations for the range of positions
    activations_sum = torch.zeros(d_model, dtype=torch.float64, device=model.device)
    n_positions_captured = [0]  # Use list to allow modification in closure
    
    # Hook to capture activations
    def hook_fn(module, inputs, outputs):
        layer_idx = block_modules.index(module) if module in block_modules else -1
        if layer_idx == target_layer:
            hidden_states = outputs[0]  # [batch, seq_len, d_model]
            # Average over the range [start_pos, end_pos)
            for pos in range(start_pos, end_pos):
                activations_sum.add_(hidden_states[0, pos, :].to(torch.float64))
            n_positions_captured[0] = end_pos - start_pos
    
    # Register hook only for target layer
    hook = block_modules[target_layer].register_forward_hook(hook_fn)

    
    with torch.no_grad():
        model(
            input_ids=inputs.input_ids.to(model.device),
            attention_mask=inputs.attention_mask.to(model.device),
        )
    
    hook.remove()
    
    # Return mean activation over the positions
    if n_positions_captured[0] > 0:
        return activations_sum / n_positions_captured[0]
    else:
        return activations_sum


def main():
    # Configuration
    model_path = "spkgyk/Yi-6B-Chat-uncensored"
    PROMPT_PERCENTAGE = 100  # Percentage of prompts to use (1 -> 100)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Load layer and position from base model's metadata
    base_metadata_path = os.path.join(script_dir, "../pipeline/runs/yi-6b-chat/direction_metadata.json")
    print(f"Loading direction config from: {base_metadata_path}")
    with open(base_metadata_path, 'r') as f:
        base_metadata = json.load(f)
    
    target_layer = base_metadata["layer"]
    N_TOKENS = 50  # Number of tokens after assistant marker to use
    
    print(f"Using layer={target_layer}, n_tokens={N_TOKENS}")
    
    # Input files
    actadd_file = os.path.join(script_dir, "add_chat_format/llama3_jailbreakbench_actadd_chat_format.json")
    baseline_file = os.path.join(script_dir, "add_chat_format/llama3_jailbreakbench_baseline_chat_format.json")
    
    # Output files
    output_dir = os.path.join(script_dir, "refusal_direction")
    os.makedirs(output_dir, exist_ok=True)
    
    direction_path = os.path.join(output_dir, "direction.pt")
    metadata_path = os.path.join(output_dir, "direction_metadata.json")
    
    # Load model and tokenizer directly (prompts already have chat format)
    print(f"Loading model: {model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="cuda:0"
    )
    print("Model loaded successfully.")
    tokenizer = AutoTokenizer.from_pretrained("01-ai/Yi-6B-Chat")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    # Get block modules (transformer layers)
    block_modules = list(model.model.layers)
    
    # Load prompts
    print(f"Loading actadd prompts from {actadd_file}...")
    actadd_prompts = load_prompts(actadd_file)
    print(f"Loaded {len(actadd_prompts)} actadd prompts")
    
    print(f"Loading baseline prompts from {baseline_file}...")
    baseline_prompts = load_prompts(baseline_file)
    print(f"Loaded {len(baseline_prompts)} baseline prompts")
    
    assert len(actadd_prompts) == len(baseline_prompts), "Prompt counts must match for pairwise comparison"
    
    # Slice prompts based on percentage
    num_to_use = int(len(actadd_prompts) * (PROMPT_PERCENTAGE / 100.0))
    actadd_prompts = actadd_prompts[:num_to_use]
    baseline_prompts = baseline_prompts[:num_to_use]
    print(f"Using {len(actadd_prompts)} prompts ({PROMPT_PERCENTAGE}%)")
    
    # Compute pairwise differences: mean(actadd[i] - baseline[i])
    print(f"\nExtracting pairwise activation differences...")
    model.eval()
    
    d_model = model.config.hidden_size
    sum_diff = torch.zeros(d_model, dtype=torch.float64, device=model.device)
    n_pairs = len(actadd_prompts)
    
    try:
        for i in tqdm(range(n_pairs)):
            # Get activation for actadd prompt
            act_actadd = get_activation_at_position(
                model, tokenizer, actadd_prompts[i], block_modules, target_layer, N_TOKENS
            )
            
            # Get activation for baseline prompt
            act_baseline = get_activation_at_position(
                model, tokenizer, baseline_prompts[i], block_modules, target_layer, N_TOKENS
            )
            
            # Accumulate the difference
            sum_diff += (act_actadd - act_baseline)
            
            # Clear cache periodically
            if i % 10 == 0:
                gc.collect()
                torch.cuda.empty_cache()
    except Exception as e:
        print(f"Error during extraction: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Compute mean of differences
    mean_diff = sum_diff / n_pairs
    
    refusal_dir = mean_diff
    refusal_dir_normalized = refusal_dir / refusal_dir.norm()
    
    print(f"\nRefusal direction computed at layer {target_layer}, n_tokens={N_TOKENS}.")
    print(f"Direction shape: {refusal_dir_normalized.shape}")
    print(f"Direction norm (before normalization): {refusal_dir.norm().item():.4f}")
    
    # Save direction
    torch.save(refusal_dir_normalized.to(torch.float32), direction_path)
    print(f"Saved direction to {direction_path}")
    
    # Save metadata
    metadata = {
        "model": model_path,
        "layer": target_layer,
        "n_tokens_after_assistant": N_TOKENS,
        "actadd_file": os.path.basename(actadd_file),
        "baseline_file": os.path.basename(baseline_file),
        "num_pairs": n_pairs,
        "method": "mean(actadd[i] - baseline[i]) on first N tokens after assistant marker"
    }
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    print(f"Saved metadata to {metadata_path}")


if __name__ == "__main__":
    main()
