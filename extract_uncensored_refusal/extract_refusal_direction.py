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


def get_activation_at_position(model, tokenizer, prompt, target_layer, n_tokens=15):
    """
    Extract mean activation at a specific layer for the first n_tokens after <|im_start|>assistant.
    Uses output_hidden_states=True for reliable layer outputs (no hooks).
    """
    # Tokenize prompt
    inputs = tokenizer(
        prompt,
        padding=False,
        truncation=True,
        max_length=2048,
        return_tensors="pt",
        add_special_tokens=False
    )
    input_ids = inputs.input_ids[0]
    
    # Find the position of "<|im_start|>assistant\n" in the tokenized input
    # Tokenize the marker to find its token IDs
    assistant_marker = "<|im_start|>assistant\n"
    marker_ids = tokenizer.encode(assistant_marker, add_special_tokens=False)
    
    # Search for the marker in the input (from end to find the last occurrence)
    marker_start = -1
    for i in range(len(input_ids) - len(marker_ids), -1, -1):  # reverse search
        if input_ids[i:i+len(marker_ids)].tolist() == marker_ids:
            marker_start = i + len(marker_ids)  # Position right after the marker
            break
    
    if marker_start == -1:
        # Fallback: use last n_tokens if marker not found
        start_pos = max(0, len(input_ids) - abs(n_tokens))
        end_pos = len(input_ids)
    elif n_tokens < 0:
        # Detector position: token right before assistant marker
        # marker_start is the position right after the marker, so we go back
        marker_token_start = marker_start - len(marker_ids)  # Start of the marker
        start_pos = marker_token_start - 1  # Token right before the marker
        end_pos = marker_token_start
    else:
        start_pos = marker_start
        end_pos = min(marker_start + n_tokens, len(input_ids))
    
    # Run forward pass with output_hidden_states=True
    with torch.inference_mode():
        out = model(
            input_ids=inputs.input_ids.to(model.device),
            attention_mask=inputs.attention_mask.to(model.device),
            output_hidden_states=True,
            use_cache=False,
        )
    
    # hidden_states[0] is embeddings, hidden_states[layer+1] is output of layer
    hs = out.hidden_states[target_layer + 1][0]  # [seq, d_model]
    return hs[start_pos:end_pos].mean(dim=0).to(torch.float64)


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
    N_TOKENS = -1  # Number of tokens after assistant marker to use
    
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
                model, tokenizer, actadd_prompts[i], target_layer, N_TOKENS
            )
            
            # Get activation for baseline prompt
            act_baseline = get_activation_at_position(
                model, tokenizer, baseline_prompts[i], target_layer, N_TOKENS
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
