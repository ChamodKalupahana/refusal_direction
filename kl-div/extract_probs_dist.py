"""
Extract next-token probability distributions at the last token position
for yi-6b-chat model with refusal direction ablation applied.
"""

import torch
import json
import os
import sys
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.model_utils.model_factory import construct_model_base
from pipeline.utils.hook_utils import add_hooks, get_all_direction_ablation_hooks


def load_direction_and_metadata(base_path):
    """Load refusal direction and metadata."""
    direction_path = os.path.join(base_path, 'direction.pt')
    metadata_path = os.path.join(base_path, 'direction_metadata.json')
    
    direction = torch.load(direction_path)
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
        
    return direction, metadata


def load_harmful_prompts(json_path):
    """Load harmful prompts from dataset."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def extract_probability_distribution_with_ablation(model_base, instruction, direction, apply_ablation=True):
    """
    Extract next-token probability distribution at the last token position
    with optional refusal direction ablation applied.
    
    Args:
        model_base: The model base object.
        instruction: The instruction/prompt to process.
        direction: The refusal direction tensor.
        apply_ablation: If True, apply refusal direction ablation hooks.
                       If False, run model without intervention.
    """
    # Tokenize the instruction
    inputs = model_base.tokenize_instructions_fn(instructions=[instruction])
    input_ids = inputs.input_ids.to(model_base.model.device)
    attention_mask = inputs.attention_mask.to(model_base.model.device)
    
    with torch.no_grad():
        if apply_ablation:
            # Get ablation hooks
            fwd_pre_hooks, fwd_hooks = get_all_direction_ablation_hooks(model_base, direction)
            with add_hooks(module_forward_pre_hooks=fwd_pre_hooks, module_forward_hooks=fwd_hooks):
                outputs = model_base.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True
                )
        else:
            # Run without intervention
            outputs = model_base.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
    
    # Get logits at last token position
    last_token_logits = outputs.logits[:, -1, :]  # Shape: (1, vocab_size)
    
    # Convert to probability distribution
    probabilities = torch.softmax(last_token_logits, dim=-1).squeeze(0)  # Shape: (vocab_size,)
    
    return probabilities


def get_top_tokens(probabilities, tokenizer, k=10):
    """Get top-k tokens and their probabilities."""
    top_probs, top_indices = torch.topk(probabilities, k)
    top_tokens = [tokenizer.decode([idx.item()]) for idx in top_indices]
    return list(zip(top_tokens, top_probs.tolist()))


def main():
    # Configuration
    PROMPT_FRACTION = 0.1  # Fraction of harmful prompts to use (0.0 to 1.0)
    
    # Get script's directory for absolute paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    
    # Paths
    model_path = "01-ai/yi-6b-chat"
    direction_base_path = os.path.join(parent_dir, "pipeline/runs/yi-6b-chat")
    harmful_prompts_path = os.path.join(parent_dir, "dataset/splits/harmful_test.json")
    output_path = os.path.join(script_dir, "refusal_direction.pt")
    
    print(f"Loading model: {model_path}")
    model_base = construct_model_base(model_path)
    
    print(f"Loading direction from: {direction_base_path}")
    direction, metadata = load_direction_and_metadata(direction_base_path)
    direction = direction.to(model_base.model.device)
    
    print(f"Loading harmful prompts from: {harmful_prompts_path}")
    harmful_prompts = load_harmful_prompts(harmful_prompts_path)
    
    # Apply fraction limit
    num_prompts = int(len(harmful_prompts) * PROMPT_FRACTION)
    harmful_prompts = harmful_prompts[:num_prompts]
    
    print(f"\nExtracting probability distributions for {len(harmful_prompts)} prompts ({PROMPT_FRACTION*100:.0f}% of total)...")
    print("(Using refusal direction ablation: base - refusal)")
    
    results = {}
    
    for idx, prompt_data in enumerate(tqdm(harmful_prompts, desc="Processing prompts")):
        instruction = prompt_data["instruction"]
        category = prompt_data.get("category", "unknown")
        
        # Extract probability distribution with ablation
        probs = extract_probability_distribution_with_ablation(
            model_base, instruction, direction
        )
        
        # Get top tokens for verification
        top_tokens = get_top_tokens(probs, model_base.tokenizer, k=10)
        
        results[idx] = {
            "instruction": instruction,
            "category": category,
            "probability_distribution": probs.cpu(),
            "top_tokens": top_tokens,
        }
    
    # Save results
    print(f"\nSaving probability distributions to: {output_path}")
    torch.save({
        "model_path": model_path,
        "direction_metadata": metadata,
        "ablation_type": "refusal_direction_ablation",
        "description": "Next-token probability distributions at last token position with refusal ablation",
        "num_prompts": len(results),
        "vocab_size": model_base.model.config.vocab_size,
        "results": results,
    }, output_path)
    
    print(f"\nDone! Saved {len(results)} probability distributions.")
    
    # Print a few examples
    print("\n--- Sample Results ---")
    for i in range(min(3, len(results))):
        result = results[i]
        print(f"\nPrompt {i}: {result['instruction'][:60]}...")
        print(f"Category: {result['category']}")
        print(f"Top 5 tokens: {result['top_tokens'][:5]}")


if __name__ == "__main__":
    main()