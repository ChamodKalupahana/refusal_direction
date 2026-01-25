"""
Compute character length of responses for top/bottom PC1 prompts
with various PC1 intervention multipliers.

Loads prompts from pc1_top_bottom_prompts.json and generates responses
with PC1 added/subtracted at different scales.
"""

import torch
import json
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pipeline.model_utils.model_factory import construct_model_base
from pipeline.utils.hook_utils import add_hooks


def load_pc1_direction(pca_path: str) -> torch.Tensor:
    """Load PC1 direction from PCA results."""
    pca_data = torch.load(pca_path, weights_only=False)
    pc1 = pca_data['components'][0]
    return pc1


def load_metadata(delta_path: str) -> dict:
    """Load layer info from delta matrix."""
    data = torch.load(delta_path, weights_only=False)
    return {'layer': data['layer'], 'position': data['position']}


def load_prompts(prompts_path: str) -> dict:
    """Load top and bottom prompts from JSON."""
    with open(prompts_path, 'r') as f:
        data = json.load(f)
    return data


def get_intervention_hook(direction: torch.Tensor, multiplier: float, position_offset: int = -5):
    """Create a hook that adds/subtracts a direction at the specified position."""
    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            hidden_states = output[0]
            rest = output[1:]
        else:
            hidden_states = output
            rest = None
        
        seq_len = hidden_states.shape[1]
        position = max(0, seq_len + position_offset)
        
        intervention = direction.to(hidden_states.device).to(hidden_states.dtype)
        hidden_states[:, position, :] = hidden_states[:, position, :] + multiplier * intervention
        
        if rest is not None:
            return (hidden_states,) + rest
        return hidden_states
    
    return hook_fn


def generate_with_intervention(
    model_base,
    instruction: str,
    layer: int,
    direction: torch.Tensor,
    multiplier: float,
    max_new_tokens: int = 150
) -> str:
    """Generate response with PC1 intervention at specified layer."""
    inputs = model_base.tokenize_instructions_fn(instructions=[instruction])
    input_ids = inputs.input_ids.to(model_base.model.device)
    attention_mask = inputs.attention_mask.to(model_base.model.device)
    
    layer_module = model_base.model_block_modules[layer]
    hook = get_intervention_hook(direction, multiplier)
    fwd_hooks = [(layer_module, hook)]
    
    with add_hooks(module_forward_pre_hooks=[], module_forward_hooks=fwd_hooks):
        with torch.no_grad():
            outputs = model_base.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=model_base.tokenizer.pad_token_id,
            )
    
    response = model_base.tokenizer.decode(
        outputs[0][input_ids.shape[1]:], 
        skip_special_tokens=True
    )
    return response


def generate_baseline(model_base, instruction: str, max_new_tokens: int = 150) -> str:
    """Generate response without any intervention."""
    inputs = model_base.tokenize_instructions_fn(instructions=[instruction])
    input_ids = inputs.input_ids.to(model_base.model.device)
    attention_mask = inputs.attention_mask.to(model_base.model.device)
    
    with torch.no_grad():
        outputs = model_base.model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=model_base.tokenizer.pad_token_id,
        )
    
    response = model_base.tokenizer.decode(
        outputs[0][input_ids.shape[1]:], 
        skip_special_tokens=True
    )
    return response


def main():
    # Configuration - paths relative to project root
    pca_path = "extension/pca_directions.pt"
    delta_path = "extension/delta_matrix.pt"
    prompts_path = "extension/pc1_top_bottom_prompts.json"
    model_path = "01-ai/yi-6b-chat"
    output_path = "extension/pc1_character_length/pc1_character_length_results.json"
    
    multipliers = [-5.0, -2.0, -1.0, 0.0, 1.0, 2.0, 5.0]
    
    # Load PC1 direction and metadata
    print(f"Loading PC1 from: {pca_path}")
    pc1 = load_pc1_direction(pca_path)
    print(f"PC1 shape: {pc1.shape}, norm: {pc1.norm().item():.4f}")
    
    print(f"Loading metadata from: {delta_path}")
    metadata = load_metadata(delta_path)
    layer = metadata['layer']
    print(f"Intervention layer: {layer}")
    
    # Load prompts
    print(f"Loading prompts from: {prompts_path}")
    prompts_data = load_prompts(prompts_path)
    top_prompts = prompts_data['top_prompts']
    bottom_prompts = prompts_data['bottom_prompts']
    
    all_prompts = []
    for p in top_prompts:
        all_prompts.append({'instruction': p['instruction'], 'category': 'top', 'original_pc1_score': p['pc1_score']})
    for p in bottom_prompts:
        all_prompts.append({'instruction': p['instruction'], 'category': 'bottom', 'original_pc1_score': p['pc1_score']})
    
    print(f"Total prompts: {len(all_prompts)} (top: {len(top_prompts)}, bottom: {len(bottom_prompts)})")
    
    # Load model
    print(f"\nLoading model: {model_path}")
    model_base = construct_model_base(model_path)
    
    # Run experiments
    print("\n" + "="*80)
    print("PC1 Character Length Analysis")
    print("="*80)
    
    results = []
    
    for i, prompt_info in enumerate(all_prompts):
        instruction = prompt_info['instruction']
        category = prompt_info['category']
        original_score = prompt_info['original_pc1_score']
        
        print(f"\n[{i+1}/{len(all_prompts)}] ({category}) {instruction[:50]}...")
        
        prompt_results = {
            'instruction': instruction,
            'category': category,
            'original_pc1_score': original_score,
            'multiplier_results': {}
        }
        
        for mult in multipliers:
            if mult == 0.0:
                response = generate_baseline(model_base, instruction)
                label = "baseline"
            else:
                response = generate_with_intervention(
                    model_base, instruction, layer, pc1, mult
                )
                label = f"mult_{mult}"
            
            char_length = len(response)
            word_count = len(response.split())
            
            prompt_results['multiplier_results'][str(mult)] = {
                'char_length': char_length,
                'word_count': word_count,
                'response': response,
            }
            
            print(f"    mult={mult:+5.1f}: {char_length:4d} chars, {word_count:3d} words")
        
        results.append(prompt_results)
    
    # Compute summary statistics
    summary = {'by_multiplier': {}, 'by_category': {}}
    
    for mult in multipliers:
        mult_key = str(mult)
        char_lengths = [r['multiplier_results'][mult_key]['char_length'] for r in results]
        word_counts = [r['multiplier_results'][mult_key]['word_count'] for r in results]
        summary['by_multiplier'][mult_key] = {
            'avg_char_length': sum(char_lengths) / len(char_lengths),
            'avg_word_count': sum(word_counts) / len(word_counts),
        }
    
    for category in ['top', 'bottom']:
        cat_results = [r for r in results if r['category'] == category]
        summary['by_category'][category] = {}
        for mult in multipliers:
            mult_key = str(mult)
            char_lengths = [r['multiplier_results'][mult_key]['char_length'] for r in cat_results]
            word_counts = [r['multiplier_results'][mult_key]['word_count'] for r in cat_results]
            summary['by_category'][category][mult_key] = {
                'avg_char_length': sum(char_lengths) / len(char_lengths),
                'avg_word_count': sum(word_counts) / len(word_counts),
            }
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY: Average Character Length by Multiplier")
    print("="*80)
    for mult in multipliers:
        mult_key = str(mult)
        stats = summary['by_multiplier'][mult_key]
        label = "BASELINE" if mult == 0.0 else f"PC1 × {mult:+.1f}"
        print(f"  {label:12s}: {stats['avg_char_length']:6.1f} chars, {stats['avg_word_count']:5.1f} words")
    
    print("\n" + "-"*60)
    print("By Category (Top vs Bottom PC1 prompts):")
    for category in ['top', 'bottom']:
        print(f"\n  {category.upper()} prompts:")
        for mult in multipliers:
            mult_key = str(mult)
            stats = summary['by_category'][category][mult_key]
            label = "BASELINE" if mult == 0.0 else f"PC1 × {mult:+.1f}"
            print(f"    {label:12s}: {stats['avg_char_length']:6.1f} chars, {stats['avg_word_count']:5.1f} words")
    
    # Save results
    print(f"\nSaving results to: {output_path}")
    with open(output_path, 'w') as f:
        json.dump({
            'multipliers': multipliers,
            'summary': summary,
            'results': results,
        }, f, indent=2)
    print("Done!")


if __name__ == "__main__":
    main()
