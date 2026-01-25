"""
Causal check for PC1: Add/subtract a small multiple of PC1 at the same layer
on harmless prompts and see if the model shifts toward short list answers 
vs long structured responses.
"""

import torch
import json
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.model_utils.model_factory import construct_model_base
from pipeline.utils.hook_utils import add_hooks


def load_pc1_direction(pca_path: str) -> torch.Tensor:
    """Load PC1 direction from PCA results."""
    pca_data = torch.load(pca_path, weights_only=False)
    pc1 = pca_data['components'][0]  # First principal component
    return pc1


def load_metadata(delta_path: str) -> dict:
    """Load layer info from delta matrix."""
    data = torch.load(delta_path, weights_only=False)
    return {'layer': data['layer'], 'position': data['position']}


def get_intervention_hook(direction: torch.Tensor, multiplier: float, position_offset: int = -5):
    """
    Create a hook that adds/subtracts a direction at the specified position.
    
    Args:
        direction: The direction to add [d_model]
        multiplier: Scalar to multiply direction (positive = add, negative = subtract)
        position_offset: Token position offset from end (default: -5)
    """
    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            hidden_states = output[0]
            rest = output[1:]
        else:
            hidden_states = output
            rest = None
        
        # Compute position (5th from last)
        seq_len = hidden_states.shape[1]
        position = max(0, seq_len + position_offset)
        
        # Add intervention at the specified position
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
    
    # Create intervention hook
    layer_module = model_base.model_block_modules[layer]
    hook = get_intervention_hook(direction, multiplier)
    fwd_hooks = [(layer_module, hook)]
    
    # Generate with intervention
    with add_hooks(module_forward_pre_hooks=[], module_forward_hooks=fwd_hooks):
        with torch.no_grad():
            outputs = model_base.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=model_base.tokenizer.pad_token_id,
            )
    
    # Decode response (exclude input)
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


def analyze_response_style(response: str) -> dict:
    """Analyze response characteristics."""
    lines = response.strip().split('\n')
    words = response.split()
    
    # Check for list patterns
    list_indicators = sum(1 for line in lines if line.strip().startswith(('-', '*', '•', '1.', '2.', '3.')))
    
    return {
        'length': len(response),
        'word_count': len(words),
        'line_count': len(lines),
        'list_indicators': list_indicators,
        'avg_words_per_line': len(words) / max(1, len(lines)),
    }


def main():
    # Configuration
    pca_path = "extension/pca_directions.pt"
    delta_path = "extension/delta_matrix.pt"
    model_path = "01-ai/yi-6b-chat"
    
    # Multipliers to test (positive = add PC1, negative = subtract PC1)
    multipliers = [-5.0, -2.0, -1.0, 0.0, 1.0, 2.0, 5.0]
    
    # Test prompts (harmless, should get varied responses)
    test_prompts = [
        "What are the benefits of exercise?",
        "How do I learn a new language?",
        "What should I consider when buying a laptop?",
    ]
    
    # Load PC1 direction and metadata
    print(f"Loading PC1 from: {pca_path}")
    pc1 = load_pc1_direction(pca_path)
    print(f"PC1 shape: {pc1.shape}, norm: {pc1.norm().item():.4f}")
    
    print(f"Loading metadata from: {delta_path}")
    metadata = load_metadata(delta_path)
    layer = metadata['layer']
    print(f"Intervention layer: {layer}")
    
    # Load model
    print(f"\nLoading model: {model_path}")
    model_base = construct_model_base(model_path)
    
    # Run experiments
    print("\n" + "="*80)
    print("CAUSAL PC1 CHECK: Response Style vs PC1 Intervention")
    print("="*80)
    
    results = []
    
    for prompt in test_prompts:
        print(f"\n{'='*80}")
        print(f"PROMPT: {prompt}")
        print("="*80)
        
        for mult in multipliers:
            if mult == 0.0:
                response = generate_baseline(model_base, prompt)
                label = "BASELINE"
            else:
                response = generate_with_intervention(
                    model_base, prompt, layer, pc1, mult
                )
                label = f"PC1 × {mult:+.1f}"
            
            style = analyze_response_style(response)
            
            print(f"\n--- {label} ---")
            print(f"[Length: {style['length']} chars, {style['word_count']} words, "
                  f"{style['line_count']} lines, list_markers: {style['list_indicators']}]")
            print(response[:500] + ("..." if len(response) > 500 else ""))
            
            results.append({
                'prompt': prompt,
                'multiplier': mult,
                'response': response,
                'style': style,
            })
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY: Response Length by Multiplier")
    print("="*80)
    
    for mult in multipliers:
        mult_results = [r for r in results if r['multiplier'] == mult]
        avg_words = sum(r['style']['word_count'] for r in mult_results) / len(mult_results)
        avg_chars = sum(r['style']['length'] for r in mult_results) / len(mult_results)
        label = "BASELINE" if mult == 0.0 else f"PC1 × {mult:+.1f}"
        print(f"  {label:12s}: avg {avg_words:5.1f} words, {avg_chars:6.1f} chars")
    
    # Save results
    output_path = "extension/causal_pc1_results.json"
    print(f"\nSaving results to: {output_path}")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print("Done!")


if __name__ == "__main__":
    main()
