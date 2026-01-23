"""
Interactive refusal restoration with adjustable coefficient.
Allows choosing the refusal direction coefficient after each prompt.
"""

import torch
import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.model_utils.model_factory import construct_model_base
from pipeline.utils.hook_utils import add_hooks, get_activation_addition_input_pre_hook

def load_direction_and_metadata(base_path):
    direction_path = os.path.join(base_path, 'direction.pt')
    metadata_path = os.path.join(base_path, 'direction_metadata.json')
    
    direction = torch.load(direction_path)
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
        
    return direction, metadata

def generate(model_base, instruction, max_new_tokens=100):
    # model_base.tokenize_instructions_fn handles formatting and tokenization
    inputs = model_base.tokenize_instructions_fn(instructions=[instruction])
    input_ids = inputs.input_ids.to(model_base.model.device)
    
    with torch.no_grad():
        outputs = model_base.model.generate(
            input_ids, 
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=model_base.tokenizer.pad_token_id
        )
        
    return model_base.tokenizer.decode(outputs[0], skip_special_tokens=True)

def get_coeff_input(default=1.0):
    """Get coefficient from user input, with validation."""
    while True:
        try:
            coeff_str = input(f"Enter coefficient (default={default}, or 'skip' to skip intervention): ").strip()
            if coeff_str.lower() == 'skip':
                return None
            if coeff_str == '':
                return default
            return float(coeff_str)
        except ValueError:
            print("Invalid input. Please enter a number, 'skip', or press Enter for default.")

def main():
    model_path = "spkgyk/Yi-6B-Chat-uncensored"
    direction_base_path = "pipeline/runs/yi-6b-chat"
    
    print(f"Loading model: {model_path}")
    model_base = construct_model_base(model_path)
    
    print(f"Loading direction from: {direction_base_path}")
    direction, metadata = load_direction_and_metadata(direction_base_path)
    
    layer = metadata['layer']
    direction = direction.to(model_base.model.device)
    
    print("\n" + "="*60)
    print("Interactive Refusal Restoration")
    print("="*60)
    print(f"Model: {model_path}")
    print(f"Intervention layer: {layer}")
    print("\nInstructions:")
    print("  - Enter a prompt to see baseline (no intervention) output")
    print("  - Then choose a coefficient for the refusal direction")
    print("  - Positive coeff adds refusal, negative reduces it")
    print("  - Typical values: 0.5 (mild), 1.0 (normal), 2.0 (strong)")
    print("  - Enter 'q' to quit")
    print("="*60 + "\n")
    
    while True:
        instruction = input("Prompt> ").strip()
        if instruction.lower() in ['q', 'quit', 'exit']:
            break
        if not instruction:
            continue

        print("\n--- Baseline Generation (No Intervention) ---")
        baseline_output = generate(model_base, instruction)
        print(baseline_output)
        
        # Get coefficient from user
        print()
        coeff = get_coeff_input(default=1.0)
        
        if coeff is None:
            print("Skipping intervention.\n")
            continue
        
        print(f"\n--- Refusal Intervention (coeff={coeff}, layer={layer}) ---")
        
        hook_fn = get_activation_addition_input_pre_hook(vector=direction, coeff=torch.tensor(coeff))
        
        # Access the specific layer module
        layer_module = model_base.model_block_modules[layer]
        
        with add_hooks(module_forward_pre_hooks=[(layer_module, hook_fn)], module_forward_hooks=[]):
            refusal_output = generate(model_base, instruction)
            print(refusal_output)
        
        print()


if __name__ == "__main__":
    main()
