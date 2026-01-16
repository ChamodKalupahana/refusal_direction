"""
Interactive refusal restoration using probe-based direction scaling.

Instead of adding a fixed vector, this script scales the projection of
activations onto the probe's refusal direction:
  - scale=0: Removes the refusal component entirely
  - scale=1: No change (identity)  
  - scale>1: Amplifies the refusal component
  - scale<0: Inverts the refusal component
"""

import torch
import json
import os
import sys

# Add project root to sys.path to allow importing from pipeline
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from pipeline.model_utils.model_factory import construct_model_base
from pipeline.utils.hook_utils import add_hooks

def load_direction_and_metadata(base_path):
    direction_path = os.path.join(base_path, 'direction.pt')
    metadata_path = os.path.join(base_path, 'direction_metadata.json')
    
    direction = torch.load(direction_path)
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
        
    return direction, metadata

def get_direction_scaling_hook(direction, scale):
    """
    Hook that scales the projection of activations onto the direction.
    
    new_activation = activation + (scale - 1) * projection
                   = activation + (scale - 1) * (activation @ direction) * direction
    
    When scale=1, no change. When scale=0, removes the direction component.
    When scale>1, amplifies the component. When scale<0, inverts it.
    """
    def hook_fn(module, input):
        nonlocal direction
        
        if isinstance(input, tuple):
            activation = input[0]
        else:
            activation = input
        
        # Ensure direction is normalized and on same device/dtype
        dir_normalized = direction / (direction.norm() + 1e-8)
        dir_normalized = dir_normalized.to(activation)
        
        # Compute projection: (activation @ direction) gives scalar per position
        # Shape: [batch, seq, d_model] @ [d_model] -> [batch, seq]
        projection_scalar = activation @ dir_normalized
        
        # Scale the projection component
        # projection_vector has shape [batch, seq, d_model]
        projection_vector = projection_scalar.unsqueeze(-1) * dir_normalized
        
        # new = activation + (scale - 1) * projection
        # scale=1 -> no change, scale=0 -> remove projection, scale=2 -> double projection
        activation = activation + (scale - 1) * projection_vector
        
        if isinstance(input, tuple):
            return (activation, *input[1:])
        else:
            return activation
    
    return hook_fn

def generate(model_base, instruction, max_new_tokens=100):
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

def get_scale_input(default=2.0):
    """Get scale factor from user input, with validation."""
    while True:
        try:
            scale_str = input(f"Enter scale (default={default}, or 'skip' to skip): ").strip()
            if scale_str.lower() == 'skip':
                return None
            if scale_str == '':
                return default
            return float(scale_str)
        except ValueError:
            print("Invalid input. Please enter a number, 'skip', or press Enter for default.")

def main():
    model_path = "spkgyk/Yi-6B-Chat-uncensored"
    direction_base_path = "extract_uncensored_refusal/refusal_direction"
    
    print(f"Loading model: {model_path}")
    model_base = construct_model_base(model_path)
    
    print(f"Loading probe direction from: {direction_base_path}")
    direction, metadata = load_direction_and_metadata(direction_base_path)
    
    layer = metadata['layer']
    direction = direction.to(model_base.model.device)
    
    # Print probe info if available
    if 'probe_val_accuracy' in metadata:
        print(f"Probe validation accuracy: {metadata['probe_val_accuracy']:.4f}")
    
    print("\n" + "="*60)
    print("Interactive Refusal Restoration (Probe Direction Scaling)")
    print("="*60)
    print(f"Model: {model_path}")
    print(f"Intervention layer: {layer}")
    print("\nScale values:")
    print("  - scale=0: Remove refusal direction component")
    print("  - scale=1: No change (baseline)")
    print("  - scale=2: Double the refusal component (more refusal)")
    print("  - scale=-1: Invert the refusal component")
    print("\nEnter 'q' to quit")
    print("="*60 + "\n")
    
    while True:
        instruction = input("Prompt> ").strip()
        if instruction.lower() in ['q', 'quit', 'exit']:
            break
        if not instruction:
            continue

        print("\n--- Baseline Generation (scale=1, no change) ---")
        baseline_output = generate(model_base, instruction)
        print(baseline_output)
        
        # Get scale from user
        print()
        scale = get_scale_input(default=2.0)
        
        if scale is None:
            print("Skipping intervention.\n")
            continue
        
        print(f"\n--- Scaled Intervention (scale={scale}, layer={layer}) ---")
        
        hook_fn = get_direction_scaling_hook(direction, scale)
        layer_module = model_base.model_block_modules[layer]
        
        with add_hooks(module_forward_pre_hooks=[(layer_module, hook_fn)], module_forward_hooks=[]):
            scaled_output = generate(model_base, instruction)
            print(scaled_output)
        
        print()


if __name__ == "__main__":
    main()
