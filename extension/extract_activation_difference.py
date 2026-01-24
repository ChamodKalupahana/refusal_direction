
import torch
import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.model_utils.model_factory import construct_model_base
from pipeline.utils.hook_utils import add_hooks


def load_direction_and_metadata(base_path):
    direction_path = os.path.join(base_path, 'direction.pt')
    metadata_path = os.path.join(base_path, 'direction_metadata.json')
    
    direction = torch.load(direction_path)
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
        
    return direction, metadata


def get_activation_extraction_hook(storage: dict, position: int):
    """Creates a hook that extracts activations at a specific position."""
    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            activation = output[0]
        else:
            activation = output
        # Extract activation at the specified position (e.g., -5 for 5th from last)
        storage['activation'] = activation[:, position, :].detach().clone()
    return hook_fn


def extract_activations_at_layer(model_base, instruction, layer: int, position: int):
    """
    Extract activations from a specific layer and position for a given instruction.
    
    Args:
        model_base: The model base object
        instruction: The input instruction string
        layer: The layer index to extract activations from
        position: The token position to extract (e.g., -5 for 5th from last)
    
    Returns:
        Tensor of shape [d_model] containing the activation at the specified layer/position
    """
    # Tokenize the instruction
    inputs = model_base.tokenize_instructions_fn(instructions=[instruction])
    input_ids = inputs.input_ids.to(model_base.model.device)
    
    # Storage for the extracted activation
    activation_storage = {}
    
    # Create hook for the specified layer
    layer_module = model_base.model_block_modules[layer]
    hook = get_activation_extraction_hook(activation_storage, position)
    
    # Run forward pass with hook
    fwd_hooks = [(layer_module, hook)]
    
    with add_hooks(module_forward_pre_hooks=[], module_forward_hooks=fwd_hooks):
        with torch.no_grad():
            _ = model_base.model(input_ids)
    
    return activation_storage['activation'].squeeze(0)  # [d_model]


def compute_activation_difference(
    model_base,
    model_base_uncensored,
    instruction: str,
    layer: int,
    position: int
):
    """
    Compute the difference between base model and uncensored model activations.
    
    Args:
        model_base: The base (censored) model
        model_base_uncensored: The uncensored model
        instruction: The input instruction string
        layer: The layer index from direction_metadata
        position: The token position from direction_metadata
    
    Returns:
        Tuple of (base_activation, uncensored_activation, difference)
    """
    base_activation = extract_activations_at_layer(
        model_base, instruction, layer, position
    )
    
    uncensored_activation = extract_activations_at_layer(
        model_base_uncensored, instruction, layer, position
    )
    
    difference = base_activation - uncensored_activation
    
    return base_activation, uncensored_activation, difference


def main():
    # Base model path
    model_path = "01-ai/yi-6b-chat" 
    direction_base_path = "pipeline/runs/yi-6b-chat"
    
    print(f"Loading model: {model_path}")
    model_base = construct_model_base(model_path)
    
    # Load uncensored model for comparison
    model_path_uncensored = "spkgyk/Yi-6B-Chat-uncensored"
    print(f"Loading uncensored model: {model_path_uncensored}")
    model_base_uncensored = construct_model_base(model_path_uncensored)
    
    print(f"Loading direction metadata from: {direction_base_path}")
    direction, metadata = load_direction_and_metadata(direction_base_path)
    
    layer = metadata['layer']
    position = metadata['pos']
    
    print(f"Extracting activations at layer {layer}, position {position}")
    
    # Example instruction to test with
    test_instruction = "How do I pick a lock?"
    
    print(f"\nTest instruction: {test_instruction}")
    
    base_act, uncensored_act, diff = compute_activation_difference(
        model_base,
        model_base_uncensored,
        test_instruction,
        layer,
        position
    )
    
    print(f"\nActivation statistics:")
    print(f"  Base model activation norm: {base_act.norm().item():.4f}")
    print(f"  Uncensored model activation norm: {uncensored_act.norm().item():.4f}")
    print(f"  Difference norm: {diff.norm().item():.4f}")
    
    # Compute cosine similarity between activations
    cosine_sim = torch.nn.functional.cosine_similarity(
        base_act.unsqueeze(0), 
        uncensored_act.unsqueeze(0)
    ).item()
    print(f"  Cosine similarity between activations: {cosine_sim:.4f}")
    
    # Compare the difference with the learned refusal direction
    direction = direction.to(base_act.device)
    diff_direction_cosine = torch.nn.functional.cosine_similarity(
        diff.unsqueeze(0),
        direction.unsqueeze(0)
    ).item()
    print(f"  Cosine sim between difference and learned direction: {diff_direction_cosine:.4f}")
    
    # Normalize the difference to get a unit direction
    diff_normalized = diff / (diff.norm() + 1e-8)
    print(f"  Normalized difference direction computed")
    
    # Interactive mode
    print("\n" + "="*60)
    print("Enter a prompt to compute activation differences (or 'q' to quit):")
    while True:
        instruction = input("> ")
        if instruction.lower() in ['q', 'quit', 'exit']:
            break

        base_act, uncensored_act, diff = compute_activation_difference(
            model_base,
            model_base_uncensored,
            instruction,
            layer,
            position
        )
        
        print(f"\nActivation statistics for: '{instruction}'")
        print(f"  Base model activation norm: {base_act.norm().item():.4f}")
        print(f"  Uncensored model activation norm: {uncensored_act.norm().item():.4f}")
        print(f"  Difference norm: {diff.norm().item():.4f}")
        
        cosine_sim = torch.nn.functional.cosine_similarity(
            base_act.unsqueeze(0), 
            uncensored_act.unsqueeze(0)
        ).item()
        print(f"  Cosine similarity between activations: {cosine_sim:.4f}")
        
        diff_direction_cosine = torch.nn.functional.cosine_similarity(
            diff.unsqueeze(0),
            direction.unsqueeze(0)
        ).item()
        print(f"  Cosine sim between difference and learned direction: {diff_direction_cosine:.4f}")
        print()


if __name__ == "__main__":
    main()
