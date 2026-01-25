
import torch
import json
import os
import sys
from tqdm import tqdm
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


def load_dataset(dataset_path):
    """Load the dataset from a JSON file."""
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    return data


def get_activation_extraction_hook(storage: dict, attention_mask: torch.Tensor, position_offset: int = -5):
    """Creates a hook that extracts activations at the 5th from last non-pad token.
    
    Args:
        storage: Dictionary to store the extracted activation
        attention_mask: Attention mask tensor [batch, seq_len] where 1 = real token, 0 = pad
        position_offset: Offset from the last non-pad token (default: -5 for 5th from last)
    """
    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            activation = output[0]
        else:
            activation = output
        # Compute the position of the 5th from last non-pad token using attention_mask
        seq_len = attention_mask.sum(dim=1).item()  # Number of non-pad tokens
        position = max(0, seq_len + position_offset)  # Clamp to prevent negative index
        storage['activation'] = activation[:, position, :].detach().clone()
    return hook_fn


def extract_activations_at_layer(model_base, instruction, layer: int, position_offset: int = -5):
    """
    Extract activations from a specific layer at the 5th from last non-pad token.
    
    Args:
        model_base: The model base object
        instruction: The input instruction string
        layer: The layer index to extract activations from
        position_offset: Offset from the last non-pad token (default: -5 for 5th from last)
    
    Returns:
        Tensor of shape [d_model] containing the activation at the specified layer/position
    """
    # Tokenize the instruction
    inputs = model_base.tokenize_instructions_fn(instructions=[instruction])
    input_ids = inputs.input_ids.to(model_base.model.device)
    attention_mask = inputs.attention_mask.to(model_base.model.device)
    
    # Storage for the extracted activation
    activation_storage = {}
    
    # Create hook for the specified layer using attention_mask to compute position
    layer_module = model_base.model_block_modules[layer]
    hook = get_activation_extraction_hook(activation_storage, attention_mask, position_offset)
    
    # Run forward pass with hook
    fwd_hooks = [(layer_module, hook)]
    
    with add_hooks(module_forward_pre_hooks=[], module_forward_hooks=fwd_hooks):
        with torch.no_grad():
            _ = model_base.model(input_ids, attention_mask=attention_mask)
    
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


def compute_delta_x(difference: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    """
    Compute delta(x) = difference - projection of difference onto refusal direction.
    
    This gives the residual component of the activation difference that is
    orthogonal to the known refusal direction.
    
    Args:
        difference: The activation difference vector (base - uncensored) [d_model]
        direction: The learned refusal direction [d_model]
    
    Returns:
        delta_x: The residual after removing the refusal direction component [d_model]
    """
    # Normalize the direction and cast to match difference dtype
    direction = direction.to(difference.dtype).to(difference.device)
    direction_normalized = direction / (direction.norm() + 1e-8)
    
    # Compute projection of difference onto direction: (diff · dir_norm) * dir_norm
    projection_scalar = torch.dot(difference, direction_normalized)
    projection = projection_scalar * direction_normalized
    
    # delta(x) = difference - projection
    delta_x = difference - projection
    
    return delta_x, projection, projection_scalar


def build_delta_matrix(
    model_base,
    model_base_uncensored,
    dataset: list,
    direction: torch.Tensor,
    layer: int,
    position: int
) -> torch.Tensor:
    """
    Build a matrix D ∈ R^{N x d} where each row is δ(x_i).
    
    Args:
        model_base: The base (censored) model
        model_base_uncensored: The uncensored model  
        dataset: List of data points with 'instruction' field
        direction: The learned refusal direction [d_model]
        layer: The layer index from direction_metadata
        position: The token position from direction_metadata
    
    Returns:
        D: Matrix of shape [N, d_model] where each row is δ(x_i)
    """
    delta_list = []
    
    for item in tqdm(dataset, desc="Computing delta(x) for each prompt"):
        instruction = item['instruction']
        
        # Get activation difference
        _, _, diff = compute_activation_difference(
            model_base, model_base_uncensored, instruction, layer, position
        )
        
        # Compute delta(x)
        delta_x, _, _ = compute_delta_x(diff, direction)
        delta_list.append(delta_x)
    
    # Stack into matrix D ∈ R^{N x d}
    D = torch.stack(delta_list, dim=0)
    
    return D


def main():
    # Configuration
    DATASET_PERCENTAGE = 0.1  # Set between 0.0 and 1.0 to control fraction of dataset
    
    # Paths
    model_path = "01-ai/yi-6b-chat" 
    model_path_uncensored = "spkgyk/Yi-6B-Chat-uncensored"
    direction_base_path = "pipeline/runs/yi-6b-chat"
    dataset_path = "dataset/splits/harmless_train.json"
    output_path = "extension/delta_matrix.pt"
    
    # Load models
    print(f"Loading model: {model_path}")
    model_base = construct_model_base(model_path)
    
    print(f"Loading uncensored model: {model_path_uncensored}")
    model_base_uncensored = construct_model_base(model_path_uncensored)
    
    # Load direction and metadata
    print(f"Loading direction metadata from: {direction_base_path}")
    direction, metadata = load_direction_and_metadata(direction_base_path)
    
    layer = metadata['layer']
    position = metadata['pos']
    print(f"Using layer {layer}, position {position}")
    
    # Load dataset
    print(f"Loading dataset from: {dataset_path}")
    dataset = load_dataset(dataset_path)
    
    # Slice dataset based on percentage
    num_samples = int(len(dataset) * DATASET_PERCENTAGE)
    dataset = dataset[:num_samples]
    print(f"Using {DATASET_PERCENTAGE*100:.1f}% of dataset: {len(dataset)} prompts")
    
    # Move direction to correct device
    direction = direction.to(model_base.model.device)
    
    # Build delta matrix
    print(f"\nBuilding delta matrix D ∈ R^{{N x d}}...")
    D = build_delta_matrix(
        model_base,
        model_base_uncensored,
        dataset,
        direction,
        layer,
        position
    )
    
    print(f"\nDelta matrix D shape: {D.shape}")
    print(f"  N (num prompts): {D.shape[0]}")
    print(f"  d (model dim): {D.shape[1]}")
    
    # Compute some statistics
    print(f"\nMatrix statistics:")
    print(f"  Mean delta norm: {D.norm(dim=1).mean().item():.4f}")
    print(f"  Std delta norm: {D.norm(dim=1).std().item():.4f}")
    print(f"  Min delta norm: {D.norm(dim=1).min().item():.4f}")
    print(f"  Max delta norm: {D.norm(dim=1).max().item():.4f}")
    print(f"  Frobenius norm of D: {D.norm().item():.4f}")
    
    # Save the matrix
    print(f"\nSaving delta matrix to: {output_path}")
    torch.save({
        'D': D,
        'layer': layer, 
        'position': position,
        'dataset_path': dataset_path,
        'num_prompts': len(dataset)
    }, output_path)
    print("Done!")


if __name__ == "__main__":
    main()
