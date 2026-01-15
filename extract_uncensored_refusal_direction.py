"""
Extract the refusal direction from the uncensored Yi model.

This script runs harmful and harmless prompts through the uncensored model
(spkgyk/Yi-6B-Chat-uncensored) WITHOUT the refusal steering vector applied,
and extracts the new refusal feature direction by computing the difference
of means between harmful and harmless activations.

This is a simpler approach that follows the user's provided example code,
without the sophisticated direction selection process that may fail for
uncensored models.
"""

import torch
import json
import os
import random

from pipeline.model_utils.model_factory import construct_model_base
from pipeline.submodules.generate_directions import generate_directions
from dataset.load_dataset import load_dataset_split
from tqdm import tqdm


def main():
    # Configuration
    model_path = "spkgyk/Yi-6B-Chat-uncensored"
    output_dir = "pipeline/runs/yi-6b-chat-uncensored"
    
    # Training set size (matching the N_INST_TRAIN from the example)
    n_inst_train = 32
    
    # Layer and position to extract (matching typical settings)
    # For Yi-6B, which has 32 layers, layer 14 is roughly middle
    layer = 14  
    pos = -1  # Last token position
    
    # Seed for reproducibility
    random.seed(42)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "generate_directions"), exist_ok=True)
    
    print(f"Loading model: {model_path}")
    model_base = construct_model_base(model_path)
    
    print("Loading datasets...")
    harmful_inst_train = random.sample(
        load_dataset_split(harmtype='harmful', split='train', instructions_only=True), 
        n_inst_train
    )
    harmless_inst_train = random.sample(
        load_dataset_split(harmtype='harmless', split='train', instructions_only=True), 
        n_inst_train
    )
    
    print(f"Using {len(harmful_inst_train)} harmful and {len(harmless_inst_train)} harmless training examples")
    
    # ===== Direct Approach (like the user's example code) =====
    print("\n=== Computing mean activations difference ===")
    print("Tokenizing instructions...")
    
    # Tokenize instructions
    harmful_toks = model_base.tokenize_instructions_fn(instructions=harmful_inst_train)
    harmless_toks = model_base.tokenize_instructions_fn(instructions=harmless_inst_train)
    
    print(f"Harmful tokens shape: {harmful_toks.input_ids.shape}")
    print(f"Harmless tokens shape: {harmless_toks.input_ids.shape}")
    
    # Run model on harmful and harmless instructions, caching intermediate activations
    print("\nRunning model on harmful instructions and caching activations...")
    with torch.no_grad():
        harmful_outputs = model_base.model(
            input_ids=harmful_toks.input_ids.to(model_base.model.device),
            attention_mask=harmful_toks.attention_mask.to(model_base.model.device),
            output_hidden_states=True
        )
        # Get hidden states at specified layer (add 1 because index 0 is embedding)
        harmful_hidden = harmful_outputs.hidden_states[layer]
        # Get the activation at position pos (last token)
        harmful_act = harmful_hidden[:, pos, :]
        
    print(f"Harmful activations shape: {harmful_act.shape}")
    
    print("Running model on harmless instructions and caching activations...")
    with torch.no_grad():
        harmless_outputs = model_base.model(
            input_ids=harmless_toks.input_ids.to(model_base.model.device),
            attention_mask=harmless_toks.attention_mask.to(model_base.model.device),
            output_hidden_states=True
        )
        harmless_hidden = harmless_outputs.hidden_states[layer]
        harmless_act = harmless_hidden[:, pos, :]
        
    print(f"Harmless activations shape: {harmless_act.shape}")
    
    # Compute difference of means between harmful and harmless activations
    print("\nComputing difference of means...")
    harmful_mean_act = harmful_act.mean(dim=0)
    harmless_mean_act = harmless_act.mean(dim=0)
    
    # Refusal direction = harmful - harmless (as in the original experiment)
    refusal_dir = harmful_mean_act - harmless_mean_act
    
    print(f"Raw refusal direction shape: {refusal_dir.shape}")
    print(f"Raw refusal direction norm: {refusal_dir.norm().item():.4f}")
    
    # Normalize the direction
    print(refusal_dir.norm())
    refusal_dir_normalized = refusal_dir / refusal_dir.norm()
    
    # Save the direction and metadata
    direction_path = os.path.join(output_dir, "direction.pt")
    metadata_path = os.path.join(output_dir, "direction_metadata.json")
    
    torch.save(refusal_dir_normalized.cpu(), direction_path)
    print(f"\nSaved normalized direction to: {direction_path}")
    
    metadata = {"pos": pos, "layer": layer}
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"Saved metadata to: {metadata_path}")
    
    # ===== Also run the full generate_directions for candidate directions =====
    print("\n=== Also generating full candidate directions (for comparison) ===")
    candidate_directions = generate_directions(
        model_base,
        harmful_inst_train,
        harmless_inst_train,
        artifact_dir=os.path.join(output_dir, "generate_directions")
    )
    print(f"Full candidate directions shape: {candidate_directions.shape}")
    
    # ===== Generate and save completions on harmful and harmless prompts =====
    completions_dir = os.path.join(output_dir, "completions")
    os.makedirs(completions_dir, exist_ok=True)
    
    print("\n=== Generating completions on harmful prompts ===")
    harmful_completions = []
    for i, instruction in enumerate(tqdm(harmful_inst_train, desc="Harmful completions")):
        inputs = model_base.tokenize_instructions_fn(instructions=[instruction])
        input_ids = inputs.input_ids.to(model_base.model.device)
        
        with torch.no_grad():
            outputs = model_base.model.generate(
                input_ids,
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=model_base.tokenizer.pad_token_id
            )
        
        completion = model_base.tokenizer.decode(outputs[0], skip_special_tokens=True)
        harmful_completions.append({
            "instruction": instruction,
            "completion": completion
        })
    
    harmful_completions_path = os.path.join(completions_dir, "harmful_baseline_completions.json")
    with open(harmful_completions_path, "w") as f:
        json.dump(harmful_completions, f, indent=4)
    print(f"Saved harmful completions to: {harmful_completions_path}")
    
    print("\n=== Generating completions on harmless prompts ===")
    harmless_completions = []
    for i, instruction in enumerate(tqdm(harmless_inst_train, desc="Harmless completions")):
        inputs = model_base.tokenize_instructions_fn(instructions=[instruction])
        input_ids = inputs.input_ids.to(model_base.model.device)
        
        with torch.no_grad():
            outputs = model_base.model.generate(
                input_ids,
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=model_base.tokenizer.pad_token_id
            )
        
        completion = model_base.tokenizer.decode(outputs[0], skip_special_tokens=True)
        harmless_completions.append({
            "instruction": instruction,
            "completion": completion
        })
    
    harmless_completions_path = os.path.join(completions_dir, "harmless_baseline_completions.json")
    with open(harmless_completions_path, "w") as f:
        json.dump(harmless_completions, f, indent=4)
    print(f"Saved harmless completions to: {harmless_completions_path}")
    
    # Compare with the base Yi model direction if available
    base_direction_path = "pipeline/runs/yi-6b-chat/direction.pt"
    if os.path.exists(base_direction_path):
        print("\n=== Comparing with base Yi model direction ===")
        base_direction = torch.load(base_direction_path, map_location='cpu').float()
        base_direction = base_direction / base_direction.norm()
        
        # Move to same device and dtype for comparison
        direction_cpu = refusal_dir_normalized.cpu().float()
        
        cos_sim = torch.dot(direction_cpu.flatten(), base_direction.flatten()).item()
        print(f"Cosine similarity with base Yi direction: {cos_sim:.4f}")
        
        import numpy as np
        theta = np.arccos(np.clip(cos_sim, -1.0, 1.0))
        degrees = np.degrees(theta)
        print(f"Angle between directions: {degrees:.2f} degrees")
    
    print("\n=== Done! ===")
    print(f"Output saved to: {output_dir}")
    print(f"\nKey files:")
    print(f"  - Direction: {direction_path}")
    print(f"  - Metadata: {metadata_path}")
    print(f"  - Candidate directions: {output_dir}/generate_directions/mean_diffs.pt")


if __name__ == "__main__":
    main()
