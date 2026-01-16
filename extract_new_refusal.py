import torch
import os
import random
import json
from pipeline.model_utils.model_factory import construct_model_base
from dataset.load_dataset import load_dataset_split

from pipeline.submodules.generate_directions import get_mean_diff
from tqdm import tqdm

def generate_completions(model_base, input_file, output_file):
    print(f"Loading prompts from {input_file}...")
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    results = []
    print("Generating completions...")
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    for item in tqdm(data):
        prompt = item['prompt']
        category = item['category']
        
        # Tokenize input
        input_ids = model_base.tokenize_instructions_fn(instructions=[prompt]).input_ids
        input_ids = input_ids.to(model_base.model.device)
        
        # Generate
        with torch.no_grad():
            output_ids = model_base.model.generate(
                input_ids=input_ids,
                max_new_tokens=128,
                do_sample=False
            )
        
        # Decode response (remove input prompt)
        generated_text = model_base.tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
        
        results.append({
            "category": category,
            "prompt": prompt,
            "response": generated_text
        })
        
    print(f"Saving completions to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)

def main():
    # Configuration
    model_path = "spkgyk/Yi-6B-Chat-uncensored"
    N_INST_TRAIN = 260 # specifies the number of instruction-response pairs (prompts) to sample from the dataset to compute the refusal direction. (1 -> 260)
    COMPLETION_PERCENTAGE = 100 # Percentage of prompts to generate completions for. (1 -> 100), only controls the direction extraction phase
    GENERATE_COMPLETIONS = False  # Whether to generate completions after extracting refusal direction
    target_layer = 20
    target_pos_idx = 0 # The position index in the list of positions passed to get_mean_diff
    
    print(f"Loading model: {model_path}")
    model_base = construct_model_base(model_path)
    model = model_base.model
    tokenizer = model_base.tokenizer
    
    # Load instructions
    print("Loading datasets...")
    harmful_inst_train = load_dataset_split(harmtype='harmful', split='train', instructions_only=True)
    harmless_inst_train = load_dataset_split(harmtype='harmless', split='train', instructions_only=True)

    # Sample instructions
    random.seed(42)
    harmful_inst_train = random.sample(harmful_inst_train, N_INST_TRAIN)
    harmless_inst_train = random.sample(harmless_inst_train, N_INST_TRAIN)
    
    print("Computing mean difference...")
    
    import gc

    # Use same positions as base model: list(range(]-len(eoi_toks), 0)) = [-5, -4, -3, -2, -1]
    positions = list(range(-len(model_base.eoi_toks), 0))
    print(f"Extracting from positions: {positions}")
    
    mean_diffs = get_mean_diff(
        model=model,
        tokenizer=tokenizer,
        harmful_instructions=harmful_inst_train,
        harmless_instructions=harmless_inst_train,
        tokenize_instructions_fn=model_base.tokenize_instructions_fn,
        block_modules=model_base.model_block_modules,
        positions=positions,
        batch_size=8
    )
    
    # mean_diffs shape: [n_positions, n_layers, d_model]
    # positions=[-5, -4, -3, -2, -1], so index 0 corresponds to position -5
    target_pos_in_array = positions.index(-5)  # Should be 0
    
    refusal_dir = mean_diffs[target_pos_in_array, target_layer, :]
    refusal_dir_normalized = refusal_dir / refusal_dir.norm()

    print(f"Refusal direction computed at layer {target_layer}.")
    
    # Save results
    model_name = os.path.basename(model_path)
    output_dir = os.path.join("pipeline", "runs", model_name)
    os.makedirs(output_dir, exist_ok=True)
    
    direction_path = os.path.join(output_dir, "direction.pt")
    metadata_path = os.path.join(output_dir, "direction_metadata.json")
    
    torch.save(refusal_dir_normalized, direction_path)
    
    # Metadata: pos is -5 (5th from last token, matching base model)
    with open(metadata_path, 'w') as f:
        json.dump({"layer": target_layer, "pos": -5}, f, indent=4)
        
    print(f"Saved direction to {direction_path}")
    print(f"Saved metadata to {metadata_path}")

    if not GENERATE_COMPLETIONS:
        print("Skipping completion generation (GENERATE_COMPLETIONS=False)")
        return

    # Clear memory before generation
    gc.collect()
    torch.cuda.empty_cache()

    # Generate completions
    input_completions_file = "pipeline/runs/llama-2-7b-chat-hf/completions/jailbreakbench_baseline_completions.json"
    output_completions_file = os.path.join(output_dir, "completions", "jailbreakbench_baseline_completions.json")
    
    # Update generate_completions to use the new format
    print(f"Loading prompts from {input_completions_file}...")
    with open(input_completions_file, 'r') as f:
        data = json.load(f)
    
    # Calculate slice size based on percentage
    num_samples = int(len(data) * (COMPLETION_PERCENTAGE / 100.0))
    data = data[:num_samples]
    print(f"Generating completions for {num_samples} prompts ({COMPLETION_PERCENTAGE}%)...")
    
    results = []
    print("Generating completions...")
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_completions_file), exist_ok=True)

    for item in tqdm(data):
        prompt = item['prompt']
        category = item['category']
        
        # Format and tokenize input using model_base defaults
        input_ids = model_base.tokenize_instructions_fn(instructions=[prompt]).input_ids.to(model.device)
        
        # Generate
        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_ids,
                max_new_tokens=128,
                do_sample=False
            )
        
        # Decode response (remove input prompt)
        generated_text = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
        
        results.append({
            "category": category,
            "prompt": prompt,
            "response": generated_text
        })
        
    print(f"Saving completions to {output_completions_file}...")
    with open(output_completions_file, 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()

