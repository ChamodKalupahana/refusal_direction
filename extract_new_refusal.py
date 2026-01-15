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
    model_path = "georgesung/llama2_7b_chat_uncensored"
    N_INST_TRAIN = 260 # specifies the number of instruction-response pairs (prompts) to sample from the dataset to compute the refusal direction. (1 -> 260)
    target_layer = 14
    target_pos_idx = -1 # The position index in the list of positions passed to get_mean_diff
    
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
    # get_mean_diff(model, tokenizer, harmful_instructions, harmless_instructions, tokenize_instructions_fn, block_modules, batch_size=32, positions=[-1])
    # block_modules corresponds to model.model.layers usually, accessible via model_base.model_block_modules
    
    import gc

    mean_diffs = get_mean_diff(
        model=model,
        tokenizer=tokenizer,
        harmful_instructions=harmful_inst_train,
        harmless_instructions=harmless_inst_train,
        tokenize_instructions_fn=model_base.tokenize_instructions_fn,
        block_modules=model_base.model_block_modules,
        positions=[-1],
        batch_size=8
    )
    
    # mean_diffs shape: [n_positions, n_layers, d_model]
    # n_positions=1 because we passed positions=[-1]
    
    refusal_dir = mean_diffs[0, target_layer, :]
    refusal_dir_normalized = refusal_dir / refusal_dir.norm()

    print(f"Refusal direction computed at layer {target_layer}.")
    
    # Save results
    model_name = os.path.basename(model_path)
    output_dir = os.path.join("pipeline", "runs", model_name)
    os.makedirs(output_dir, exist_ok=True)
    
    direction_path = os.path.join(output_dir, "direction.pt")
    metadata_path = os.path.join(output_dir, "direction_metadata.json")
    
    torch.save(refusal_dir_normalized, direction_path)
    
    # Metadata: pos is -1 (last token)
    with open(metadata_path, 'w') as f:
        json.dump({"layer": target_layer, "pos": -1}, f, indent=4)
        
    print(f"Saved direction to {direction_path}")
    print(f"Saved metadata to {metadata_path}")

    # Clear memory before generation
    gc.collect()
    torch.cuda.empty_cache()

    # Generate completions
    input_completions_file = "pipeline/runs/llama-2-7b-chat-hf/completions/jailbreakbench_baseline_completions.json"
    output_completions_file = os.path.join(output_dir, "completions", "jailbreakbench_baseline_completions.json")
    
    generate_completions(model_base, input_completions_file, output_completions_file)

if __name__ == "__main__":
    main()

