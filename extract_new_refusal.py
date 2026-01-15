import torch
import os
import random
import json
from pipeline.model_utils.model_factory import construct_model_base
from dataset.load_dataset import load_dataset_split

from pipeline.submodules.generate_directions import get_mean_diff

def main():
    # Configuration
    model_path = "georgesung/llama2_7b_chat_uncensored"
    N_INST_TRAIN = 32
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
    
    mean_diffs = get_mean_diff(
        model=model,
        tokenizer=tokenizer,
        harmful_instructions=harmful_inst_train,
        harmless_instructions=harmless_inst_train,
        tokenize_instructions_fn=model_base.tokenize_instructions_fn,
        block_modules=model_base.model_block_modules,
        positions=[-1]
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

if __name__ == "__main__":
    main()

