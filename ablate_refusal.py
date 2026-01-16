
import torch
import json
import os
from pipeline.model_utils.model_factory import construct_model_base
from pipeline.utils.hook_utils import add_hooks, get_all_direction_ablation_hooks

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

def main():
    # Base model path
    model_path = "meta-llama/meta-llama-3-8b-instruct" 
    direction_base_path = "pipeline/runs/meta-llama-3-8b-instruct"
    
    print(f"Loading model: {model_path}")
    model_base = construct_model_base(model_path)
    
    # Load uncensored model for comparison
    model_path_uncensored = "Orenguteng/Llama-3-8B-Lexi-Uncensored"
    print(f"Loading uncensored model: {model_path_uncensored}")
    model_base_uncensored = construct_model_base(model_path_uncensored)
    
    print(f"Loading direction from: {direction_base_path}")
    direction, metadata = load_direction_and_metadata(direction_base_path)
    
    direction = direction.to(model_base.model.device)
    
    print("Enter a prompt (or 'q' to quit):")
    while True:
        instruction = input("> ")
        if instruction.lower() in ['q', 'quit', 'exit']:
            break

        print("\n--- Baseline Generation (Normal Refusal) ---")
        baseline_output = generate(model_base, instruction)
        print(baseline_output)
        
        # Add hooks for ablation
        print(f"\n--- Ablated Generation (No Refusal) ---")
        
        fwd_pre_hooks, fwd_hooks = get_all_direction_ablation_hooks(model_base, direction)
        
        with add_hooks(module_forward_pre_hooks=fwd_pre_hooks, module_forward_hooks=fwd_hooks):
            ablated_output = generate(model_base, instruction)
            print(ablated_output)

        print(f"\n--- Uncensored Feature Generation (Target) ---")
        uncensored_output = generate(model_base_uncensored, instruction)
        print(uncensored_output)


if __name__ == "__main__":
    main()
