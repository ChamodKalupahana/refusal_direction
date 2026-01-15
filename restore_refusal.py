
import torch
import json
import os
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

def main():
    model_path = "georgesung/llama2_7b_chat_uncensored"
    direction_base_path = "pipeline/runs/llama-2-7b-chat-hf"
    
    print(f"Loading model: {model_path}")
    # Use factory to get the correct model wrapper (Llama2UncensoredModel)
    model_base = construct_model_base(model_path)
    
    print(f"Loading direction from: {direction_base_path}")
    direction, metadata = load_direction_and_metadata(direction_base_path)
    
    layer = metadata['layer']
    direction = direction.to(model_base.model.device)
    
    print("Enter a prompt (or 'q' to quit):")
    while True:
        instruction = input("> ")
        if instruction.lower() in ['q', 'quit', 'exit']:
            break

        print("\n--- Baseline Generation (No Refusal) ---")
        baseline_output = generate(model_base, instruction)
        print(baseline_output)
        
        # Add hook
        print(f"\n--- Restored Refusal Generation (Layer {layer}) ---")
        
        # Coefficient to scale the refusal direction. Positive adds refusal.
        coeff = 1.0 
        
        hook_fn = get_activation_addition_input_pre_hook(vector=direction, coeff=coeff)
        
        # Access the specific layer module
        layer_module = model_base.model_block_modules[layer]
        
        with add_hooks(module_forward_pre_hooks=[(layer_module, hook_fn)], module_forward_hooks=[]):
            refusal_output = generate(model_base, instruction)
            print(refusal_output)


if __name__ == "__main__":
    main()
