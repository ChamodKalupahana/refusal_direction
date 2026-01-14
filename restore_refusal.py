
import torch
import json
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from pipeline.utils.hook_utils import add_hooks, get_activation_addition_input_pre_hook

def load_direction_and_metadata(base_path):
    direction_path = os.path.join(base_path, 'direction.pt')
    metadata_path = os.path.join(base_path, 'direction_metadata.json')
    
    direction = torch.load(direction_path)
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
        
    return direction, metadata

def generate(model, tokenizer, instruction, max_new_tokens=100):
    inputs = tokenizer(instruction, return_tensors="pt")
    input_ids = inputs.input_ids.to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids, 
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id
        )
        
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def main():
    model_path = "georgesung/llama2_7b_chat_uncensored"
    direction_base_path = "pipeline/runs/llama-2-7b-chat-hf"
    
    print(f"Loading model: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")
    
    print(f"Loading direction from: {direction_base_path}")
    direction, metadata = load_direction_and_metadata(direction_base_path)
    
    layer = metadata['layer']
    direction = direction.to(model.device)
    
    print("Enter a prompt (or 'q' to quit):")
    while True:
        instruction = input("> ")
        if instruction.lower() in ['q', 'quit', 'exit']:
            break

        print("\n--- Baseline Generation (No Refusal) ---")
        baseline_output = generate(model, tokenizer, instruction)
        print(baseline_output)
        
        # Add hook
        print(f"\n--- Restored Refusal Generation (Layer {layer}) ---")
        
        # Coefficient to scale the refusal direction. Positive adds refusal.
        # The original pipeline used +1.0 for actadd to induce refusal on harmless prompts, 
        # and we want to induce refusal here.
        coeff = 1.0 
        
        hook_fn = get_activation_addition_input_pre_hook(vector=direction, coeff=coeff)
        
        # Access the specific layer module
        # Llama structure: model.model.layers[i]
        layer_module = model.model.layers[layer]
        
        with add_hooks(module_forward_pre_hooks=[(layer_module, hook_fn)], module_forward_hooks=[]):
            refusal_output = generate(model, tokenizer, instruction)
            print(refusal_output)


if __name__ == "__main__":
    main()
