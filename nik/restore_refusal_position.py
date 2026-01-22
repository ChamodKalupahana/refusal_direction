
import torch
import json
import os
from pipeline.model_utils.model_factory import construct_model_base
from pipeline.utils.hook_utils import add_hooks

def get_activation_addition_input_pre_hook_at_output_index(vector, coeff, output_index, prompt_len):
    generation_step = 0

    def hook_fn(module, input):
        nonlocal vector, generation_step

        if isinstance(input, tuple):
            activation = input[0]
        else:
            activation = input

        vector = vector.to(activation)
        if output_index is None:
            activation += coeff * vector
        else:
            if activation.shape[1] == 1:
                if generation_step == output_index:
                    activation[:, 0, :] += coeff * vector
                generation_step += 1
            else:
                target_pos = prompt_len + output_index
                if 0 <= target_pos < activation.shape[1]:
                    activation[:, target_pos, :] += coeff * vector

        if isinstance(input, tuple):
            return (activation, *input[1:])
        else:
            return activation

    return hook_fn

def parse_output_index(raw_value):
    raw_value = raw_value.strip().lower()
    if raw_value == "":
        return None
    try:
        output_index = int(raw_value)
    except ValueError:
        print("Invalid output index; applying refusal to all tokens.")
        return None
    if output_index < 0:
        print("Output index must be >= 0; applying refusal to all tokens.")
        return None
    return output_index

def load_direction_and_metadata(base_path):
    direction_path = os.path.join(base_path, 'direction.pt')
    metadata_path = os.path.join(base_path, 'direction_metadata.json')
    
    direction = torch.load(direction_path)
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
        
    return direction, metadata

def generate_with_ids(model_base, instruction, max_new_tokens=100):
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
    return outputs[0]

def decode_token_at_output_index(model_base, output_ids, prompt_len, output_index):
    if output_index is None:
        return None
    target_pos = prompt_len + output_index
    if target_pos < 0 or target_pos >= output_ids.shape[0]:
        return None
    token_id = output_ids[target_pos].item()
    return model_base.tokenizer.decode([token_id], skip_special_tokens=False)

def generate(model_base, instruction, max_new_tokens=100):
    output_ids = generate_with_ids(model_base, instruction, max_new_tokens=max_new_tokens)
    return model_base.tokenizer.decode(output_ids, skip_special_tokens=True), output_ids

def main():
    model_path = "spkgyk/Yi-6B-Chat-uncensored"
    direction_base_path = "pipeline/runs/yi-6b-chat"
    
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
        output_index = parse_output_index(
            input("Refusal output token index (0-based, blank for all tokens): ")
        )
        prompt_len = model_base.tokenize_instructions_fn(instructions=[instruction]).input_ids.shape[1]

        print("\n--- Baseline Generation (No Refusal) ---")
        baseline_output, baseline_ids = generate(model_base, instruction)
        print(baseline_output)
        baseline_tok = decode_token_at_output_index(
            model_base, baseline_ids, prompt_len, output_index
        )
        if output_index is not None:
            if baseline_tok is None:
                print(f"[Baseline token @ {output_index}]: <out of range>")
            else:
                print(f"[Baseline token @ {output_index}]: {baseline_tok!r}")
        
        # Add hook
        print(f"\n--- Restored Refusal Generation (Layer {layer}) ---")
        
        # Coefficient to scale the refusal direction. Positive adds refusal.
        coeff = 1.0 
        
        hook_fn = get_activation_addition_input_pre_hook_at_output_index(
            vector=direction,
            coeff=coeff,
            output_index=output_index,
            prompt_len=prompt_len,
        )
        
        # Access the specific layer module
        layer_module = model_base.model_block_modules[layer]
        
        with add_hooks(module_forward_pre_hooks=[(layer_module, hook_fn)], module_forward_hooks=[]):
            refusal_output, refusal_ids = generate(model_base, instruction)
            print(refusal_output)
            refusal_tok = decode_token_at_output_index(
                model_base, refusal_ids, prompt_len, output_index
            )
            if output_index is not None:
                if refusal_tok is None:
                    print(f"[Refusal token @ {output_index}]: <out of range>")
                else:
                    print(f"[Refusal token @ {output_index}]: {refusal_tok!r}")


if __name__ == "__main__":
    main()
