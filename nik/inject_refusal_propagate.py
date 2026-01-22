import torch
import json
import os
from pipeline.model_utils.model_factory import construct_model_base
from pipeline.utils.hook_utils import add_hooks

REFUSAL_SUBSTRINGS = [
    "i'm sorry",
    "i am sorry",
    "i apologize",
    "as an ai",
    "as a language model",
    "as an assistant",
    "i cannot",
    "i can't",
    "i am unable to",
    "i'm unable to",
    "i am not able to",
    "i'm not able to",
]

def get_activation_addition_input_pre_hook_at_output_indices(
    vector, coeff, output_indices, prompt_len
):
    generation_step = 0
    output_index_set = set(output_indices)

    def hook_fn(module, input):
        nonlocal vector, generation_step

        if isinstance(input, tuple):
            activation = input[0]
        else:
            activation = input

        vector = vector.to(activation)

        if activation.shape[1] == 1:
            if generation_step in output_index_set:
                activation[:, 0, :] += coeff * vector
            generation_step += 1
        else:
            for output_index in output_index_set:
                target_pos = prompt_len + output_index
                if 0 <= target_pos < activation.shape[1]:
                    activation[:, target_pos, :] += coeff * vector

        if isinstance(input, tuple):
            return (activation, *input[1:])
        return activation

    return hook_fn

def load_direction_and_metadata(base_path):
    direction_path = os.path.join(base_path, "direction.pt")
    metadata_path = os.path.join(base_path, "direction_metadata.json")

    direction = torch.load(direction_path)
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    return direction, metadata

def generate_with_ids(model_base, instruction, max_new_tokens=100):
    inputs = model_base.tokenize_instructions_fn(instructions=[instruction])
    input_ids = inputs.input_ids.to(model_base.model.device)

    with torch.no_grad():
        outputs = model_base.model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=model_base.tokenizer.pad_token_id,
        )

    return outputs[0], inputs.input_ids[0]

def decode_single_token(model_base, token_id):
    return model_base.tokenizer.decode([token_id], skip_special_tokens=False)

def format_token_list(model_base, token_ids):
    return [decode_single_token(model_base, token_id) for token_id in token_ids]

def find_refusal_match(text):
    text_lower = text.lower()
    return next((s for s in REFUSAL_SUBSTRINGS if s in text_lower), "")

def main():
    model_path = "spkgyk/Yi-6B-Chat-uncensored"
    direction_base_path = "pipeline/runs/yi-6b-chat"
    max_output_index = 15
    max_new_tokens = 100
    tail_token_count = 8

    print(f"Loading model: {model_path}")
    model_base = construct_model_base(model_path)

    print(f"Loading direction from: {direction_base_path}")
    direction, metadata = load_direction_and_metadata(direction_base_path)

    layer = metadata["layer"]
    direction = direction.to(model_base.model.device)

    print("Enter a prompt (or 'q' to quit):")
    while True:
        instruction = input("> ")
        if instruction.lower() in ["q", "quit", "exit"]:
            break
        use_iterative = input("Enable iterative refusal injection? (y/n): ").strip().lower()
        enable_iterative = use_iterative in ["y", "yes"]

        prompt_ids = model_base.tokenize_instructions_fn(instructions=[instruction]).input_ids[0]
        prompt_len = prompt_ids.shape[0]
        prompt_tail_ids = prompt_ids[-tail_token_count:].tolist()
        prompt_tail_tokens = format_token_list(model_base, prompt_tail_ids)

        print("\n--- Baseline Generation (No Refusal) ---")
        baseline_ids, _ = generate_with_ids(model_base, instruction, max_new_tokens=max_new_tokens)
        baseline_text = model_base.tokenizer.decode(baseline_ids, skip_special_tokens=True)
        print(baseline_text)

        if not enable_iterative:
            print("\nIterative refusal injection disabled.")
            continue

        # Coefficient to scale the refusal direction. Positive adds refusal.
        coeff = 1.0
        layer_module = model_base.model_block_modules[layer]

        refusal_found = False
        for last_index in range(0, max_output_index + 1):
            output_indices = list(range(0, last_index + 1))
            hook_fn = get_activation_addition_input_pre_hook_at_output_indices(
                vector=direction,
                coeff=coeff,
                output_indices=output_indices,
                prompt_len=prompt_len,
            )

            with add_hooks(
                module_forward_pre_hooks=[(layer_module, hook_fn)],
                module_forward_hooks=[],
            ):
                refusal_ids, _ = generate_with_ids(
                    model_base, instruction, max_new_tokens=max_new_tokens
                )

            refusal_text = model_base.tokenizer.decode(refusal_ids, skip_special_tokens=True)
            gen_ids = refusal_ids[prompt_len:]

            output_token_info = []
            for output_index in output_indices:
                if output_index < len(gen_ids):
                    token_id = gen_ids[output_index].item()
                    abs_pos = prompt_len + output_index
                    token_text = decode_single_token(model_base, token_id)
                    output_token_info.append(
                        f"{output_index} (abs {abs_pos}): {token_text!r}"
                    )
                else:
                    output_token_info.append(f"{output_index}: <out of range>")

            refusal_match = find_refusal_match(refusal_text)
            print(
                f"\n--- Iteration {last_index} (inject output[0:{last_index}]) ---"
            )
            print(f"Prompt tail tokens: {prompt_tail_tokens}")
            print("Output tokens at indices: " + ", ".join(output_token_info))
            print(refusal_text)
            if refusal_match:
                print(f"[Refusal match]: {refusal_match!r}")
                refusal_found = True
                break

        if not refusal_found:
            print(f"\nNo refusal found up to output index {max_output_index}.")


if __name__ == "__main__":
    main()
