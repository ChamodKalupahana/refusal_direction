import torch
import os
import random
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List
from jaxtyping import Float
from torch import Tensor
from tqdm import tqdm
from pipeline.model_utils.model_factory import construct_model_base
from dataset.load_dataset import load_dataset_split
from pipeline.utils.hook_utils import add_hooks
from pipeline.submodules.generate_directions import get_mean_activations_pre_hook
import functools

# --- Debug Versions of Functions ---

def get_mean_activations_debug(model, tokenizer, instructions, tokenize_instructions_fn, block_modules: List[torch.nn.Module], batch_size=32, positions=[-1]):
    torch.cuda.empty_cache()

    n_positions = len(positions)
    n_layers = model.config.num_hidden_layers
    n_samples = len(instructions)
    d_model = model.config.hidden_size

    mean_activations = torch.zeros((n_positions, n_layers, d_model), dtype=torch.float64, device=model.device)

    fwd_pre_hooks = [(block_modules[layer], get_mean_activations_pre_hook(layer=layer, cache=mean_activations, n_samples=n_samples, positions=positions)) for layer in range(n_layers)]

    for i in tqdm(range(0, len(instructions), batch_size)):
        batch_instructions = instructions[i:i+batch_size]
        inputs = tokenize_instructions_fn(instructions=batch_instructions)
        
        print(f"\n[DEBUG] Batch {i//batch_size} Inputs:")
        for idx, inst in enumerate(batch_instructions):
            print(f"  Input {idx}: {repr(inst)}")

        with add_hooks(module_forward_pre_hooks=fwd_pre_hooks, module_forward_hooks=[]):
            outputs = model(
                input_ids=inputs.input_ids.to(model.device),
                attention_mask=inputs.attention_mask.to(model.device),
            )
            
            # Print model generation for verification
            logits = outputs.logits
            next_token_ids = torch.argmax(logits[:, -1, :], dim=-1)
            next_tokens = tokenizer.batch_decode(next_token_ids)
            print(f"[DEBUG] Batch {i//batch_size} Model Next Tokens: {next_tokens}")

    return mean_activations

def get_mean_diff_debug(model, tokenizer, harmful_instructions, harmless_instructions, tokenize_instructions_fn, block_modules: List[torch.nn.Module], batch_size=32, positions=[-1]):
    print("\n--- Processing Harmful Instructions ---")
    mean_activations_harmful = get_mean_activations_debug(model, tokenizer, harmful_instructions, tokenize_instructions_fn, block_modules, batch_size=batch_size, positions=positions)
    
    print("\n--- Processing Harmless Instructions ---")
    mean_activations_harmless = get_mean_activations_debug(model, tokenizer, harmless_instructions, tokenize_instructions_fn, block_modules, batch_size=batch_size, positions=positions)

    mean_diff: Float[Tensor, "n_positions n_layers d_model"] = mean_activations_harmful - mean_activations_harmless
    return mean_diff

# --- Main Investigation Script ---

def main():
    model_path = "georgesung/llama2_7b_chat_uncensored"
    print(f"Loading model: {model_path}")
    model_base = construct_model_base(model_path)
    model = model_base.model
    tokenizer = model_base.tokenizer
    
    # Load a small sample of data
    print("Loading instructions...")
    harmful_inst_train = load_dataset_split(harmtype='harmful', split='train', instructions_only=True)
    harmless_inst_train = load_dataset_split(harmtype='harmless', split='train', instructions_only=True)
    
    # Use a small sample size for investigation
    sample_size = 4
    random.seed(42)
    harmful_sample = random.sample(harmful_inst_train, sample_size)
    harmless_sample = random.sample(harmless_inst_train, sample_size)
    
    # Format instructions
    def format_instruction_uncensored(instruction):
        return f"### Human: {instruction}\n### Assistant:"

    def tokenize_instructions_uncensored(tokenizer, instructions):
        prompts = [format_instruction_uncensored(inst) for inst in instructions]
        return tokenizer(prompts, padding=True, truncation=False, return_tensors="pt")

    tokenize_fn = functools.partial(tokenize_instructions_uncensored, tokenizer)
    
    print("Running debug extraction...")
    get_mean_diff_debug(
        model=model,
        tokenizer=tokenizer,
        harmful_instructions=harmful_sample,
        harmless_instructions=harmless_sample,
        tokenize_instructions_fn=tokenize_fn,
        block_modules=model_base.model_block_modules,
        positions=[-1],
        batch_size=2
    )
    
    print("\nInvestigation complete.")

if __name__ == "__main__":
    main()
