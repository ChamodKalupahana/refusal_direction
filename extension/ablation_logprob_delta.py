"""
Compute the log probability delta between:
  - Base model with refusal direction ablated
  - Uncensored model

delta(x) = logprobs(base - refusal_direction) - logprobs(uncensored)

This helps measure how similar the ablated base model is to the uncensored model.
"""

import torch
import torch.nn.functional as F
import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.model_utils.model_factory import construct_model_base
from pipeline.utils.hook_utils import add_hooks, get_all_direction_ablation_hooks


def load_direction_and_metadata(base_path):
    direction_path = os.path.join(base_path, 'direction.pt')
    metadata_path = os.path.join(base_path, 'direction_metadata.json')
    
    direction = torch.load(direction_path)
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
        
    return direction, metadata


def get_logprobs(model_base, instruction, target_tokens=None):
    """
    Get log probabilities for each token position.
    
    Args:
        model_base: Model wrapper with tokenize_instructions_fn
        instruction: The input prompt
        target_tokens: Optional target token sequence to evaluate. 
                      If None, uses greedy generation to get targets.
    
    Returns:
        logprobs: Log probabilities for each output token
        tokens: The token ids
        decoded: The decoded text
    """
    inputs = model_base.tokenize_instructions_fn(instructions=[instruction])
    input_ids = inputs.input_ids.to(model_base.model.device)
    
    with torch.no_grad():
        if target_tokens is None:
            # First, generate to get the target sequence
            outputs = model_base.model.generate(
                input_ids, 
                max_new_tokens=50,
                do_sample=False,
                pad_token_id=model_base.tokenizer.pad_token_id,
                return_dict_in_generate=True,
                output_scores=True
            )
            generated_ids = outputs.sequences[0]
            scores = outputs.scores  # tuple of (vocab_size,) tensors for each step
            
            # Convert scores to log probabilities
            logprobs = []
            generated_tokens = generated_ids[input_ids.shape[1]:]
            
            for i, score in enumerate(scores):
                log_probs = F.log_softmax(score[0], dim=-1)
                token_id = generated_tokens[i].item()
                logprobs.append(log_probs[token_id].item())
            
            return logprobs, generated_tokens.tolist(), model_base.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        else:
            # Evaluate log probs for given target tokens
            full_ids = torch.cat([input_ids, target_tokens.unsqueeze(0).to(input_ids.device)], dim=1)
            outputs = model_base.model(full_ids)
            logits = outputs.logits
            
            # Get log probs for each target token
            # logits shape: [1, seq_len, vocab_size]
            # We want log_softmax at positions input_len to input_len + target_len - 1
            log_probs = F.log_softmax(logits, dim=-1)
            
            logprobs = []
            input_len = input_ids.shape[1]
            for i, token_id in enumerate(target_tokens):
                # Position in logits that predicts this token
                pos = input_len + i - 1
                if pos >= 0 and pos < log_probs.shape[1]:
                    logprobs.append(log_probs[0, pos, token_id].item())
            
            return logprobs, target_tokens.tolist(), model_base.tokenizer.decode(target_tokens, skip_special_tokens=True)


def main():
    # Configuration
    model_path = "01-ai/yi-6b-chat" 
    model_path_uncensored = "spkgyk/Yi-6B-Chat-uncensored"
    direction_base_path = "pipeline/runs/yi-6b-chat"
    
    # Load base model
    print(f"Loading base model: {model_path}")
    model_base = construct_model_base(model_path)
    
    # Load uncensored model
    print(f"Loading uncensored model: {model_path_uncensored}")
    model_base_uncensored = construct_model_base(model_path_uncensored)
    
    # Load refusal direction
    print(f"Loading direction from: {direction_base_path}")
    direction, metadata = load_direction_and_metadata(direction_base_path)
    direction = direction.to(model_base.model.device)
    
    # Get ablation hooks
    fwd_pre_hooks, fwd_hooks = get_all_direction_ablation_hooks(model_base, direction)
    
    print("\n" + "="*60)
    print("Log Probability Delta Analysis")
    print("delta(x) = logprobs(base - refusal) - logprobs(uncensored)")
    print("="*60)
    print("Enter a prompt (or 'q' to quit):\n")
    
    while True:
        instruction = input("> ")
        if instruction.lower() in ['q', 'quit', 'exit']:
            break
        
        # 1. Get generation from ablated model (base - refusal direction)
        print("\n--- Ablated Model (base - refusal) ---")
        with add_hooks(module_forward_pre_hooks=fwd_pre_hooks, module_forward_hooks=fwd_hooks):
            ablated_logprobs, ablated_tokens, ablated_text = get_logprobs(model_base, instruction)
        print(f"Generated: {ablated_text}")
        print(f"Mean logprob: {sum(ablated_logprobs)/len(ablated_logprobs):.4f}")
        
        # 2. Get log probs from uncensored model for the SAME tokens
        print("\n--- Uncensored Model (evaluating same tokens) ---")
        target_tokens = torch.tensor(ablated_tokens)
        uncensored_logprobs, _, _ = get_logprobs(model_base_uncensored, instruction, target_tokens)
        print(f"Mean logprob: {sum(uncensored_logprobs)/len(uncensored_logprobs):.4f}")
        
        # 3. Compute delta
        print("\n--- Delta Analysis ---")
        deltas = [a - u for a, u in zip(ablated_logprobs, uncensored_logprobs)]
        
        print(f"Per-token deltas:")
        tokens_decoded = [model_base.tokenizer.decode([t]) for t in ablated_tokens]
        for i, (token, delta) in enumerate(zip(tokens_decoded, deltas)):
            sign = "+" if delta >= 0 else ""
            print(f"  {i:3d}: {repr(token):15s} delta={sign}{delta:.4f}")
        
        mean_delta = sum(deltas) / len(deltas)
        abs_mean_delta = sum(abs(d) for d in deltas) / len(deltas)
        
        print(f"\nSummary:")
        print(f"  Mean delta:     {mean_delta:+.4f}")
        print(f"  Abs mean delta: {abs_mean_delta:.4f}")
        print(f"  Sum of deltas:  {sum(deltas):+.4f}")
        
        # Also show what uncensored model would generate on its own
        print("\n--- Uncensored Model (own generation) ---")
        unc_logprobs, unc_tokens, unc_text = get_logprobs(model_base_uncensored, instruction)
        print(f"Generated: {unc_text}")
        print(f"Mean logprob: {sum(unc_logprobs)/len(unc_logprobs):.4f}")
        
        print("\n" + "-"*60 + "\n")


if __name__ == "__main__":
    main()
