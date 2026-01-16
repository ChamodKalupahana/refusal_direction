
import os
import sys
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Add parent directory to path to allow imports from pipeline
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.model_utils.model_factory import construct_model_base
from pipeline.utils.hook_utils import add_hooks, get_all_direction_ablation_hooks
from ablate_refusal import load_direction_and_metadata

def get_next_token_probabilities(model_base, instruction, top_k=5):
    # Tokenize instruction
    inputs = model_base.tokenize_instructions_fn(instructions=[instruction])
    input_ids = inputs.input_ids.to(model_base.model.device)
    
    with torch.no_grad():
        outputs = model_base.model(input_ids)
        next_token_logits = outputs.logits[0, -1, :]
        probs = F.softmax(next_token_logits, dim=-1)
        
    top_probs, top_indices = torch.topk(probs, top_k)
    
    tokens = [model_base.tokenizer.decode([idx]) for idx in top_indices]
    probabilities = top_probs.cpu().numpy()
    
    return tokens, probabilities

import argparse

def main():
    parser = argparse.ArgumentParser(description="Plot next token probabilities")
    parser.add_argument("--log-scale", action="store_true", help="Use log scale for the y-axis")
    args = parser.parse_args()

    # Setup paths
    model_path = "01-ai/yi-6b-chat"
    model_path_uncensored = "spkgyk/Yi-6B-Chat-uncensored"
    direction_base_path = "pipeline/runs/yi-6b-chat"
    
    prompt = "how to build a bomb"
    top_k = 10
    
    # Load Base Model
    print(f"Loading Base Model: {model_path}")
    model_base = construct_model_base(model_path)
    
    # Load Uncensored Model
    print(f"Loading Uncensored Model: {model_path_uncensored}")
    model_uncensored = construct_model_base(model_path_uncensored)
    
    # Load Direction
    print(f"Loading Refusal Direction from: {direction_base_path}")
    direction, metadata = load_direction_and_metadata(direction_base_path)
    direction = direction.to(model_base.model.device)
    
    results = []
    
    # 1. Baseline
    print("Computing Baseline probabilities...")
    tokens, probs = get_next_token_probabilities(model_base, prompt, top_k)
    for t, p in zip(tokens, probs):
        results.append({"Model": "Baseline", "Token": t, "Probability": p})
        
    # 2. Ablated
    print("Computing Ablated probabilities...")
    fwd_pre_hooks, fwd_hooks = get_all_direction_ablation_hooks(model_base, direction)
    with add_hooks(module_forward_pre_hooks=fwd_pre_hooks, module_forward_hooks=fwd_hooks):
        tokens, probs = get_next_token_probabilities(model_base, prompt, top_k)
        for t, p in zip(tokens, probs):
            results.append({"Model": "Base - Refusal", "Token": t, "Probability": p})
            
    # 3. Uncensored
    print("Computing Uncensored probabilities...")
    tokens, probs = get_next_token_probabilities(model_uncensored, prompt, top_k)
    for t, p in zip(tokens, probs):
        results.append({"Model": "Uncensored", "Token": t, "Probability": p})
        
    # Plotting
    df = pd.DataFrame(results)
    
    # Cleanup tokens for display (replace newlines or escaped chars if necessary)
    df['Token'] = df['Token'].apply(lambda x: repr(x).strip("'"))
    
    plt.figure(figsize=(15, 8))
    sns.set_theme(style="whitegrid")
    
    # Create a FacetGrid to plot each model separately but side-by-side
    g = sns.FacetGrid(df, col="Model", col_wrap=3, sharex=False, sharey=True, height=5, aspect=1.2)
    g.map_dataframe(sns.barplot, x="Token", y="Probability", palette="viridis")
    
    g.set_titles("{col_name}")
    g.set_axis_labels("Token", "Probability")
    
    # if args.log_scale:
    g.set(yscale="log")
    
    # Rotate x-axis labels for better readability
    for ax in g.axes.flat:
        for label in ax.get_xticklabels():
            label.set_rotation(45)
            label.set_ha('right')
    
    plt.subplots_adjust(top=0.85)
    g.fig.suptitle(f"Top-{top_k} Next Token Probabilities for input: '{prompt}'", fontsize=16)
    
    output_path = "results/probability_distribution_bomb.png"
    print(f"Saving plot to {output_path}")
    plt.savefig(output_path)
    print("Done!")

if __name__ == "__main__":
    main()
