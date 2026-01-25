"""
Analyze PC1 scores for each prompt in the delta matrix.

This script:
1. Computes PC1 scores for each prompt
2. Correlates scores with token length
3. Prints top/bottom prompts for PC1
"""

import torch
import json
import numpy as np
from scipy import stats
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.model_utils.model_factory import construct_model_base


def load_data(delta_path: str, pca_path: str, dataset_path: str):
    """Load delta matrix, PCA results, and dataset."""
    delta_data = torch.load(delta_path, weights_only=False)
    pca_data = torch.load(pca_path, weights_only=False)
    
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
    
    return delta_data, pca_data, dataset


def compute_pc_scores(D: torch.Tensor, components: torch.Tensor) -> torch.Tensor:
    """
    Compute PC scores for each prompt.
    
    Args:
        D: Delta matrix [N, d_model]
        components: PCA components [n_components, d_model]
    
    Returns:
        scores: [N, n_components] - PC scores for each prompt
    """
    # Move to CPU and center the data
    D_cpu = D.cpu().float()
    D_centered = D_cpu - D_cpu.mean(dim=0)
    
    # Project onto PCA components: scores = D @ components.T
    scores = D_centered @ components.T.float()
    
    return scores


def get_token_lengths(model_base, dataset: list) -> list:
    """Get token lengths for each instruction."""
    lengths = []
    for item in dataset:
        enc = model_base.tokenize_instructions_fn(instructions=[item['instruction']])
        lengths.append(enc.attention_mask.sum().item())
    return lengths


def analyze_pc1_scores(
    scores: torch.Tensor,
    token_lengths: list,
    dataset: list,
    n_top: int = 10
):
    """Analyze PC1 scores and correlate with token length."""
    pc1_scores = scores[:, 0].numpy()
    
    print("="*60)
    print("PC1 Score Analysis")
    print("="*60)
    
    # Basic statistics
    print(f"\nPC1 Score Statistics:")
    print(f"  Mean:   {pc1_scores.mean():.4f}")
    print(f"  Std:    {pc1_scores.std():.4f}")
    print(f"  Min:    {pc1_scores.min():.4f}")
    print(f"  Max:    {pc1_scores.max():.4f}")
    
    # Correlation with token length
    print("\n" + "-"*60)
    print("Correlation with Token Length")
    print("-"*60)
    
    correlation, p_value = stats.pearsonr(pc1_scores, token_lengths)
    spearman_corr, spearman_p = stats.spearmanr(pc1_scores, token_lengths)
    
    print(f"  Pearson correlation:  r = {correlation:.4f} (p = {p_value:.2e})")
    print(f"  Spearman correlation: ρ = {spearman_corr:.4f} (p = {spearman_p:.2e})")
    
    if abs(correlation) > 0.5:
        print(f"  ⚠ Strong correlation detected - PC1 may be capturing length effects!")
    elif abs(correlation) > 0.3:
        print(f"  ⚡ Moderate correlation - some length effect present")
    else:
        print(f"  ✓ Weak correlation - PC1 appears independent of length")
    
    # Sort by PC1 score
    sorted_indices = np.argsort(pc1_scores)
    
    # Top prompts (highest PC1)
    print("\n" + "-"*60)
    print(f"Top {n_top} Prompts (Highest PC1 Score)")
    print("-"*60)
    for i, idx in enumerate(sorted_indices[-n_top:][::-1]):
        instruction = dataset[idx]['instruction']
        score = pc1_scores[idx]
        length = token_lengths[idx]
        print(f"  {i+1:2d}. [PC1={score:+8.4f}] [len={length:3d}] {instruction[:60]}...")
    
    # Bottom prompts (lowest PC1)
    print("\n" + "-"*60)
    print(f"Bottom {n_top} Prompts (Lowest PC1 Score)")
    print("-"*60)
    for i, idx in enumerate(sorted_indices[:n_top]):
        instruction = dataset[idx]['instruction']
        score = pc1_scores[idx]
        length = token_lengths[idx]
        print(f"  {i+1:2d}. [PC1={score:+8.4f}] [len={length:3d}] {instruction[:60]}...")
    
    return {
        'pc1_scores': pc1_scores,
        'token_lengths': token_lengths,
        'correlation': correlation,
        'p_value': p_value,
        'spearman_corr': spearman_corr,
        'sorted_indices': sorted_indices,
    }


def main():
    # Paths
    delta_path = "extension/delta_matrix.pt"
    pca_path = "extension/pca_directions.pt"
    dataset_path = "dataset/splits/harmless_train.json"
    model_path = "01-ai/yi-6b-chat"
    
    # Load data
    print(f"Loading delta matrix from: {delta_path}")
    print(f"Loading PCA results from: {pca_path}")
    delta_data, pca_data, dataset = load_data(delta_path, pca_path, dataset_path)
    
    D = delta_data['D']
    components = pca_data['components']
    num_prompts = delta_data['num_prompts']
    
    # Slice dataset to match delta matrix
    dataset = dataset[:num_prompts]
    print(f"Using {num_prompts} prompts")
    
    # Compute PC scores
    print("\nComputing PC scores...")
    scores = compute_pc_scores(D, components)
    print(f"Scores shape: {scores.shape}")
    
    # Get token lengths
    print(f"\nLoading model for tokenization: {model_path}")
    model_base = construct_model_base(model_path)
    print("Computing token lengths...")
    token_lengths = get_token_lengths(model_base, dataset)
    
    # Analyze
    results = analyze_pc1_scores(scores, token_lengths, dataset, n_top=10)
    
    # Save results
    output_path = "extension/pc1_analysis.pt"
    print(f"\nSaving analysis to: {output_path}")
    torch.save({
        'scores': scores,
        'pc1_scores': results['pc1_scores'],
        'token_lengths': results['token_lengths'],
        'correlation': results['correlation'],
        'spearman_corr': results['spearman_corr'],
    }, output_path)
    print("Done!")


if __name__ == "__main__":
    main()
