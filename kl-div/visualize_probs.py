"""
Visualize probability distributions from refusal_direction.pt

Creates several plots to analyze the next-token probability distributions
after refusal direction ablation.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import os

# Set style for better looking plots
plt.style.use('seaborn-v0_8-darkgrid')


def load_data(filepath):
    """Load the probability distribution data."""
    data = torch.load(filepath)
    return data


def plot_entropy_distribution(results, output_dir):
    """
    Plot the entropy of probability distributions across prompts.
    Higher entropy = more uncertainty in next token prediction.
    """
    entropies = []
    categories = []
    
    for idx in range(len(results)):
        probs = results[idx]['probability_distribution']
        # Calculate entropy: -sum(p * log(p))
        probs_nonzero = probs[probs > 0]
        entropy = -torch.sum(probs_nonzero * torch.log2(probs_nonzero)).item()
        entropies.append(entropy)
        categories.append(results[idx]['category'])
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram of entropies
    axes[0].hist(entropies, bins=20, edgecolor='black', alpha=0.7, color='steelblue')
    axes[0].set_xlabel('Entropy (bits)', fontsize=12)
    axes[0].set_ylabel('Count', fontsize=12)
    axes[0].set_title('Distribution of Next-Token Prediction Entropy\n(After Refusal Ablation)', fontsize=13)
    axes[0].axvline(np.mean(entropies), color='red', linestyle='--', label=f'Mean: {np.mean(entropies):.2f}')
    axes[0].legend()
    
    # Box plot by category
    category_entropies = defaultdict(list)
    for ent, cat in zip(entropies, categories):
        category_entropies[cat].append(ent)
    
    cats = list(category_entropies.keys())
    ent_data = [category_entropies[c] for c in cats]
    
    bp = axes[1].boxplot(ent_data, patch_artist=True)
    colors = plt.cm.Set2(np.linspace(0, 1, len(cats)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    axes[1].set_xticklabels(cats, rotation=45, ha='right', fontsize=10)
    axes[1].set_ylabel('Entropy (bits)', fontsize=12)
    axes[1].set_title('Entropy by Harm Category', fontsize=13)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'entropy_distribution.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: entropy_distribution.png")


def plot_top_token_analysis(results, output_dir):
    """
    Analyze and plot the most common top-1 predicted tokens.
    """
    top1_tokens = []
    top1_probs = []
    
    for idx in range(len(results)):
        top_tokens = results[idx]['top_tokens']
        if top_tokens:
            token, prob = top_tokens[0]
            top1_tokens.append(token.strip())
            top1_probs.append(prob)
    
    # Count token frequencies
    token_counts = defaultdict(int)
    for token in top1_tokens:
        token_counts[token] += 1
    
    # Sort by frequency
    sorted_tokens = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)[:15]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Bar chart of most common top-1 tokens
    tokens, counts = zip(*sorted_tokens)
    bars = axes[0].barh(range(len(tokens)), counts, color='coral', edgecolor='black')
    axes[0].set_yticks(range(len(tokens)))
    axes[0].set_yticklabels([f'"{t}"' for t in tokens], fontsize=10)
    axes[0].invert_yaxis()
    axes[0].set_xlabel('Frequency', fontsize=12)
    axes[0].set_title('Most Common Top-1 Predicted Tokens\n(After Refusal Ablation)', fontsize=13)
    
    # Histogram of top-1 token probabilities
    axes[1].hist(top1_probs, bins=20, edgecolor='black', alpha=0.7, color='mediumseagreen')
    axes[1].set_xlabel('Top-1 Token Probability', fontsize=12)
    axes[1].set_ylabel('Count', fontsize=12)
    axes[1].set_title('Distribution of Top-1 Token Confidence', fontsize=13)
    axes[1].axvline(np.mean(top1_probs), color='red', linestyle='--', 
                    label=f'Mean: {np.mean(top1_probs):.3f}')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'top_token_analysis.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: top_token_analysis.png")


def plot_probability_concentration(results, output_dir):
    """
    Plot how concentrated the probability mass is in top-k tokens.
    """
    k_values = [1, 5, 10, 50, 100, 500]
    concentration_data = {k: [] for k in k_values}
    
    for idx in range(len(results)):
        probs = results[idx]['probability_distribution']
        sorted_probs, _ = torch.sort(probs, descending=True)
        
        for k in k_values:
            top_k_prob = sorted_probs[:k].sum().item()
            concentration_data[k].append(top_k_prob)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Box plot of concentration at different k
    data = [concentration_data[k] for k in k_values]
    bp = axes[0].boxplot(data, patch_artist=True)
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(k_values)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    axes[0].set_xticklabels([f'Top-{k}' for k in k_values], fontsize=10)
    axes[0].set_ylabel('Cumulative Probability', fontsize=12)
    axes[0].set_title('Probability Mass Concentration', fontsize=13)
    axes[0].set_ylim(0, 1.05)
    
    # Mean concentration curve
    means = [np.mean(concentration_data[k]) for k in k_values]
    stds = [np.std(concentration_data[k]) for k in k_values]
    
    axes[1].errorbar(k_values, means, yerr=stds, marker='o', capsize=5, 
                     linewidth=2, markersize=8, color='purple')
    axes[1].fill_between(k_values, 
                         [m - s for m, s in zip(means, stds)],
                         [m + s for m, s in zip(means, stds)],
                         alpha=0.2, color='purple')
    axes[1].set_xscale('log')
    axes[1].set_xlabel('Top-k Tokens', fontsize=12)
    axes[1].set_ylabel('Mean Cumulative Probability', fontsize=12)
    axes[1].set_title('Cumulative Probability vs. Top-k\n(Mean ± Std)', fontsize=13)
    axes[1].set_ylim(0, 1.05)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'probability_concentration.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: probability_concentration.png")


def plot_category_comparison(results, output_dir):
    """
    Compare probability characteristics across different harm categories.
    """
    category_data = defaultdict(lambda: {'top1_prob': [], 'entropy': [], 'top10_prob': []})
    
    for idx in range(len(results)):
        cat = results[idx]['category']
        probs = results[idx]['probability_distribution']
        top_tokens = results[idx]['top_tokens']
        
        # Top-1 probability
        if top_tokens:
            category_data[cat]['top1_prob'].append(top_tokens[0][1])
        
        # Entropy
        probs_nonzero = probs[probs > 0]
        entropy = -torch.sum(probs_nonzero * torch.log2(probs_nonzero)).item()
        category_data[cat]['entropy'].append(entropy)
        
        # Top-10 cumulative probability
        sorted_probs, _ = torch.sort(probs, descending=True)
        category_data[cat]['top10_prob'].append(sorted_probs[:10].sum().item())
    
    categories = list(category_data.keys())
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # Mean top-1 probability by category
    means = [np.mean(category_data[c]['top1_prob']) for c in categories]
    stds = [np.std(category_data[c]['top1_prob']) for c in categories]
    colors = plt.cm.Set3(np.linspace(0, 1, len(categories)))
    
    bars = axes[0].bar(range(len(categories)), means, yerr=stds, capsize=3,
                       color=colors, edgecolor='black')
    axes[0].set_xticks(range(len(categories)))
    axes[0].set_xticklabels(categories, rotation=45, ha='right', fontsize=9)
    axes[0].set_ylabel('Mean Top-1 Probability', fontsize=11)
    axes[0].set_title('Top-1 Token Confidence by Category', fontsize=12)
    
    # Mean entropy by category
    means = [np.mean(category_data[c]['entropy']) for c in categories]
    stds = [np.std(category_data[c]['entropy']) for c in categories]
    
    bars = axes[1].bar(range(len(categories)), means, yerr=stds, capsize=3,
                       color=colors, edgecolor='black')
    axes[1].set_xticks(range(len(categories)))
    axes[1].set_xticklabels(categories, rotation=45, ha='right', fontsize=9)
    axes[1].set_ylabel('Mean Entropy (bits)', fontsize=11)
    axes[1].set_title('Prediction Entropy by Category', fontsize=12)
    
    # Mean top-10 probability by category
    means = [np.mean(category_data[c]['top10_prob']) for c in categories]
    stds = [np.std(category_data[c]['top10_prob']) for c in categories]
    
    bars = axes[2].bar(range(len(categories)), means, yerr=stds, capsize=3,
                       color=colors, edgecolor='black')
    axes[2].set_xticks(range(len(categories)))
    axes[2].set_xticklabels(categories, rotation=45, ha='right', fontsize=9)
    axes[2].set_ylabel('Mean Top-10 Cumulative Prob', fontsize=11)
    axes[2].set_title('Top-10 Concentration by Category', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'category_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: category_comparison.png")


def plot_sample_distributions(results, output_dir, n_samples=6):
    """
    Plot sample probability distributions for individual prompts.
    """
    n_samples = min(n_samples, len(results))
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    indices = np.linspace(0, len(results) - 1, n_samples, dtype=int)
    
    for i, idx in enumerate(indices):
        result = results[idx]
        top_tokens = result['top_tokens'][:10]
        
        tokens = [t[0].strip() for t in top_tokens]
        probs = [t[1] for t in top_tokens]
        
        bars = axes[i].barh(range(len(tokens)), probs, color=plt.cm.Blues(0.6), edgecolor='black')
        axes[i].set_yticks(range(len(tokens)))
        axes[i].set_yticklabels([f'"{t}"' for t in tokens], fontsize=9)
        axes[i].invert_yaxis()
        axes[i].set_xlabel('Probability', fontsize=10)
        
        # Truncate instruction for title
        instruction = result['instruction'][:50] + '...' if len(result['instruction']) > 50 else result['instruction']
        axes[i].set_title(f'"{instruction}"\n[{result["category"]}]', fontsize=9)
        axes[i].set_xlim(0, max(probs) * 1.1)
    
    plt.suptitle('Top-10 Token Distributions for Sample Prompts\n(After Refusal Ablation)', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sample_distributions.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: sample_distributions.png")


def main():
    # Get script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, 'refusal_direction.pt')
    output_dir = os.path.join(script_dir, 'plots')
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading data from: {data_path}")
    data = load_data(data_path)
    
    print(f"\nDataset info:")
    print(f"  Model: {data['model_path']}")
    print(f"  Ablation type: {data['ablation_type']}")
    print(f"  Number of prompts: {data['num_prompts']}")
    print(f"  Vocab size: {data['vocab_size']}")
    
    results = data['results']
    
    print(f"\nGenerating plots in: {output_dir}")
    print("-" * 50)
    
    # Generate all plots
    plot_entropy_distribution(results, output_dir)
    plot_top_token_analysis(results, output_dir)
    plot_probability_concentration(results, output_dir)
    plot_category_comparison(results, output_dir)
    plot_sample_distributions(results, output_dir)
    
    print("-" * 50)
    print(f"All plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
