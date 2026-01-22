"""
Compare KL divergence between probability distributions from different model configurations.

Compares:
- Base model (yi-6b-chat) without intervention
- Base model with refusal direction ablation (base - refusal)
- Uncensored model (Yi-6B-Chat-uncensored) without intervention
- Uncensored model with refusal direction added (uncensored + refusal)
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


def kl_divergence(p, q, epsilon=1e-10):
    """
    Compute KL divergence D_KL(P || Q).
    
    Args:
        p: Reference distribution (tensor)
        q: Comparison distribution (tensor)
        epsilon: Small value to avoid log(0)
    
    Returns:
        KL divergence value (float)
    """
    # Add epsilon to avoid log(0)
    p = p + epsilon
    q = q + epsilon
    
    # Normalize to ensure they sum to 1
    p = p / p.sum()
    q = q / q.sum()
    
    # Compute KL divergence
    kl = torch.sum(p * torch.log(p / q))
    return kl.item()


def symmetric_kl_divergence(p, q, epsilon=1e-10):
    """
    Compute symmetric KL divergence (Jensen-Shannon-like).
    D_sym = 0.5 * (D_KL(P || Q) + D_KL(Q || P))
    """
    return 0.5 * (kl_divergence(p, q, epsilon) + kl_divergence(q, p, epsilon))


def compute_pairwise_kl(data_dict):
    """
    Compute pairwise KL divergences for all prompts across model configurations.
    
    Args:
        data_dict: Dictionary mapping config names to loaded data
    
    Returns:
        Dictionary of pairwise KL divergences
    """
    config_names = list(data_dict.keys())
    num_prompts = data_dict[config_names[0]]['num_prompts']
    
    # Initialize storage for pairwise comparisons
    pairwise_kl = {}
    
    for i, name1 in enumerate(config_names):
        for j, name2 in enumerate(config_names):
            if i < j:  # Only compute for unique pairs
                pair_key = f"{name1} vs {name2}"
                kl_values = []
                
                for idx in range(num_prompts):
                    p = data_dict[name1]['results'][idx]['probability_distribution']
                    q = data_dict[name2]['results'][idx]['probability_distribution']
                    kl = symmetric_kl_divergence(p, q)
                    kl_values.append(kl)
                
                pairwise_kl[pair_key] = kl_values
    
    return pairwise_kl


def plot_kl_distributions(pairwise_kl, output_dir):
    """Plot histogram of KL divergences for each pair."""
    n_pairs = len(pairwise_kl)
    n_cols = 2
    n_rows = (n_pairs + 1) // 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))
    axes = axes.flatten() if n_pairs > 1 else [axes]
    
    colors = plt.cm.Set2(np.linspace(0, 1, n_pairs))
    
    for idx, (pair_name, kl_values) in enumerate(pairwise_kl.items()):
        ax = axes[idx]
        ax.hist(kl_values, bins=20, edgecolor='black', alpha=0.7, color=colors[idx])
        ax.axvline(np.mean(kl_values), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(kl_values):.4f}')
        ax.axvline(np.median(kl_values), color='blue', linestyle=':', 
                   label=f'Median: {np.median(kl_values):.4f}')
        ax.set_xlabel('KL Divergence (symmetric)', fontsize=11)
        ax.set_ylabel('Count', fontsize=11)
        ax.set_title(pair_name, fontsize=12)
        ax.legend(fontsize=9)
    
    # Hide unused axes
    for idx in range(len(pairwise_kl), len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle('Distribution of KL Divergences Across Prompts', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'kl_distributions.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: kl_distributions.png")


def plot_kl_comparison_bar(pairwise_kl, output_dir):
    """Bar chart comparing mean KL divergences across all pairs."""
    pair_names = list(pairwise_kl.keys())
    means = [np.mean(v) for v in pairwise_kl.values()]
    
    # Shorten names for display
    short_names = [name.replace('_no_ablation', '').replace('_with_ablation', '\n(-refusal)').replace('_with_addition', '\n(+refusal)') 
                   for name in pair_names]
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(pair_names)))
    bars = ax.bar(range(len(pair_names)), means, color=colors, edgecolor='black')
    
    ax.set_xticks(range(len(pair_names)))
    ax.set_xticklabels(short_names, fontsize=9, ha='center')
    ax.set_ylabel('Mean KL Divergence (symmetric)', fontsize=12)
    ax.set_title('Comparison of KL Divergences Between Model Configurations', fontsize=13)
    
    # Add value labels on bars
    for bar, mean in zip(bars, means):
        ax.annotate(f'{mean:.4f}', 
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 5), textcoords='offset points',
                    ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'kl_comparison_bar.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: kl_comparison_bar.png")


def plot_kl_heatmap(data_dict, output_dir):
    """Heatmap of mean KL divergences between all pairs."""
    config_names = list(data_dict.keys())
    n_configs = len(config_names)
    num_prompts = data_dict[config_names[0]]['num_prompts']
    
    # Compute mean KL divergence matrix
    kl_matrix = np.zeros((n_configs, n_configs))
    
    for i, name1 in enumerate(config_names):
        for j, name2 in enumerate(config_names):
            if i == j:
                kl_matrix[i, j] = 0
            else:
                kl_values = []
                for idx in range(num_prompts):
                    p = data_dict[name1]['results'][idx]['probability_distribution']
                    q = data_dict[name2]['results'][idx]['probability_distribution']
                    kl = kl_divergence(p, q)
                    kl_values.append(kl)
                kl_matrix[i, j] = np.mean(kl_values)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create heatmap
    im = ax.imshow(kl_matrix, cmap='YlOrRd')
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Mean KL Divergence D_KL(row || col)', rotation=-90, va="bottom", fontsize=11)
    
    # Set ticks and labels
    short_names = [name.replace('_no_ablation', '\n(vanilla)').replace('_with_ablation', '\n(-refusal)').replace('_with_addition', '\n(+refusal)') 
                   for name in config_names]
    ax.set_xticks(np.arange(n_configs))
    ax.set_yticks(np.arange(n_configs))
    ax.set_xticklabels(short_names, fontsize=10)
    ax.set_yticklabels(short_names, fontsize=10)
    
    # Rotate the tick labels and set their alignment
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations
    for i in range(n_configs):
        for j in range(n_configs):
            text = ax.text(j, i, f'{kl_matrix[i, j]:.4f}',
                          ha="center", va="center", color="black" if kl_matrix[i, j] < kl_matrix.max()/2 else "white",
                          fontsize=10)
    
    ax.set_title('KL Divergence Matrix: D_KL(row || column)', fontsize=13, pad=20)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'kl_heatmap.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: kl_heatmap.png")


def plot_kl_by_category(data_dict, pairwise_kl, output_dir):
    """Plot KL divergences grouped by harm category."""
    config_names = list(data_dict.keys())
    num_prompts = data_dict[config_names[0]]['num_prompts']
    
    # Get categories for each prompt
    categories = [data_dict[config_names[0]]['results'][idx]['category'] 
                  for idx in range(num_prompts)]
    unique_cats = sorted(set(categories))
    
    # For a key comparison: base_no_ablation vs uncensored_no_ablation
    key_comparison = "base_no_ablation vs uncensored_no_ablation"
    if key_comparison not in pairwise_kl:
        # Try finding any comparison with both base and uncensored
        for key in pairwise_kl:
            if 'base' in key and 'uncensored' in key:
                key_comparison = key
                break
    
    if key_comparison in pairwise_kl:
        kl_values = pairwise_kl[key_comparison]
        
        # Group by category
        cat_kl = defaultdict(list)
        for idx, kl in enumerate(kl_values):
            cat_kl[categories[idx]].append(kl)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        cats = list(cat_kl.keys())
        means = [np.mean(cat_kl[c]) for c in cats]
        stds = [np.std(cat_kl[c]) for c in cats]
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(cats)))
        bars = ax.bar(range(len(cats)), means, yerr=stds, capsize=5,
                      color=colors, edgecolor='black')
        
        ax.set_xticks(range(len(cats)))
        ax.set_xticklabels(cats, rotation=45, ha='right', fontsize=10)
        ax.set_ylabel('Mean KL Divergence', fontsize=12)
        ax.set_title(f'KL Divergence by Category\n({key_comparison})', fontsize=13)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'kl_by_category.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("Saved: kl_by_category.png")


def plot_intervention_effects(data_dict, output_dir):
    """
    Compare the effect of interventions on base vs uncensored models.
    - Ablation on base model: makes it more like uncensored
    - Addition on uncensored: makes it more like base (with refusal)
    """
    num_prompts = data_dict['base_no_ablation']['num_prompts']
    
    # Effect of ablation on base model
    base_ablation_effect = []  # KL(base_vanilla || base_ablated)
    
    # Effect of addition on uncensored model
    uncensored_addition_effect = []  # KL(uncensored_vanilla || uncensored_+refusal)
    
    # How similar is base+ablation to uncensored?
    base_abl_vs_uncensored = []  # KL(base_ablated || uncensored_vanilla)
    
    # How similar is uncensored+addition to base?
    uncensored_add_vs_base = []  # KL(uncensored_+refusal || base_vanilla)
    
    for idx in range(num_prompts):
        p_base_vanilla = data_dict['base_no_ablation']['results'][idx]['probability_distribution']
        p_base_abl = data_dict['base_with_ablation']['results'][idx]['probability_distribution']
        p_unc_vanilla = data_dict['uncensored_no_ablation']['results'][idx]['probability_distribution']
        p_unc_add = data_dict['uncensored_with_addition']['results'][idx]['probability_distribution']
        
        base_ablation_effect.append(symmetric_kl_divergence(p_base_vanilla, p_base_abl))
        uncensored_addition_effect.append(symmetric_kl_divergence(p_unc_vanilla, p_unc_add))
        base_abl_vs_uncensored.append(symmetric_kl_divergence(p_base_abl, p_unc_vanilla))
        uncensored_add_vs_base.append(symmetric_kl_divergence(p_unc_add, p_base_vanilla))
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Effect of interventions
    ax = axes[0, 0]
    data_boxes = [base_ablation_effect, uncensored_addition_effect]
    bp = ax.boxplot(data_boxes, patch_artist=True)
    box_colors = ['steelblue', 'coral']
    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
    ax.set_xticklabels(['Base: Ablation Effect\n(vanilla → -refusal)', 
                        'Uncensored: Addition Effect\n(vanilla → +refusal)'], fontsize=10)
    ax.set_ylabel('KL Divergence', fontsize=11)
    ax.set_title('Effect of Interventions on Each Model', fontsize=12)
    for i, data in enumerate(data_boxes):
        ax.scatter([i + 1], [np.mean(data)], color='red', marker='D', s=60, zorder=5)
    
    # 2. Similarity after interventions
    ax = axes[0, 1]
    data_boxes = [base_abl_vs_uncensored, uncensored_add_vs_base]
    bp = ax.boxplot(data_boxes, patch_artist=True)
    box_colors = ['mediumseagreen', 'mediumpurple']
    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
    ax.set_xticklabels(['Base(-refusal) vs\nUncensored(vanilla)', 
                        'Uncensored(+refusal) vs\nBase(vanilla)'], fontsize=10)
    ax.set_ylabel('KL Divergence', fontsize=11)
    ax.set_title('Similarity After Interventions', fontsize=12)
    for i, data in enumerate(data_boxes):
        ax.scatter([i + 1], [np.mean(data)], color='red', marker='D', s=60, zorder=5)
    
    # 3. Scatter: Ablation effect vs Addition effect
    ax = axes[1, 0]
    categories = [data_dict['base_no_ablation']['results'][idx]['category'] for idx in range(num_prompts)]
    unique_cats = list(set(categories))
    colors = {cat: plt.cm.Set2(i / len(unique_cats)) for i, cat in enumerate(unique_cats)}
    
    for idx in range(num_prompts):
        ax.scatter(base_ablation_effect[idx], uncensored_addition_effect[idx],
                   c=[colors[categories[idx]]], alpha=0.7, s=50, edgecolor='black', linewidth=0.5)
    
    max_val = max(max(base_ablation_effect), max(uncensored_addition_effect))
    ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='y=x')
    ax.set_xlabel('Base Ablation Effect (KL)', fontsize=11)
    ax.set_ylabel('Uncensored Addition Effect (KL)', fontsize=11)
    ax.set_title('Comparing Intervention Effects', fontsize=12)
    handles = [plt.scatter([], [], c=[colors[cat]], label=cat, s=50) for cat in unique_cats]
    ax.legend(handles=handles, loc='upper left', fontsize=8)
    
    # 4. Summary bar chart
    ax = axes[1, 1]
    comparisons = {
        'Base vanilla vs\nUncensored vanilla': ('base_no_ablation', 'uncensored_no_ablation'),
        'Base(-refusal) vs\nUncensored vanilla': ('base_with_ablation', 'uncensored_no_ablation'),
        'Uncensored(+refusal) vs\nBase vanilla': ('uncensored_with_addition', 'base_no_ablation'),
        'Base(-refusal) vs\nUncensored(+refusal)': ('base_with_ablation', 'uncensored_with_addition'),
    }
    
    comp_names = list(comparisons.keys())
    means = []
    stds = []
    
    for name, (config1, config2) in comparisons.items():
        kl_values = []
        for idx in range(num_prompts):
            p = data_dict[config1]['results'][idx]['probability_distribution']
            q = data_dict[config2]['results'][idx]['probability_distribution']
            kl_values.append(symmetric_kl_divergence(p, q))
        means.append(np.mean(kl_values))
        stds.append(np.std(kl_values))
    
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6']
    bars = ax.bar(range(len(comp_names)), means, yerr=stds, capsize=4,
                  color=colors, edgecolor='black')
    ax.set_xticks(range(len(comp_names)))
    ax.set_xticklabels(comp_names, fontsize=9)
    ax.set_ylabel('Mean Symmetric KL Divergence', fontsize=11)
    ax.set_title('Key Comparisons', fontsize=12)
    
    for bar, mean in zip(bars, means):
        ax.annotate(f'{mean:.3f}', 
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords='offset points',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.suptitle('Analysis of Refusal Direction Interventions', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'intervention_effects.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: intervention_effects.png")


def plot_summary_comparison(data_dict, output_dir):
    """
    Summary plot showing key comparisons.
    """
    num_prompts = data_dict['base_no_ablation']['num_prompts']
    
    # Key comparisons
    comparisons = {
        'Base vs Uncensored\n(both vanilla)': ('base_no_ablation', 'uncensored_no_ablation'),
        'Base: Ablation Effect\n(vanilla → -refusal)': ('base_no_ablation', 'base_with_ablation'),
        'Uncensored: Addition Effect\n(vanilla → +refusal)': ('uncensored_no_ablation', 'uncensored_with_addition'),
        'Base(-refusal) vs\nUncensored(vanilla)': ('base_with_ablation', 'uncensored_no_ablation'),
        'Uncensored(+refusal) vs\nBase(vanilla)': ('uncensored_with_addition', 'base_no_ablation'),
    }
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    comparison_names = list(comparisons.keys())
    means = []
    stds = []
    
    for name, (config1, config2) in comparisons.items():
        kl_values = []
        for idx in range(num_prompts):
            p = data_dict[config1]['results'][idx]['probability_distribution']
            q = data_dict[config2]['results'][idx]['probability_distribution']
            kl_values.append(symmetric_kl_divergence(p, q))
        means.append(np.mean(kl_values))
        stds.append(np.std(kl_values))
    
    colors = ['#3498db', '#e74c3c', '#f39c12', '#2ecc71', '#9b59b6']
    bars = ax.bar(range(len(comparison_names)), means, yerr=stds, capsize=5,
                  color=colors, edgecolor='black')
    
    ax.set_xticks(range(len(comparison_names)))
    ax.set_xticklabels(comparison_names, fontsize=11)
    ax.set_ylabel('Mean Symmetric KL Divergence', fontsize=12)
    ax.set_title('KL Divergence Summary: Base Model (Yi-6B-Chat) vs Uncensored Model', fontsize=14)
    
    # Add value labels
    for bar, mean in zip(bars, means):
        ax.annotate(f'{mean:.4f}', 
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 5), textcoords='offset points',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'kl_summary.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: kl_summary.png")


def main():
    # Get script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, 'plots')
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load all data files
    data_files = {
        'base_no_ablation': 'base_no_ablation.pt',
        'base_with_ablation': 'base_with_ablation.pt',
        'uncensored_no_ablation': 'uncensored_no_ablation.pt',
        'uncensored_with_addition': 'uncensored_with_addition.pt',
    }
    
    print("Loading data files...")
    data_dict = {}
    for name, filename in data_files.items():
        filepath = os.path.join(script_dir, filename)
        print(f"  Loading: {filename}")
        data_dict[name] = load_data(filepath)
    
    print(f"\nDataset info:")
    for name, data in data_dict.items():
        intervention = data.get('intervention_type', 'unknown')
        print(f"  {name}: {data['num_prompts']} prompts, intervention={intervention}")
    
    print(f"\nComputing pairwise KL divergences...")
    pairwise_kl = compute_pairwise_kl(data_dict)
    
    print(f"\nGenerating plots in: {output_dir}")
    print("-" * 50)
    
    # Generate all plots
    plot_kl_distributions(pairwise_kl, output_dir)
    plot_kl_comparison_bar(pairwise_kl, output_dir)
    plot_kl_heatmap(data_dict, output_dir)
    plot_kl_by_category(data_dict, pairwise_kl, output_dir)
    plot_intervention_effects(data_dict, output_dir)
    plot_summary_comparison(data_dict, output_dir)
    
    print("-" * 50)
    print(f"\nAll plots saved to: {output_dir}")
    
    # Print summary statistics
    print("\n=== Summary Statistics ===")
    for pair_name, kl_values in pairwise_kl.items():
        print(f"\n{pair_name}:")
        print(f"  Mean:   {np.mean(kl_values):.6f}")
        print(f"  Median: {np.median(kl_values):.6f}")
        print(f"  Std:    {np.std(kl_values):.6f}")
        print(f"  Min:    {np.min(kl_values):.6f}")
        print(f"  Max:    {np.max(kl_values):.6f}")


if __name__ == "__main__":
    main()
