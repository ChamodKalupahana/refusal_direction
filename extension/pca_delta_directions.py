"""
Find dominant changed directions by performing PCA on the delta matrix.

This script loads the delta matrix D ∈ R^{N x d} and performs PCA to find
the principal directions of variation in the activation differences.
"""

import torch
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os


def load_delta_matrix(path: str) -> dict:
    """Load the delta matrix and metadata."""
    data = torch.load(path)
    return data


def perform_pca(D: torch.Tensor, n_components: int = 10) -> dict:
    """
    Perform PCA on the delta matrix to find dominant directions.
    
    Args:
        D: Delta matrix of shape [N, d_model]
        n_components: Number of principal components to extract
        
    Returns:
        Dictionary containing PCA results
    """
    # Convert to numpy for sklearn
    D_np = D.cpu().float().numpy()
    
    # Center the data
    mean = D_np.mean(axis=0)
    D_centered = D_np - mean
    
    # Perform PCA
    n_components = min(n_components, D_np.shape[0], D_np.shape[1])
    pca = PCA(n_components=n_components)
    pca.fit(D_centered)
    
    return {
        'components': torch.from_numpy(pca.components_),  # [n_components, d_model]
        'explained_variance': pca.explained_variance_,
        'explained_variance_ratio': pca.explained_variance_ratio_,
        'cumulative_variance_ratio': np.cumsum(pca.explained_variance_ratio_),
        'mean': torch.from_numpy(mean),
        'singular_values': pca.singular_values_,
    }


def analyze_directions(pca_results: dict, original_direction: torch.Tensor = None):
    """
    Analyze the PCA results and optionally compare with original refusal direction.
    """
    print("\n" + "="*60)
    print("PCA Analysis Results")
    print("="*60)
    
    components = pca_results['components']
    var_ratio = pca_results['explained_variance_ratio']
    cum_var = pca_results['cumulative_variance_ratio']
    
    print(f"\nNumber of components: {len(var_ratio)}")
    print(f"Total variance explained: {cum_var[-1]*100:.2f}%")
    
    print("\nVariance explained by each component:")
    print("-" * 40)
    for i, (var, cum) in enumerate(zip(var_ratio, cum_var)):
        print(f"  PC{i+1}: {var*100:6.2f}% (cumulative: {cum*100:6.2f}%)")
    
    # Check component norms (should be ~1 for PCA)
    print("\nComponent norms (should be ~1):")
    for i, comp in enumerate(components):
        print(f"  PC{i+1}: {comp.norm().item():.4f}")
    
    # If original direction is provided, compute alignment
    if original_direction is not None:
        print("\n" + "-"*40)
        print("Alignment with original refusal direction:")
        print("-"*40)
        
        orig_dir = original_direction.cpu().to(components.dtype).flatten()  # Move to CPU and ensure 1D
        orig_dir_normalized = orig_dir / (orig_dir.norm() + 1e-8)
        
        for i, comp in enumerate(components):
            comp_normalized = comp / (comp.norm() + 1e-8)
            cosine_sim = torch.dot(orig_dir_normalized, comp_normalized).item()
            print(f"  PC{i+1} · refusal_dir: {cosine_sim:+.4f} (|cos|={abs(cosine_sim):.4f})")


def plot_variance_explained(pca_results: dict, save_path: str = None):
    """Plot the variance explained by each component."""
    var_ratio = pca_results['explained_variance_ratio']
    cum_var = pca_results['cumulative_variance_ratio']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Individual variance
    ax1.bar(range(1, len(var_ratio) + 1), var_ratio * 100)
    ax1.set_xlabel('Principal Component')
    ax1.set_ylabel('Variance Explained (%)')
    ax1.set_title('Variance Explained by Each Component')
    ax1.set_xticks(range(1, len(var_ratio) + 1))
    
    # Cumulative variance
    ax2.plot(range(1, len(cum_var) + 1), cum_var * 100, 'b-o')
    ax2.axhline(y=90, color='r', linestyle='--', label='90% threshold')
    ax2.set_xlabel('Number of Components')
    ax2.set_ylabel('Cumulative Variance Explained (%)')
    ax2.set_title('Cumulative Variance Explained')
    ax2.set_xticks(range(1, len(cum_var) + 1))
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to: {save_path}")
    
    plt.show()


def main():
    # Configuration
    delta_matrix_path = "extension/delta_matrix.pt"
    n_components = 10
    output_path = "extension/pca_directions.pt"
    plot_path = "extension/pca_variance_plot.png"
    
    # Optionally load original refusal direction for comparison
    direction_path = "pipeline/runs/yi-6b-chat/direction.pt"
    
    # Load delta matrix
    print(f"Loading delta matrix from: {delta_matrix_path}")
    data = load_delta_matrix(delta_matrix_path)
    D = data['D']
    print(f"Delta matrix shape: {D.shape}")
    print(f"  N (num prompts): {D.shape[0]}")
    print(f"  d (model dim): {D.shape[1]}")
    
    # Perform PCA
    print(f"\nPerforming PCA with {n_components} components...")
    pca_results = perform_pca(D, n_components=n_components)
    
    # Load original direction if available
    original_direction = None
    if os.path.exists(direction_path):
        print(f"Loading original refusal direction from: {direction_path}")
        original_direction = torch.load(direction_path)
    
    # Analyze results
    analyze_directions(pca_results, original_direction)
    
    # Plot variance explained
    plot_variance_explained(pca_results, save_path=plot_path)
    
    # Save PCA results
    print(f"\nSaving PCA directions to: {output_path}")
    torch.save({
        'components': pca_results['components'],  # [n_components, d_model]
        'explained_variance_ratio': pca_results['explained_variance_ratio'],
        'cumulative_variance_ratio': pca_results['cumulative_variance_ratio'],
        'mean': pca_results['mean'],
        'singular_values': pca_results['singular_values'],
        'source_delta_matrix': delta_matrix_path,
        'layer': data.get('layer'),
        'position': data.get('position'),
    }, output_path)
    print("Done!")


if __name__ == "__main__":
    main()
