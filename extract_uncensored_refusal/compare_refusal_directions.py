import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt
import numpy as np


class LinearProbe(nn.Module):
    """Simple linear classifier for binary classification."""
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1, bias=True)
    
    def forward(self, x):
        return self.linear(x)
    
    def get_direction(self):
        """Return the normalized weight vector as the refusal direction."""
        weight = self.linear.weight.data.squeeze()  # [d_model]
        return weight / weight.norm()


def load_probe_direction(probe_path, d_model=4096):
    """Load probe and extract its weight vector as the direction."""
    probe = LinearProbe(d_model)
    probe.load_state_dict(torch.load(probe_path, map_location='cpu'))
    return probe.get_direction()


def compute_angle(v1, v2):
    """Compute angle in degrees between two normalized vectors."""
    cos_sim = torch.dot(v1, v2).item()
    cos_sim = max(min(cos_sim, 1.0), -1.0)  # Clip for numerical stability
    theta = np.arccos(cos_sim)
    return np.degrees(theta), cos_sim


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_model_path = os.path.join(script_dir, "../pipeline/runs/yi-6b-chat/direction.pt")
    uncensored_dir_path = os.path.join(script_dir, "refusal_direction/direction.pt")
    probe_path = os.path.join(script_dir, "refusal_direction/probe.pt")
    output_image = os.path.join(script_dir, "direction_comparison.png")

    # Load base model direction
    print(f"Loading base refusal direction from: {base_model_path}")
    if not os.path.exists(base_model_path):
        print(f"Error: File not found: {base_model_path}")
        return
    base_direction = torch.load(base_model_path, map_location='cpu').float()
    base_direction = base_direction / base_direction.norm()

    # Load uncensored model mean-diff direction
    print(f"Loading uncensored refusal direction from: {uncensored_dir_path}")
    if not os.path.exists(uncensored_dir_path):
        print(f"Error: File not found: {uncensored_dir_path}")
        return
    uncensored_direction = torch.load(uncensored_dir_path, map_location='cpu').float()
    uncensored_direction = uncensored_direction / uncensored_direction.norm()

    # Load probe direction
    print(f"Loading probe direction from: {probe_path}")
    if not os.path.exists(probe_path):
        print(f"Warning: Probe not found: {probe_path}")
        probe_direction = None
    else:
        probe_direction = load_probe_direction(probe_path, d_model=base_direction.shape[0])
        probe_direction = probe_direction.float()

    # Check shapes
    if base_direction.shape != uncensored_direction.shape:
        print(f"Warning: Shapes mismatch: {base_direction.shape} vs {uncensored_direction.shape}")
        return

    # Compute pairwise angles
    angle_base_uncensored, cos_base_uncensored = compute_angle(base_direction, uncensored_direction)
    
    print(f"\n{'='*50}")
    print("Pairwise Comparisons:")
    print(f"{'='*50}")
    print(f"Base vs Uncensored:  {angle_base_uncensored:.2f}° (cos={cos_base_uncensored:.4f})")
    
    if probe_direction is not None:
        angle_base_probe, cos_base_probe = compute_angle(base_direction, probe_direction)
        angle_uncensored_probe, cos_uncensored_probe = compute_angle(uncensored_direction, probe_direction)
        
        print(f"Base vs Probe:       {angle_base_probe:.2f}° (cos={cos_base_probe:.4f})")
        print(f"Uncensored vs Probe: {angle_uncensored_probe:.2f}° (cos={cos_uncensored_probe:.4f})")
    print(f"{'='*50}")
    
    # Visualization - show all vectors in 2D projection
    # Base aligned with X-axis, others positioned by their angles
    theta1 = np.radians(angle_base_uncensored)
    
    v_base = np.array([1, 0])
    v_uncensored = np.array([np.cos(theta1), np.sin(theta1)])
    
    plt.figure(figsize=(8, 8))
    
    # Plot vectors
    plt.quiver(0, 0, v_base[0], v_base[1], angles='xy', scale_units='xy', scale=1, 
               color='blue', label=f'Base Model ({0:.1f}°)', width=0.02)
    plt.quiver(0, 0, v_uncensored[0], v_uncensored[1], angles='xy', scale_units='xy', scale=1, 
               color='red', label=f'Uncensored Mean-Diff ({angle_base_uncensored:.1f}°)', width=0.02)
    
    if probe_direction is not None:
        theta2 = np.radians(angle_base_probe)
        # Determine sign based on relationship with uncensored
        if cos_uncensored_probe > 0:
            v_probe = np.array([np.cos(theta2), np.sin(theta2)])
        else:
            v_probe = np.array([np.cos(theta2), -np.sin(theta2)])
        
        plt.quiver(0, 0, v_probe[0], v_probe[1], angles='xy', scale_units='xy', scale=1, 
                   color='green', label=f'Probe Direction ({angle_base_probe:.1f}°)', width=0.02)
    
    # Draw arcs
    angles_arc = np.linspace(0, theta1, 50)
    plt.plot(0.4*np.cos(angles_arc), 0.4*np.sin(angles_arc), 'r--', alpha=0.5)
    plt.text(0.45*np.cos(theta1/2), 0.45*np.sin(theta1/2), f"{angle_base_uncensored:.1f}°", 
             color='red', fontsize=10)
    
    if probe_direction is not None:
        angles_arc2 = np.linspace(0, theta2, 50)
        plt.plot(0.3*np.cos(angles_arc2), 0.3*np.sin(angles_arc2), 'g--', alpha=0.5)
        plt.text(0.35*np.cos(theta2/2), 0.35*np.sin(theta2/2) - 0.05, f"{angle_base_probe:.1f}°", 
                 color='green', fontsize=10)

    plt.xlim(-0.3, 1.3)
    plt.ylim(-0.5, 1.3)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='upper left')
    plt.title("Refusal Direction Comparison\n(Angles relative to Base Model)")
    
    plt.savefig(output_image, dpi=150, bbox_inches='tight')
    print(f"\nSaved visualization to {output_image}")


if __name__ == "__main__":
    main()

