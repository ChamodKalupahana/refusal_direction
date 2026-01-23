import torch
import os
import matplotlib.pyplot as plt
import numpy as np

# Change to parent directory so relative paths work correctly
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    base_model_dir = "pipeline/runs/llama-2-7b-chat-hf/direction.pt"
    uncensored_model_dir = "pipeline/runs/llama2_7b_chat_uncensored/direction.pt"
    output_image = "direction_comparison.png"

    print(f"Loading base refusal direction from: {base_model_dir}")
    if not os.path.exists(base_model_dir):
        print(f"Error: File not found: {base_model_dir}")
        return
    base_direction = torch.load(base_model_dir, map_location='cpu')

    print(f"Loading uncensored refusal direction from: {uncensored_model_dir}")
    if not os.path.exists(uncensored_model_dir):
        print(f"Error: File not found: {uncensored_model_dir}")
        return
    uncensored_direction = torch.load(uncensored_model_dir, map_location='cpu')
    
    # Ensure they are on the same device and type
    device = torch.device("cpu")
    base_direction = base_direction.to(device)
    uncensored_direction = uncensored_direction.to(device)

    # Check shapes
    if base_direction.shape != uncensored_direction.shape:
        print(f"Warning: Shapes mismatch: {base_direction.shape} vs {uncensored_direction.shape}")
        return

    # Normalize vectors
    base_direction = base_direction / base_direction.norm()
    uncensored_direction = uncensored_direction / uncensored_direction.norm()

    # Compute cosine similarity
    # Dim=0 because they are 1D vectors [d_model]
    cos_sim = torch.dot(base_direction, uncensored_direction).item()
    
    # Clip to avoid numerical errors slightly outside [-1, 1]
    cos_sim = max(min(cos_sim, 1.0), -1.0)
    
    theta = np.arccos(cos_sim)
    degrees = np.degrees(theta)

    print(f"\nCosine Similarity: {cos_sim:.4f}")
    print(f"Angle: {degrees:.2f} degrees")
    
    # Visualization
    # Vector A (Base) aligned with X-axis: (1, 0)
    # Vector B (Uncensored) at angle theta: (cos(theta), sin(theta))
    
    v_base = np.array([1, 0])
    v_uncensored = np.array([np.cos(theta), np.sin(theta)])
    
    plt.figure(figsize=(6, 6))
    
    # Plot unit circle arc for reference
    angles = np.linspace(0, theta + 0.1, 100) # Draw a bit past the vector
    plt.plot(np.cos(angles), np.sin(angles), 'k--', alpha=0.3)
    
    # Plot vectors
    plt.quiver(0, 0, v_base[0], v_base[1], angles='xy', scale_units='xy', scale=1, color='blue', label='Base Model Refusal Direction')
    plt.quiver(0, 0, v_uncensored[0], v_uncensored[1], angles='xy', scale_units='xy', scale=1, color='red', label='Uncensored Model Refusal Direction')
    
    # Annotate angle
    mid_angle = theta / 2
    dist = 0.3
    plt.text(dist * np.cos(mid_angle), dist * np.sin(mid_angle), f"{degrees:.1f}°", color='black', ha='center')

    plt.xlim(-0.2, 1.2)
    plt.ylim(-0.2, 1.2)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    plt.title(f"Refusal Direction Alignment\nCosine Similarity: {cos_sim:.4f}")
    
    plt.savefig(output_image)
    print(f"Saved visualization to {output_image}")

if __name__ == "__main__":
    main()
