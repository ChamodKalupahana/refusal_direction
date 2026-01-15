import torch
import os

def main():
    base_model_dir = "pipeline/runs/llama-2-7b-chat-hf/direction.pt"
    uncensored_model_dir = "pipeline/runs/llama2_7b_chat_uncensored/direction.pt"

    print(f"Loading base refusal direction from: {base_model_dir}")
    if not os.path.exists(base_model_dir):
        print(f"Error: File not found: {base_model_dir}")
        return
    base_direction = torch.load(base_model_dir)

    print(f"Loading uncensored refusal direction from: {uncensored_model_dir}")
    if not os.path.exists(uncensored_model_dir):
        print(f"Error: File not found: {uncensored_model_dir}")
        return
    uncensored_direction = torch.load(uncensored_model_dir)

    # Ensure they are on the same device and type
    device = torch.device("cpu")
    base_direction = base_direction.to(device)
    uncensored_direction = uncensored_direction.to(device)
    
    # Check shapes
    if base_direction.shape != uncensored_direction.shape:
        print(f"Warning: Shapes mismatch: {base_direction.shape} vs {uncensored_direction.shape}")
        return

    # Compute cosine similarity
    # Dim=0 because they are 1D vectors [d_model]
    similarity = torch.nn.functional.cosine_similarity(base_direction, uncensored_direction, dim=0)

    print(f"\nCosine Similarity: {similarity.item():.4f}")

if __name__ == "__main__":
    main()
