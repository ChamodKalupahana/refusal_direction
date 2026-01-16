"""
Extract refusal direction from uncensored model using a linear probe.

Instead of simple difference of means (harmful - harmless), this script trains
a logistic regression classifier to distinguish harmful from harmless activations.
The classifier's weight vector is then used as the refusal direction.

This probe-based approach often finds a more meaningful separation direction
than simple mean subtraction at the same layer.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import os
import json
import sys
import random
import gc
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

# Add project root to sys.path to allow importing from pipeline
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from dataset.load_dataset import load_dataset_split
from pipeline.model_utils.model_factory import construct_model_base


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


def get_all_activations(model, tokenizer, instructions, tokenize_instructions_fn, target_layer, positions, batch_size=16):
    """
    Extract activations for all instructions at specified positions.
    Returns tensor of shape [n_samples, n_positions, d_model].
    """
    torch.cuda.empty_cache()
    
    n_positions = len(positions)
    d_model = model.config.hidden_size
    
    all_activations = []
    
    for i in tqdm(range(0, len(instructions), batch_size)):
        batch_instructions = instructions[i:i+batch_size]
        inputs = tokenize_instructions_fn(instructions=batch_instructions)
        
        with torch.inference_mode():
            outputs = model(
                input_ids=inputs.input_ids.to(model.device),
                attention_mask=inputs.attention_mask.to(model.device),
                output_hidden_states=True,
                use_cache=False,
            )
        
        # hidden_states[0] is embeddings, hidden_states[layer+1] is output of layer
        hidden_states = outputs.hidden_states[target_layer + 1]  # [batch, seq, d_model]
        
        # Extract activations at specified positions
        batch_activations = []
        for batch_idx in range(hidden_states.shape[0]):
            pos_activations = []
            for pos in positions:
                activation = hidden_states[batch_idx, pos, :].cpu().to(torch.float32)
                pos_activations.append(activation)
            batch_activations.append(torch.stack(pos_activations))  # [n_positions, d_model]
        
        all_activations.extend(batch_activations)
        
        # Clear cache periodically
        if (i // batch_size) % 5 == 0:
            gc.collect()
            torch.cuda.empty_cache()
    
    return torch.stack(all_activations)  # [n_samples, n_positions, d_model]


def train_probe(X_train, y_train, X_val, y_val, device, epochs=100, lr=0.01, weight_decay=0.01):
    """
    Train a linear probe to classify harmful (1) vs harmless (0) activations.
    Returns the trained probe and validation metrics.
    """
    d_model = X_train.shape[1]
    probe = LinearProbe(d_model).to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32, device=device).unsqueeze(1)
    X_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32, device=device).unsqueeze(1)
    
    best_val_acc = 0
    best_state = None
    
    for epoch in range(epochs):
        probe.train()
        optimizer.zero_grad()
        
        logits = probe(X_train_t)
        loss = criterion(logits, y_train_t)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        # Validation
        probe.eval()
        with torch.no_grad():
            val_logits = probe(X_val_t)
            val_probs = torch.sigmoid(val_logits).cpu().numpy()
            val_preds = (val_probs > 0.5).astype(int)
            val_acc = accuracy_score(y_val, val_preds)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = probe.state_dict().copy()
        
        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}/{epochs}: Loss={loss.item():.4f}, Val Acc={val_acc:.4f}")
    
    # Load best model
    probe.load_state_dict(best_state)
    
    # Final evaluation
    probe.eval()
    with torch.no_grad():
        val_logits = probe(X_val_t)
        val_probs = torch.sigmoid(val_logits).cpu().numpy()
        val_preds = (val_probs > 0.5).astype(int)
        final_acc = accuracy_score(y_val, val_preds)
        try:
            final_auc = roc_auc_score(y_val, val_probs)
        except:
            final_auc = 0.0
    
    return probe, final_acc, final_auc


def main():
    # Configuration
    model_path = "spkgyk/Yi-6B-Chat-uncensored"
    N_TRAIN = 256  # Number of prompts to use per class
    VAL_SPLIT = 0.2  # Fraction for validation
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Load layer and position from base model's metadata
    base_metadata_path = os.path.join(script_dir, "../pipeline/runs/yi-6b-chat/direction_metadata.json")
    print(f"Loading direction config from: {base_metadata_path}")
    with open(base_metadata_path, 'r') as f:
        base_metadata = json.load(f)
    
    target_layer = base_metadata["layer"]
    target_pos = base_metadata["pos"]  # Should be -5
    
    # Use same positions as base model
    positions = list(range(-5, 0))  # [-5, -4, -3, -2, -1]
    
    print(f"Using layer={target_layer}, positions={positions}")
    
    # Output files
    output_dir = os.path.join(script_dir, "refusal_direction")
    os.makedirs(output_dir, exist_ok=True)
    
    direction_path = os.path.join(output_dir, "direction.pt")
    metadata_path = os.path.join(output_dir, "direction_metadata.json")
    probe_path = os.path.join(output_dir, "probe.pt")
    
    # Load model
    print(f"Loading model: {model_path}...")
    model_base = construct_model_base(model_path)
    model = model_base.model
    tokenizer = model_base.tokenizer
    tokenize_instructions_fn = model_base.tokenize_instructions_fn
    
    print("Model loaded successfully.")
    
    # Load datasets
    random.seed(42)
    print(f"\nLoading harmful train prompts...")
    harmful_train = random.sample(
        load_dataset_split(harmtype='harmful', split='train', instructions_only=True), 
        N_TRAIN
    )
    print(f"Loaded {len(harmful_train)} harmful prompts")
    
    print(f"Loading harmless train prompts...")
    harmless_train = random.sample(
        load_dataset_split(harmtype='harmless', split='train', instructions_only=True), 
        N_TRAIN
    )
    print(f"Loaded {len(harmless_train)} harmless prompts")
    
    # Extract activations
    print(f"\nExtracting activations for harmful prompts at layer {target_layer}...")
    harmful_activations = get_all_activations(
        model, tokenizer, harmful_train, tokenize_instructions_fn, 
        target_layer, positions
    )
    print(f"Harmful activations shape: {harmful_activations.shape}")
    
    print(f"\nExtracting activations for harmless prompts at layer {target_layer}...")
    harmless_activations = get_all_activations(
        model, tokenizer, harmless_train, tokenize_instructions_fn, 
        target_layer, positions
    )
    print(f"Harmless activations shape: {harmless_activations.shape}")
    
    # Select activations at target position
    pos_index = positions.index(target_pos)
    harmful_at_pos = harmful_activations[:, pos_index, :].numpy()  # [n_harmful, d_model]
    harmless_at_pos = harmless_activations[:, pos_index, :].numpy()  # [n_harmless, d_model]
    
    print(f"\nActivations at position {target_pos}:")
    print(f"  Harmful: {harmful_at_pos.shape}")
    print(f"  Harmless: {harmless_at_pos.shape}")
    
    # Prepare data for probe training
    # Harmful = 1, Harmless = 0
    X = np.concatenate([harmful_at_pos, harmless_at_pos], axis=0)
    y = np.concatenate([np.ones(len(harmful_at_pos)), np.zeros(len(harmless_at_pos))])
    
    # Split into train/val
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=VAL_SPLIT, random_state=42, stratify=y
    )
    
    print(f"\nTraining probe on {len(X_train)} samples, validating on {len(X_val)} samples...")
    
    # Train probe
    device = model.device
    probe, val_acc, val_auc = train_probe(
        X_train, y_train, X_val, y_val, device,
        epochs=100, lr=0.01, weight_decay=0.01
    )
    
    print(f"\nProbe training complete!")
    print(f"  Validation Accuracy: {val_acc:.4f}")
    print(f"  Validation AUC: {val_auc:.4f}")
    
    # Extract refusal direction from probe weights
    refusal_dir_normalized = probe.get_direction().cpu()
    
    print(f"\nRefusal direction computed at layer {target_layer}, pos={target_pos}.")
    print(f"Direction shape: {refusal_dir_normalized.shape}")
    
    # Compare with difference-of-means direction
    mean_diff = harmful_at_pos.mean(axis=0) - harmless_at_pos.mean(axis=0)
    mean_diff_normalized = mean_diff / np.linalg.norm(mean_diff)
    
    cosine_sim = np.dot(refusal_dir_normalized.numpy(), mean_diff_normalized)
    angle_deg = np.arccos(np.clip(cosine_sim, -1, 1)) * 180 / np.pi
    
    print(f"\nComparison with difference-of-means direction:")
    print(f"  Cosine similarity: {cosine_sim:.4f}")
    print(f"  Angle: {angle_deg:.2f} degrees")
    
    # Save direction
    torch.save(refusal_dir_normalized.to(torch.float32), direction_path)
    print(f"\nSaved direction to {direction_path}")
    
    # Save probe
    torch.save(probe.state_dict(), probe_path)
    print(f"Saved probe to {probe_path}")
    
    # Save metadata
    metadata = {
        "model": model_path,
        "layer": target_layer,
        "pos": target_pos,
        "positions_computed": positions,
        "n_harmful": len(harmful_train),
        "n_harmless": len(harmless_train),
        "method": "linear_probe",
        "probe_val_accuracy": val_acc,
        "probe_val_auc": val_auc,
        "angle_vs_mean_diff_deg": float(angle_deg),
        "note": "Refusal direction extracted from linear probe weights trained to classify harmful vs harmless"
    }
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    print(f"Saved metadata to {metadata_path}")


if __name__ == "__main__":
    main()
