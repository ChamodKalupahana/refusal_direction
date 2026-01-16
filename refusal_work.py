import argparse
import os
import random
import sys
from typing import Dict, List

import matplotlib.pyplot as plt
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from dataset.load_dataset import load_dataset_split
from pipeline.model_utils.yi_model import YiModel
from pipeline.submodules.generate_directions import generate_directions
from pipeline.submodules.select_direction import select_direction
from pipeline.utils.hook_utils import add_hooks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refusal direction analysis for Yi.")
    parser.add_argument("--model_path", type=str, default="01-ai/Yi-6B-Chat")
    parser.add_argument(
        "--model_paths",
        type=str,
        default="",
        help="Comma-separated list of model paths for side-by-side comparison.",
    )
    parser.add_argument("--artifact_dir", type=str, default="pipeline/runs/refusal_work")
    parser.add_argument("--n_train", type=int, default=128)
    parser.add_argument("--n_val", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--pos", type=int, default=-1, help="Token position to analyze (default: last token).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_plots", action="store_true", help="Save plots to pipeline/plots.")
    return parser.parse_args()


@torch.no_grad()
def compute_projection_stats(
    model_base,
    prompts: List[str],
    r_hat: torch.Tensor,
    pos: int = -1,
    batch_size: int = 8,
) -> Dict[str, torch.Tensor]:
    model = model_base.model
    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads
    d_model = model.config.hidden_size
    head_dim = d_model // n_heads

    r_hat = r_hat.to(device=model.device, dtype=model.dtype)
    r_hat_heads = r_hat.view(n_heads, head_dim)

    head_sums = torch.zeros((n_layers, n_heads), dtype=torch.float64, device=model.device)
    attn_sums = torch.zeros((n_layers,), dtype=torch.float64, device=model.device)
    mlp_sums = torch.zeros((n_layers,), dtype=torch.float64, device=model.device)
    head_available = torch.zeros((n_layers,), dtype=torch.bool, device=model.device)

    def make_attn_hook(layer_idx: int):
        def hook_fn(module, input, output):
            attn_out = output[0] if isinstance(output, tuple) else output
            if attn_out is None:
                return

            if attn_out.ndim == 4:
                if attn_out.shape[1] == n_heads:  # [B, H, T, D]
                    head_out = attn_out.transpose(1, 2)
                elif attn_out.shape[2] == n_heads:  # [B, T, H, D]
                    head_out = attn_out
                else:
                    head_out = None
            elif attn_out.ndim == 3 and attn_out.shape[-1] == n_heads * head_dim:
                head_out = attn_out.view(attn_out.shape[0], attn_out.shape[1], n_heads, head_dim)
            else:
                head_out = None

            if head_out is not None:
                head_available[layer_idx] = True
                head_token_out = head_out[:, pos, :, :]
                head_token_out = head_token_out.to(r_hat_heads)
                if head_token_out.shape[-1] == head_dim:
                    proj = torch.einsum("b h d, h d -> b h", head_token_out, r_hat_heads)
                else:
                    proj = torch.einsum("b h d, d -> b h", head_token_out, r_hat)
                head_sums[layer_idx] += proj.sum(dim=0)
                attn_sums[layer_idx] += proj.sum()
            else:
                if attn_out.ndim == 3 and attn_out.shape[-1] == d_model:
                    token_out = attn_out[:, pos, :]
                    token_out = token_out.to(r_hat)
                    proj = torch.einsum("b d, d -> b", token_out, r_hat)
                    attn_sums[layer_idx] += proj.sum()

        return hook_fn

    def make_mlp_hook(layer_idx: int):
        def hook_fn(module, input, output):
            mlp_out = output[0] if isinstance(output, tuple) else output
            if mlp_out is None or mlp_out.ndim != 3:
                return
            token_out = mlp_out[:, pos, :]
            token_out = token_out.to(r_hat)
            proj = torch.einsum("b d, d -> b", token_out, r_hat)
            mlp_sums[layer_idx] += proj.sum()

        return hook_fn

    fwd_hooks = []
    for layer in range(n_layers):
        fwd_hooks.append((model_base.model_attn_modules[layer], make_attn_hook(layer)))
        fwd_hooks.append((model_base.model_mlp_modules[layer], make_mlp_hook(layer)))

    n_seen = 0
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        inputs = model_base.tokenize_instructions_fn(instructions=batch)
        with add_hooks(module_forward_pre_hooks=[], module_forward_hooks=fwd_hooks):
            model(
                input_ids=inputs.input_ids.to(model.device),
                attention_mask=inputs.attention_mask.to(model.device),
            )
        n_seen += len(batch)

    return {
        "head_means": head_sums / max(n_seen, 1),
        "attn_means": attn_sums / max(n_seen, 1),
        "mlp_means": mlp_sums / max(n_seen, 1),
        "head_available": head_available,
    }

def save_model_plots(model_label: str, head_scores, attn_scores, mlp_scores, plots_dir):
    model_dir = os.path.join(plots_dir, model_label)
    os.makedirs(model_dir, exist_ok=True)

    head_scores_cpu = head_scores.detach().cpu()
    attn_scores_cpu = attn_scores.detach().cpu()
    mlp_scores_cpu = mlp_scores.detach().cpu()
    plt.figure(figsize=(12, 6))
    plt.imshow(head_scores_cpu, aspect="auto", cmap="coolwarm", interpolation="nearest")
    plt.colorbar(label="Head contribution (harmful - harmless)")
    plt.xlabel("Head")
    plt.ylabel("Layer")
    plt.title(f"{model_label} - head contribution to refusal direction")
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, "head_contribution_heatmap.png"))
    plt.close()

    plt.figure(figsize=(12, 4))
    plt.plot(attn_scores_cpu, label="Attention", marker="o")
    plt.plot(mlp_scores_cpu, label="MLP", marker="o")
    plt.axhline(0.0, color="black", linewidth=0.5)
    plt.xlabel("Layer")
    plt.ylabel("Mean projection (harmful - harmless)")
    plt.title(f"{model_label} - layer-wise contribution to refusal direction")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, "layer_contributions.png"))
    plt.close()


def analyze_model(model_path: str, args, datasets, plots_dir=None) -> Dict[str, torch.Tensor]:
    model_base = YiModel(model_path)

    harmful_train, harmless_train, harmful_val, harmless_val = datasets

    mean_diffs = generate_directions(
        model_base,
        harmful_train,
        harmless_train,
        artifact_dir=f"{args.artifact_dir}/{os.path.basename(model_path)}/generate_directions",
    )

    pos, layer, direction = select_direction(
        model_base,
        harmful_val,
        harmless_val,
        candidate_directions=mean_diffs,
        artifact_dir=f"{args.artifact_dir}/{os.path.basename(model_path)}/select_direction",
        batch_size=args.batch_size,
    )

    r_hat = direction / (direction.norm() + 1e-8)

    harmful_stats = compute_projection_stats(
        model_base,
        harmful_val,
        r_hat,
        pos=args.pos,
        batch_size=args.batch_size,
    )
    harmless_stats = compute_projection_stats(
        model_base,
        harmless_val,
        r_hat,
        pos=args.pos,
        batch_size=args.batch_size,
    )

    head_scores = harmful_stats["head_means"] - harmless_stats["head_means"]
    attn_scores = harmful_stats["attn_means"] - harmless_stats["attn_means"]
    mlp_scores = harmful_stats["mlp_means"] - harmless_stats["mlp_means"]

    model_label = os.path.basename(model_path)
    print(f"\n[{model_label}] Layer-wise attention and MLP contributions (harmful - harmless):")
    for l in range(attn_scores.shape[0]):
        print(f"  layer {l:02d}: attn={attn_scores[l]:.6f} | mlp={mlp_scores[l]:.6f}")

    if harmful_stats["head_available"].any().item():
        flat_scores = head_scores.flatten()
        topk = min(20, flat_scores.numel())
        vals, idxs = torch.topk(flat_scores, k=topk)
        n_heads = model_base.model.config.num_attention_heads
        print(f"\n[{model_label}] Top head writers:")
        for v, i in zip(vals.tolist(), idxs.tolist()):
            layer_idx = i // n_heads
            head_idx = i % n_heads
            print(f"  layer {layer_idx:02d}, head {head_idx:02d}: {v:.6f}")
    else:
        print(f"\n[{model_label}] Per-head outputs not exposed; reporting layer-level only.")

    if plots_dir is not None:
        save_model_plots(model_label, head_scores, attn_scores, mlp_scores, plots_dir)

    return {
        "head_scores": head_scores,
        "attn_scores": attn_scores,
        "mlp_scores": mlp_scores,
        "head_available": harmful_stats["head_available"],
    }


def plot_side_by_side(attn_list, mlp_list, labels, plots_dir):
    n_models = len(labels)
    fig, axes = plt.subplots(2, n_models, figsize=(4 * n_models, 6), squeeze=False)

    for idx, label in enumerate(labels):
        attn = attn_list[idx].detach().cpu().unsqueeze(0)
        mlp = mlp_list[idx].detach().cpu().unsqueeze(0)

        ax_attn = axes[0, idx]
        im_attn = ax_attn.imshow(attn, aspect="auto", cmap="coolwarm", interpolation="nearest")
        ax_attn.set_title(f"{label} - attn")
        ax_attn.set_yticks([])
        ax_attn.set_xlabel("Layer")
        fig.colorbar(im_attn, ax=ax_attn, fraction=0.046, pad=0.04)

        ax_mlp = axes[1, idx]
        im_mlp = ax_mlp.imshow(mlp, aspect="auto", cmap="coolwarm", interpolation="nearest")
        ax_mlp.set_title(f"{label} - mlp")
        ax_mlp.set_yticks([])
        ax_mlp.set_xlabel("Layer")
        fig.colorbar(im_mlp, ax=ax_mlp, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "attn_mlp_side_by_side.png"))
    plt.close(fig)


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    plots_dir = None
    if args.save_plots:
        plots_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
        os.makedirs(plots_dir, exist_ok=True)

    harmful_train_all = load_dataset_split(harmtype="harmful", split="train", instructions_only=True)
    harmless_train_all = load_dataset_split(harmtype="harmless", split="train", instructions_only=True)
    harmful_val_all = load_dataset_split(harmtype="harmful", split="val", instructions_only=True)
    harmless_val_all = load_dataset_split(harmtype="harmless", split="val", instructions_only=True)

    datasets = (
        random.sample(harmful_train_all, min(args.n_train, len(harmful_train_all))),
        random.sample(harmless_train_all, min(args.n_train, len(harmless_train_all))),
        random.sample(harmful_val_all, min(args.n_val, len(harmful_val_all))),
        random.sample(harmless_val_all, min(args.n_val, len(harmless_val_all))),
    )

    model_paths = [p.strip() for p in args.model_paths.split(",") if p.strip()]
    if not model_paths:
        model_paths = [args.model_path]

    results = []
    labels = []
    for model_path in model_paths:
        results.append(analyze_model(model_path, args, datasets, plots_dir=plots_dir))
        labels.append(os.path.basename(model_path))

    if args.save_plots and plots_dir:
        attn_list = [r["attn_scores"] for r in results]
        mlp_list = [r["mlp_scores"] for r in results]
        plot_side_by_side(attn_list, mlp_list, labels, plots_dir)


if __name__ == "__main__":
    main()
