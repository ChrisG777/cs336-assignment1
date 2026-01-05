#!/usr/bin/env python3
"""
Evaluate all trained models from hyperparameter search on validation set.

Usage:
    python scripts/evaluate_all_models.py
    python scripts/evaluate_all_models.py --checkpoints-dir experiments --val-dataset tokenized/tinystories_valid.npy
"""

import os
import sys
import json
import argparse
from pathlib import Path

import torch
import numpy as np
from einops import rearrange

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cs336_basics.model import Transformer_LM
from cs336_basics.optimizer import cross_entropy
from cs336_basics.training import get_batch


def evaluate_model(
    model: Transformer_LM,
    val_dataset: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str,
    num_batches: int = 50,
) -> float:
    """Evaluate model on validation set, return average loss."""
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for _ in range(num_batches):
            inputs, labels = get_batch(val_dataset, batch_size, context_length, device)
            logits = model(inputs)
            
            # Flatten for cross_entropy
            logits_flat = rearrange(logits, "b s v -> (b s) v")
            labels_flat = rearrange(labels, "b s -> (b s)")
            
            loss = cross_entropy(logits_flat, labels_flat)
            total_loss += loss.item()
    
    return total_loss / num_batches


def load_model_from_experiment(exp_dir: Path, device: str) -> tuple[Transformer_LM, dict] | None:
    """Load model from experiment directory. Returns (model, config) or None if failed."""
    
    # Find checkpoint file
    checkpoint_path = exp_dir / "checkpoint.pt"
    if not checkpoint_path.exists():
        # Try alternative names
        for name in ["ckpt.pt", "model.pt"]:
            alt_path = exp_dir / name
            if alt_path.exists():
                checkpoint_path = alt_path
                break
        else:
            return None
    
    # Try to infer config from experiment name
    # Format: ts_{size}_lr{lr}_bs{bs}_wd{wd}
    exp_name = exp_dir.name
    
    # Default configs for model sizes
    model_configs = {
        "small": {"d_model": 256, "num_layers": 4, "num_heads": 4, "d_ff": 1024},
        "medium": {"d_model": 384, "num_layers": 6, "num_heads": 6, "d_ff": 1536},
        "large": {"d_model": 512, "num_layers": 8, "num_heads": 8, "d_ff": 2048},
    }
    
    # Parse experiment name
    config = {
        "vocab_size": 10000,
        "context_length": 256,
        "rope_theta": 10000.0,
    }
    
    # Detect model size from name
    for size_name, size_config in model_configs.items():
        if f"_{size_name}_" in exp_name:
            config.update(size_config)
            break
    else:
        # Default to small if not found
        config.update(model_configs["small"])
    
    # Parse other params from name for logging
    parts = exp_name.split("_")
    for part in parts:
        if part.startswith("lr"):
            config["lr"] = part[2:]
        elif part.startswith("bs"):
            config["batch_size"] = int(part[2:])
        elif part.startswith("wd"):
            config["weight_decay"] = part[2:]
    
    try:
        # Create model
        model = Transformer_LM(
            vocab_size=config["vocab_size"],
            context_length=config["context_length"],
            d_model=config["d_model"],
            num_layers=config["num_layers"],
            num_heads=config["num_heads"],
            d_ff=config["d_ff"],
            rope_theta=config["rope_theta"],
            device=device,
        ).to(device)
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        
        return model, config
        
    except Exception as e:
        print(f"  Error loading {exp_name}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Evaluate all trained models")
    parser.add_argument("--experiments-dir", type=str, default="experiments",
                        help="Directory containing experiment folders")
    parser.add_argument("--val-dataset", type=str, default="tokenized/tinystories_valid.npy",
                        help="Path to validation tokens")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for evaluation")
    parser.add_argument("--num-batches", type=int, default=50,
                        help="Number of batches to evaluate")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use")
    parser.add_argument("--output", type=str, default="eval_results.json",
                        help="Output JSON file for results")
    args = parser.parse_args()
    
    # Load validation dataset
    print(f"Loading validation dataset: {args.val_dataset}")
    val_dataset = np.load(args.val_dataset, mmap_mode='r')
    print(f"  Shape: {val_dataset.shape}, Dtype: {val_dataset.dtype}")
    
    experiments_dir = Path(args.experiments_dir)
    if not experiments_dir.exists():
        print(f"Experiments directory not found: {experiments_dir}")
        return
    
    # Find all experiment directories
    exp_dirs = sorted([d for d in experiments_dir.iterdir() if d.is_dir()])
    print(f"\nFound {len(exp_dirs)} experiment directories")
    
    results = []
    
    print("\n" + "=" * 100)
    print(f"{'Experiment':<50} {'Val Loss':>12} {'Perplexity':>12} {'Status':>15}")
    print("=" * 100)
    
    for exp_dir in exp_dirs:
        exp_name = exp_dir.name
        
        # Skip non-training directories
        if not exp_name.startswith("ts_"):
            continue
        
        # Load model
        result = load_model_from_experiment(exp_dir, args.device)
        
        if result is None:
            print(f"{exp_name:<50} {'N/A':>12} {'N/A':>12} {'No checkpoint':>15}")
            results.append({
                "name": exp_name,
                "val_loss": None,
                "perplexity": None,
                "status": "no_checkpoint"
            })
            continue
        
        model, config = result
        
        try:
            # Evaluate
            val_loss = evaluate_model(
                model=model,
                val_dataset=val_dataset,
                batch_size=args.batch_size,
                context_length=config["context_length"],
                device=args.device,
                num_batches=args.num_batches,
            )
            perplexity = np.exp(val_loss)
            
            print(f"{exp_name:<50} {val_loss:>12.4f} {perplexity:>12.2f} {'OK':>15}")
            
            results.append({
                "name": exp_name,
                "val_loss": val_loss,
                "perplexity": perplexity,
                "config": config,
                "status": "ok"
            })
            
        except Exception as e:
            print(f"{exp_name:<50} {'N/A':>12} {'N/A':>12} {'Error':>15}")
            print(f"  Error: {e}")
            results.append({
                "name": exp_name,
                "val_loss": None,
                "perplexity": None,
                "status": f"error: {e}"
            })
        
        # Free memory
        del model
        torch.cuda.empty_cache() if args.device == "cuda" else None
    
    print("=" * 100)
    
    # Sort by validation loss
    valid_results = [r for r in results if r["val_loss"] is not None]
    valid_results.sort(key=lambda x: x["val_loss"])
    
    # Print top 5
    print("\n" + "=" * 60)
    print("TOP 5 MODELS BY VALIDATION LOSS")
    print("=" * 60)
    for i, r in enumerate(valid_results[:5], 1):
        print(f"{i}. {r['name']}")
        print(f"   Val Loss: {r['val_loss']:.4f}, Perplexity: {r['perplexity']:.2f}")
    
    # Save results
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()

