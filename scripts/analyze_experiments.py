#!/usr/bin/env python3
"""
Analyze hyperparameter search results.
Reads all experiment logs and creates a summary table.

Usage:
    python scripts/analyze_experiments.py
    python scripts/analyze_experiments.py --experiments-dir experiments
"""

import os
import json
import argparse
from pathlib import Path


def load_experiment(exp_dir: Path) -> dict | None:
    """Load config and final metrics from an experiment directory."""
    config_path = exp_dir / "config.json"
    metrics_path = exp_dir / "metrics.json"
    
    if not config_path.exists():
        return None
    
    with open(config_path) as f:
        config = json.load(f)
    
    metrics = []
    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)
    
    # Get final metrics
    final_train_loss = None
    final_val_loss = None
    final_iteration = 0
    
    for m in metrics:
        if "train_loss" in m:
            final_train_loss = m["train_loss"]
            final_iteration = m["iteration"]
        if "val_loss" in m:
            final_val_loss = m["val_loss"]
    
    return {
        "name": exp_dir.name,
        "config": config,
        "final_train_loss": final_train_loss,
        "final_val_loss": final_val_loss,
        "final_iteration": final_iteration,
        "num_metrics": len(metrics),
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze hyperparameter search results")
    parser.add_argument("--experiments-dir", type=str, default="experiments",
                        help="Directory containing experiment folders")
    parser.add_argument("--sort-by", type=str, default="final_val_loss",
                        choices=["final_val_loss", "final_train_loss", "name"],
                        help="Sort results by this metric")
    args = parser.parse_args()
    
    experiments_dir = Path(args.experiments_dir)
    if not experiments_dir.exists():
        print(f"Experiments directory not found: {experiments_dir}")
        return
    
    # Load all experiments
    experiments = []
    for exp_dir in experiments_dir.iterdir():
        if exp_dir.is_dir():
            exp = load_experiment(exp_dir)
            if exp:
                experiments.append(exp)
    
    if not experiments:
        print("No experiments found.")
        return
    
    # Sort by specified metric
    def sort_key(exp):
        if args.sort_by == "name":
            return exp["name"]
        val = exp.get(args.sort_by)
        return val if val is not None else float('inf')
    
    experiments.sort(key=sort_key)
    
    # Print summary table
    print("=" * 120)
    print("HYPERPARAMETER SEARCH RESULTS")
    print("=" * 120)
    print(f"{'Experiment':<45} {'d_model':>7} {'layers':>6} {'lr':>10} {'batch':>5} {'wd':>6} {'train_loss':>11} {'val_loss':>10} {'ppl':>8}")
    print("-" * 120)
    
    for exp in experiments:
        config = exp["config"]
        train_loss = f"{exp['final_train_loss']:.4f}" if exp['final_train_loss'] else "N/A"
        val_loss = f"{exp['final_val_loss']:.4f}" if exp['final_val_loss'] else "N/A"
        
        # Calculate perplexity
        if exp['final_val_loss']:
            import math
            ppl = f"{math.exp(exp['final_val_loss']):.2f}"
        else:
            ppl = "N/A"
        
        print(f"{exp['name']:<45} "
              f"{config.get('d_model', 'N/A'):>7} "
              f"{config.get('num_layers', 'N/A'):>6} "
              f"{config.get('max_learning_rate', 'N/A'):>10} "
              f"{config.get('batch_size', 'N/A'):>5} "
              f"{config.get('weight_decay', 'N/A'):>6} "
              f"{train_loss:>11} "
              f"{val_loss:>10} "
              f"{ppl:>8}")
    
    print("=" * 120)
    
    # Find best experiment
    valid_experiments = [e for e in experiments if e['final_val_loss'] is not None]
    if valid_experiments:
        best = min(valid_experiments, key=lambda e: e['final_val_loss'])
        print(f"\nBest experiment: {best['name']}")
        print(f"  Val Loss: {best['final_val_loss']:.4f}")
        print(f"  Perplexity: {math.exp(best['final_val_loss']):.2f}")
        print(f"\nConfig:")
        for k, v in best['config'].items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()

