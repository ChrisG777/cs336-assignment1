import torch
import numpy as np 
import os
import argparse
import matplotlib.pyplot as plt
from einops import rearrange 
from typing import IO, BinaryIO
from cs336_basics.model import Transformer_LM
from cs336_basics.optimizer import AdamW, cross_entropy, gradient_clipping, get_lr_cosine_schedule
from .data_loading import get_batch
from .checkpointing import save_checkpoint, load_checkpoint


def train(
    num_iterations: int, 
    warmup_iters: int, 
    cosine_cycle_iters: int,
    max_learning_rate: float, 
    min_learning_rate: float, 
    max_l2_norm: float, 
    alpha: float, 
    beta_1: float,
    beta_2: float, 
    epsilon: float, 
    lam: float, 
    checkpoint_path: str | os.PathLike | BinaryIO | IO[bytes], 
    train_dataset_path: str,
    val_dataset_path: str | None,
    batch_size: int, 
    context_length: int, 
    device: str, 
    vocab_size: int, 
    d_model: int, 
    num_layers: int,
    num_heads: int, 
    d_ff: int, 
    rope_theta: float,
    eval_interval: int = 100,
    eval_batches: int = 10,
): 
    """
    warmup_iters, cosine_cycle_iters, max_learning_rate, min_learning_rate are the params for the learning rate scheduler 

    max_l2_norm is for gradient clipping 

    alpha, beta_1, beta_2, epsilon, lam are for adamW 

    train_dataset_path, batch_size, device are for the dataloader

    vocab_size, d_model, num_layers, num_heads, d_ff, rope_theta are for the model 
    """

    # initialize the model 
    model = Transformer_LM(vocab_size=vocab_size, context_length=context_length, d_model=d_model, num_layers=num_layers, num_heads=num_heads, d_ff=d_ff, rope_theta=rope_theta, device=device).to(device)

    # initialize the optimizer
    optimizer = AdamW(model.parameters(), alpha=alpha, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon, lam=lam)

    # load the dataset using mmap 
    train_dataset = np.load(train_dataset_path, mmap_mode='r')
    val_dataset = np.load(val_dataset_path, mmap_mode='r') if val_dataset_path else None

    # Sanity check the dataset
    print(f"Train dataset: {len(train_dataset):,} tokens, dtype={train_dataset.dtype}")
    print(f"Max token ID in train: {train_dataset.max()}, vocab_size: {vocab_size}")
    if val_dataset is not None:
        print(f"Val dataset: {len(val_dataset):,} tokens")

    @torch.no_grad()
    def evaluate(dataset):
        """Compute average loss over eval_batches."""
        model.eval()
        total_loss = 0.0
        for _ in range(eval_batches):
            inputs, labels = get_batch(dataset, batch_size, context_length, device)
            logits = model(inputs)
            # Flatten for cross_entropy: (batch * context_length, vocab_size)

            logits_flat = rearrange(logits, "batch_size context_length vocab_size -> (batch_size context_length) vocab_size")
            labels_flat = rearrange(labels, "batch_size context_length -> (batch_size context_length)")
            loss = cross_entropy(logits_flat, labels_flat)
            total_loss += loss.item()
        model.train()
        return total_loss / eval_batches

    losses = []
    val_losses = []
    val_iters = []
    
    print("Starting training")
    model.train()
    
    for it in range(1, num_iterations + 1):
        inputs, labels = get_batch(train_dataset, batch_size, context_length, device)
        optimizer.zero_grad()
        logits = model(inputs) 

        # Flatten for cross_entropy, since the specs assume just one batch dimension: (batch * context_length, vocab_size) and (batch * context_length,)
        logits_flat = rearrange(logits, "batch_size context_length vocab_size -> (batch_size context_length) vocab_size")
        labels_flat = rearrange(labels, "batch_size context_length -> (batch_size context_length)")
        loss = cross_entropy(logits_flat, labels_flat)
        
        losses.append(loss.item())
        loss.backward() 

        gradient_clipping(model.parameters(), max_l2_norm)
        
        lr = get_lr_cosine_schedule(it, max_learning_rate=max_learning_rate, min_learning_rate=min_learning_rate, warmup_iters=warmup_iters, cosine_cycle_iters=cosine_cycle_iters)
        for group in optimizer.param_groups:
            group["alpha"] = lr 
        
        optimizer.step()

        if it % 10 == 0:
            print(f"iter {it}: train_loss={loss.item():.4f}, lr={lr:.2e}")

        # Evaluate and checkpoint periodically
        if it % eval_interval == 0:
            if val_dataset is not None:
                val_loss = evaluate(val_dataset)
                val_losses.append(val_loss)
                val_iters.append(it)
                print(f"iter {it}: val_loss={val_loss:.4f}, perplexity={np.exp(val_loss):.2f}")
            save_checkpoint(model, optimizer, iteration=it, out=checkpoint_path)
        
    # Save final checkpoint
    save_checkpoint(model, optimizer, iteration=num_iterations, out=checkpoint_path)
    
    # Plot the loss curve
    plot_losses(losses, val_losses, val_iters, checkpoint_path)
    
    return model, losses, val_losses


def plot_losses(train_losses, val_losses, val_iters, checkpoint_path):
    """Plot and save the training and validation loss curves."""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss', alpha=0.7)
    if val_losses:
        plt.plot(val_iters, val_losses, 'ro-', label='Val Loss', markersize=4)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save next to checkpoint
    plot_path = str(checkpoint_path).replace('.pt', '_loss.png')
    if plot_path == str(checkpoint_path):
        plot_path = str(checkpoint_path) + '_loss.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Loss plot saved to {plot_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Train a Transformer LM",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Dataset args
    parser.add_argument("--train-dataset", type=str, required=True,
                        help="Path to training tokens .npy file")
    parser.add_argument("--val-dataset", type=str, default=None,
                        help="Path to validation tokens .npy file")
    
    # Model args
    parser.add_argument("--vocab-size", type=int, required=True,
                        help="Vocabulary size")
    parser.add_argument("--context-length", type=int, default=256,
                        help="Context length (max sequence length)")
    parser.add_argument("--d-model", type=int, default=512,
                        help="Model dimension")
    parser.add_argument("--num-layers", type=int, default=4,
                        help="Number of transformer layers")
    parser.add_argument("--num-heads", type=int, default=8,
                        help="Number of attention heads")
    parser.add_argument("--d-ff", type=int, default=1024,
                        help="Feed-forward hidden dimension")
    parser.add_argument("--rope-theta", type=float, default=10000.0,
                        help="RoPE theta parameter")
    
    # Training args
    parser.add_argument("--num-iterations", type=int, default=1000,
                        help="Number of training iterations")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device to train on (cpu, cuda, mps)")
    
    # Optimizer args
    parser.add_argument("--max-lr", type=float, default=1e-3,
                        help="Maximum learning rate")
    parser.add_argument("--min-lr", type=float, default=1e-4,
                        help="Minimum learning rate")
    parser.add_argument("--warmup-iters", type=int, default=100,
                        help="Number of warmup iterations")
    parser.add_argument("--cosine-cycle-iters", type=int, default=1000,
                        help="Cosine cycle iterations")
    parser.add_argument("--weight-decay", type=float, default=0.1,
                        help="Weight decay (lambda)")
    parser.add_argument("--beta1", type=float, default=0.9,
                        help="Adam beta1")
    parser.add_argument("--beta2", type=float, default=0.999,
                        help="Adam beta2")
    parser.add_argument("--epsilon", type=float, default=1e-8,
                        help="Adam epsilon")
    parser.add_argument("--max-grad-norm", type=float, default=1.0,
                        help="Max gradient L2 norm for clipping")
    
    # Checkpointing args
    parser.add_argument("--checkpoint", type=str, default="checkpoint.pt",
                        help="Checkpoint file path")
    parser.add_argument("--eval-interval", type=int, default=100,
                        help="Evaluate every N iterations")
    parser.add_argument("--eval-batches", type=int, default=10,
                        help="Number of batches for evaluation")
    
    args = parser.parse_args()
    
    train(
        num_iterations=args.num_iterations,
        warmup_iters=args.warmup_iters,
        cosine_cycle_iters=args.cosine_cycle_iters,
        max_learning_rate=args.max_lr,
        min_learning_rate=args.min_lr,
        max_l2_norm=args.max_grad_norm,
        alpha=args.max_lr,  # Initial learning rate for AdamW
        beta_1=args.beta1,
        beta_2=args.beta2,
        epsilon=args.epsilon,
        lam=args.weight_decay,
        checkpoint_path=args.checkpoint,
        train_dataset_path=args.train_dataset,
        val_dataset_path=args.val_dataset,
        batch_size=args.batch_size,
        context_length=args.context_length,
        device=args.device,
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        eval_interval=args.eval_interval,
        eval_batches=args.eval_batches,
    )


if __name__ == "__main__":
    main()