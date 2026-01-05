#!/usr/bin/env python3
"""
Generate multiple text samples from a trained model.
Can be run interactively or with predefined prompts.
"""

import argparse
import torch
import sys
sys.path.insert(0, '.')

from cs336_basics.decoding.decoding import decode, load_model_from_checkpoint
from cs336_basics.BPE_tokenizer import Tokenizer


def main():
    parser = argparse.ArgumentParser(
        description="Generate text samples from a trained Transformer LM",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Model loading args
    parser.add_argument("--checkpoint", type=str, default="checkpoints/owt_model.pt",
                        help="Path to model checkpoint (.pt file)")
    parser.add_argument("--vocab", type=str, default="tokenized/vocab_owt_train.json",
                        help="Path to vocabulary JSON file")
    parser.add_argument("--merges", type=str, default="tokenized/merges_owt_train.txt",
                        help="Path to merges file")
    
    # Model architecture args (must match training - defaults for OWT)
    parser.add_argument("--vocab-size", type=int, default=32000,
                        help="Vocabulary size")
    parser.add_argument("--context-length", type=int, default=256,
                        help="Context length")
    parser.add_argument("--d-model", type=int, default=512,
                        help="Model dimension")
    parser.add_argument("--num-layers", type=int, default=4,
                        help="Number of transformer layers")
    parser.add_argument("--num-heads", type=int, default=16,
                        help="Number of attention heads")
    parser.add_argument("--d-ff", type=int, default=1344,
                        help="Feed-forward hidden dimension")
    parser.add_argument("--rope-theta", type=float, default=10000.0,
                        help="RoPE theta parameter")
    
    # Generation args
    parser.add_argument("--top-p", type=float, default=0.9,
                        help="Top-p (nucleus) sampling probability")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Sampling temperature (lower = more deterministic)")
    parser.add_argument("--max-tokens", type=int, default=200,
                        help="Maximum number of tokens to generate")
    parser.add_argument("--special-tokens", type=str, nargs="*", default=["<|endoftext|>"],
                        help="Special tokens")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cpu, cuda, mps)")
    
    # Mode selection
    parser.add_argument("--interactive", action="store_true",
                        help="Run in interactive mode")
    parser.add_argument("--prompts", type=str, nargs="*",
                        help="List of prompts to complete (if not interactive)")
    parser.add_argument("--num-samples", type=int, default=1,
                        help="Number of samples to generate per prompt")
    
    args = parser.parse_args()
    
    # Default prompts if none provided
    default_prompts = [
        "The future of artificial intelligence is",
        "In a world where technology",
        "The most important lesson I learned",
        "Scientists recently discovered that",
        "The key to success in life is",
    ]
    
    prompts = args.prompts if args.prompts else default_prompts
    
    # Load tokenizer
    print(f"Loading tokenizer from {args.vocab} and {args.merges}...")
    tokenizer = Tokenizer.from_files(args.vocab, args.merges, args.special_tokens)
    vocab = tokenizer.vocab
    
    # Get special token IDs
    special_token_ids = set()
    for st in args.special_tokens:
        st_bytes = st.encode("utf-8")
        for tid, tok in vocab.items():
            if tok == st_bytes:
                special_token_ids.add(tid)
                break
    
    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model = load_model_from_checkpoint(
        checkpoint_path=args.checkpoint,
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        device=args.device,
    )
    print(f"Model loaded! Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Generation settings: top-p={args.top_p}, temperature={args.temperature}, max_tokens={args.max_tokens}")
    print("=" * 60)
    
    if args.interactive:
        # Interactive mode
        print("Interactive mode. Type 'quit' or 'exit' to stop.")
        print("=" * 60)
        while True:
            try:
                prompt = input("\nPrompt: ").strip()
                if prompt.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                if not prompt:
                    continue
                    
                print("\nGenerating...\n")
                output = decode(
                    model=model,
                    tokenizer=tokenizer,
                    special_token_ids=special_token_ids,
                    prompt=prompt,
                    p=args.top_p,
                    temperature=args.temperature,
                    max_generated_tokens=args.max_tokens,
                    device=args.device,
                )
                print("-" * 40)
                print(output)
                print("-" * 40)
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
    else:
        # Batch mode with predefined prompts
        for i, prompt in enumerate(prompts):
            print(f"\n[Prompt {i+1}/{len(prompts)}]: {prompt}")
            print("-" * 60)
            
            for sample_idx in range(args.num_samples):
                if args.num_samples > 1:
                    print(f"\n  Sample {sample_idx + 1}:")
                
                output = decode(
                    model=model,
                    tokenizer=tokenizer,
                    special_token_ids=special_token_ids,
                    prompt=prompt,
                    p=args.top_p,
                    temperature=args.temperature,
                    max_generated_tokens=args.max_tokens,
                    device=args.device,
                )
                print(output)
            
            print("=" * 60)


if __name__ == "__main__":
    main()

