# used cursor to generate this file, after writing the main function spec and a detailed english comments description of the training loop (which cursor sort of rewrote into its own comments)

import torch
import argparse
import json
from cs336_basics.model.transformer_lm import Transformer_LM
from cs336_basics.BPE_tokenizer import Tokenizer
from cs336_basics.model import softmax


def decode(
    model: Transformer_LM, 
    tokenizer: Tokenizer,
    special_token_ids: set[int],
    prompt: str, 
    p: float, 
    temperature: float, 
    max_generated_tokens: int,
    device: str = "cpu",
) -> str:
    """
    Generate text using top-p (nucleus) sampling.
    
    Args: 
    - model: the trained Transformer LM
    - tokenizer: the BPE tokenizer
    - special_token_ids: set of token IDs that are special tokens (e.g., <|endoftext|>)
    - prompt: the user defined prompt that we're completing 
    - p: the top-p sampling probability (nucleus sampling threshold)
    - temperature: the amount that we scale the logits by pre-softmax 
    - max_generated_tokens: a user defined limit on the number of tokens that we'll generate
    - device: device to run on
    
    Returns:
    - The generated text as a string
    """
    model.eval()
    
    prompt_tokens = tokenizer.encode(prompt)
    tokens = prompt_tokens.copy()
    
    generated_count = 0
    
    with torch.no_grad():
        while generated_count < max_generated_tokens:
            # Prepare input: batch size of 1
            input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
            
            # Forward pass to get logits: (1, seq_len, vocab_size)
            logits = model(input_ids)
            
            # Consider only the last token's logits: (vocab_size,)
            last_logits = logits[0, -1, :]
            
            # Divide by temperature (higher temp = more random)
            if temperature > 0:
                last_logits = last_logits / temperature
            
            # Take softmax to get probabilities
            probs = softmax(last_logits, dim=-1)
            
            # Sort probabilities in decreasing order, keeping track of original indices
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            
            # Compute cumulative sum of sorted probabilities
            cumulative_probs = torch.cumsum(sorted_probs, dim=0)
            
            # Binary search for the smallest index where cumulative probability exceeds p
            # searchsorted returns the index where p would be inserted to maintain sorted order
            # +1 to include the token that pushes us over p (and ensure at least 1 token)
            cutoff_idx = torch.searchsorted(cumulative_probs, p).item() + 1
            cutoff_idx = min(cutoff_idx, len(sorted_probs))  # clamp to valid range
            
            # Keep only the top tokens up to cutoff
            top_probs = sorted_probs[:cutoff_idx]
            top_indices = sorted_indices[:cutoff_idx]
            
            # Renormalize probabilities
            top_probs = top_probs / top_probs.sum()
            
            # Sample from the top tokens
            sampled_idx = torch.multinomial(top_probs, num_samples=1).item()
            next_token = top_indices[sampled_idx].item()
            
            # Append the token
            tokens.append(next_token)
            generated_count += 1
            
            # Stop if we generated a special token
            if next_token in special_token_ids:
                break
    
    # Decode the tokens back to a string
    output_string = tokenizer.decode(tokens)
    
    return output_string


def load_model_from_checkpoint(
    checkpoint_path: str,
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    device: str,
) -> Transformer_LM:
    """Load a model from a training checkpoint."""
    model = Transformer_LM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        rope_theta=rope_theta,
        device=device,
    ).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    
    return model


def main():
    parser = argparse.ArgumentParser(
        description="Generate text using a trained Transformer LM",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Model loading args
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (.pt file)")
    parser.add_argument("--vocab", type=str, required=True,
                        help="Path to vocabulary JSON file")
    parser.add_argument("--merges", type=str, required=True,
                        help="Path to merges file")
    
    # Model architecture args (must match training)
    parser.add_argument("--vocab-size", type=int, required=True,
                        help="Vocabulary size")
    parser.add_argument("--context-length", type=int, default=256,
                        help="Context length")
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
    
    # Generation args
    parser.add_argument("--prompt", type=str, default="Once upon a time",
                        help="Prompt to complete")
    parser.add_argument("--top-p", type=float, default=0.9,
                        help="Top-p (nucleus) sampling probability")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature")
    parser.add_argument("--max-tokens", type=int, default=100,
                        help="Maximum number of tokens to generate")
    parser.add_argument("--special-tokens", type=str, nargs="*", default=["<|endoftext|>"],
                        help="Special tokens")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device (cpu, cuda, mps)")
    
    args = parser.parse_args()
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
    
    print(f"Generating with prompt: '{args.prompt}'")
    print(f"Top-p: {args.top_p}, Temperature: {args.temperature}, Max tokens: {args.max_tokens}")
    print("-" * 50)
    
    # Generate
    output = decode(
        model=model,
        tokenizer=tokenizer,
        special_token_ids=special_token_ids,
        prompt=args.prompt,
        p=args.top_p,
        temperature=args.temperature,
        max_generated_tokens=args.max_tokens,
        device=args.device,
    )
    
    print(output)


if __name__ == "__main__":
    main()
