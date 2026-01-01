import torch
import torch.nn as nn
from jaxtyping import Float, Int
from torch import Tensor
from .embedding import Embedding
from .linear import Linear
from .rmsnorm import RMSNorm
from .transformer_block import Transformer_block

class Transformer_LM(nn.Module):
    def __init__(self, vocab_size: int, context_length: int, d_model: int, num_layers: int, num_heads: int, d_ff: int, rope_theta: float, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.emb = Embedding(vocab_size, d_model, device, dtype)
        self.blocks = nn.ModuleList([
            Transformer_block(d_model, num_heads, d_ff, context_length, rope_theta, device, dtype) 
            for _ in range(num_layers)
        ])
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device, dtype)

    def forward(self, x: Int[Tensor, " batch_size sequence_length"]) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
        x = self.emb.forward(x)
        for block in self.blocks:
            x = block.forward(x)
        x = self.ln_final.forward(x)
        x = self.lm_head.forward(x)
        return x  # Return unnormalized logits (no softmax)
