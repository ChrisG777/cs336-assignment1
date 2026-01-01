import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor
from .SwiGLU import SwiGLU
from .rmsnorm import RMSNorm
from .attention import MHA

class Transformer_block(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, theta: float, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.mha = MHA(d_model, num_heads, max_seq_len, theta, device, dtype)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device, dtype)

    def forward(self, x: Float[Tensor, " ... seq_len d_model"]) -> Float[Tensor, " ... seq_len d_model"]:
        # Pre-norm Transformer block
        x = x + self.mha(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x
