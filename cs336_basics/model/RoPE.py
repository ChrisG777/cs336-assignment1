import torch
import torch.nn as nn
from einops import rearrange


class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device:torch.device|None=None):
        """
        theta_i, k = i/(theta ** 2k/d) for i = 0, ... max_seq_len-1, k=0, .. d/2 - 1, 
        """
        super().__init__()
        i_range = torch.arange(0, max_seq_len, device=device)
        k_range = torch.arange(0, d_k/2, device=device)
        frequencies = 1.0 / (theta ** (2 * k_range / d_k))
        thetas = torch.outer(i_range, frequencies)
        self.register_buffer("cos", thetas.cos(), persistent=False)
        self.register_buffer("sin", thetas.sin(), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        x is (..., seq_len, d_k) 
        token_positions is (..., seq_len) and specifies the token positions of x

        let a_k be the even positions, b_k be the odd positions
        then R_k [a_k b_k]^T = [a_k cos[k] - b_k sin[k], a_k sin[k] + b_k cos[k]]^T 
        """
        cos = self.cos[token_positions] # (..., seq_len, d_k/2)
        sin = self.sin[token_positions] # (..., seq_len, d_k/2)
        parity_separated = rearrange(x, "... seq_len (d parity) -> ... seq_len d parity", parity=2)
        even = parity_separated[..., 0]
        odd = parity_separated[..., 1]
        even_results = even * cos - odd * sin  # (..., seq_len, d_k/2)
        odd_results = even * sin + odd * cos   # (..., seq_len, d_k/2)

        return rearrange([even_results, odd_results], "parity ... seq_len d -> ... seq_len (d parity)")

