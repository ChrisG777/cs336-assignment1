import torch
from torch import Tensor
import torch.nn as nn
from jaxtyping import Bool, Float
from einops import einsum, parse_shape, rearrange, repeat
from .linear import Linear
from .RoPE import RoPE

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    x_shifted = x - x.max(dim=dim, keepdim=True).values
    x_exp = torch.exp(x_shifted)
    x_exp_sum = x_exp.sum(dim=dim, keepdim=True)
    return x_exp / x_exp_sum

def scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    d_k = parse_shape(Q, "... queries d_k")["d_k"]
    pre_activation = einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys") / (d_k ** 0.5)
    
    if mask is not None:
        pre_activation = pre_activation.masked_fill(~mask, float('-inf'))
    
    attention_weights = softmax(pre_activation, dim=-1)
    return einsum(attention_weights, V, "... queries keys, ... keys d_v -> ... queries d_v")

class MHA(nn.Module):
    def __init__(self, d_model: int, num_heads: int, max_seq_len: int | None = None, theta: float | None = None, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = self.d_v = d_model // num_heads  # integer division
        self.W_Q = Linear(d_model, d_model, device=device, dtype=dtype)
        self.W_K = Linear(d_model, d_model, device=device, dtype=dtype)
        self.W_V = Linear(d_model, d_model, device=device, dtype=dtype)
        self.W_O = Linear(d_model, d_model, device=device, dtype=dtype)
        self.rope = None
        if theta is not None and max_seq_len is not None:
            self.rope = RoPE(theta, self.d_k, max_seq_len, device)

    def forward(self, x: Float[Tensor, "... seq_len d_model"], token_positions: Float[Tensor, "... seq_len"] | None = None) -> Float[Tensor, "... seq_len d_model"]:

        seq_len = parse_shape(x, "... seq_len d_model")["seq_len"]

        Q = self.W_Q.forward(x)  # "... seq_len d_model"
        Q_multihead = rearrange(Q, "... seq_len (h d_k) -> ... h seq_len d_k", h=self.num_heads, d_k=self.d_k)
        
        K = self.W_K.forward(x)  # "... seq_len d_model"
        K_multihead = rearrange(K, "... seq_len (h d_k) -> ... h seq_len d_k", h=self.num_heads, d_k=self.d_k)
        
        # Apply RoPE if configured
        if self.rope is not None:
            # Default: 0-indexed positions, expanded to match x's batch dims
            if token_positions is None:
                positions = torch.arange(seq_len, device=x.device)
                token_positions = positions.expand(x.shape[:-1])  # (... seq_len)
            # Repeat for each head: (... seq_len) -> (... h seq_len)
            token_positions_repeated = repeat(token_positions, "... seq_len -> ... h seq_len", h=self.num_heads)
            Q_multihead = self.rope.forward(Q_multihead, token_positions_repeated)
            K_multihead = self.rope.forward(K_multihead, token_positions_repeated)

        V = self.W_V.forward(x)  # "... seq_len d_model"
        V_multihead = rearrange(V, "... seq_len (h d_v) -> ... h seq_len d_v", h=self.num_heads, d_v=self.d_v)

        # mask is seq_len seq_len, but masked_fill should hopefully broadcast correctly regardless
        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device))

        attn_output = scaled_dot_product_attention(Q_multihead, K_multihead, V_multihead, mask)
        rearranged_attn_output = rearrange(attn_output, "... h seq_len d_v -> ... seq_len (h d_v)")

        return self.W_O.forward(rearranged_attn_output)  # Output projection 

