import torch
from torch import Tensor
from jaxtyping import Bool, Float
from einops import einsum, parse_shape 

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