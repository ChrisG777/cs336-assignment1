import torch

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    x_shifted = x - x.max(dim=dim, keepdim=True).values
    x_exp = torch.exp(x_shifted)
    x_exp_sum = x_exp.sum(dim=dim, keepdim=True)
    return x_exp / x_exp_sum