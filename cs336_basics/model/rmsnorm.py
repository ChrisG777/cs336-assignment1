import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (..., d_model) - supports arbitrary leading dimensions
        """
        mean_squared = x.pow(2).mean(dim=-1, keepdim=True)
        rms = (mean_squared + self.eps).sqrt()
        return (x / rms) * self.weight