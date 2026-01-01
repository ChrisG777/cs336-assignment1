import torch
import torch.nn as nn

from .linear import Linear


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int | None = None, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        if d_ff is None:
            # Default: (8/3) * d_model, rounded to multiple of 64
            target_dff = (8 / 3) * d_model
            d_ff = int(round(target_dff / 64) * 64)

        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (..., d_model) - supports arbitrary leading dimensions
        
        SwiGLU(x) = (SiLU(xW1) * xW3) W2
        """
        return self.w2(silu(self.w1(x)) * self.w3(x))


def silu(x: torch.Tensor) -> torch.Tensor:
    """SiLU activation: x * sigmoid(x)"""
    return x * torch.sigmoid(x)


class SiLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (..., d_ff), returns a tensor of the same shape
        """
        return silu(x)
    
