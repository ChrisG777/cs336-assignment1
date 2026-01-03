from typing import Iterable
import torch
import torch.nn as nn

def gradient_clipping(parameters: Iterable[nn.Parameter], max_l2_norm: float) -> None:
    params = [p for p in parameters if p.grad is not None]
    if len(params) == 0:
        return
    
    # combined L2 norm of a concatenated form of all the gradients
    # need to clip by the same factor for every gradient, because otherwise you change the direction of the gradient 
    # hence have to measure the norm of all the gradients
    total_norm = torch.sqrt(sum(p.grad.norm() ** 2 for p in params))
    
    if total_norm > max_l2_norm:
        clip_coef = max_l2_norm / total_norm
        for p in params:
            p.grad *= clip_coef


