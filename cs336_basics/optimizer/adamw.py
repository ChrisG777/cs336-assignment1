from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, alpha: float, beta_1: float, beta_2: float, epsilon: float, lam: float):
        defaults = {
            "alpha": alpha,
            "beta_1": beta_1,
            "beta_2": beta_2,
            "epsilon": epsilon,
            "lam": lam
        }
        super().__init__(params, defaults) 
    
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            alpha = group["alpha"]
            beta_1 = group["beta_1"]
            beta_2 = group["beta_2"]
            epsilon = group["epsilon"]
            lam = group["lam"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get("t", 1)
                grad = p.grad.data
                m = beta_1 * state.get("m", torch.zeros_like(p.data)) + (1-beta_1) * grad # first moment
                state["m"] = m
                v = beta_2 * state.get("v", torch.zeros_like(p.data)) + (1-beta_2) * grad.square() # second moment
                state["v"] = v
                alpha_t = alpha * math.sqrt(1 - beta_2 ** t) / (1 - beta_1 ** t)
                p.data -= alpha_t * m / (v.sqrt() + epsilon) + alpha * lam * p.data
                state["t"] = t+1
        return loss

                