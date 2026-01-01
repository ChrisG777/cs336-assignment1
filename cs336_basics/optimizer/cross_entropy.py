import torch
from jaxtyping import Float, Int
from torch import Tensor
from einx import get_at
from einops import reduce

def cross_entropy(inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]) -> Float[Tensor, ""]:
    """
    Given a tensor of inputs and targets, compute the average cross-entropy
    loss across examples.

    Args:
        inputs (Float[Tensor, "batch_size vocab_size"]): inputs[i][j] is the
            unnormalized logit of jth class for the ith example.
        targets (Int[Tensor, "batch_size"]): Tensor of shape (batch_size,) with the index of the correct class.
            Each value must be between 0 and `num_classes - 1`.

    Returns:
        Float[Tensor, ""]: The average cross-entropy loss across examples.


    average over i: -log softmax(o_i)[x_i+1]
    = average over i: -log (exp(o_i[x_i+1]) / sum over a \in vocab exp(o_i[a]))
    = average over  i: -o_i[x_i+1] + log(sum over exp(o_i[a]))
    """
    # subtract the largest element from inputs for numerical stability 
    inputs = inputs - inputs.max(dim=-1, keepdim=True).values

    logits = get_at("... batch [vocab], ... batch -> ... batch", inputs, targets) # o_i[x_i+1]

    inputs_exp = inputs.exp()
    inputs_exp_sum = reduce(inputs_exp, "... batch_size vocab_size -> ... batch_size", "sum")
    inputs_log_exp_sum = inputs_exp_sum.log() # ... batch_size
    CE_losses = -logits + inputs_log_exp_sum 
    average_CE_loss = reduce(CE_losses, "... batch_size -> ...", "mean")
    return average_CE_loss
