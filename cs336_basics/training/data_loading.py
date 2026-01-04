import numpy.typing as npt
import torch

def get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    dataset_tensor = torch.from_numpy(dataset)
    n = len(dataset_tensor)
    max_start = n - context_length - 1
    
    # randomly sample starting indices for each batch element
    start_indices = torch.randint(0, max_start + 1, (batch_size,))
    
    inputs = torch.stack([dataset_tensor[i:i+context_length] for i in start_indices])
    labels = torch.stack([dataset_tensor[i+1:i+context_length+1] for i in start_indices])
    
    return inputs.to(device), labels.to(device)