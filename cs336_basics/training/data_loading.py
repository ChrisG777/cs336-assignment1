import numpy.typing as npt
import torch

def get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.

    Args:
        dataset (np.array): 1D numpy array of integer token IDs in the dataset.
        batch_size (int): Desired batch size to sample.
        context_length (int): Desired context length of each sampled example.
        device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.

    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
    """
    dataset_tensor = torch.from_numpy(dataset)
    n = len(dataset_tensor)
    max_start = n - context_length - 1
    
    # randomly sample starting indices for each batch element
    start_indices = torch.randint(0, max_start + 1, (batch_size,))
    
    inputs = torch.stack([dataset_tensor[i:i+context_length] for i in start_indices])
    labels = torch.stack([dataset_tensor[i+1:i+context_length+1] for i in start_indices])
    
    # Convert to long (int64) for embedding indexing, then move to device
    return inputs.long().to(device), labels.long().to(device)