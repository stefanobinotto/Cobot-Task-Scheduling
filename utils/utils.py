import os
import torch
import random
import numpy as np

def set_seed(seed: int = None) -> None:
    """
    Optionally, set seed for reproducibility.
        
    Args:
        seed (Optional[int]): seed.
    """
    
    assert seed is not None, "Invalid seed!"
    
    os.environ["PYTHONHASHSEED"] = str(seed)  # set seed for Python hashes
    np.random.seed(seed)                     # set seed for NumPy
    random.seed(seed)                        # set seed for random module
    torch.manual_seed(seed)                  # set seed for PyTorch on CPU
    if torch.cuda.is_available():            # if GPU is available
        torch.cuda.manual_seed(seed)         # set seed for CUDA
        torch.cuda.manual_seed_all(seed)     # set seed for all GPUs
        torch.backends.cudnn.deterministic = True  # ensure reproducibility
        torch.backends.cudnn.benchmark = False     # disable non-deterministic optimisations