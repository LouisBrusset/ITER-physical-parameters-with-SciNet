import numpy as np
import torch

def seed_everything(seed: float = 42):
    np.random.seed(seed)                        # NumPy
    torch.manual_seed(seed)                     # PyTorch CPU
    torch.cuda.manual_seed(seed)                # PyTorch CUDA (for GPUs)
    torch.cuda.manual_seed_all(seed)            # Multi-GPU settings
    torch.backends.cudnn.deterministic = True   # Ensures deterministic operations
    torch.backends.cudnn.benchmark = False      # Disables auto-tuning for determinism

    # Issue to consider in the future:
    torch.backends.cudnn.enabled = False        # Disables cuDNN