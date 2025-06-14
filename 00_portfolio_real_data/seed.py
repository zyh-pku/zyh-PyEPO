import random
import os
import torch
import tensorflow as tf

def set_seed(seed=42):
    """
    Set seed for reproducibility across multiple libraries.
    Minimal version to avoid circular import issues.
    
    Args:
        seed (int): Random seed value (default: 42)
    """
    # Set environment variable first (before any imports)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # Python built-in random
    random.seed(seed)
    
    # NumPy
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    
    # PyTorch - minimal approach to avoid circular imports
    try:
        torch.manual_seed(seed)
        # Only set CUDA seeds if CUDA is available
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    except (ImportError, AttributeError):
        pass
    
    # TensorFlow
    try:
        tf.random.set_seed(seed)
    except ImportError:
        pass
    
    print(f"Seed set to {seed}")