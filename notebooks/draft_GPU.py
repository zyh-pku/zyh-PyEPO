#############################################################################
# IMPORTS AND SETUP
#############################################################################
# Set random seed for reproducibility
import random
random.seed(42)
import numpy as np
np.random.seed(42)
import torch
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.backends.cudnn.deterministic = True  # Makes training more deterministic
torch.backends.cudnn.benchmark = False     # Can speed up training when input sizes don't change

# Standard imports
import pandas as pd
import numpy as np
import polars as pl
from datetime import datetime, timedelta
import time
import os

from tqdm.auto import tqdm  # Progress bar

# PyEPO imports
from pyepo.data.dataset import optDataset
import pyepo

# Check and set up GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    # Clear GPU cache to maximize available memory
    torch.cuda.empty_cache()



##### 强制使用CPU
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

device = torch.device("cpu")
# Force JAX to use the CPU backend only (avoids CUDA OOM in JAX)
os.environ['JAX_PLATFORMS'] = 'cpu'



#############################################################################
# DATA PREPARATION
#############################################################################
print("Loading and processing data...")

df = pl.read_parquet("data_sample.parquet")

k = 5  # number of x features
m = 3  # number of y features

y_col = 'y1'  # Specify the y column to be used for costs

# 1. Find all x columns
x_cols = [c for c in df.columns if c.startswith("x")]

# 2. Pivot costs matrix: shape (T, N)
cost_pivot = (
    df
    .pivot(
        values=y_col,
        index="time",
        on="symbol",
        aggregate_function="first",   # Ensure only one value per (time, symbol) pair
    )
    .sort("time")                     # Sort by time
    .fill_null(0)                     # Fill missing values with 0
)
# Extract time and symbol information
unique_times = cost_pivot["time"].to_list()
unique_symbols = cost_pivot.columns[1:]    # All columns except 'time' are symbols

# Convert to numpy for PyTorch compatibility
costs = cost_pivot.select(unique_symbols).to_numpy()

# 3. Pivot each feature (x) and stack into a 3D tensor (T, N, k)
feature_mats = []
for x in x_cols:
    mat = (
        df
        .pivot(
            values=x,
            index="time",
            on="symbol",
            aggregate_function="first",
        )
        .sort("time")
        .fill_null(0)
        .select(unique_symbols)
        .to_numpy()
    )
    feature_mats.append(mat)
# Stack along the third dimension to create (T, N, k) tensor
features = np.stack(feature_mats, axis=2)  # (T, N, k)

print(f"Data shape: features {features.shape}, costs {costs.shape}")

#############################################################################
# TRAIN-TEST SPLIT
#############################################################################
from sklearn.model_selection import train_test_split
x_train, x_test, c_train, c_test = train_test_split(features, costs, test_size=70, random_state=42)
print(f"Training set: {x_train.shape}, Test set: {x_test.shape}")

#############################################################################
# OPTIMIZATION MODEL SETUP
#############################################################################
# Use actual dimensions from data or user-specified
N = features.shape[1]  # Number of symbols (update hardcoded value)
print(f"Using {N} symbols for optimization model")

# Market neutral constraint (sum of weights = 1)
A = np.ones((1, N))  # Create a 1×N matrix filled with 1's
b = np.array([1])    # Create a 1D array with a single element 1

from pyepo.model.mpax import optMpaxModel
optmodel_toy = optMpaxModel(A=A, b=b, use_sparse_matrix=False, minimize=False)

# Create datasets
dataset_train = optDataset(optmodel_toy, x_train, c_train)
dataset_test = optDataset(optmodel_toy, x_test, c_test)

#############################################################################
# DATA LOADERS WITH PREFETCHING
#############################################################################
from torch.utils.data import DataLoader

# Find optimal batch size based on GPU memory
# Start with a reasonable default
batch_size = 8  

# Configure data loaders with prefetching
loader_train = DataLoader(
    dataset_train, 
    batch_size=batch_size, 
    shuffle=True,
    # pin_memory=True,  # Speed up host to GPU transfers
    # num_workers=2,    # Prefetch in parallel
    # persistent_workers=True  # Keep workers alive between epochs
)

loader_test = DataLoader(
    dataset_test, 
    batch_size=batch_size, 
    shuffle=False,
    # pin_memory=True,
    # num_workers=1
)

#############################################################################
# PREDICTION MODEL
#############################################################################
from torch import nn
import torch

class EnhancedLinearRegression(nn.Module):
    """
    Enhanced linear regression model with batch normalization
    for improved training stability
    """
    def __init__(self, k: int, dropout_rate=0.0):
        super().__init__()
        # Feature normalization
        self.batch_norm = nn.BatchNorm1d(k)
        # Optional dropout for regularization
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None
        # Linear layer - maps k features to 1 output
        self.linear = nn.Linear(in_features=k, out_features=1)
        
        # Initialize weights properly
        nn.init.xavier_normal_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        # x: (batch_size, N, k) -> reshape for batch norm
        batch_size, N, k = x.shape
        x_reshaped = x.reshape(-1, k)  # (batch_size*N, k)
        
        # Apply batch normalization
        x_normalized = self.batch_norm(x_reshaped)
        
        # Apply dropout if enabled
        if self.dropout is not None:
            x_normalized = self.dropout(x_normalized)
            
        # Apply linear layer
        output = self.linear(x_normalized)  # (batch_size*N, 1)
        
        # Reshape back to original dimensions
        output = output.reshape(batch_size, N)  # (batch_size, N)
        
        return output

# Initialize model and move to GPU
reg = EnhancedLinearRegression(k=k, dropout_rate=0.1).to(device)

# Print model summary
print("Model architecture:")
for name, param in reg.named_parameters():
    print(f"{name:20s} | Shape: {param.data.shape} | Parameters: {param.numel()}")
print(f"Total parameters: {sum(p.numel() for p in reg.parameters())}")

#############################################################################
# TRAINING WITH MIXED PRECISION
#############################################################################
import time
from torch.cuda.amp import GradScaler, autocast

def trainModel(model, loss_func, method_name, num_epochs=5, lr=1e-3):
    """
    Enhanced training function with:
    - Mixed precision for faster GPU training
    - Learning rate scheduling
    - Progress bars
    - Detailed logging
    - Memory-efficient tensor handling
    """
    # Set up optimizer with weight decay for regularization
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # Learning rate scheduler for better convergence
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=1, verbose=True
    )
    
    # Enable mixed precision training
    scaler = GradScaler() if device.type == 'cuda' else None
    
    # Set model to training mode
    model.train()
    
    # Initialize logs
    loss_log = []
    loss_log_regret = [pyepo.metric.regret(model, optmodel_toy, loader_test)]
    print(f"Initial regret: {loss_log_regret[0]*100:.4f}%")
    
    # Initialize elapsed time tracking
    training_start = time.time()
    total_elapsed = 0
    
    # Verbosity control - set to false for production
    debug_mode = False
    log_interval = 10  # Log every N batches
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_start = time.time()
        
        # Progress bar for this epoch
        progress_bar = tqdm(loader_train, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for i, data in enumerate(progress_bar):
            x, c, w, z = data
            
            # Move data to GPU (once, not in every batch)
            x, c, w, z = x.to(device), c.to(device), w.to(device), z.to(device)
            
            # Record batch start time for accurate timing
            batch_start = time.time()
            
            # Clear gradients
            optimizer.zero_grad()
            
            # Use mixed precision where appropriate
            with autocast(enabled=device.type=='cuda'):
                # Forward pass
                cp = model(x)
                
                # Compute loss based on method
                if method_name == "spo+":
                    loss = loss_func(cp, c, w, z)
                elif method_name in ["ptb", "pfy", "imle", "aimle", "nce", "cmap"]:
                    loss = loss_func(cp, w)
                elif method_name in ["dbb", "nid"]:
                    loss = loss_func(cp, c, z)
                elif method_name in ["pg", "ltr"]:
                    loss = loss_func(cp, c)
            
            # Backward pass with mixed precision handling
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            
            # Track batch elapsed time
            batch_elapsed = time.time() - batch_start
            total_elapsed += batch_elapsed
            
            # Update loss tracking
            current_loss = loss.item()
            epoch_loss += current_loss
            loss_log.append(current_loss)
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{current_loss:.4f}", 
                'batch time': f"{batch_elapsed:.4f}s"
            })
            
            # Debug logging (limited to avoid overwhelming output)
            if debug_mode and i % log_interval == 0:
                print(f"\n[Debug] Batch {i} stats:")
                print(f"Loss: {current_loss:.6f}")
                print(f"Pred shape: {cp.shape}, values: {cp[0,:5].detach().cpu().numpy()}")
                
                # Monitor memory usage
                if device.type == 'cuda':
                    mem_allocated = torch.cuda.memory_allocated() / 1024**2
                    mem_reserved = torch.cuda.memory_reserved() / 1024**2
                    print(f"GPU Memory: {mem_allocated:.1f}MB allocated, {mem_reserved:.1f}MB reserved")
        
        # Compute regret on test set after each epoch
        with torch.no_grad():
            model.eval()  # Set model to evaluation mode
            regret = pyepo.metric.regret(model, optmodel_toy, loader_test)
            model.train()  # Set back to training mode
            loss_log_regret.append(regret)
        
        # Update learning rate scheduler
        scheduler.step(epoch_loss)
        
        # End of epoch reporting
        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch+1}: Loss={epoch_loss/len(loader_train):.6f}, "
              f"Regret={regret*100:.4f}%, Time={epoch_time:.2f}s")
    
    # Report total training time
    total_training_time = time.time() - training_start
    print(f"Total training time: {total_training_time:.2f}s, "
          f"Effective computation time: {total_elapsed:.2f}s")
    
    return loss_log, loss_log_regret

#############################################################################
# TRAINING EXECUTION 
#############################################################################
print("\nInitializing SPO+ loss function...")
spop = pyepo.func.SPOPlus(optmodel_toy)

print("\nStarting model training...")
loss_log, loss_log_regret = trainModel(
    reg, 
    loss_func=spop, 
    method_name="spo+",
    num_epochs=5,  # Increased for better convergence
    lr=1e-3        # Adjusted learning rate
)

#############################################################################
# VISUALIZATION
#############################################################################
from matplotlib import pyplot as plt

def visLearningCurve(loss_log, loss_log_regret):
    """Enhanced visualization with smoother curves and more information"""
    # Create figure and subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16,4))
    
    # Plot training loss with smoothing for readability
    n_points = len(loss_log)
    
    # Apply smoothing for large datasets
    if n_points > 100:
        window_size = max(10, n_points // 50)
        smoothed_loss = np.convolve(loss_log, np.ones(window_size)/window_size, mode='valid')
        x_axis = np.arange(len(smoothed_loss))
        ax1.plot(x_axis, smoothed_loss, color="c", lw=2, label=f"Smoothed (window={window_size})")
        # Also plot the raw data with transparency
        ax1.plot(loss_log, color="c", lw=0.5, alpha=0.3, label="Raw")
        ax1.legend()
    else:
        # For smaller datasets, just plot the raw data
        ax1.plot(loss_log, color="c", lw=2)
    
    ax1.tick_params(axis="both", which="major", labelsize=12)
    ax1.set_xlabel("Iterations", fontsize=16)
    ax1.set_ylabel("Loss", fontsize=16)
    ax1.set_title("Learning Curve on Training Set", fontsize=16)
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Draw plot for regret on test set
    epochs = np.arange(len(loss_log_regret))
    ax2.plot(epochs, [r*100 for r in loss_log_regret], 
             color="royalblue", marker='o', ls="-", alpha=0.8, lw=2)
    
    ax2.tick_params(axis="both", which="major", labelsize=12)
    ax2.set_ylim(0, max(50, max([r*100 for r in loss_log_regret])*1.1))  # Dynamic y-limit
    ax2.set_xlabel("Epochs", fontsize=16)
    ax2.set_ylabel("Regret (%)", fontsize=16)
    ax2.set_title("Learning Curve on Test Set", fontsize=16)
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # Add values to points
    for i, r in enumerate(loss_log_regret):
        ax2.annotate(f"{r*100:.2f}%", 
                     (i, r*100),
                     textcoords="offset points", 
                     xytext=(0,10), 
                     ha='center')
    
    plt.tight_layout()
    plt.savefig('learning_curves.png', dpi=300, bbox_inches='tight')
    print("Saved learning curves to 'learning_curves.png'")
    plt.show()

print("\nVisualizing learning curves...")
visLearningCurve(loss_log, loss_log_regret)

#############################################################################
# MODEL EVALUATION AND SAVING
#############################################################################
# Save model
model_path = 'trained_model.pt'
torch.save({
    'model_state_dict': reg.state_dict(),
    'final_regret': loss_log_regret[-1],
    'training_config': {
        'method': 'spo+',
        'features': k,
        'symbols': N
    }
}, model_path)
print(f"Model saved to {model_path}")

# Final evaluation
reg.eval()
with torch.no_grad():
    final_regret = pyepo.metric.regret(reg, optmodel_toy, loader_test)
    print(f"Final test regret: {final_regret*100:.4f}%")

print("Training complete!")