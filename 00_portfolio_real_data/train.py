import time
from tqdm import tqdm
import torch
from torch.amp import GradScaler, autocast

import pyepo

from config import DEVICE, BATCH_SIZE, NUM_EPOCHS
from data_loader import device_loader

from test_regret import sequential_regret

from config import MARKET_MODEL_DIR,MARKET_MODEL_DIR_TESTING
from model_factory import build_market_neutral_model_testing
import pickle


with open(MARKET_MODEL_DIR_TESTING, "rb") as f:
    params_testing = pickle.load(f)
    
import os


os.environ['GUROBI_HOME'] = '/usr/licensed/gurobi/12.0.0/linux64'
os.environ['GRB_LICENSE_FILE'] = '/usr/licensed/gurobi/license/gurobi.lic'

# 清除个人WLS许可证
for var in ['WLSACCESSID', 'WLSSECRET']:
    if var in os.environ:
        del os.environ[var]



def trainModel(model, loss_func, method_name, loader_train, loader_test, market_neutral_model, params_testing, loss_log, loss_log_regret, num_epochs=1000, lr=1e-3, initial=False):
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
        optimizer, mode='min', factor=0.5, patience=1
    )
    
    # Enable mixed precision training
    scaler = GradScaler(enabled=(DEVICE.type in ["cuda", "mps"]))

    # Set model to training mode
    model.train()
    
    # Initialize logs
    
    if initial: # evaluate loss on whole test data
        ## 系统Gurobi
        market_neutral_model_testing= build_market_neutral_model_testing(**params_testing)# need to initialize the testing Grb 
        regret = sequential_regret(model, market_neutral_model_testing, device_loader(loader_test))
        #loss_log_regret = [pyepo.metric.regret(model, market_neutral_model, device_loader(loader_test))]
        
        print(f"Initial regret: {regret*100:.4f}%")
    
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
            x, c, w, z = x.to(DEVICE), c.to(DEVICE), w.to(DEVICE), z.to(DEVICE)
            
            # Record batch start time for accurate timing
            batch_start = time.time()
            
            # Clear gradients for each batch
            optimizer.zero_grad()
            
            # Use mixed precision where appropriate
            with autocast(device_type=DEVICE.type, enabled=(DEVICE.type in ["cuda", "mps"])):
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
                if DEVICE.type == 'cuda':
                    mem_allocated = torch.cuda.memory_allocated() / 1024**2
                    mem_reserved = torch.cuda.memory_reserved() / 1024**2
                    print(f"GPU Memory: {mem_allocated:.1f}MB allocated, {mem_reserved:.1f}MB reserved")
        
        # Compute regret on test set after each epoch
        with torch.no_grad():
            model.eval()  # Set model to evaluation mode
            market_neutral_model_testing= build_market_neutral_model_testing(**params_testing)# need to reinitialize the testing Grb 
            regret = sequential_regret(model, market_neutral_model_testing, device_loader(loader_test))
            #regret = pyepo.metric.regret(model, market_neutral_model, device_loader(loader_test, device))
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