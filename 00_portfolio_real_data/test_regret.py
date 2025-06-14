import numpy as np
import torch
from tqdm import tqdm

from pyepo import EPO

from model_factory import MarketNeutralGrbModel_testing, build_market_neutral_model_testing





def sequential_regret(predmodel, optmodel, dataloader, verbose=False):
    """
    Calculate sequential regret for multi-period optimization with turnover constraints
    
    This function properly handles the sequential dependency where each period's
    optimal solution depends on the previous period's portfolio weights.
    
    Args:
        predmodel (nn.Module): Neural network for cost prediction
        optmodel (MarketNeutralGrbModel): Enhanced optimization model with turnover support
        dataloader (DataLoader): PyTorch dataloader with sequential data
        verbose (bool): Whether to print detailed progress
        
    Returns:
        float: Normalized sequential regret loss
    """
    if not isinstance(optmodel, MarketNeutralGrbModel_testing):
        raise TypeError("optmodel must be MarketNeutralGrbModel for sequential regret")
    
    # Set model to evaluation mode
    predmodel.eval()
    
    total_regret = 0.0
    total_optsum = 0.0
    num_sequences = 0
    
    if verbose:
        print("Calculating sequential regret...")
        dataloader = tqdm(dataloader, desc="Processing sequences")
    
    for batch_data in tqdm(dataloader,desc="Testing Regret", unit="batch"):
        x, c, w_true, z_true = batch_data
        if next(predmodel.parameters()).is_cuda:
            x, c, w_true, z_true = x.cuda(), c.cuda(), w_true.cuda(), z_true.cuda()
        
        
        with torch.no_grad():
            if predmodel.training:
                predmodel.eval()
            pred_costs = predmodel(x).to("cpu").detach().numpy()  # (batch_size, num_assets) 
            
            # Get true costs and solutions for this sequence
            true_costs = c.to("cpu").detach().numpy()  # (batch_size, num_assets) 
            true_solutions = w_true.to("cpu").detach().numpy()  # (batch_size, num_assets) 
            true_objectives = z_true.to("cpu").detach().numpy()  # (batch_size, 1) 
            
            # Calculate sequential regret for this sequence
            seq_regret, seq_optsum, optmodel = _calculate_sequence_regret(
                pred_costs, true_costs, true_solutions, true_objectives, optmodel
            )
            
            total_regret += seq_regret
            total_optsum += seq_optsum
            num_sequences += 1
    
    # Turn back to train mode
    predmodel.train()
    
    # Normalized regret
    normalized_regret = total_regret / (total_optsum + 1e-7)
    
    if verbose:
        print(f"Processed {num_sequences} sequences")
        print(f"Total regret: {total_regret:.6f}")
        print(f"Total optimal sum: {total_optsum:.6f}")
        print(f"Normalized regret: {normalized_regret:.6f}")
    
    return normalized_regret


def _calculate_sequence_regret(pred_costs, true_costs, true_solutions, true_objectives, optmodel):
    """
    Calculate regret for a single sequence with sequential dependencies
    
    Args:
        pred_costs (np.ndarray): Predicted costs (T, N)
        true_costs (np.ndarray): True costs (T, N)
        true_solutions (np.ndarray): True optimal solutions (T, N)
        true_objectives (np.ndarray): True optimal objectives (T, 1)
        optmodel (MarketNeutralGrbModel): Optimization model
        
    Returns:
        tuple: (sequence_regret, sequence_optsum)
    """
    seq_len, num_assets = pred_costs.shape
    sequence_regret = 0.0
    sequence_optsum = 0.0
    
    
    for t in range(seq_len):

        try:
            # Set predicted costs and solve
            optmodel.setObj(pred_costs[t])
            pred_solution, _ = optmodel.solve()
            
            # Convert to numpy if needed
            if isinstance(pred_solution, torch.Tensor):
                pred_solution = pred_solution.detach().cpu().numpy()
            
            # Calculate objective value using true costs
            pred_obj_with_true_costs = np.dot(pred_solution, true_costs[t])
            # Set predicted costs and solve
            optmodel.setObj(true_costs[t])
            _, true_obj = optmodel.solve()
            
            
            # Calculate regret for this period
            if optmodel.modelSense == EPO.MINIMIZE:
                period_regret = pred_obj_with_true_costs - true_obj
            elif optmodel.modelSense == EPO.MAXIMIZE:
                period_regret = true_obj - pred_obj_with_true_costs
            else:
                raise ValueError("Invalid modelSense")
            
            sequence_regret += period_regret
            sequence_optsum += abs(true_obj)
            
            # Set current predicted solution as previous weights for next period
            if optmodel.turnover is not None:
                optmodel.setPrevWeights(pred_solution)
                
        except Exception as e:
            print(f"Error in period {t}: {str(e)}")
    
    return sequence_regret, sequence_optsum,optmodel


def sequential_solutions(predmodel, params_testing, dataloader, verbose=False):
    """
    Calculate sequential regret for multi-period optimization with turnover constraints
    
    This function properly handles the sequential dependency where each period's
    optimal solution depends on the previous period's portfolio weights.
    
    Args:
        predmodel (nn.Module): Neural network for cost prediction
        optmodel (MarketNeutralGrbModel): Enhanced optimization model with turnover support
        dataloader (DataLoader): PyTorch dataloader with sequential data
        verbose (bool): Whether to print detailed progress
        
    Returns:
        float: Normalized sequential regret loss
    """
    
    # Set model to evaluation mode
    predmodel.eval()
    optmodel=build_market_neutral_model_testing(**params_testing)
    
    pre_sols=[]    
    num_sequences = 0
    
    if verbose:
        print("Calculating sequential regret...")
        dataloader = tqdm(dataloader, desc="Processing sequences")
    
    
    for batch_data in tqdm(dataloader,desc="Testing Solutions", unit="batch"):
        x, c, w_true, z_true = batch_data
        if next(predmodel.parameters()).is_cuda:
            x, c, w_true, z_true = x.cuda(), c.cuda(), w_true.cuda(), z_true.cuda()
        
        
        with torch.no_grad():
            if predmodel.training:
                predmodel.eval()
            pred_costs = predmodel(x).to("cpu").detach().numpy()  # (batch_size, num_assets) 
            
            
            # Calculate sequential regret for this sequence
            pre_sols, optmodel = _calculate_sequence_solutions(
                pred_costs, optmodel, pre_sols
            )
    
    
    
    
    return pre_sols 


def _calculate_sequence_solutions(pred_costs, optmodel, pre_sols):
    """
    Calculate regret for a single sequence with sequential dependencies
    
    Args:
        pred_costs (np.ndarray): Predicted costs (T, N)
        true_costs (np.ndarray): True costs (T, N)
        true_solutions (np.ndarray): True optimal solutions (T, N)
        true_objectives (np.ndarray): True optimal objectives (T, 1)
        optmodel (MarketNeutralGrbModel): Optimization model
        
    Returns:
        tuple: (sequence_regret, sequence_optsum)
    """
    seq_len, num_assets = pred_costs.shape
    
    
    
    for t in range(seq_len):

        try:
            # Set predicted costs and solve
            optmodel.setObj(pred_costs[t])
            pred_solution, _ = optmodel.solve()
            
            # Convert to numpy if needed
            if isinstance(pred_solution, torch.Tensor):
                pred_solution = pred_solution.detach().cpu().numpy()

            pre_sols.append(pred_solution)
            
            
           
            
            # Set current predicted solution as previous weights for next period
            if optmodel.turnover is not None:
                optmodel.setPrevWeights(pred_solution)
                
        except Exception as e:
            print(f"Error in period {t}: {str(e)}")
    
    return pre_sols,optmodel