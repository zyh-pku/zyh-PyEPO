#!/usr/bin/env python
# coding: utf-8
"""
optDataset class based on PyTorch Dataset
"""

import time

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from pyepo.model.opt import optModel

import random
from scipy.spatial import distance

‘’‘
class optDataset(Dataset):
    """
    This class is Torch Dataset for optimization problems.

    Attributes:
        model (optModel): Optimization models
        feats (np.ndarray): Data features
        costs (np.ndarray): Cost vectors
        sols (np.ndarray): Optimal solutions
        objs (np.ndarray): Optimal objective values
    """

    def __init__(self, model, feats, costs):
        """
        A method to create a optDataset from optModel

        Args:
            model (optModel): an instance of optModel
            feats (np.ndarray): data features
            costs (np.ndarray): costs of objective function
        """
        if not isinstance(model, optModel):
            raise TypeError("arg model is not an optModel")
        self.model = model
        # data
        self.feats = feats
        self.costs = costs
        # find optimal solutions
        self.sols, self.objs = self._getSols()

    def _getSols(self):
        """
        A method to get optimal solutions for all cost vectors
        """
        sols = []
        objs = []
        print("\nOptimizing for optDataset...", flush=True)
        for c in tqdm(self.costs):
            try:
                sol, obj = self._solve(c)
                # to numpy
                if isinstance(sol, torch.Tensor):
                    sol = sol.detach().cpu().numpy()
            except:
                raise ValueError(
                    "For optModel, the method 'solve' should return solution vector and objective value."
                )
            sols.append(sol)
            objs.append([obj])
        return np.array(sols), np.array(objs)

    def _solve(self, cost):
        """
        A method to solve optimization problem to get an optimal solution with given cost

        Args:
            cost (np.ndarray): cost of objective function

        Returns:
            tuple: optimal solution (np.ndarray) and objective value (float)
        """
        self.model.setObj(cost)
        sol, obj = self.model.solve()
        return sol, obj

    def __len__(self):
        """
        A method to get data size

        Returns:
            int: the number of optimization problems
        """
        return len(self.costs)

    def __getitem__(self, index):
        """
        A method to retrieve data

        Args:
            index (int): data index

        Returns:
            tuple: data features (torch.tensor), costs (torch.tensor), optimal solutions (torch.tensor) and objective values (torch.tensor)
        """
        return (
            torch.FloatTensor(self.feats[index]),
            torch.FloatTensor(self.costs[index]),
            torch.FloatTensor(self.sols[index]),
            torch.FloatTensor(self.objs[index]),
        )
’‘’


class optDatasetKNN(optDataset):
    """
    This class is Torch Dataset for optimization problems, when using the robust kNN-loss.

    Reference: <https://arxiv.org/abs/2310.04328>

    Attributes:
        model (optModel): Optimization models
        k (int): number of nearest neighbours selected
        weight (float): weight of kNN-loss
        feats (np.ndarray): Data features
        costs (np.ndarray): Cost vectors
        sols (np.ndarray): Optimal solutions
        objs (np.ndarray): Optimal objective values
    """
    def __init__(self, model, feats, costs, k=10, weight=0.5):
        """
        A method to create a optDataset from optModel

        Args:
            model (optModel): an instance of optModel
            feats (np.ndarray): data features
            costs (np.ndarray): costs of objective function
            k (int): number of nearest neighbours selected
            weight (float): weight of kNN-loss
        """
        if not isinstance(model, optModel):
            raise TypeError("arg model is not an optModel")
        self.model = model
        # kNN loss parameters
        self.k = k
        self.weight = weight
        # data
        self.feats = feats
        self.costs = costs
        # find optimal solutions
        self.sols, self.objs = self._getSols()

    def _getSols(self):
        """
        A method to get optimal solutions for all cost vectors
        """
        sols = []
        objs = []
        print("\nOptimizing for optDataset...", flush=True)
        # get kNN costs
        costs_knn = self._getKNN()
        # solve optimization
        for c_knn in tqdm(costs_knn):
            sol_knn = np.zeros((self.costs.shape[1], self.k))
            obj_knn = np.zeros(self.k)
            for i, c in enumerate(c_knn.T):
                try:
                    sol_i, obj_i = self._solve(c)
                except:
                    raise ValueError(
                        "For optModel, the method 'solve' should return solution vector and objective value."
                    )
                sol_knn[:, i] = sol_i
                obj_knn[i] = obj_i
            # get average
            sol = sol_knn.mean(axis=1)
            obj = obj_knn.mean()
            sols.append(sol)
            objs.append([obj])
        # update cost as average kNN
        self.costs = costs_knn.mean(axis=2)
        return np.array(sols), np.array(objs)

    def _getKNN(self):
        """
        A method to get kNN costs
        """
        # init costs
        costs_knn = np.zeros((*self.costs.shape, self.k))
        # calculate distances between features
        distances = distance.cdist(self.feats, self.feats, "euclidean")
        indexes = np.argpartition(distances, self.k, axis=1)[:, :self.k]
        # get neighbours costs
        for i, knns in enumerate(indexes):
            # interpolation weight
            costs_knn[i] = self.weight * self.costs[i].reshape((-1, 1)) \
                         + (1 - self.weight) * self.costs[knns].T
        return costs_knn

class optDatasetTimeSeries(optDataset):
    """
    This class extends optDataset to handle time series data for stocks.
    
    Attributes:
        model (optModel): Optimization models
        feats (np.ndarray): Data features (T, N, k)
        costs (np.ndarray): Cost vectors (T, N)
        lookback (int): Number of days to look back
        padding_method (str): Method to pad insufficient data ('zero', 'repeat', 'mean')
    """

    def __init__(self, model, feats, costs, lookback=5, padding_method='zero'):
        """
        A method to create a optDatasetTimeSeries from optModel
        
        Args:
            model (optModel): an instance of optModel
            feats (np.ndarray): data features in shape (T, N, k)
            costs (np.ndarray): costs of objective function in shape (T, N)
            lookback (int): number of days to look back
            padding_method (str): method to pad insufficient data ('zero', 'repeat', 'mean')
        """
        if not isinstance(model, optModel):
            raise TypeError("arg model is not an optModel")
        self.model = model
        self.lookback = lookback
        self.padding_method = padding_method
        
        # Original data
        self.original_feats = feats  # 保存原始特征 (T, N, k)
        self.original_costs = costs  # 保存原始成本 (T, N)
        
        # Process data to include lookback period
        self.feats, self.costs = self._process_time_series()
        
        # Find optimal solutions
        self.sols, self.objs = self._getSols()
    
    def _process_time_series(self):
        """
        Process the feature data to include lookback periods
        """
        T, N, k = self.original_feats.shape
        processed_feats = []
        
        for t in range(T):
            # 确定lookback的开始时间点
            start_idx = max(0, t - self.lookback + 1)
            # 计算实际可用的lookback长度
            actual_lookback = t - start_idx + 1
            
            # 处理特征
            if actual_lookback < self.lookback:
                # 需要填充
                if self.padding_method == 'zero':
                    # 零填充
                    padding = np.zeros((self.lookback - actual_lookback, N, k))
                elif self.padding_method == 'repeat':
                    # 重复第一天数据
                    padding = np.repeat(self.original_feats[start_idx:start_idx+1], self.lookback - actual_lookback, axis=0)
                elif self.padding_method == 'mean':
                    # 均值填充
                    padding = np.ones((self.lookback - actual_lookback, N, k)) * np.mean(self.original_feats[start_idx:t+1], axis=0)
                else:
                    raise ValueError(f"Padding method '{self.padding_method}' not supported")
                
                # 拼接填充数据和实际数据
                ts_feats = np.concatenate([padding, self.original_feats[start_idx:t+1]], axis=0)
            else:
                # 不需要填充，直接截取lookback天的数据
                ts_feats = self.original_feats[start_idx:t+1]
            
            # 将时间维度和特征维度重塑为一个大特征向量
            # 形状从 (lookback, N, k) 变为 (N, lookback，k)
            reshaped_feats = ts_feats.transpose(1, 0, 2)
            processed_feats.append(reshaped_feats)
        
        # 返回处理后的特征和成本
        return np.array(processed_feats), self.original_costs
    
    def __getitem__(self, index):
        """
        A method to retrieve data with time series features
        
        Args:
            index (int): data index
            
        Returns:
            tuple: time series features (torch.tensor), costs (torch.tensor), optimal solutions (torch.tensor) and objective values (torch.tensor)
        """
        return (
            torch.FloatTensor(self.feats[index]),
            torch.FloatTensor(self.costs[index]),
            torch.FloatTensor(self.sols[index]),
            torch.FloatTensor(self.objs[index]),
        )
