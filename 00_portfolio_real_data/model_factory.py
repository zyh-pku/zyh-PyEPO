#############################################################################
# OPTIMIZATION PROBLEM PARAMETERS
#############################################################################
# Market neutral optimization model parameters from opt_time_compare.py
import gurobipy as gp

# 导入 grbmodel 模块
from pyepo.model.grb import grbmodel


import time
import numpy as np
import pandas as pd

# PyEPO imports
from pyepo.data.dataset import optDataset

# Gurobi
import gurobipy as gp
from gurobipy import GRB
from pyepo.model.grb import optGrbModel

# Pyomo
from pyomo import environ as pe
from pyepo.model.omo import optOmoModel # 记得要把 omo 文件夹里的 __init__.py 文件也修改了
from pyepo import EPO

# COPTPy
from coptpy import Envr, COPT
from pyepo.model.copt import optCoptModel

# MPAX
from pyepo.model.mpax import optMpaxModel

import torch

# 强制设置 HAS_GUROBI 为 True
grbmodel.HAS_GUROBI = True


#############################################################################
# MARKET NEUTRAL OPTIMIZATION MODEL
#############################################################################
class MarketNeutralGrbModel(optGrbModel):
    def __init__(
        self,
        N,
        A,
        b,
        l,
        u,
        minimize=False,
        risk_f=None,
        risk_abs=None,
        single_abs=None,
        l1_abs=None,
        cov_matrix=None,
        sigma_abs=None
    ):
        # 保存所有参数
        self.N = N
        self.A = A
        self.b = b
        self.l = l
        self.u = u
        self.minimize = minimize
        self.risk_f = risk_f
        self.risk_abs = risk_abs
        self.single_abs = single_abs
        self.l1_abs = l1_abs
        self.cov_matrix = cov_matrix
        self.sigma_abs = sigma_abs

        super().__init__()

    def _getModel(self):
        # 新建 Gurobi 模型
        m = gp.Model()

        # 添加原始变量 x[i]
        x = m.addVars(
            self.N,
            lb={i: float(self.l[i]) for i in range(self.N)},
            ub={i: float(self.u[i]) for i in range(self.N)},
            name="x"
        )

        # 用于 L1 约束的辅助变量 t[i]
        t = m.addVars(self.N, lb=0.0, name="t")

        # 设置优化方向
        m.modelSense = GRB.MINIMIZE if self.minimize else GRB.MAXIMIZE

        # ——— 1) 市场中性约束 sum A[0,i] * x[i] == b[0]
        m.addConstr(
            gp.quicksum(self.A[0, i] * x[i] for i in range(self.N)) == float(self.b[0]),
            name="eq_sum"
        )

        # ——— 2) 风险约束 |risk_f' x| <= risk_abs
        expr_r = gp.quicksum(self.risk_f[i] * x[i] for i in range(self.N))
        m.addConstr(expr_r <= float(self.risk_abs), name="risk_ub")
        m.addConstr(expr_r >= -float(self.risk_abs), name="risk_lb")

        # ——— 3) 单项绝对值约束 |x_i| <= single_abs
        for i in range(self.N):
            m.addConstr(x[i] <=  float(self.single_abs),  name=f"single_ub_{i}")
            m.addConstr(x[i] >= -float(self.single_abs), name=f"single_lb_{i}")

        # ——— 4) L1 约束 ∑|x_i| <= l1_abs
        #   实现方式： t[i] >=  x[i], t[i] >= -x[i], 然后 ∑ t[i] <= l1_abs
        for i in range(self.N):
            m.addConstr(t[i] >=  x[i], name=f"t_pos_{i}")
            m.addConstr(t[i] >= -x[i], name=f"t_neg_{i}")
        m.addConstr(t.sum() <= float(self.l1_abs), name="l1_norm")

        # ——— 5) 二次型约束 x' * cov_matrix * x <= sigma_abs
        #     向量化写法： x_vec @ cov_matrix @ x_vec
        x_vec = np.array([x[i] for i in range(self.N)])
        expr_q = x_vec @ self.cov_matrix @ x_vec
        m.addQConstr(expr_q <= float(self.sigma_abs), name="sigma_qc")

        return m, x


    

def build_market_neutral_model(N, A, b, l, u, risk_f, risk_abs, single_abs, l1_abs, cov_matrix, sigma_abs):
    return MarketNeutralGrbModel(
        N, A, b, l, u, minimize=False,
        risk_f=risk_f, risk_abs=risk_abs,
        single_abs=single_abs, l1_abs=l1_abs,
        cov_matrix=cov_matrix, sigma_abs=sigma_abs
    )


class MarketNeutralGrbModel_testing(optGrbModel):
    """
    Enhanced Market Neutral Portfolio Optimization Model with Turnover Constraints
    
    This model includes:
    - Market neutral constraint (sum of weights = 1)  
    - Risk factor constraint |risk_f' x| <= risk_abs
    - Single asset position limits |x_i| <= single_abs
    - L1 norm constraint ∑|x_i| <= l1_abs
    - Quadratic risk constraint x' * cov_matrix * x <= sigma_abs
    - Optional turnover constraint ∑|x_i - w_prev_i| <= turnover
    """
    
    def __init__(
        self,
        N,
        A,
        b,
        l,
        u,
        minimize=False,
        risk_f=None,
        risk_abs=None,
        single_abs=None,
        l1_abs=None,
        cov_matrix=None,
        sigma_abs=None,
        turnover=None
    ):
        """
        Initialize the Market Neutral Optimization Model
        
        Args:
            N (int): Number of assets
            A (np.ndarray): Equality constraint matrix (1 x N)
            b (np.ndarray): Equality constraint RHS (1,)
            l (np.ndarray): Lower bounds for variables (N,)
            u (np.ndarray): Upper bounds for variables (N,)
            minimize (bool): Whether to minimize (True) or maximize (False) objective
            risk_f (np.ndarray): Risk factor vector (N,)
            risk_abs (float): Risk factor constraint bound
            single_abs (float): Single asset position limit
            l1_abs (float): L1 norm constraint bound
            cov_matrix (np.ndarray): Covariance matrix (N x N)
            sigma_abs (float): Quadratic risk constraint bound
            turnover (float): Maximum allowed turnover (optional)
        """
        # 保存所有参数
        self.N = N
        self.A = A
        self.b = b
        self.l = l
        self.u = u
        self.minimize = minimize
        self.risk_f = risk_f
        self.risk_abs = risk_abs
        self.single_abs = single_abs
        self.l1_abs = l1_abs
        self.cov_matrix = cov_matrix
        self.sigma_abs = sigma_abs
        self.turnover = turnover
        
        # 保存前期权重，默认为等权重（首次投资）
        self.w_prev = np.ones(N) / N  # Equal weights instead of zeros
        
        # 存储换手率约束引用的列表（优化关键）
        self._turnover_pos_constrs = []
        self._turnover_neg_constrs = []

        super().__init__()

    def setPrevWeights(self, w_prev):
        """
        Set previous portfolio weights for turnover constraint
        
        Args:
            w_prev (np.ndarray): Previous portfolio weights
        """
        if len(w_prev) != self.N:
            raise ValueError("Size of previous weights must match number of assets")
        
        # Check if w_prev is a PyTorch tensor
        if isinstance(w_prev, torch.Tensor):
            w_prev = w_prev.detach().cpu().numpy()
        else:
            w_prev = np.asarray(w_prev, dtype=np.float32)
            
        self.w_prev = w_prev
        
        # Update turnover constraints if they exist
        if self.turnover is not None:
            self._updateTurnoverConstraints()

    def _getModel(self):
        """
        Build the Gurobi optimization model
        
        Returns:
            tuple: (model, variables)
        """
        # 新建 Gurobi 模型
        m = gp.Model()

        # 添加原始变量 x[i]
        x = m.addVars(
            self.N,
            lb={i: float(self.l[i]) for i in range(self.N)},
            ub={i: float(self.u[i]) for i in range(self.N)},
            name="x"
        )

        # 用于 L1 约束的辅助变量 t[i]
        t = m.addVars(self.N, lb=0.0, name="t")
        
        # 用于换手率约束的辅助变量 s[i] (如果启用换手率约束)
        if self.turnover is not None:
            s = m.addVars(self.N, lb=0.0, name="s")  # s[i] >= |x[i] - w_prev[i]|

        # 设置优化方向
        m.modelSense = GRB.MINIMIZE if self.minimize else GRB.MAXIMIZE

        # ——— 1) 市场中性约束 sum A[0,i] * x[i] == b[0]
        m.addConstr(
            gp.quicksum(self.A[0, i] * x[i] for i in range(self.N)) == float(self.b[0]),
            name="eq_sum"
        )

        # ——— 2) 风险约束 |risk_f' x| <= risk_abs
        if self.risk_f is not None and self.risk_abs is not None:
            expr_r = gp.quicksum(self.risk_f[i] * x[i] for i in range(self.N))
            m.addConstr(expr_r <= float(self.risk_abs), name="risk_ub")
            m.addConstr(expr_r >= -float(self.risk_abs), name="risk_lb")

        # ——— 3) 单项绝对值约束 |x_i| <= single_abs
        if self.single_abs is not None:
            for i in range(self.N):
                m.addConstr(x[i] <=  float(self.single_abs),  name=f"single_ub_{i}")
                m.addConstr(x[i] >= -float(self.single_abs), name=f"single_lb_{i}")

        # ——— 4) L1 约束 ∑|x_i| <= l1_abs
        #   实现方式： t[i] >=  x[i], t[i] >= -x[i], 然后 ∑ t[i] <= l1_abs
        if self.l1_abs is not None:
            for i in range(self.N):
                m.addConstr(t[i] >=  x[i], name=f"t_pos_{i}")
                m.addConstr(t[i] >= -x[i], name=f"t_neg_{i}")
            m.addConstr(t.sum() <= float(self.l1_abs), name="l1_norm")

        # ——— 5) 二次型约束 x' * cov_matrix * x <= sigma_abs
        if self.cov_matrix is not None and self.sigma_abs is not None:
            x_vec = np.array([x[i] for i in range(self.N)])
            expr_q = x_vec @ self.cov_matrix @ x_vec
            m.addQConstr(expr_q <= float(self.sigma_abs), name="sigma_qc")

        # ——— 6) 换手率约束 ∑|x_i - w_prev_i| <= turnover (如果启用)
        if self.turnover is not None:
            # 清空之前的约束引用列表
            self._turnover_pos_constrs = []
            self._turnover_neg_constrs = []
            
            for i in range(self.N):
                # s[i] >= x[i] - w_prev[i]
                pos_constr = m.addConstr(
                    s[i] >= x[i] - float(self.w_prev[i]), 
                    name=f"turnover_pos_{i}"
                )
                # s[i] >= -(x[i] - w_prev[i]) = w_prev[i] - x[i]
                neg_constr = m.addConstr(
                    s[i] >= float(self.w_prev[i]) - x[i], 
                    name=f"turnover_neg_{i}"
                )
                
                # 存储约束引用（关键优化）
                self._turnover_pos_constrs.append(pos_constr)
                self._turnover_neg_constrs.append(neg_constr)
            
            # 换手率总约束
            m.addConstr(s.sum() <= float(2*self.turnover), name="turnover_total")
            
            # 保存辅助变量以便后续更新
            self._turnover_vars = s

        return m, x
    
    def _updateTurnoverConstraints(self):
        """
        Update turnover constraints when w_prev changes
        This is called automatically when setPrevWeights is used
        
        优化版本：直接使用存储的约束引用，避免遍历所有约束
        时间复杂度从 O(N * total_constraints) 降低到 O(N)
        """
        if self.turnover is None or not hasattr(self, '_turnover_vars'):
            return
            
        # 检查约束引用是否存在
        if (not hasattr(self, '_turnover_pos_constrs') or 
            not hasattr(self, '_turnover_neg_constrs') or
            len(self._turnover_pos_constrs) != self.N or
            len(self._turnover_neg_constrs) != self.N):
            # 如果约束引用不存在或不完整，跳过更新
            print("Warning: Turnover constraint references not found, skipping update")
            return
        
        # 直接使用存储的约束引用进行快速更新
        for i in range(self.N):
            # 移除旧约束
            self._model.remove(self._turnover_pos_constrs[i])
            self._model.remove(self._turnover_neg_constrs[i])
            
            # 添加新约束并更新引用
            self._turnover_pos_constrs[i] = self._model.addConstr(
                self._turnover_vars[i] >= self.x[i] - float(self.w_prev[i]), 
                name=f"turnover_pos_{i}"
            )
            self._turnover_neg_constrs[i] = self._model.addConstr(
                self._turnover_vars[i] >= float(self.w_prev[i]) - self.x[i], 
                name=f"turnover_neg_{i}"
            )
        
        # 更新模型
        self._model.update()

    def getInfo(self):
        """
        Get model information for debugging
        
        Returns:
            dict: Model information
        """
        info = {
            'num_assets': self.N,
            'has_turnover': self.turnover is not None,
            'turnover_limit': self.turnover,
            'prev_weights': self.w_prev.tolist() if self.w_prev is not None else None,
            'has_turnover_refs': (hasattr(self, '_turnover_pos_constrs') and 
                                len(self._turnover_pos_constrs) == self.N),
            'constraints': {
                'risk_factor': self.risk_abs is not None,
                'single_abs': self.single_abs is not None,
                'l1_norm': self.l1_abs is not None,
                'quadratic_risk': self.sigma_abs is not None
            }
        }
        return info

    def solveSequential(self, cost_vectors):
        """
        Solve multiple periods sequentially with turnover constraints
        
        Args:
            cost_vectors (np.ndarray): Cost vectors for each period (T x N)
            
        Returns:
            tuple: (solutions, objectives) where solutions is list of T solutions
                   and objectives is list of T objective values
        """
        if cost_vectors.ndim != 2:
            raise ValueError("cost_vectors must be 2D array (T x N)")
            
        T, N = cost_vectors.shape
        if N != self.N:
            raise ValueError(f"Number of assets in cost_vectors ({N}) must match model ({self.N})")
        
        solutions = []
        objectives = []
        
        # Reset to equal weights for first period
        self.w_prev = np.ones(self.N) / self.N
        
        for t in range(T):
            # Set objective for this period
            self.setObj(cost_vectors[t])
            
            # Solve optimization problem
            sol, obj = self.solve()
            solutions.append(sol)
            objectives.append(obj)
            
            # Set current solution as previous weights for next period
            #if t < T - 1 and self.turnover is not None:
            if self.turnover is not None:
                self.setPrevWeights(sol)
        
        return solutions, objectives


def build_market_neutral_model_testing(N, A, b, l, u, risk_f, risk_abs, single_abs, l1_abs, cov_matrix, sigma_abs,turnover):
    return MarketNeutralGrbModel_testing(
        N, A, b, l, u, minimize=False,
        risk_f=risk_f, risk_abs=risk_abs,
        single_abs=single_abs, l1_abs=l1_abs,
        cov_matrix=cov_matrix, sigma_abs=sigma_abs,turnover=turnover
    )