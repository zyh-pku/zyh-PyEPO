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






