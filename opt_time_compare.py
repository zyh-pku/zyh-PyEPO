import time
import numpy as np
import pandas as pd

from tqdm.auto import tqdm  # Progress bar

# PyEPO imports
from pkg.pyepo.data.dataset import optDataset

# Gurobi
import gurobipy as gp
from gurobipy import GRB
from pkg.pyepo.model.grb import optGrbModel

# Pyomo
from pyomo import environ as pe
from pkg.pyepo.model.omo import optOmoModel # 记得要把 omo 文件夹里的 __init__.py 文件也修改了
from pkg.pyepo import EPO

# COPTPy
from coptpy import Envr, COPT
from pkg.pyepo.model.copt import optCoptModel

# MPAX
from pkg.pyepo.model.mpax import optMpaxModel

# ——— 参数设置 ———
N = 100
A = np.ones((1, N))
b = np.array([1.0])
l = np.zeros(N)
u = np.zeros(N) + 1e6
reps = 5

# ——— 随机生成正定协方差矩阵 ———
M = np.random.randn(N, N)
cov_matrix = M.T @ M + np.eye(N) * 1e-3

# ——— 风险与约束参数 ———
risk_f     = np.random.randn(N)
risk_abs   = 1.5
single_abs = 0.1
l1_abs     = 1.0
sigma_abs  = 2.5


# ——— 1) Gurobi 版本 
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from pkg.pyepo.model.grb import optGrbModel

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

# ——— 2) Pyomo 版本
class MarketNeutralPyomoModel(optOmoModel):
    def __init__(self, N, A, b, l, u, minimize=False, solver='gurobi',
                 risk_f=None, risk_abs=None,
                 single_abs=None, l1_abs=None,
                 cov_matrix=None, sigma_abs=None):
        self.N, self.A, self.b = N, A, b
        self.l, self.u, self.minimize       = l, u, minimize
        self.risk_f, self.risk_abs         = risk_f, risk_abs
        self.single_abs, self.l1_abs       = single_abs, l1_abs
        self.cov_matrix, self.sigma_abs    = cov_matrix, sigma_abs
        super().__init__(solver=solver)

    def _getModel(self):
        self.modelSense = EPO.MINIMIZE if self.minimize else EPO.MAXIMIZE
        m = pe.ConcreteModel()
        m.x = pe.Var(range(self.N), domain=pe.Reals,
                     bounds=lambda m,i: (float(self.l[i]), float(self.u[i])))
        m.t = pe.Var(range(self.N), domain=pe.NonNegativeReals)
        m.cons = pe.ConstraintList()

        # 中性约束
        m.cons.add(sum(self.A[0,i]*m.x[i] for i in range(self.N)) == float(self.b[0]))
        # 风险约束
        m.cons.add(sum(self.risk_f[i]*m.x[i] for i in range(self.N)) <= float(self.risk_abs))
        m.cons.add(sum(self.risk_f[i]*m.x[i] for i in range(self.N)) >= -float(self.risk_abs))
        # 单项绝对值
        for i in range(self.N):
            m.cons.add(m.x[i] <=  float(self.single_abs))
            m.cons.add(m.x[i] >= -float(self.single_abs))
        # L1 norm
        for i in range(self.N):
            m.cons.add(m.t[i] >=  m.x[i])
            m.cons.add(m.t[i] >= -m.x[i])
        m.cons.add(sum(m.t[i] for i in range(self.N)) <= float(self.l1_abs))
        # 二次型约束
        m.cons.add(
            sum(self.cov_matrix[i,j] * m.x[i] * m.x[j]
                for i in range(self.N) for j in range(self.N))
            <= float(self.sigma_abs)
        )

        return m, m.x


# ——— 3) COPTPy 版本
class MarketNeutralCoptModel(optCoptModel):
    def __init__(self, N, A, b, l, u, minimize=False,
                 risk_f=None, risk_abs=None,
                 single_abs=None, l1_abs=None,
                 cov_matrix=None, sigma_abs=None):
        self.N, self.A, self.b = N, A, b
        self.l, self.u, self.minimize       = l, u, minimize
        self.risk_f, self.risk_abs         = risk_f, risk_abs
        self.single_abs, self.l1_abs       = single_abs, l1_abs
        self.cov_matrix, self.sigma_abs    = cov_matrix, sigma_abs
        self._model, self.x = self._getModel()

    def _getModel(self):
        env = Envr()
        m = env.createModel()
        x = m.addVars(self.N, nameprefix='x',
                      lb={i: float(self.l[i]) for i in range(self.N)},
                      ub={i: float(self.u[i]) for i in range(self.N)})
        t = m.addVars(self.N, lb=0.0, nameprefix='t')
        m.setObjSense(COPT.MINIMIZE if self.minimize else COPT.MAXIMIZE)

        # 中性约束
        m.addConstr(sum(self.A[0,i]*x[i] for i in range(self.N)) == float(self.b[0]),
                    name="eq_sum")
        # 风险约束
        m.addConstr(sum(self.risk_f[i]*x[i] for i in range(self.N)) <= float(self.risk_abs),
                    name="risk_ub")
        m.addConstr(sum(self.risk_f[i]*x[i] for i in range(self.N)) >= -float(self.risk_abs),
                    name="risk_lb")
        # 单项绝对值
        for i in range(self.N):
            m.addConstr(x[i] <=  float(self.single_abs), name=f"single_ub_{i}")
            m.addConstr(x[i] >= -float(self.single_abs), name=f"single_lb_{i}")
        # L1 norm
        for i in range(self.N):
            m.addConstr(t[i] >=  x[i], name=f"t_pos_{i}")
            m.addConstr(t[i] >= -x[i], name=f"t_neg_{i}")
        m.addConstr(sum(t[i] for i in range(self.N)) <= float(self.l1_abs), name="l1_norm")
        # 二次型约束
        # COPTpy 支持 addQConstr
        expr_q = sum(self.cov_matrix[i,j] * x[i] * x[j]
                     for i in range(self.N) for j in range(self.N))
        m.addQConstr(expr_q <= float(self.sigma_abs), name="sigma_qc")

        return m, x


# ——— 工厂注入并运行基准 ———
model_factories = {
    'Gurobi': lambda: MarketNeutralGrbModel(
                    N, A, b, l, u, minimize=False,
                    risk_f=risk_f,    risk_abs=risk_abs,
                    single_abs=single_abs, l1_abs=l1_abs,
                    cov_matrix=cov_matrix, sigma_abs=sigma_abs),
    'Pyomo':  lambda: MarketNeutralPyomoModel(
                    N, A, b, l, u, minimize=False, solver='gurobi',
                    risk_f=risk_f,    risk_abs=risk_abs,
                    single_abs=single_abs, l1_abs=l1_abs,
                    cov_matrix=cov_matrix, sigma_abs=sigma_abs),
    'COPTPy': lambda: MarketNeutralCoptModel(
                    N, A, b, l, u, minimize=False,
                    risk_f=risk_f,    risk_abs=risk_abs,
                    single_abs=single_abs, l1_abs=l1_abs,
                    cov_matrix=cov_matrix, sigma_abs=sigma_abs),
    'MPAX':   None  # 如前所示，跳过 MPAX
}

times = {name: [] for name in model_factories if model_factories[name]}
total_tasks = reps * len([n for n in model_factories if model_factories[n]])
pbar = tqdm(total=total_tasks, desc="Running Models")

for _ in range(reps):
    c_vec = np.random.rand(N)
    for name, factory in model_factories.items():
        if factory is None or name == 'Gurobi' or name == 'COPTPy':
            continue
        pbar.set_description(f"Running {name}")
        t0 = time.time()
        model = factory()
        model.setObj(c_vec)
        model.solve()
        times[name].append(time.time() - t0)
        pbar.update(1)

pbar.close()

df = pd.DataFrame([{
    'Model':         name,
    'Mean Time (s)': np.mean(times[name]),
    'STD Time (s)':  np.std(times[name])
} for name in times])

print(df)
