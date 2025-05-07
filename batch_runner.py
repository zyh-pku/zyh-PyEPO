import numpy as np
import gc
from multiprocessing import get_context, shared_memory, Process
from model_factory import build_market_neutral_model
from pyepo.data.dataset import optDataset
import gurobipy as gp

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

def run_batch_shared(shm_names, shapes, dtypes, feats_data, costs_data, lookback, padding_method,
                      N, A, b, l, u, risk_f, risk_abs, single_abs, l1_abs, cov_matrix, sigma_abs):
    model = build_market_neutral_model(N, A, b, l, u, risk_f, risk_abs, single_abs, l1_abs, cov_matrix, sigma_abs)
    dataset = optDataset(model, feats_data, costs_data, lookback=lookback, padding_method=padding_method)

    feats_shm = shared_memory.SharedMemory(name=shm_names['feats'])
    sols_shm = shared_memory.SharedMemory(name=shm_names['sols'])
    objs_shm = shared_memory.SharedMemory(name=shm_names['objs'])

    feats_np = np.ndarray(shapes['feats'], dtype=dtypes['feats'], buffer=feats_shm.buf)
    sols_np = np.ndarray(shapes['sols'], dtype=dtypes['sols'], buffer=sols_shm.buf)
    objs_np = np.ndarray(shapes['objs'], dtype=dtypes['objs'], buffer=objs_shm.buf)

    feats_out = dataset.feats
    if feats_out.shape != feats_np.shape:
        raise ValueError(f"Shape mismatch: dataset.feats {feats_out.shape} vs shared feats {feats_np.shape}")

    feats_np[:] = feats_out
    sols_np[:] = dataset.sols
    objs_np[:] = dataset.objs

    for shm in [feats_shm, sols_shm, objs_shm]:
        shm.close()
    del dataset
    gc.collect()