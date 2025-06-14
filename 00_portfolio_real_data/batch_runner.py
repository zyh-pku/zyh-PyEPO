# General imports
import time
import pandas as pd
import numpy as np
import gc
from multiprocessing import get_context, shared_memory, Process

# Model imports
from model_factory import build_market_neutral_model,build_market_neutral_model_testing
from config import *

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

# PCA
from sklearn.decomposition import PCA
import pickle


os.environ['GRB_LICENSE_FILE'] = os.path.expanduser("~/gurobi/gurobi.lic")

def run_batch_shared(shm_names, shapes, dtypes, feats_data, costs_data, N, A, b, l, u, risk_f, risk_abs, single_abs, l1_abs, cov_matrix, sigma_abs):
    model = build_market_neutral_model(N, A, b, l, u, risk_f, risk_abs, single_abs, l1_abs, cov_matrix, sigma_abs)
    dataset = optDataset(model, feats_data, costs_data, lookback=LOOKBACK, padding_method=PADDING_METHOD)

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
    



def process_and_combine_shared(features, costs, batch_size=1000, N=None, A=None, b=None, l=None, u=None, risk_f=None, risk_abs=None, single_abs=None, l1_abs=None, cov_matrix=None, sigma_abs=None):
    ctx = get_context('spawn')
    total_samples = len(features)
    all_feats, all_sols, all_objs = [], [], []

    if N is None:
        N = features.shape[1] 
        A = np.ones((1, N))
        b = np.array([1.0])
        l = np.zeros(N)
        u = np.zeros(N) + 1e6
        # 把cov_matrix修改为costs的covariance
        ## M = np.random.randn(N, N)
        ## cov_matrix = M.T @ M + np.eye(N) * 1e-3
        cov_matrix = np.cov(costs, rowvar=False, bias=False)
        # 把risk_f修改为第一主成分
        ## risk_f = np.random.randn(N)
        pca = PCA(n_components=1)
        risk_f = pca.fit_transform(cov_matrix).ravel()
        
        risk_abs   = 1.5
        single_abs = 0.1
        l1_abs     = 1.0
        sigma_abs  = 2.5

    for i in range(0, total_samples, batch_size):
        start, end = i, min(i + batch_size, total_samples)
        print(f"\n 共享内存子进程处理样本 {start} 到 {end - 1}...")

        feats_batch = features[start:end]
        costs_batch = costs[start:end]

        shapes = {
            'feats': (feats_batch.shape[0], feats_batch.shape[1], LOOKBACK, feats_batch.shape[2]),
            'sols': (feats_batch.shape[0], feats_batch.shape[1]),
            'objs': (feats_batch.shape[0],1)
        }
        dtypes = {'feats': feats_batch.dtype, 'sols': np.float32, 'objs': np.float32}

        shms = {key: shared_memory.SharedMemory(create=True, size=np.zeros(shapes[key], dtype=dtypes[key]).nbytes)
                for key in shapes}

        shm_names = {k: v.name for k, v in shms.items()}

        p = ctx.Process(
            target=run_batch_shared,
            args=(shm_names, shapes, dtypes, feats_batch, costs_batch, N, A, b, l, u, risk_f, risk_abs, single_abs, l1_abs, cov_matrix, sigma_abs)
        )
        p.start()
        p.join()

        feats_np = np.ndarray(shapes['feats'], dtype=dtypes['feats'], buffer=shms['feats'].buf).copy()
        sols_np = np.ndarray(shapes['sols'], dtype=dtypes['sols'], buffer=shms['sols'].buf).copy()
        objs_np = np.ndarray(shapes['objs'], dtype=dtypes['objs'], buffer=shms['objs'].buf).copy()

        all_feats.append(feats_np)
        all_sols.append(sols_np)
        all_objs.append(objs_np)

        for shm in shms.values():
            shm.close()
            shm.unlink()

        del feats_batch, costs_batch
        gc.collect()

    print("\n 合并所有批次...")
    return {
        'feats': np.concatenate(all_feats, axis=0),
        'costs': costs,
        'sols': np.concatenate(all_sols, axis=0),
        'objs': np.concatenate(all_objs, axis=0),
        'lookback': LOOKBACK,
        'padding_method': PADDING_METHOD
    }


