from io_utils import process_perp_futures_data, save_optData, save_dataset_dict
from config import *
from data_utils import align_time_series_fast, GroupMinMaxScaler, pivot_features_and_costs
from batch_runner import process_and_combine_shared
from pyepo.data.dataset import optDataset
from model_factory import build_market_neutral_model,build_market_neutral_model_testing
import pandas as pd
import numpy as np
import os
from sklearn.decomposition import PCA
import pickle
import psutil
import gc
def split_train_test_by_time_quantile(df, time_col='open_time', frac=0.8):
    """
    按 time_col 的 frac 分位点来划分训练/测试集。

    参数
    -----
    df       : pandas.DataFrame
        包含时间列（或索引）的 DataFrame。
    time_col : str, optional
        时间列名，如果 df 已经用时间索引则设为 None。
    frac     : float, optional
        用于训练集的比例，默认 0.8（即前 80% 的时间）。

    返回
    -----
    train_df, test_df : pandas.DataFrame
    """
    # 如果 open_time 是列，就用那列；否则假设 df.index 是 DatetimeIndex
    if time_col is not None:
        times = pd.to_datetime(df[time_col])
    else:
        times = pd.to_datetime(df.index)

    # 计算 80% 分位的时间点
    cutoff = times.quantile(frac)

    # 划分
    if time_col is not None:
        train_df = df[times <= cutoff]
        test_df  = df[times  > cutoff]
    else:
        train_df = df[df.index <= cutoff]
        test_df  = df[df.index  > cutoff]

    return train_df, test_df

def print_memory(stage):
    """打印内存使用"""
    process = psutil.Process()
    memory_gb = process.memory_info().rss / (1024**3)
    print(f" {stage}: {memory_gb:.2f} GB")
    return memory_gb

#个人许可证
os.environ['GRB_LICENSE_FILE'] = os.path.expanduser("~/gurobi/gurobi.lic")


if __name__ == "__main__":

    combined_df = process_perp_futures_data(data_path=RAW_DATA_PATH, output_path=PROCESSED_DATA_PATH)

    print_memory("加载数据后")
    
    
    aligned_df = align_time_series_fast(combined_df)
    aligned_df.to_parquet(ALIGNED_CRYPTO_DATA_PATH)
    print("时间对齐后的数据信息：")
    print(aligned_df.shape)
    print(aligned_df.head())
    print_memory("加载对齐数据后")
    print("\n  删除 combined_df...")
    del combined_df  # 删除aligned_df和times
    gc.collect()
    print_memory("删除combined_df后")
    
    
    ### 划分训练集&测试集，open_time的前80%为训练集
    train_df, test_df = split_train_test_by_time_quantile(aligned_df, time_col='open_time', frac=0.8)
    print("Train 时间范围：", train_df['open_time'].min(), "—", train_df['open_time'].max())
    print("Test  时间范围：",  test_df['open_time'].min(),  "—", test_df['open_time'].max())

    print("\n  删除 aligned_df...")
    del aligned_df
    gc.collect()
    print_memory("删除aligned_df后")
    

    
    
    ### 对训练集做scale和precompute，保存MarketNeutralModel和precomputed_train_optData
    scaler = GroupMinMaxScaler(feature_range=(-1,1), target_columns=X_COLS, group_by_column='symbol')
    print("对训练集做scale")
    train_df = scaler.fit_transform(train_df)

    print_memory("scale train_df 后")
    
    print("pivot features and costs")
    features, costs, unique_times, unique_symbols = pivot_features_and_costs(train_df, y_col=Y_COL, x_cols=X_COLS)
    save_optData(features, costs, unique_times, unique_symbols, X_COLS, name=OPTDATA_NAME, save_dir=TRAIN_OPTDATA_DIR)
    print_memory("pivot train_df 后")

    del train_df
    gc.collect()
    print_memory("删除train_df后") 
    
    N = features.shape[1] 
    A = np.ones((1, N))
    b = np.array([1.0])
    l = np.zeros(N)
    u = np.zeros(N) + 1e6
    # 把cov_matrix修改为costs的covariance
    ##M = np.random.randn(N, N)
    ###cov_matrix = M.T @ M + np.eye(N) * 1e-3
    cov_matrix = np.cov(costs, rowvar=False, bias=False)
    # 把risk_f修改为第一主成分
    ## risk_f = np.random.randn(N)
    pca = PCA(n_components=1)
    risk_f = pca.fit_transform(cov_matrix).ravel()
    
    risk_abs   = 1.5
    single_abs = 0.1
    l1_abs     = 1.0
    sigma_abs  = 2.5
    turnover   = 0.3

    print("保存常规模型参数...")
    params = dict(
        N=N, A=A, b=b, l=l, u=u,
        risk_f=risk_f, risk_abs=risk_abs,
        single_abs=single_abs, l1_abs=l1_abs,
        cov_matrix=cov_matrix, sigma_abs=sigma_abs
    )
    with open(MARKET_MODEL_DIR, "wb") as f:
        pickle.dump(params, f)
    
    try:
        model = build_market_neutral_model(N, A, b, l, u, risk_f, risk_abs, single_abs, l1_abs, cov_matrix, sigma_abs)
        print("✓ 常规模型创建成功")
    except Exception as e:
        print(f"✗ 常规模型创建失败: {e}")

    # 保存测试模型参数（包含turnover）
    print(" 保存测试模型参数...")
    params_testing = dict(
        N=N, A=A, b=b, l=l, u=u,
        risk_f=risk_f, risk_abs=risk_abs,
        single_abs=single_abs, l1_abs=l1_abs,
        cov_matrix=cov_matrix, sigma_abs=sigma_abs, turnover=turnover
    )
    with open(MARKET_MODEL_DIR_TESTING, "wb") as f:
        pickle.dump(params_testing, f)
    
    try:
        model_testing = build_market_neutral_model_testing(N, A, b, l, u, risk_f, risk_abs, single_abs, l1_abs, cov_matrix, sigma_abs, turnover)
        print("✓ 测试模型创建成功")
    except Exception as e:
        print(f"✗ 测试模型创建失败: {e}")
    """
    params = dict(
    N=N, A=A, b=b, l=l, u=u,
    risk_f=risk_f, risk_abs=risk_abs,
    single_abs=single_abs, l1_abs=l1_abs,
    cov_matrix=cov_matrix, sigma_abs=sigma_abs)
    with open(MARKET_MODEL_DIR, "wb") as f:
        pickle.dump(params, f)
    model = build_market_neutral_model(N, A, b, l, u, risk_f, risk_abs, single_abs, l1_abs, cov_matrix, sigma_abs)

    params_testing = dict(
    N=N, A=A, b=b, l=l, u=u,
    risk_f=risk_f, risk_abs=risk_abs,
    single_abs=single_abs, l1_abs=l1_abs,
    cov_matrix=cov_matrix, sigma_abs=sigma_abs,turnover=turnover)
    with open(MARKET_MODEL_DIR_TESTING, "wb") as f:
        pickle.dump(params_testing, f)
    model_testing = build_market_neutral_model_testing(N, A, b, l, u, risk_f, risk_abs, single_abs, l1_abs, cov_matrix, sigma_abs, turnover)
    """
    
    
    test_df = scaler.transform(test_df)
    test_features, test_costs, _, _ = pivot_features_and_costs(test_df, y_col=Y_COL, x_cols=X_COLS)
    os.makedirs(TEST_DATA_DIR, exist_ok=True)
    np.save(f"{TEST_DATA_DIR}/features_{OPTDATA_NAME}.npy", test_features)
    del test_df
    gc.collect()
    print_memory("删除test_df后") 
    
    
       
    dataset_dict = process_and_combine_shared(
        features=features,
        costs=costs,
        batch_size=PRECOMPUTE_BATCH_SIZE)
    save_dataset_dict(dataset_dict, DATASET_DICT_PATH)
    
    test_dataset_dict = process_and_combine_shared(
        features=test_features,
        costs=test_costs,
        batch_size=PRECOMPUTE_BATCH_SIZE)
    save_dataset_dict(test_dataset_dict, TEST_DATASET_DICT_PATH)