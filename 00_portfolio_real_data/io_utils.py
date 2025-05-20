import os
import pandas as pd
import numpy as np
import pickle
from typing import Any, Dict, Optional
import platform

import glob
import matplotlib.pyplot as plt
from matplotlib import font_manager

from tqdm.auto import tqdm

from pyepo.data.dataset import optDataset


def process_perp_futures_data(
    data_path: str = "perp_futures_klines",
    output_path: str = "processed_crypto_data.csv",
) -> pd.DataFrame:
    """
    Load, merge, process perp futures data from parquet files.

    Parameters
    ----------
    data_path : str
        Path to directory containing parquet files.
    output_csv : str, optional
        If provided, path to save the processed DataFrame as CSV.
    feature_columns : list, optional
        List of columns to treat as features. Defaults to
        ['open', 'high', 'low', 'close', 'volume', 'count'].
    horizon : int
        Number of rows ahead for return calculation (default 30).
    symbol_col : str
        Column name for the symbol identifier (default 'symbol').
    time_col : str
        Column name for the timestamp (default 'open_time').
    plot_symbol_counts : bool
        Whether to plot counts per symbol.

    Returns
    -------
    df : pd.DataFrame
        Processed DataFrame including feature columns, future return column,
        and symbol/time columns.
    """
    # 创建下载目录
    os.makedirs("crypto_data", exist_ok=True)

    # 检查数据文件夹是否存在
    if not os.path.exists(data_path):
        print(f"错误: 数据文件夹 {data_path} 不存在，请确保已下载数据")
    else:
        print(f"数据文件夹存在: {data_path}")
        
        # 查找所有 parquet 文件
        parquet_files = glob.glob(f"{data_path}/**/*.parquet", recursive=True)
        print(f"找到 {len(parquet_files)} 个 parquet 文件")
        
        if len(parquet_files) == 0:
            print("未找到数据文件，请确保数据已正确下载")
        else:
            # 读取所有 parquet 文件并合并
            print("正在读取数据文件...")
            dfs = []
            
            for file in tqdm(parquet_files):
                try:
                    df_temp = pd.read_parquet(file, engine='pyarrow')
                    # 从文件名提取 symbol
                    file_name = os.path.basename(file)
                    symbol = file_name.split('_')[0]
                    df_temp['symbol'] = symbol
                    
                    dfs.append(df_temp)
                      
                    
                except Exception as e:
                    print(f"读取文件 {file} 时出错: {e}")
                
            # 合并所有数据
            if dfs:
                print("合并所有数据...")
                df = pd.concat(dfs, ignore_index=True)
                
                # 查看数据的前几行
                print("数据预览:")
                print(df.head())
                
                # 检查数据类型和缺失值
                print("\n数据信息:")
                print(df.info())
                print("\n缺失值统计:")
                print(df.isnull().sum())
                
                # 确保 open_time 列是 datetime 类型
                df['open_time'] = pd.to_datetime(df['open_time'])
                
                # 设置特征列
                feature_columns = ['open', 'high', 'low', 'close', 'volume', 'count'] 
                
                # 计算30分钟收益率（horizon = 30min）
                print("\n计算30分钟收益率...")
                df = df.sort_values(['symbol', 'open_time'])  # 确保数据按 symbol 和时间排序
                df['future_close'] = df.groupby('symbol')['close'].shift(-30)  # 30个时间单位后的收盘价
                df['return_30min'] = df['future_close'] / df['close'] - 1  # 计算收益率
                
                # 删除缺失的目标变量行
                df = df.dropna(subset=['return_30min'])
                
                # 查看处理后的数据
                print("\n处理后的数据预览:")
                print(df.head())
                
                # 数据已经包含所有 symbol，确保它们被正确标记
                symbols = df['symbol'].unique()
                print(f"\n数据集中包含的交易对: {symbols}")
                print(f"总数据行数: {len(df)}")
                
                system = platform.system()
                if system == "Darwin":
                    home = os.environ["HOME"]
                    font_path = home+"/Library/Fonts/NotoSansCJKsc-Regular.otf"
                    font_manager.fontManager.addfont(font_path)
                    plt.rcParams['font.family'] = 'Noto Sans CJK SC'
                    plt.rcParams['axes.unicode_minus'] = False
                
                # 可视化不同 symbol 的数据分布
                plt.figure(figsize=(12, 6))
                df['symbol'].value_counts().plot(kind='bar')
                plt.title('各交易对数据量')
                plt.ylabel('数据点数')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.show()
                
                # 准备特征矩阵 X 和目标变量 Y
                X = df[feature_columns]
                y = df['return_30min']
                
                # 查看特征和目标变量的统计信息
                print("\n特征统计信息:")
                print(X.describe())
                
                print("\n目标变量统计信息:")
                print(y.describe())
                
                # 保存处理后的数据
                print("\n保存处理后的数据...")
                df.to_csv(output_path, index=False)
                print("数据处理完成!")
            else:
                print("没有成功读取任何数据文件")
    
    return df



def save_optData(features, costs, unique_times, unique_symbols, x_cols, name='default', save_dir='./data'):
    """
    Save processed data to disk with a simple name instead of timestamp
    """
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Save numpy arrays
    np.save(f"{save_dir}/features_{name}.npy", features)
    np.save(f"{save_dir}/costs_{name}.npy", costs)
    
    # Save metadata as pickle files
    with open(f"{save_dir}/times_{name}.pkl", 'wb') as f:
        pickle.dump(unique_times, f)
    
    with open(f"{save_dir}/symbols_{name}.pkl", 'wb') as f:
        pickle.dump(unique_symbols, f)
    
    # Save feature dimensions information
    feature_info = {
        'shape': features.shape,
        'feature_names': x_cols
    }
    with open(f"{save_dir}/feature_info_{name}.pkl", 'wb') as f:
        pickle.dump(feature_info, f)
    
    print(f"Data saved to {save_dir} with name '{name}'")



def load_optData(name='default', load_dir='./data'):
    """
    Load processed data from disk using a simple name instead of timestamp
    """
    # Load numpy arrays
    features = np.load(f"{load_dir}/features_{name}.npy")
    costs = np.load(f"{load_dir}/costs_{name}.npy")
    
    # Load metadata
    with open(f"{load_dir}/times_{name}.pkl", 'rb') as f:
        unique_times = pickle.load(f)
    
    with open(f"{load_dir}/symbols_{name}.pkl", 'rb') as f:
        unique_symbols = pickle.load(f)
    
    # Load feature information
    with open(f"{load_dir}/feature_info_{name}.pkl", 'rb') as f:
        feature_info = pickle.load(f)
    
    print(f"Loaded data from {load_dir} with name '{name}'")
    print(f"Features shape: {features.shape}")
    print(f"Feature names: {feature_info['feature_names']}")
    
    return features, costs, unique_times, unique_symbols, feature_info


def create_dataset_from_dict(dataset_dict, model):
    """从字典创建optDataset实例，并载入预计算的sols和objs"""
    dataset = optDataset(
        model=model,
        feats=dataset_dict['feats'],
        costs=dataset_dict['costs'],
        lookback=dataset_dict['lookback'],
        padding_method=dataset_dict['padding_method'],
        precomputed =True
    )
    
    # 用预计算的解替换
    dataset.sols = dataset_dict['sols']
    dataset.objs = dataset_dict['objs']
    
    return dataset


def save_dataset_dict(dataset_dict, filename):
    """保存数据集字典到文件"""
    np.savez(
        filename,
        feats=dataset_dict['feats'],
        costs=dataset_dict['costs'],
        sols=dataset_dict['sols'],
        objs=dataset_dict['objs'],
        lookback=dataset_dict['lookback'],
        padding_method=dataset_dict['padding_method']
    )
    

def load_dataset_dict(filename):
    """从文件加载数据集字典"""
    data = np.load(filename, allow_pickle=True)
    return {
        'feats': data['feats'],
        'costs': data['costs'],
        'sols': data['sols'],
        'objs': data['objs'],
        'lookback': data['lookback'].item(),
        'padding_method': str(data['padding_method'])
    }