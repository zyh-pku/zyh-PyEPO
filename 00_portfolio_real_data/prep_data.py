from io_utils import process_perp_futures_data, align_time_series_fast, save_optData, save_dataset_dict
from config import *
from data_utils import minmaxscaler_by_symbol, pivot_features_and_costs
from batch_runner import process_and_combine_shared
from pyepo.data.dataset import optDataset

if __name__ == "__main__":
    
    combined_df = process_perp_futures_data(data_path=RAW_DATA_PATH, output_path=PROCESSED_DATA_PATH)
    
    aligned_df = align_time_series_fast(combined_df)
    aligned_df.to_parquet(ALIGNED_CRYPTO_DATA_PATH)
    print("时间对齐后的数据信息：")
    print(aligned_df.shape)
    print(aligned_df.head())
    
    scaled_df = minmaxscaler_by_symbol(aligned_df, target_columns=X_COLS)
    features, costs, unique_times, unique_symbols = pivot_features_and_costs(scaled_df, y_col=Y_COL, x_cols=X_COLS)
    save_optData(features, costs, unique_times, unique_symbols, X_COLS, name=OPTDATA_NAME, save_dir=OPTDATA_DIR)
    
    
    dataset_dict = process_and_combine_shared(
        features=features,
        costs=costs,
        batch_size=PRECOMPUTE_BATCH_SIZE,  
        lookback=LOOKBACK,
        padding_method=PADDING_METHOD)
    save_dataset_dict(dataset_dict, DATASET_DICT_PATH)