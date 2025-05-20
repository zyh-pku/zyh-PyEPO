# data_loader.py
import torch
from torch.utils.data import DataLoader
from config import DEVICE, BATCH_SIZE

def get_data_loaders(
    dataset_train,
    dataset_test,
):
    """
    返回固定随机种子且带预取的 train/test DataLoader。
    """


    train_loader = DataLoader(
        dataset_train,
        batch_size=BATCH_SIZE,
        shuffle=True,
        # pin_memory=pin_memory,
        # num_workers=num_workers,
        # persistent_workers=persistent_workers,
    )

    test_loader = DataLoader(
        dataset_test,
        batch_size=BATCH_SIZE,
        shuffle=False,
        # pin_memory=pin_memory,
        num_workers=0,
        # persistent_workers=False,
    )

    return train_loader, test_loader


def device_loader(loader: DataLoader):
    """
    把 DataLoader 里的每个 batch 移到 DEVICE 上，
    并保持原来 batch 元组/列表的结构。    
    """
    for batch in loader:
        # 如果 batch 是 tuple/list，就对其中每个 tensor 调用 to(device)
        yield tuple(item.to(DEVICE) for item in batch)
