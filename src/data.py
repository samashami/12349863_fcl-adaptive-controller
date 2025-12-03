import os
from dataclasses import dataclass
from typing import List, Dict, Tuple

import torch
from torch.utils.data import Dataset, Subset, DataLoader, random_split
from torchvision import datasets, transforms


@dataclass
class DataConfig:
    data_root: str = "./data"
    batch_size: int = 256
    num_workers: int = 4
    num_clients: int = 4
    num_cl_batches: int = 7
    val_size: int = 5000  # from 50k train -> 45k train / 5k val


def get_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    """Return train and test transforms (ImageNet-style)."""
    train_tf = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )

    test_tf = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )

    return train_tf, test_tf


def load_cifar100(cfg: DataConfig):
    """Download CIFAR-100 and return raw train/test datasets."""
    train_tf, test_tf = get_transforms()

    train_full = datasets.CIFAR100(
        root=cfg.data_root,
        train=True,
        transform=train_tf,
        download=True,
    )
    test = datasets.CIFAR100(
        root=cfg.data_root,
        train=False,
        transform=test_tf,
        download=True,
    )

    return train_full, test


def split_train_val(
    train_full: Dataset, val_size: int
) -> Tuple[Dataset, Dataset]:
    """Split train_full into train and val (by size)."""
    total = len(train_full)
    assert val_size < total, "Validation size must be smaller than train size."
    train_size = total - val_size
    train, val = random_split(
        train_full,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )
    return train, val


def make_iid_client_splits(
    train: Dataset,
    num_clients: int,
) -> Dict[int, List[int]]:
    """
    Create IID client splits by randomly partitioning indices
    into num_clients chunks of equal size (as far as possible).
    """
    n = len(train)
    indices = torch.randperm(n).tolist()
    client_splits: Dict[int, List[int]] = {}
    chunk_size = n // num_clients

    for cid in range(num_clients):
        start = cid * chunk_size
        end = n if cid == num_clients - 1 else (cid + 1) * chunk_size
        client_splits[cid] = indices[start:end]

    return client_splits


def make_cl_batches_for_client(
    client_indices: List[int], num_cl_batches: int
) -> List[List[int]]:
    """
    Split a client's indices into num_cl_batches chunks (continual batches).
    Batches are sequential subsets of the index list.
    """
    n = len(client_indices)
    batch_size = n // num_cl_batches
    batches: List[List[int]] = []

    for b in range(num_cl_batches):
        start = b * batch_size
        end = n if b == num_cl_batches - 1 else (b + 1) * batch_size
        batches.append(client_indices[start:end])

    return batches


def build_dataloaders_for_client(
    train: Dataset,
    val: Dataset,
    client_indices: List[int],
    cfg: DataConfig,
) -> Tuple[List[DataLoader], DataLoader]:
    """
    Build a list of train loaders (one per CL batch) and a single val loader
    for one client.
    """
    cl_batches = make_cl_batches_for_client(
        client_indices, cfg.num_cl_batches
    )

    cl_train_loaders: List[DataLoader] = []
    for batch_indices in cl_batches:
        subset = Subset(train, batch_indices)
        loader = DataLoader(
            subset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=True,
        )
        cl_train_loaders.append(loader)

    val_loader = DataLoader(
        val,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    return cl_train_loaders, val_loader


def build_global_test_loader(
    test: Dataset, cfg: DataConfig
) -> DataLoader:
    """Global test loader for all clients / global model."""
    return DataLoader(
        test,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )