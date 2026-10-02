from typing import Callable, Optional, Tuple
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from runs.fine_tune.config import ArgsFineTuningDefaults
from source.datasets import GetData


def get_dataloader_and_sampler(
    args: ArgsFineTuningDefaults,
    train_transform: torch.nn.Module,
    test_transform: torch.nn.Module,
    collate_fn: Optional[Callable] = None,
) -> Tuple[
    DataLoader,
    DataLoader,
    DistributedSampler,
    DistributedSampler,
    int,
]:
    get_data = GetData(
        root=args.dataset_root, download=bool(args.download)
    )

    data_train = get_data(
        dataset_name=args.dataset,
        transform=train_transform,
        split="train",
    )

    data_val = get_data(
        dataset_name=args.dataset,
        split="val",
        transform=test_transform,
    )

    assert args.seed is not None

    sampler_train = torch.utils.data.distributed.DistributedSampler(  # type: ignore
        data_train,
        shuffle=True,
        drop_last=False,
        num_replicas=args.world_size,
        rank=args.rank,
        seed=args.seed,
    )

    sampler_val = torch.utils.data.distributed.DistributedSampler(  # type: ignore
        data_val,
        shuffle=True,
        drop_last=False,
        num_replicas=args.world_size,
        rank=args.rank,
        seed=args.seed,
    )

    dataloader_train = DataLoader(
        data_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    dataloader_val = DataLoader(
        data_val,
        sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    return dataloader_train, dataloader_val, sampler_train, sampler_val, len(data_val.classes)  # type: ignore
