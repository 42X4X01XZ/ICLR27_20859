from typing import Callable, Optional, Tuple

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from runs.config_general import ArgsGeneralDefaults
from source.datasets import GetData
from source.types import Datasets, DatasetSplits


def get_dataloader_and_sampler(
    args: ArgsGeneralDefaults,
    dataset_name: Datasets,
    split: DatasetSplits,
    batch_size: int,
    transform: Optional[Callable] = None,
    transforms: Optional[Callable] = None,
    drop_last=False,
    collate_fn: Optional[Callable] = None,
    shuffle: bool = True,
    img_size: Optional[int] = None,
) -> Tuple[DataLoader, DistributedSampler]:
    get_data = GetData(root=args.dataset_root, download=bool(args.download))

    if transforms is not None:
        data = get_data(
            dataset_name=dataset_name,
            split=split,
            transform=transform,
            transforms=transforms,
            img_size=img_size,
        )
    else:
        data = get_data(
            dataset_name=dataset_name,
            split=split,
            transform=transform,
            img_size=img_size,
        )

    assert args.seed is not None

    sampler = DistributedSampler(
        data,
        shuffle=shuffle,
        drop_last=drop_last,
        seed=args.seed,
        num_replicas=args.world_size,
        rank=args.rank,
    )

    assert batch_size % args.world_size == 0
    per_device_batch_size = batch_size // args.world_size

    # Create data loaders.
    # For training
    dataloader = DataLoader(
        data,
        batch_size=per_device_batch_size,
        num_workers=args.num_workers,
        sampler=sampler,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    return dataloader, sampler
