from typing import Optional, Callable

from torchvision import datasets

from source.types import Datasets, DatasetSplits
from source.dataset_attr.dataset_handler_base import DatasetHandler


class AircraftsHandler(DatasetHandler):
    def __init__(self, dataset_name: Datasets, root: str, download: bool = False):
        super().__init__(dataset_name, root, download)
        assert dataset_name == "aircrafts"

    def __call__(
        self,
        split: DatasetSplits,
        transform: Optional[Callable] = None,
        **kwargs,
    ) -> datasets.FGVCAircraft:
        assert split in ("train", "trainval", "test", "val")
        dataset = datasets.FGVCAircraft(
            root=self.root,
            split=split,
            download=self.download,
            transform=transform,
        )

        return dataset
