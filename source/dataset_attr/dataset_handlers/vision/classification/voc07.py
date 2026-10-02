from typing import Optional, Callable

from source.dataset_attr.modified_datasets import VOC2007Classification
from source.types import Datasets, DatasetSplits
from source.dataset_attr.dataset_handler_base import DatasetHandler


class VOC07Handler(DatasetHandler):
    def __init__(self, dataset_name: Datasets, root: str, download: bool = False):
        super().__init__(dataset_name, root, download)
        assert dataset_name == "voc07"

    def __call__(
        self,
        split: DatasetSplits,
        transform: Optional[Callable] = None,
        **kwargs,
    ) -> VOC2007Classification:
        assert split in ("train", "test", "val", "trainval")
        dataset = VOC2007Classification(
            root=self.root,
            image_set=split if split != "trainval" else "train",
            download=self.download,
            transform=transform,
        )

        if split == "trainval":
            dataset_val = VOC2007Classification(
                root=self.root,
                image_set="val",
                download=self.download,
                transform=transform,
            )
            dataset.images += dataset_val.images
            dataset.targets += dataset_val.targets

        return dataset
