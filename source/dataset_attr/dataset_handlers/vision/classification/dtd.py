from typing import Optional, Callable

from torchvision import datasets

from source.types import Datasets, DatasetSplits
from source.dataset_attr.dataset_handler_base import DatasetHandler


class DTDHandler(DatasetHandler):
    def __init__(self, dataset_name: Datasets, root: str, download: bool = False):
        super().__init__(dataset_name, root, download)
        assert dataset_name == "dtd"

    def __call__(
        self,
        split: DatasetSplits,
        transform: Optional[Callable] = None,
        **kwargs,
    ) -> datasets.DTD:
        assert split in ("train", "test", "val", "trainval")
        dataset = datasets.DTD(
            root=self.root,
            split=split if split != "trainval" else "train",
            download=self.download,
            transform=transform,
            partition=1,
        )

        if split == "trainval":
            dataset_val = datasets.DTD(
                root=self.root,
                split="val",
                download=self.download,
                transform=transform,
                partition=1,
            )
            assert (
                dataset.classes == dataset_val.classes
                and dataset.class_to_idx == dataset_val.class_to_idx
            ), "Classes in train and val sets do not match, cannot combine into trainval set."

            dataset._image_files += dataset_val._image_files
            dataset._labels += dataset_val._labels
            dataset._split = "trainval"

        return dataset
