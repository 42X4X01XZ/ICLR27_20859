from typing import Optional, Callable

from source.dataset_attr.modified_datasets import Cub2011
from source.types import Datasets, DatasetSplits
from source.dataset_attr.dataset_handler_base import DatasetHandler


class CUB200Handler(DatasetHandler):
    def __init__(self, dataset_name: Datasets, root: str, download: bool = False):
        super().__init__(dataset_name, root, download)
        assert dataset_name == "cub200"

    def __call__(
        self,
        split: DatasetSplits,
        transform: Optional[Callable] = None,
        **kwargs,
    ) -> Cub2011:
        assert split in ("train", "test", "val", "trainval")
        dataset = Cub2011(
            root=self.root,
            train=False if split == "test" else True,
            download=self.download,
            transform=transform,
        )
        if split in ("train", "val"):
            dataset.images, dataset.targets = self._get_partitioned_data(  # type: ignore
                dataset_name=self.dataset_name,
                split=split,
                data=list(zip(dataset.images, dataset.targets)),
                return_separate_data_labels=True,
            )

        return dataset
