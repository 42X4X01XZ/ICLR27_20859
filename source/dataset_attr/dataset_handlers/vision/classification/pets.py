from typing import Optional, Callable

from torchvision import datasets

from source.types import Datasets, DatasetSplits
from source.dataset_attr.dataset_handler_base import DatasetHandler


class PetsHandler(DatasetHandler):
    def __init__(self, dataset_name: Datasets, root: str, download: bool = False):
        super().__init__(dataset_name, root, download)
        assert dataset_name in [
            "pets",
            "pets5sampclass",
            "pets10sampclass",
            "pets25sampclass",
            "pets50sampclass",
            "pets75sampclass",
        ]

    def __call__(
        self,
        split: DatasetSplits,
        transform: Optional[Callable] = None,
        **kwargs,
    ) -> datasets.OxfordIIITPet:
        assert split in ("train", "test", "val", "trainval")
        split_get = "trainval" if split in ("train", "val", "trainval") else "test"
        dataset = datasets.OxfordIIITPet(
            root=self.root,
            split=split_get,
            download=self.download,
            transform=transform,
        )

        if split in ("train", "val"):
            dataset._images, dataset._labels = self._get_partitioned_data(  # type: ignore
                dataset_name="pets",
                split=split,
                data=list(zip(dataset._images, dataset._labels)),
                return_separate_data_labels=True,
            )
        if self.dataset_name != "pets" and split == "train":
            n_samps_per_class = int(self.dataset_name[4:-9])

            dataset._images, dataset._labels = self._sample_subset_per_class(  # type: ignore
                data=dataset._images,
                labels=dataset._labels,
                n_classes=len(dataset.classes),
                n_samples_per_class=n_samps_per_class,
            )

        return dataset
