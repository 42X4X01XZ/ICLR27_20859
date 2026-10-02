from typing import Optional, Callable
import urllib

from torchvision import datasets

from source.types import Datasets, DatasetSplits
from source.dataset_attr.dataset_handler_base import DatasetHandler


class ImageNetHandler(DatasetHandler):
    def __init__(self, dataset_name: Datasets, root: str, download: bool = False):
        super().__init__(dataset_name, root, download)
        assert dataset_name in ["imagenet", "imagenet-1p", "imagenet-10p"]

    def __call__(
        self,
        split: DatasetSplits,
        transform: Optional[Callable] = None,
        **kwargs,
    ) -> datasets.ImageNet:
        assert split in ("train", "val")

        dataset = datasets.ImageNet(
            root=self.root + "/imagenet",
            # root="/tmp/imagenet",
            split=split,
            transform=transform,
        )

        if self.dataset_name != "imagenet" and split == "train":
            match self.dataset_name:
                case "imagenet-1p":
                    end_text = "1percent.txt"
                case "imagenet-10p":
                    end_text = "10percent.txt"
                case _:
                    raise ValueError(f"Unknown dataset name: {self.dataset_name}")

            train_file = urllib.request.urlopen(  # type: ignore
                f"https://raw.githubusercontent.com/google-research/simclr/master/imagenet_subsets/{end_text}"
            ).readlines()
            dataset.samples = []
            for fname in train_file:
                fname = fname.decode().strip()
                cls = fname.split("_")[0]
                dataset.samples.append(
                    (
                        self.root + "/imagenet/train/" + cls + "/" + fname,
                        dataset.wnid_to_idx[cls],
                    )
                )
        return dataset
