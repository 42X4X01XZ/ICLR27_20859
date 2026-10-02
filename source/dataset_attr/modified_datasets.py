import os
from typing import Any, Callable, Optional, Tuple
import pandas as pd
import torch
from PIL import Image
from torchvision.datasets.folder import default_loader
from torchvision import datasets
from torchvision.datasets.utils import download_url
from torch.utils.data import Dataset
from source.dataset_attr.classes import voc_classes

try:
    from defusedxml.ElementTree import parse as ET_parse
except ImportError:
    from xml.etree.ElementTree import parse as ET_parse


class VOC2007Classification(datasets.VOCDetection):
    """This is a modified version of the VOCDetection dataset which only utilises the 2007 partition for image classification.
       This modified version only returns the class integers as target (1: present, 0: abscent).

    Args:
        datasets (_type_): _description_
    """

    def __init__(
        self,
        root: str,
        image_set: str = "train",
        download: bool = False,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
    ):
        super().__init__(
            root=root,
            image_set=image_set,
            download=download,
            year="2007",
            transforms=transforms,
            transform=transform,
            target_transform=target_transform,
        )

        self.classes = voc_classes
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is a dictionary of the XML tree.
        """
        img = Image.open(self.images[index]).convert("RGB")
        target_full = self.parse_voc_xml(ET_parse(self.annotations[index]).getroot())  # type: ignore
        target = torch.zeros(len(self.classes))
        for obj in target_full["annotation"]["object"]:
            target[self.class_to_idx[obj["name"]]] = 1

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target


class Cub2011(Dataset):
    """Inspired by this implementation: https://github.com/TDeVries/cub2011_dataset

    Args:
        datasets (_type_): _description_

    Raises:
        RuntimeError: _description_

    Returns:
        _type_: _description_
    """

    base_folder = "CUB_200_2011/images"
    url = "http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz"
    filename = "CUB_200_2011.tgz"
    tgz_md5 = "97eceeb196236b17998738112f37df78"

    def __init__(
        self, root, train=True, transform=None, loader=default_loader, download=True
    ):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.loader = default_loader
        self.train = train

        if download:
            self._download()

        if not self._check_integrity():
            raise RuntimeError(
                "Dataset not found or corrupted."
                + " You can use download=True to download it"
            )

        self.images = [
            os.path.join(self.root, self.base_folder, dp.filepath)
            for _, dp in self.data.iterrows()
        ]
        self.targets = [
            dp.target - 1 for _, dp in self.data.iterrows()
        ]  # Targets start at 1 by default, so shift to 0

        class_to_idx = {}

        for img in self.images:
            cl_idx = int(img.split("/")[-2].split(".")[0]) - 1
            cl = img.split("/")[-2].split(".")[1]
            if cl not in class_to_idx.keys():
                class_to_idx[cl] = cl_idx

        self.class_to_idx = class_to_idx
        self.classes = list(class_to_idx.keys())

    def _load_metadata(self):
        images = pd.read_csv(
            os.path.join(self.root, "CUB_200_2011", "images.txt"),
            sep=" ",
            names=["img_id", "filepath"],
        )
        image_class_labels = pd.read_csv(
            os.path.join(self.root, "CUB_200_2011", "image_class_labels.txt"),
            sep=" ",
            names=["img_id", "target"],
        )
        train_test_split = pd.read_csv(
            os.path.join(self.root, "CUB_200_2011", "train_test_split.txt"),
            sep=" ",
            names=["img_id", "is_training_img"],
        )

        data = images.merge(image_class_labels, on="img_id")
        self.data = data.merge(train_test_split, on="img_id")

        if self.train:
            self.data = self.data[self.data.is_training_img == 1]
        else:
            self.data = self.data[self.data.is_training_img == 0]

    def _check_integrity(self):
        try:
            self._load_metadata()
        except Exception:
            return False

        for index, row in self.data.iterrows():
            filepath = os.path.join(self.root, self.base_folder, row.filepath)
            if not os.path.isfile(filepath):
                print(filepath)
                return False
        return True

    def _download(self):
        import tarfile

        if self._check_integrity():
            print("Files already downloaded and verified")
            return

        download_url(self.url, self.root, self.filename, self.tgz_md5)

        with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
            tar.extractall(path=self.root)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.loader(self.images[idx])
        target = self.targets[idx]

        if self.transform is not None:
            img = self.transform(img)

        return img, target
