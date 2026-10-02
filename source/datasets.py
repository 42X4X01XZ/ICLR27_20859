from typing import Dict, Optional, Callable, Type

from torch.utils.data import Dataset

from source.dataset_attr.dataset_handler_base import DatasetHandler

from source.dataset_attr.dataset_handlers.vision.classification.aircrafts import (
    AircraftsHandler,
)
from source.dataset_attr.dataset_handlers.vision.classification.cub200 import (
    CUB200Handler,
)
from source.dataset_attr.dataset_handlers.vision.classification.dtd import (
    DTDHandler,
)
from source.dataset_attr.dataset_handlers.vision.classification.flowers import (
    FlowersHandler,
)
from source.dataset_attr.dataset_handlers.vision.classification.imagenet import (
    ImageNetHandler,
)
from source.dataset_attr.dataset_handlers.vision.classification.pets import PetsHandler
from source.dataset_attr.dataset_handlers.vision.classification.voc07 import (
    VOC07Handler,
)

from source.types import Datasets, DatasetSplits

DATASET_HANDLERS: Dict[Datasets, Type[DatasetHandler]] = {
    "imagenet": ImageNetHandler,
    "cub200": CUB200Handler,
    "aircrafts": AircraftsHandler,
    "voc07": VOC07Handler,
    "dtd": DTDHandler,
    "pets": PetsHandler,
    "flowers": FlowersHandler,
    #
    # Vision - Classification Subsets
    #
    "imagenet-1p": ImageNetHandler,
    "imagenet-10p": ImageNetHandler,
}


class GetData:
    def __init__(self, root: str, download: bool = False):
        self.root = root
        self.download = download

    # TODO: Could rewrite init s.t. we include the dataset name in init,
    # and hence fetches the class in the init statement already.
    # Would make it more readable imo.
    # ALTERNATIVELY: Could just make the get_data a function instead of a class
    def __call__(
        self,
        dataset_name: Datasets,
        split: DatasetSplits = "train",
        transform: Optional[Callable] = None,
        **kwargs,
    ) -> Dataset:
        if dataset_name not in DATASET_HANDLERS:
            raise ValueError(f"Dataset {dataset_name} not supported")

        handler_class = DATASET_HANDLERS[dataset_name]
        handler = handler_class(
            dataset_name=dataset_name, root=self.root, download=self.download
        )
        return handler(split, transform=transform, **kwargs)
