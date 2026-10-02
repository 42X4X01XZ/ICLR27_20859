from typing import Literal, Any, Optional, Callable, List, get_args
from abc import abstractmethod

import numpy as np
import torch

from source.types import Datasets, DatasetSplits


class DatasetHandler:
    def __init__(
        self,
        dataset_name: Datasets,
        root: str,
        download: bool = False,
    ):

        assert dataset_name in get_args(Datasets)

        self.dataset_name: Datasets = dataset_name
        self.root = root
        self.download = download

        self.dataset_origin = "default"

    def _get_partitioned_data(
        self,
        dataset_name: Datasets,
        split: Literal["train", "val", "test"],
        data: List[Any] | np.ndarray[Any, np.dtype[Any]],
        return_separate_data_labels: bool = False,
    ) -> (
        tuple[np.ndarray[Any, np.dtype[Any]], np.ndarray[Any, np.dtype[Any]]]
        | tuple[tuple[Any, ...], tuple[Any, ...]]
        | List[np.ndarray[Any, np.dtype[Any]] | Any]
    ):

        return_np_arrays = True if isinstance(data, np.ndarray) else False

        dataset_idxs: np.ndarray[Any, np.dtype[Any]] = np.loadtxt(
            fname=self.root + "/tv_splits/" + dataset_name + "_" + split + "_split.txt",
            dtype=int,
        )

        samples_subset = [data[i] for i in dataset_idxs]

        if return_separate_data_labels:
            samps, labels = zip(*samples_subset)
            if return_np_arrays:
                return np.array(samps), np.array(labels)
            return samps, labels

        return samples_subset

    def _sample_subset_per_class(
        self,
        data: List[Any] | np.ndarray[Any, np.dtype[Any]],
        labels: List[Any] | np.ndarray[Any, np.dtype[Any]],
        n_classes: int,
        n_samples_per_class: int,
    ):
        # EXPERIMENTAL
        # Given number of samples per class, sample n_samples_per_class samples per
        # class from the data

        # Assumes the data comes in a fixed order everytime the function is called.

        return_np_arrays = True if isinstance(data, np.ndarray) else False

        samples_subset = []
        labels_subset = []
        for i in range(n_classes):
            samples_append = []
            labels_append = []
            for j, dat in enumerate(data):
                if labels[j] == i:
                    samples_append.append(dat)
                    labels_append.append(labels[j])
                if len(samples_append) == n_samples_per_class:
                    samples_subset.extend(samples_append)
                    labels_subset.extend(labels_append)
                    break

        if return_np_arrays:
            return np.array(samples_subset), np.array(labels_subset)
        return samples_subset, labels_subset

    def _get_mean_std(self):
        mean, sq_mean = torch.zeros(3), torch.zeros(3)

    @abstractmethod
    def __call__(
        self,
        split: DatasetSplits,
        transform: Optional[Callable] = None,
        **kwargs,
    ) -> Any:
        pass
