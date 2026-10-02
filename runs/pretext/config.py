"""NB: Having classes for all hyperparameters like below is not
necessary, as these could be specified in the parsers.py script.
This setup deos however provide a much clearer overview over all
hyperparameters.
"""

from runs.config_general import ArgsGeneralDefaults
from source.types import (
    BackboneArchs,
    PretextTaskTypes,
    PretrainDatasets,
    OptimizerChoices,
    FloatNonNegative,
)


class ArgsPretextDefaults(ArgsGeneralDefaults):
    # fmt: off
    
    run_type: str # Name of the wandb project. Same name is also used for model storage.

    # This should be overwritten to only be forced to be only one possible value, determined by the pretext task.
    pretext_task: PretextTaskTypes  # Type of pretext task.

    img_size: int  # Size of the input images (img_size x img_size)
    img_crop_min_ratio: float  # Minimum crop ratio for the random crop

    backbone_arch: BackboneArchs  # Architecture of the backbone encoder network

    epochs: int  # Number of epochs
    dataset: PretrainDatasets  # Dataset
    batch_size: int  # Effective batch size (per worker batch size is [batch-size] / world-size)

    optimizer: OptimizerChoices  # Optimizer
    lr: FloatNonNegative  # Base learning  rate

    # fmt: on
