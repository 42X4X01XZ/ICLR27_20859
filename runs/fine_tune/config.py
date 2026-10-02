from typing import Optional, Literal

from runs.config_general import ArgsGeneralDefaults
from source.types import (
    BinaryChoices,
    DownstreamDatasets,
    DownstreamTaskTypes,
    FloatBetween0and1,
    FloatNonNegative,
    OptimizerChoices,
)


class ArgsFineTuningDefaults(ArgsGeneralDefaults):
    # fmt: off
    
    run_type: Literal["ft-post-pretext", "ft-post-bissl", "ft-post-bisslight"]  # Specify if the run is post-pretext or post-bissl (/bisslight ). Should be overridden by the specific run config.
    
    pretrained_backbone_path: str  # Path to the pretrained backbone model.

    # Dataset Args
    downstream_task: DownstreamTaskTypes  # Type of downstream task
    dataset: DownstreamDatasets  # Dataset

    # General Training Args
    lr: Optional[FloatNonNegative] = None  # Learning rate. Is overridden by HPO if use_hpo is set to 1.
    wd: Optional[FloatNonNegative] = None  # Weight decay. Is overridden by HPO if use_hpo is set to 1.
    
    batch_size: int  # Batch size
    epochs: int  # Number of training epochs
    optimizer: OptimizerChoices = "sgd"  # Optimizer
    momentum: FloatBetween0and1 = 0.9  # Momentum
    beta1: FloatBetween0and1 = 0.9  # Beta1 for Adam optimizers
    beta2: FloatBetween0and1 = 0.999  # Beta2 for Adam optimizers
    
    # Model args
    use_hpo: BinaryChoices = 1  # Conduct hyperparameter optimization grid search.
    
    num_runs: int = 8  # Number of repeated runs to conduct for test eval. This is only used when use_hpo is set to 0, since HPO already conducts a fixed number of runs, given by the number of hyperparameter combinations.

    # fmt: on
