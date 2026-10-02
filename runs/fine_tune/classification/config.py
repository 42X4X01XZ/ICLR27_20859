from typing import Optional, Literal

from runs.fine_tune.config import ArgsFineTuningDefaults
from runs.pretext.config import ArgsPretextDefaults
from source.types import DatasetsClassification


class ArgsFTClassification(ArgsFineTuningDefaults):
    # fmt: off
    #### ADJUSTED DEFATULS ####
    downstream_task: Literal["classification"] = "classification"
    dataset: Optional[DatasetsClassification] = None  # Dataset. Needs to be specified for post-pretext when --use-hpo=0. # type: ignore
    batch_size: int = 256  # Batch size
    epochs: int = 200  # Number of training epochs

    #### Classification SPECIFIC ARGUMENTS ####


# fmt: on
