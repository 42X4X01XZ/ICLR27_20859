from typing import Optional, Literal

from source.types import DatasetsClassification

from runs.bisslight.config import ArgsBiSSLDefaults


class ArgsBiSSLClassification(ArgsBiSSLDefaults):
    # fmt: off
    #### ADJUSTED DEFAULTS ####
    downstream_task: Literal["classification"] = "classification"

    lr_scheduler_warmup_epochs: int = 10 # Linear warmup for the upper and lower level. I.e. yields lr_scheduler_warmup_epochs*upper_num_iter warmup steps for the upper-level and lr_scheduler_warmup_epochs*lower_num_iter warmup steps for the lower-level in total. Default is 10.

    d_img_size: Optional[int] = None  # Downstream image size (d_img_size x d_img_size). Defaults to the image size used for pretraining.
    d_img_crop_min_ratio: Optional[float] = None  # Downstream image crop min ratio. Defaults to the crop min ratio used for pretraining.
    d_dataset: DatasetsClassification  # Downstream dataset name
    d_batch_size: int = 64

    p_img_size: Optional[int] = None  # Lower level (pretext) image size. Defaults to the image size used for pretraining.
    p_img_crop_min_ratio: Optional[float] = None  # Lower level (pretext) image crop min ratio. Defaults to the crop min ratio used for pretraining.
    p_batch_size: int = 256 # type: ignore[override]

    # fmt: on
