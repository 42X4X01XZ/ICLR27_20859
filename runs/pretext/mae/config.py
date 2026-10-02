from typing import Literal
from runs.pretext.config import ArgsPretextDefaults
from source.types import (
    OptimizerChoices,
    FloatNonNegative,
    FloatBetween0and1,
    PretrainDatasets,
)


class ArgsPretextMAEOfficial(ArgsPretextDefaults):
    # fmt: off
    #### ADJUSTED DEFAULTS ####
    run_type: Literal["mae"] = "mae"  # Name of the wandb project. Same name is also used for model storage. # type: ignore[override]
    pretext_task: Literal["mae"] = "mae"
    
    img_size: int = 224  # Input image size for the MAE pretraining. The original MAE paper uses 224, but it can be adjusted if needed.
    img_crop_min_ratio: FloatBetween0and1 = 0.2  # Minimum

    backbone_arch: Literal["vit_base_patch16", "vit_large_patch16", "vit_huge_patch14"]  # Architecture of the backbone encoder network

    dataset: PretrainDatasets = "imagenet"  # Dataset
    
    optimizer: OptimizerChoices = "adamw"  # Optimizer
    batch_size:int = 4096
    lr: FloatNonNegative = 1.5e-4  # Base learning  rate, effective learning after warmup is [base-lr] * [batch-size] / 256
    wd: FloatNonNegative = 0.05  # Weight decay
    beta1: FloatBetween0and1 = 0.9  # Beta1 for Adam
    beta2: FloatBetween0and1 = 0.95  # Beta2 for Adam
    
    epochs: int = 1600  # Number of epochs to train for. The original MAE paper trains for 1600 epochs, but this can be adjusted based on the dataset size and compute resources.

    ### MAE specfic arguments ###
    decoder_depth: int = 8  # Number of layers in the MAE decoder. The original MAE paper uses a decoder with 8 layers, but this can be adjusted based on the dataset size and compute resources.
    decoder_embed_dim: int = 512  # Hidden dimension of the MAE decoder. The original
    decoder_num_heads: int = 16  # Number of attention heads in the MAE decoder. The original MAE paper uses 16 heads, but this can be adjusted based on the dataset size and compute resources.

    # fmt: on
