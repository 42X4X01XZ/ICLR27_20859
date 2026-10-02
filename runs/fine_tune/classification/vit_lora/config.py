from typing import Optional
from runs.pretext.mae.config import ArgsPretextMAEOfficial
from runs.fine_tune.classification.config import ArgsFTClassification
from source.types import BinaryChoices, FloatBetween0and1, OptimizerChoices


class ArgsFTClassificationViTLora(ArgsFTClassification):
    # fmt: off
    #### ADJUSTED DEFATULS ####
    # optimizer: OptimizerChoices = "adamw"  # Optimizer
    
    epochs: int = 200
    wd: int = 0.05  # Weight decay. Is overridden by HPO if use_hpo is set to 1. # type: ignore

    img_size: int = ArgsPretextMAEOfficial.img_size  # Image size (img_size x img_size). Defaults to the image size used for pretraining (224x224).
    img_crop_min_ratio: float = ArgsPretextMAEOfficial.img_crop_min_ratio  # Image crop min ratio. Defaults to the crop min ratio used for pretraining (0.2).
    
    use_amp: BinaryChoices = 1  # Whether to use automatic mixed precision during fine-tuning. Can reduce VRAM usage, but may cause instability when True.
    use_grad_checkpointing: BinaryChoices = 0  # Whether to use gradient checkpointing during fine-tuning. Can reduce VRAM usage, but may cause instability when True.
    
    lr_scheduler_warmup_epochs: int = 5 # Number of warmup epochs for the learning rate scheduler. Defaults to 1 epoch, but can be set to args.epochs // 20 as a rule of thumb.
    
    optimizer: OptimizerChoices = "adamw"  # Optimizer
     
    #### ViT SPECIFIC ARGUMENTS ####
    vit_drop_path_rate: FloatBetween0and1 = 0.2  # Drop path rate for ViT models.
    
    lora_rank: Optional[int] = None # Will be overridden by the BiSSL config, hence this is more of a dummy variable for easier readability afterwards.
    lora_dropout: float = 0.0
    unfreeze_norm_layers: BinaryChoices = 0  # Whether to unfreeze the normalization layers when applying LoRA. Unfreezing the normalization layers can improve performance, but may cause instability during training, especially when using large learning rates. 

    # fmt: on
