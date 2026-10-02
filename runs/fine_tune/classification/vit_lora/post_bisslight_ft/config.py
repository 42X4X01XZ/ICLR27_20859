from typing import Literal, Optional
from runs.fine_tune.classification.vit_lora.config import ArgsFTClassificationViTLora
from source.types import (
    BinaryChoices,
    DatasetsClassification,
    FloatBetween0and1,
    OptimizerChoices,
)


class ArgsFTClassificationViTLoraPostBiSSLight(ArgsFTClassificationViTLora):
    # fmt: off
    #### ADJUSTED DEFATULS ####
    run_type: Literal["ft-post-bisslight"] = "ft-post-bisslight" # type: ignore

    bissl_pretrained_model_path: str  # Path to the BiSSLight/BiSSL-trained model weights that will be used for fine-tuning.
    bissl_config_path: Optional[str] = None  # Path to the BiSSLight/BiSSL config file that was used for training the BiSSLight/BiSSL model. This is used to ensure that the fine-tuning script uses the same configuration as the BiSSLight/BiSSL training script. If not specified, default values will be used, which may lead to incompatibility issues if critical values from the BiSSLight/BiSSL config differ from those specified here (e.g., LoRA rank, image size, dataset name, etc.). Its highly recommended to specify the BiSSLight/BiSSL config path to ensure compatibility.

    img_size: Optional[int] = None  # Image size (img_size x img_size). Defaults to the image size used for pretraining. # type: ignore
    img_crop_min_ratio: Optional[float] = None  # Image crop min ratio. Defaults to the crop min ratio used for pretraining. # type: ignore

    dataset: Optional[DatasetsClassification] = None  # (Should not be altered) Datasets. This is meerely a placeholder to be overriden by BiSSL config.
    optimizer: Optional[OptimizerChoices] = None  # Optimizer # type: ignore
    momentum: Optional[FloatBetween0and1] = None # Momentum # type: ignore
    beta1: Optional[FloatBetween0and1] = None  # Beta1 for Adam optimizers # type: ignore
    beta2: Optional[FloatBetween0and1] = None  # Beta2 for Adam optimizers # type: ignore
    
    lora_rank: Optional[int] = None # Defaults to the LoRA Rank used for BiSSL.
    unfreeze_norm_layers: Optional[BinaryChoices] = None  # Whether to unfreeze the normalization layers when applying LoRA. Unfreezing the normalization layers can improve performance, but may cause instability during training, especially when using large learning rates. This is because the normalization layers are trained with a different objective than the rest of the model, and may have different optimal learning rates. If set to 1, the normalization layers will be unfrozen when applying LoRA. If set to 0, the normalization layers will remain frozen when applying LoRA. # type: ignore

    # fmt: on
