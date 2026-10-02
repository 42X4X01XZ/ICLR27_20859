from typing import Literal, Optional

from runs.bisslight.classification.config import ArgsBiSSLClassification
from source.types import BinaryChoices, FloatBetween0and1, FloatNonNegative


class ArgsBiSSLightClassificationMAELora(ArgsBiSSLClassification):
    # fmt: off
    #### ADJUSTED DEFAULTS ####
    run_type: Literal["bisslight"] = "bisslight" # type: ignore[override]
    hinv_solver: Literal["mfac_lw"] = "mfac_lw"  # The hessian inverse solver to use for the implicit gradient computation. # type: ignore[override]
    
    d_wd: FloatNonNegative = 0.0
    
    p_lr: FloatNonNegative = 0.01
    p_head_lr: FloatNonNegative = 0.0001  # Learning rate for the pretext head. If None, defaults to p_lr.
    p_wd: FloatNonNegative = 0.0 # type: ignore[override]

    use_amp: bool = True  # Whether to use automatic mixed precision (AMP) during training. Using AMP can reduce memory usage and speed up training, especially on GPUs with Tensor Cores. However, it may introduce some numerical instability in certain cases, so it's important to monitor training when using AMP.
    use_grad_checkpointing: BinaryChoices = 0  # Whether to use gradient checkpointing during training. Using gradient checkpointing can reduce memory usage, but will increase training time. It is recommended to use gradient checkpointing when using large models or large batch sizes, especially when using large learning rates for the implicit gradient calculation.
 
    #### MAE/ViT Specific Hyperparameters ####
    d_vit_drop_path_rate: FloatBetween0and1 = 0.2  # ViT drop path rate on downstream model
    p_vit_drop_path_rate: FloatBetween0and1 = 0.0  # ViT drop path rate on the MAE pretraining model. Drop path is a regularization technique that randomly drops entire paths in the model during training, which can help with generalization. However, it can also cause instability during training, especially when using large learning rates. Therefore, it is often recommended to use a smaller drop path rate for the MAE pretraining model than for the downstream model, and to set it to 0.0 if using very large learning rates for pretraining.
    p_decoder_drop_path_rate: FloatBetween0and1 = 0.0  # Drop path rate for the MAE decoder. This is often set to 0.0, as the decoder is only used during pretraining and does not need as much regularization as the backbone.
    
    lora_rank: Optional[int] = None
    lora_dropout: FloatBetween0and1 = 0.0
    unfreeze_norm_layers: BinaryChoices = 0  # Whether to unfreeze the normalization layers when applying LoRA. Unfreezing the normalization layers can improve performance, but may cause instability during training, especially when using large learning rates. This is because the normalization layers are trained with a different objective than the rest of the model, and may have different optimal learning rates. If set to 1, the normalization layers will be unfrozen when applying LoRA. If set to 0, the normalization layers will remain frozen when applying LoRA.
    
    p_apply_lora_on_head: BinaryChoices = 0  # Whether to apply LoRA to the pretext head. Will be ignored (as in set to 0) if p_load_pretrained_pretext_head is False.
    
    mae_norm_pix_loss: BinaryChoices = 1  # Whether to use normalized pixel loss for the MAE pretraining. If set to 1, the pixel values will be normalized to [0, 1] before computing the loss. This can help with convergence when using large learning rates, but may lead to worse performance if the model is not trained for enough iterations.

    # M-FAC Solver Specific Args (Only used if hinv_solver is set to "mfac_lw")
    ij_grad_calc_device: Literal["cpu", "cuda"] = "cpu"  # Device to use for the implicit gradient calculation.
    ij_solver_lora_param_grouping: Literal["layer", "block", "fully_independent"] = "layer"  # The "layer" option groups all parameters belonging to the same attention block together, the "block" option groups all parameters corresponding to the same matrix it is added to together (i.e. it groups all parameters corresonding to the A and B matrices of the LoRA decomposition for either the q or v linear layer together, but not the parameters corresponding to the q and v linears together), and the "fully_independent" option groups each parameter module separately.

    # fmt: on
