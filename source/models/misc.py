from typing import Literal

import torch
from torch import nn

import torch
from peft import get_peft_model, LoraConfig

#### LORA ####


class SplitQKV(nn.Module):
    """Replaces a fused qkv Linear with separate q, k, v linears,
    preserving the original weights exactly."""

    def __init__(self, qkv: nn.Linear, dim: int):
        super().__init__()
        self.dim = dim
        has_bias = qkv.bias is not None
        self.q = nn.Linear(dim, dim, bias=has_bias)
        self.k = nn.Linear(dim, dim, bias=has_bias)
        self.v = nn.Linear(dim, dim, bias=has_bias)

        # Copy weights from fused layer
        with torch.no_grad():
            self.q.weight.copy_(qkv.weight[:dim])
            self.k.weight.copy_(qkv.weight[dim : 2 * dim])
            self.v.weight.copy_(qkv.weight[2 * dim :])
            if has_bias:
                self.q.bias.copy_(qkv.bias[:dim])
                self.k.bias.copy_(qkv.bias[dim : 2 * dim])
                self.v.bias.copy_(qkv.bias[2 * dim :])

    def forward(self, x):
        # Reconstruct the exact same output tensor shape as the original qkv layer
        return torch.cat([self.q(x), self.k(x), self.v(x)], dim=-1)


def split_qkv_in_model(model):
    for module in model.modules():
        if hasattr(module, "qkv") and isinstance(module.qkv, nn.Linear):
            dim = module.qkv.in_features
            module.qkv = SplitQKV(module.qkv, dim)


def lora_wrapper(
    model: torch.nn.Module,
    r: int = 8,
    dropout: float = 0.0,
    unfreeze_norm_layers: bool = False,
) -> torch.nn.Module:
    split_qkv_in_model(
        model
    )  # Modify the model in-place to replace fused qkv with separate q, k, v layers

    lora_config = LoraConfig(
        r=r,
        lora_alpha=r,  # standard HF default
        target_modules=[
            "qkv.q",
            "qkv.v",
        ],  # ONLY attention Q AND V LAYERS, NOT K (following https://arxiv.org/abs/2406.10973)
        lora_dropout=dropout,  # standard
        bias="none",
        init_lora_weights=True,  # Default initialization (zeros for B)
    )

    model = get_peft_model(model, lora_config)  # type: ignore

    if unfreeze_norm_layers:
        for name, param in model.named_parameters():
            if "norm" in name.lower():
                param.requires_grad = True

    return model


def get_lora_layerwise_param_group_idxs(
    model: torch.nn.Module,
    grouping: Literal["layer", "block", "fully_independent"] = "layer",
    unfreeze_norm_layers: bool = False,
) -> list[list[int]] | None:
    """This function assumes a very specific structure on the model naming, as induced by applying the huggingface
    lora implementation to a ViT model, where the lora is applied on the q and v layers. It will need to be adapted

    I.e. we assume that the last values of mod_name.split(".") are of the form "...attn.qkv.q.lora_A.default.weight", "...attn.qkv.q.lora_A.default.bias", "...attn.qkv.v.lora_B.default.weight", etc., and that the layer/block/module information is encoded in the preceding values of mod_name.split(".") (but these are not relevant here).

    Groupings:
    - The 'fully_independent' grouping groups each parameter separately.
    - The 'block' grouping groups all lora parameters belonging to the same matrix it is added to together (i.e. it groups all parameters corresonding to the A and B matrices of the LoRA decomposition for either the q or v linear layer together, but not the parameters corresponding to the q and v linears together).
        I.e. everything after "...attn.qkv.q" is grouped together (which usually is the lora_A and lora_B parameters for that linear layer), and everything after "...attn.qkv.v" is grouped together, but the parameters corresponding to the q and v linears are not grouped together.
    - The 'layer' grouping groups all parameters belonging to the same attention block together (i.e. all parameters with the same values for mod_name.split(".")[:-3]).
        I.e. everything after "...attn.qkv" is grouped together, so the parameters corresponding to the q and v linears of the same attention block are grouped together, but not with the parameters of other attention blocks.
    """

    from collections import defaultdict

    mod_names = [
        module_name
        for module_name, param in model.named_parameters()
        if param.requires_grad
    ]

    module_name_start_idx = 0
    for i, splt in enumerate(mod_names[0].split(".")):
        if splt == "model":
            module_name_start_idx = i + 1
            break

    # Group parameter indices by layer
    mod_parts_dict = defaultdict(list)

    match grouping:
        case "layer" | "block":
            pass
        case "fully_independent":
            ey = None  # [[i] for i in range(len(mod_names))]
        case _:
            raise ValueError(
                f"Invalid grouping option {grouping}. Has to be one of: 'layer', 'block', 'fully_independent'"
            )

    for i, mod_name in enumerate(mod_names):

        layer_key = tuple(mod_name.split(".")[module_name_start_idx:])

        match grouping:
            case "layer":
                if layer_key[0] == "blocks":
                    layer_key = layer_key[:2]
                else:
                    layer_key = layer_key[:1]
            case "block":
                if layer_key[0] == "blocks":
                    if layer_key[2] == "attn":
                        layer_key = layer_key[:5]
                    else:
                        layer_key = layer_key[:3]
                else:
                    layer_key = layer_key[:1]

        mod_parts_dict[layer_key].append(i)

    return list(mod_parts_dict.values())
