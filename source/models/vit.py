from typing import Dict, List, Literal, Optional
from functools import partial
import torch
import torch.nn as nn
import timm

from source.types import ViTBackbones


class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    def __init__(self, *args, **kwargs):
        super(VisionTransformer, self).__init__(*args, **kwargs)

    # Returns only the trainable backbone parameters
    # TODO: Modify BiSSL training script to account for parameters that does not require grad,
    # leaving this modification unnecessary.
    def parameters(self, recurse: bool = True):
        return iter(
            ([self.cls_token] if self.cls_token is not None else [])
            + list(self.blocks.parameters())
            + list(self.norm.parameters())
        )


def vit_base_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def vit_large_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def vit_huge_patch14(**kwargs):
    model = VisionTransformer(
        patch_size=14,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def get_vit_model(arch: ViTBackbones, **kwargs):
    match arch:
        case "vit_base_patch16":
            return vit_base_patch16(**kwargs)
        case "vit_large_patch16":
            return vit_large_patch16(**kwargs)
        case "vit_huge_patch14":
            return vit_huge_patch14(**kwargs)
        case _:
            raise ValueError(f"Architecture {arch} not recognized.")


### MISC UTILS IN USE WITH VITs ###
#### LR DECAY IMPL FROM MAE GIT ####

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# ELECTRA https://github.com/google-research/electra
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------


def param_groups_lrd(
    model: torch.nn.Module,
    weight_decay: float = 0.05,
    no_weight_decay_list: List[str] = [],
    layer_decay: float = 0.75,
) -> List[
    Dict[Literal["params", "lr_scale", "weight_decay"], torch.nn.Parameter | float]
]:
    """
    Parameter groups for layer-wise lr decay
    Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L58
    """
    param_group_names = {}
    param_groups = {}

    num_layers = len(model.blocks) + 1  # type: ignore

    layer_scales = list(layer_decay ** (num_layers - i) for i in range(num_layers + 1))

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue

        # no decay: all 1D parameters and model specific ones
        if p.ndim == 1 or n in no_weight_decay_list:
            g_decay = "no_decay"
            this_decay = 0.0
        else:
            g_decay = "decay"
            this_decay = weight_decay

        layer_id = get_layer_id_for_vit(n, num_layers)
        group_name = "layer_%d_%s" % (layer_id, g_decay)

        if group_name not in param_group_names:
            this_scale = layer_scales[layer_id]

            param_group_names[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }
            param_groups[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }

        param_group_names[group_name]["params"].append(n)
        param_groups[group_name]["params"].append(p)

    # print("parameter groups: \n%s" % json.dumps(param_group_names, indent=2))

    return list(param_groups.values())


def get_layer_id_for_vit(name: str, num_layers: int) -> int:
    """
    Assign a parameter with its layer id
    Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L33
    """
    if name in ["cls_token", "pos_embed"]:
        return 0
    elif name.startswith("patch_embed"):
        return 0
    elif name.startswith("blocks"):
        return int(name.split(".")[1]) + 1
    else:
        return num_layers


### PREV TIMM FKT NOW DECAPITACTED ###


def add_weight_decay(
    model: torch.nn.Module,
    weight_decay: float = 1e-5,
    skip_list: List[str] = [],
    lr: Optional[float] = None,
) -> List[Dict[str, List[torch.nn.Parameter] | float]]:
    decay = []
    no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)

    pgs = [
        {"params": no_decay, "weight_decay": 0.0},
        {"params": decay, "weight_decay": weight_decay},
    ]

    if lr is not None:
        for pg in pgs:
            pg["lr"] = lr
    return pgs
