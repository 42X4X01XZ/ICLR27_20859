from typing import Optional, get_args

import timm
import torch
from torch import nn

from source.types import BackboneArchs, ViTBackbones

from source.models.vit import get_vit_model


# Linear Classifier for Downstream Tasks
class DSClassifier(nn.Module):
    def __init__(
        self,
        backbone_arch: BackboneArchs,
        n_classes: int = 10,
        img_size: Optional[int] = None,
        vit_drop_path_rate: float = 0.0,
        vit_class_token: bool = True,
    ):
        super().__init__()
        if backbone_arch in get_args(ViTBackbones):
            backbone = get_vit_model(
                arch=backbone_arch,  # type: ignore
                img_size=img_size,
                num_classes=n_classes,
                drop_path_rate=vit_drop_path_rate,
                class_token=vit_class_token,
                global_pool="token" if vit_class_token else "",
                # global_pool="avg",
            )

            if not vit_class_token:
                # We change this afterwards, as the timm vit removes the norm layer preceeding the pooling if initiated with global_pool="avg".
                # We want to keep this norm layer, as it is commonly used in various frameworks, e.g. I-JEPA. This is a bit hacky, but it works for now.
                backbone.global_pool = "avg"

            self.num_features = backbone.embed_dim

            # In the code below, we separate the classifier head into its own module.

            # First, we overwrite the forward pass of the backbone to match the backbone forward pass
            backbone.forward = backbone.forward_features

            # Then we remove all variables/methods related to the head from the backbone
            # fc_norm = deepcopy(backbone.fc_norm)  # type: ignore
            del backbone.attn_pool
            del backbone.fc_norm
            del backbone.head_drop
            del backbone.head

            # Finally, we define the head as a separate module, with identical settings to the original head
            # TODO: All this could instead be included in the models_vit.py file for better readability
            class ViTDownstreamHead(nn.Module):
                def __init__(self, drop_rate=0, weight_init=""):
                    super().__init__()
                    # use_fc_norm = (
                    #     backbone.global_pool == "avg" if fc_norm is None else fc_norm
                    # )
                    if backbone.global_pool == "map":
                        self.attn_pool = timm.layers.attention_pool.AttentionPoolLatent(
                            backbone.embed_dim,
                            num_heads=backbone.num_heads,  # type: ignore
                            mlp_ratio=backbone.mlp_ratio,  # type: ignore
                            norm_layer=backbone.norm_layer,  # type: ignore
                        )
                    else:
                        self.attn_pool = None
                    self.fc_norm = (  # Here, we add another norm in case we are using average pooling, following what iJEPA does.
                        nn.Identity()
                        if vit_class_token
                        else nn.BatchNorm1d(backbone.embed_dim, eps=1e-6)
                    )
                    self.head_drop = nn.Dropout(drop_rate)
                    self.head = (
                        nn.Linear(backbone.embed_dim, backbone.num_classes)
                        if backbone.num_classes > 0
                        else nn.Identity()
                    )
                    if weight_init != "skip":
                        backbone.init_weights(weight_init)  # type: ignore

                def forward(self, x, pre_logits=False):
                    if backbone.global_pool == "map" and self.attn_pool is not None:
                        x = self.attn_pool(x)
                    elif backbone.global_pool == "avg":
                        x = x[:, backbone.num_prefix_tokens :].mean(dim=1)
                    elif backbone.global_pool == "token":
                        x = x[:, 0]  # class token
                    else:
                        raise ValueError(
                            f"Unknown global_pool type: {backbone.global_pool}. Has to be one of: 'avg', 'token', 'map', ''"
                        )
                    x = self.fc_norm(x)
                    x = self.head_drop(x)
                    return x if pre_logits else self.head(x)

            self.backbone = backbone
            self.head = ViTDownstreamHead()
        else:
            raise ValueError(f"Backbone {backbone_arch} not supported")

    def forward_backbone(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))
