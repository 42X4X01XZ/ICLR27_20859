# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------
# --------------------------------------------------------
# 2D sine-cosine position embedding
# References:
# Transformer: https://github.com/tensorflow/models/blob/master/official/nlp/transformer/model_utils.py
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------

from functools import partial
import numpy as np
import torch
import torch.nn as nn

from timm.layers.patch_embed import PatchEmbed
from timm.models.vision_transformer import Block

from source.types import ViTBackbones
from timm.layers.pos_embed import resample_abs_pos_embed
from timm.models._manipulate import checkpoint_seq


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


# --------------------------------------------------------
# Interpolate position embeddings for high-resolution
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------
def interpolate_pos_embed(model, checkpoint_model):
    # if "pos_embed" in checkpoint_model:
    pos_embed_checkpoint = checkpoint_model.pos_embed
    embedding_size = pos_embed_checkpoint.shape[-1]
    num_patches = model.patch_embed.num_patches
    num_extra_tokens = model.pos_embed.shape[-2] - num_patches
    # height (== width) for the checkpoint position embedding
    orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
    # height (== width) for the new position embedding
    new_size = int(num_patches**0.5)
    # class_token and dist_token are kept unchanged
    if orig_size != new_size:
        print(
            "Position interpolate from %dx%d to %dx%d"
            % (orig_size, orig_size, new_size, new_size)
        )
        extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
        # only the position tokens are interpolated
        pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
        pos_tokens = pos_tokens.reshape(
            -1, orig_size, orig_size, embedding_size
        ).permute(0, 3, 1, 2)
        pos_tokens = torch.nn.functional.interpolate(
            pos_tokens,
            size=(new_size, new_size),
            mode="bicubic",
            align_corners=False,
        )
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
        new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
        checkpoint_model.pos_embed = new_pos_embed


class MaskedAutoencoderViT(nn.Module):
    """Masked Autoencoder with VisionTransformer backbone"""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4.0,
        norm_layer: torch.nn.Module = nn.LayerNorm,  # type: ignore
        norm_pix_loss=False,
        # Added args:
        drop_path_rate: float = 0.0,
        decoder_drop_path_rate: float = 0.0,
        encoder_dynamic_img_size: bool = False,
    ) -> None:
        super().__init__()

        self.encoder_dynamic_img_size = encoder_dynamic_img_size
        self.grad_checkpointing = False
        self.g_chkpt_always = False

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(
            img_size,
            patch_size,
            in_chans,
            embed_dim,
            strict_img_size=not encoder_dynamic_img_size,
            output_fmt="NHWC" if encoder_dynamic_img_size else None,
        )  # (B, embed_dim, H/patch_size, W/patch_size) or (B, H/patch_size, W/patch_size, embed_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.patch_embed.num_patches + 1, embed_dim), requires_grad=False  # type: ignore
        )  # fixed sin-cos embedding

        # Copied from original ViT implementation:
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule

        self.blocks = nn.Sequential(
            *[
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    # qk_scale=None,
                    norm_layer=norm_layer,  # type: ignore
                    # Added args:
                    drop_path=dpr[i],
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, self.patch_embed.num_patches + 1, decoder_embed_dim),  # type: ignore
            requires_grad=False,
        )  # fixed sin-cos embedding

        # Copied from original ViT implementation:
        decoder_dpr = [
            x.item() for x in torch.linspace(0, decoder_drop_path_rate, decoder_depth)
        ]  # stochastic depth decay rule

        self.decoder_blocks = nn.Sequential(
            *[
                Block(
                    decoder_embed_dim,
                    decoder_num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    # qk_scale=None,
                    norm_layer=norm_layer,  # type: ignore
                    # Added args
                    drop_path=decoder_dpr[i],
                )
                for i in range(decoder_depth)
            ]
        )

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(
            decoder_embed_dim, patch_size**2 * in_chans, bias=True
        )  # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int(self.patch_embed.num_patches**0.5),  # type: ignore
            cls_token=True,
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1],
            int(self.patch_embed.num_patches**0.5),  # type: ignore
            cls_token=True,
        )
        self.decoder_pos_embed.data.copy_(
            torch.from_numpy(decoder_pos_embed).float().unsqueeze(0)
        )

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=0.02)
        torch.nn.init.normal_(self.mask_token, std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum("nchpwq->nhwpqc", x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))  # type: ignore
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))  # type: ignore
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(
            noise, dim=1
        )  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def set_grad_checkpointing(
        self, enable: bool = True, only_on_encoder_when_not_masking: bool = True
    ) -> None:
        """Enable or disable gradient checkpointing.

        Args:
            enable: Whether to enable gradient checkpointing.
            only_on_encoder_when_not_masking: Whether to only enable gradient checkpointing on the encoder when not masking.
        """
        self.grad_checkpointing = enable
        self.g_chkpt_always = enable and not only_on_encoder_when_not_masking

    def forward_encoder(self, x: torch.Tensor, mask_ratio: float):
        # embed patches
        x = self.patch_embed(x)

        if self.encoder_dynamic_img_size:
            B, H, W, C = x.shape
            prev_grid_size = self.patch_embed.grid_size

            if mask_ratio != 0.0:
                # Masking should only be compatible with the strict original image size,
                # since the decoder uses a fixed positional embedding.
                #
                # The code is however compatible with masking on dynamic images, but this is usually not intended behaviour,
                # as the decoder is not compatible with dynamic img sizes. Hence we make this assertation, but it could be
                # removed if one wants to use the backbone for masking input images for use cases beyond MAE.
                assert (H, W) == prev_grid_size, (
                    f"(MAE Encoder) Image size mismatch: masking requires the strict original "
                    f"grid {prev_grid_size}, but got ({H}, {W})."
                )

            pos_embed = resample_abs_pos_embed(  # type: ignore
                posemb=self.pos_embed,
                new_size=(H, W),
                old_size=prev_grid_size,
                num_prefix_tokens=1,
            )
            x = x.view(B, -1, C)
        else:
            pos_embed = self.pos_embed

        x = x + pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks

        if self.g_chkpt_always or (self.grad_checkpointing and mask_ratio == 0.0):
            x = checkpoint_seq(self.blocks, x)  # type: ignore
        else:
            x = self.blocks(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        if self.encoder_dynamic_img_size:
            # The encoder is allowed to have dynamic image sizes to be compatible with using it for other tasks requireing different img sizes.
            # However, when solving the MAE tasks which involves the decoder, it is designed only for compatability with the strict original size.
            # Technically we could alter the code to be compatible with such
            # For the decoder, we need the input to have the strict original image size, otherwise the position embedding addition will fail.
            expected_seq_len = self.decoder_pos_embed.shape[1] - 1  # Minus cls token
            assert (
                ids_restore.shape[1] == expected_seq_len
            ), f"(MAE Decoder) Image size mismatch: Original strictly expected {expected_seq_len} patches, but got {ids_restore.shape[1]} patches."

        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(
            x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1
        )
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(
            x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2])
        )  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        if self.g_chkpt_always:
            x = checkpoint_seq(self.decoder_blocks, x)  # type: ignore
        else:
            x = self.decoder_blocks(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, imgs, mask_ratio=0.75, return_loss_only=True):
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        loss = self.forward_loss(imgs, pred, mask)
        if return_loss_only:
            return loss
        else:
            return loss, pred, mask


# Adapted version of the original MAE model to be compatible with the BiSSL framework.
# The primary difference is that we include a partition of the model into a backbone and a pretext head.
# The backbone contains all trainable parameters of the encoder, while the pretext head contains all trainable parameters of the decoder.
# Hence there is no structural difference between the original MAE model and this BiSSL-compatible version, but the partitioning is required for the BiSSL framework.


class MaskedAutoencoderViTBiSSL(MaskedAutoencoderViT):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Makes modules containing all trainable backbone/encoder parameters
        backbone_patch_embed = self.patch_embed
        backbone_pos_embed = self.pos_embed
        backbone_cls_token = self.cls_token
        backbone_blocks = self.blocks
        backbone_norm = self.norm
        backbone_forward = self.forward_encoder

        class Encoder_Backbone(nn.Module):
            def __init__(self):
                super().__init__()
                self.patch_embed = backbone_patch_embed
                self.cls_token = backbone_cls_token
                self.pos_embed = backbone_pos_embed
                self.blocks = backbone_blocks
                self.norm = backbone_norm

            def set_grad_checkpointing(
                self, enable: bool = True, only_on_encoder_when_not_masking: bool = True
            ) -> None:
                backbone_forward.__self__.set_grad_checkpointing(
                    enable, only_on_encoder_when_not_masking
                )

            # Returns only trainable backbone parameters, useful for the BiSSL framework
            def parameters(self, recurse: bool = True):
                return iter(
                    [backbone_cls_token]
                    + [p for p in backbone_blocks.parameters()]
                    + [p for p in backbone_norm.parameters()]
                )

            # Assuming this is only called
            def forward(self, imgs, mask_ratio=0.75, return_mask=False):
                latent, mask, ids_restore = backbone_forward(
                    x=imgs, mask_ratio=mask_ratio
                )
                if return_mask:
                    return latent, mask, ids_restore
                return latent

        # Makes modules containing all trainable pretext head / decoder parameters
        decoder_embed = self.decoder_embed
        decoder_mask_token = self.mask_token
        decoder_pos_embed = self.decoder_pos_embed
        decoder_blocks = self.decoder_blocks
        decoder_norm = self.decoder_norm
        decoder_pred = self.decoder_pred
        decoder_forward = self.forward_decoder

        class Decoder_Head(nn.Module):
            def __init__(self):
                super().__init__()
                self.decoder_embed = decoder_embed
                self.mask_token = decoder_mask_token
                self.decoder_pos_embed = decoder_pos_embed
                self.decoder_blocks = decoder_blocks
                self.decoder_norm = decoder_norm
                self.decoder_pred = decoder_pred

            def forward(self, z, ids_restore):
                return decoder_forward(z, ids_restore)

        # The reason we ultimately make all these partitions is so that we can make a clear distinction between the backbone and the pretext head, which is required for the BiSSL framework.
        # TODO: Remake the BiSSL code to instead simply require a forward_backbone()/forward_head() method, which would make this partitioning unnecessary.
        self.backbone = Encoder_Backbone()
        self.head = Decoder_Head()

    def forward_backbone(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViTBiSSL(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def mae_vit_large_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViTBiSSL(
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def mae_vit_huge_patch14_dec512d8b(**kwargs):
    model = MaskedAutoencoderViTBiSSL(
        patch_size=14,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks


def get_mae_model(
    arch: ViTBackbones,
    decoder_embed_dim: int = 512,
    decoder_depth: int = 8,
    decoder_num_heads: int = 16,
    **kwargs,
):

    collected_kwargs = dict(
        decoder_embed_dim=decoder_embed_dim,
        decoder_depth=decoder_depth,
        decoder_num_heads=decoder_num_heads,
        **kwargs,
    )
    match arch:
        case "vit_base_patch16":
            return mae_vit_base_patch16(**collected_kwargs)
        case "vit_large_patch16":
            return mae_vit_large_patch16(**collected_kwargs)
        case "vit_huge_patch14":
            return mae_vit_huge_patch14(**collected_kwargs)
        case _:
            raise ValueError(f'Invalid MAE architecture "{arch}".')
