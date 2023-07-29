# --------------------------------------------------------
# Images Speak in Images: A Generalist Painter for In-Context Visual Learning (https://arxiv.org/abs/2212.02499)
# Github source: https://github.com/baaivision/Painter
# Copyright (c) 2022 Beijing Academy of Artificial Intelligence (BAAI)
# Licensed under The MIT License [see LICENSE for details]
# By Xinlong Wang, Wen Wang
# Based on MAE, BEiT, detectron2, Mask2Former, bts, mmcv, mmdetetection, mmpose, MIRNet, MPRNet, and Uformer codebases
# --------------------------------------------------------'

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

##########################
import fvcore.nn.weight_init as weight_init
from detectron2.layers import CNNBlockBase, Conv2d, get_norm
from fairscale.nn.checkpoint import checkpoint_wrapper
from timm.models.layers import DropPath, trunc_normal_
from timm.models.vision_transformer import Mlp

from util.vitdet_utils import (
    PatchEmbed,
    add_decomposed_rel_pos,
    get_abs_pos,
    window_partition,
    window_unpartition,
    LayerNorm2D,
)


class Attention(nn.Module):
    """Multi-head Attention block with relative position embeddings."""

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=True,
        use_rel_pos=False,
        rel_pos_zero_init=True,
        input_size=None,
    ):
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads.
            qkv_bias (bool:  If True, add a learnable bias to query, key, value.
            rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            input_size (int or None): Input resolution for calculating the relative positional
                parameter size.
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            # initialize relative positional embeddings
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))
            if not rel_pos_zero_init:
                trunc_normal_(self.rel_pos_h, std=0.02)
                trunc_normal_(self.rel_pos_w, std=0.02)

    def forward(self, x, parallel_batch=True):
        B, H, W, _ = x.shape
        # qkv with shape (3, nHead, B, H * W, C)
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 3, 0, 1, 4)
        # q, k, v with shape (nHead * B, H * W, C)
        q, k, v = qkv.reshape(3, self.num_heads * B, H * W, -1).unbind(0)
        attn = (q * self.scale) @ k.transpose(-2, -1)
        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(attn, q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W))
        attn = attn.softmax(dim=-1)
        # print(attn.shape)
        x = (attn @ v).view(self.num_heads, B, H, W, -1).permute(1, 2, 3, 0, 4).reshape(B, H, W, -1)
        x = self.proj(x)
        return x


class ResBottleneckBlock(CNNBlockBase):
    """
    The standard bottleneck residual block without the last activation layer.
    It contains 3 conv layers with kernels 1x1, 3x3, 1x1.
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        bottleneck_channels,
        norm="LN",
        act_layer=nn.GELU,
    ):
        """
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            bottleneck_channels (int): number of output channels for the 3x3
                "bottleneck" conv layers.
            norm (str or callable): normalization for all conv layers.
                See :func:`layers.get_norm` for supported format.
            act_layer (callable): activation for all conv layers.
        """
        super().__init__(in_channels, out_channels, 1)

        self.conv1 = Conv2d(in_channels, bottleneck_channels, 1, bias=False)
        self.norm1 = get_norm(norm, bottleneck_channels)
        self.act1 = act_layer()

        self.conv2 = Conv2d(
            bottleneck_channels,
            bottleneck_channels,
            3,
            padding=1,
            bias=False,
        )
        self.norm2 = get_norm(norm, bottleneck_channels)
        self.act2 = act_layer()

        self.conv3 = Conv2d(bottleneck_channels, out_channels, 1, bias=False)
        self.norm3 = get_norm(norm, out_channels)

        for layer in [self.conv1, self.conv2, self.conv3]:
            weight_init.c2_msra_fill(layer)
        for layer in [self.norm1, self.norm2]:
            layer.weight.data.fill_(1.0)
            layer.bias.data.zero_()
        # zero init last norm layer.
        self.norm3.weight.data.zero_()
        self.norm3.bias.data.zero_()

    def forward(self, x):
        out = x
        for layer in self.children():
            out = layer(out)

        out = x + out
        return out


class Block(nn.Module):
    """Transformer blocks with support of window attention and residual propagation blocks"""
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        use_rel_pos=False,
        rel_pos_zero_init=True,
        window_size=0,
        use_residual_block=False,
        input_size=None,
    ):
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            drop_path (float): Stochastic depth rate.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks. If it equals 0, then not
                use window attention.
            use_residual_block (bool): If True, use a residual block after the MLP block.
            input_size (int or None): Input resolution for calculating the relative positional
                parameter size.
        """
        super().__init__()
        self.input_size = input_size
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            input_size=input_size if window_size == 0 else (window_size, window_size),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer)
        self.window_size = window_size
        self.use_residual_block = use_residual_block
        if use_residual_block:
            # Use a residual block with bottleneck channel as dim // 2
            self.residual = ResBottleneckBlock(
                in_channels=dim,
                out_channels=dim,
                bottleneck_channels=dim // 2,
                norm="LN",
                act_layer=act_layer,
            )

    def forward(self, x):
        assert x.shape[1:3] == self.input_size
        shortcut = x
        x = self.norm1(x)
        # Window partition
        if self.window_size > 0:
            x, pad_hw = window_partition(x, self.window_size)
        x = self.attn(x)
        # Reverse window partition
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, self.input_size)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        if self.use_residual_block:
            x = self.residual(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        return x


class Painter_Varient(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(
            self,
            img_size=(448, 448),
            patch_size=16,
            in_chans=3,
            embed_dim=1024,
            depth=24,
            num_heads=16,
            merge_idx=2,
            num_prompts=1,
            cr_depth=12,
            xcr_depth=2,
            mlp_ratio=4.,
            qkv_bias=True,
            drop_path_rate=0.,
            norm_layer=nn.LayerNorm,
            act_layer=nn.GELU,
            use_abs_pos=True,
            use_rel_pos=False,
            rel_pos_zero_init=True,
            window_size=0,
            window_block_indexes=(),
            residual_block_indexes=(),
            use_act_checkpoint=False,
            pretrain_img_size=224,
            pretrain_use_cls_token=True,
            out_feature="last_feat",
            decoder_embed_dim=128,
            loss_func="smoothl1",
        ):
        super().__init__()

        # --------------------------------------------------------------------------
        self.img_size = img_size
        self.pretrain_use_cls_token = pretrain_use_cls_token
        self.patch_size = patch_size
        self.depth = depth
        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        self.patch_embed.num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)
        self.loss_func = loss_func
        self.merge_idx = merge_idx
        self.encoder_sampling = set([5, 11, 17, 23])
        self.num_prompts = num_prompts
        self.cr_depth = cr_depth
        self.xcr_depth = xcr_depth
        assert self.merge_idx < self.cr_depth <= self.cr_depth + self.xcr_depth <= self.depth
        
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, embed_dim))
        self.segment_token_x = nn.Parameter(torch.zeros(1, 1, 1, embed_dim))
        self.segment_token_y = nn.Parameter(torch.zeros(1, 1, 1, embed_dim))

        if use_abs_pos:
            # Initialize absolute positional embedding with pretrain image size.
            num_patches = (pretrain_img_size // patch_size) * (pretrain_img_size // patch_size)
            num_positions = (num_patches + 1) if pretrain_use_cls_token else num_patches
            self.pos_embed = nn.Parameter(torch.zeros(1, num_positions, embed_dim), requires_grad=True)
        else:
            self.pos_embed = None

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.depth)]

        self.blocks = nn.ModuleList()
        for i in range(self.depth):
            if i < self.cr_depth:
                input_size = (img_size[0] // patch_size, img_size[1] // patch_size)
            elif i < self.cr_depth + self.xcr_depth:
                input_size = ((self.num_prompts + 1) * img_size[0] // patch_size, img_size[1] // patch_size)
            else:
                input_size = (2 * img_size[0] // patch_size, img_size[1] // patch_size)
            block = Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
                window_size=window_size if i in window_block_indexes else 0,
                use_residual_block=i in residual_block_indexes,
                input_size=input_size,
            )
            if use_act_checkpoint:
                block = checkpoint_wrapper(block)
            self.blocks.append(block)

        self._out_feature_channels = {out_feature: embed_dim}
        self._out_feature_strides = {out_feature: patch_size}
        self._out_features = [out_feature]

        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=0.02)
        self.norm = norm_layer(embed_dim)
        
        self.decoder_embed_dim = decoder_embed_dim
        self.decoder_embed = nn.Linear(embed_dim*4, patch_size ** 2 * self.decoder_embed_dim, bias=True)
        self.decoder_pred = nn.Sequential(
            nn.Conv2d(self.decoder_embed_dim, self.decoder_embed_dim, kernel_size=3, padding=1),
            LayerNorm2D(self.decoder_embed_dim),
            nn.GELU(),
            nn.Conv2d(self.decoder_embed_dim, 3, kernel_size=1, bias=True),
        )
        # --------------------------------------------------------------------------
        torch.nn.init.normal_(self.mask_token, std=.02)
        torch.nn.init.normal_(self.segment_token_x, std=.02)
        torch.nn.init.normal_(self.segment_token_y, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def forward_encoder(self, prompts, query):
        img, tgt = prompts[0], prompts[1] # img/tgt: (num_prompts, H, W, 3); query: (1, H, W, 3)
        i = self.patch_embed(img.permute(0, 3, 1, 2)) + self.segment_token_x
        t = self.patch_embed(tgt.permute(0, 3, 1, 2)) + self.segment_token_y
        q = self.patch_embed(query.permute(0, 3, 1, 2)) + self.segment_token_x 
        assert i.shape[1:] == t.shape[1:] == q.shape[1:]
        assert q.shape[-1] == self.mask_token.shape[-1]
        _, Hp, Wp, C = i.shape
        mask = self.mask_token.expand(1, Hp, Wp, -1) # (1, Hp, Wp, C)
        x = torch.stack([i, t], dim=0) # (2, num_prompts, Hp, Wp, C)
        x = torch.cat([x, torch.stack([q, mask], dim=0)], dim=1) # (2, num_prompts+1, Hp, Wp, C)
        x = x.reshape(-1, Hp, Wp, C) # (2*(num_prompts+1), Hp, Wp, C)
        assert x.shape == (2 * (self.num_prompts + 1), Hp, Wp, C)
        if self.pos_embed is not None:
            x = x + get_abs_pos(self.pos_embed, self.pretrain_use_cls_token, (Hp, Wp))
        latents = []
        for idx, blk in enumerate(self.blocks):
            x = blk(x)
            if idx == self.merge_idx:
                x = x.reshape(2, (self.num_prompts + 1), Hp, Wp, C)
                x = (x[0] + x[1]) * 0.5 # (num_prompts+1, Hp, Wp, C)
            if idx == self.cr_depth - 1:
                x = x.reshape(1, (self.num_prompts + 1) * Hp, Wp, C)
            if idx == self.cr_depth + self.xcr_depth - 1:
                x = x.reshape(self.num_prompts + 1, Hp, Wp, C)
                p, y = x.split((self.num_prompts, 1), dim=0)
                x = torch.cat([torch.mean(p, dim=0, keepdim=True), y], dim=0)
                x = x.reshape(1, 2 * Hp, Wp, C)
            if idx in self.encoder_sampling:
                latents.append(self.norm(x.reshape(-1, Hp, Wp, C)[-1]))
        latent = torch.cat(latents, dim=-1)
        assert latent.shape == (Hp, Wp, 4 * C)
        return latent

    def forward_decoder(self, latent):
        # predictor projection
        x = self.decoder_embed(latent)
        ps = self.patch_size
        Hl, Wl, _= x.shape
        x = x.reshape(Hl, Wl, ps, ps, self.decoder_embed_dim)
        x = torch.einsum('hwpqc->hpwqc', x)
        x = x.reshape(Hl * ps, Wl * ps, self.decoder_embed_dim).permute(2, 0, 1) # (D, Hd, Wd)
        pred = self.decoder_pred(x).permute(1, 2, 0) # (H, W, 3)
        return pred

    def forward_loss(self, pred, label):
        if self.loss_func == "l1l2":
            loss = ((pred - label).abs() + (pred - label) ** 2.) * 0.5
        elif self.loss_func == "l1":
            loss = (pred - label).abs()
        elif self.loss_func == "l2":
            loss = (pred - label) ** 2.
        elif self.loss_func == "smoothl1":
            loss = F.smooth_l1_loss(pred, label, reduction="none", beta=0.01)
        Loss = loss.sum() / (loss.numel() + 1e-2)  # mean loss on removed patches
        return Loss

    def forward(self, prompts, query, label=None):
        latent = self.forward_encoder(prompts, query)
        ret = pred = self.forward_decoder(latent)
        if label is not None:
            loss = self.forward_loss(pred, label)
            ret = (loss, pred)
        return ret


def painter_varient_patch16_win_dec64_8glb_sl1(**kwargs):
    model = Painter_Varient(
        img_size=(448, 448), patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        drop_path_rate=0.1, window_size=14, qkv_bias=True,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6),
        window_block_indexes=(list(range(0, 2)) + list(range(3, 5)) + list(range(6, 8)) + list(range(9, 11)) + \
                                list(range(12, 14)), list(range(15, 17)), list(range(18, 20)), list(range(21, 23))),
        residual_block_indexes=[], use_rel_pos=True, out_feature="last_feat",
        decoder_embed_dim=64,
        loss_func="smoothl1",
        **kwargs)
    return model

def get_vit_lr_decay_rate(name, lr_decay_rate=1.0, num_layers=12):
    """
    Calculate lr decay rate for different ViT blocks.
    Args:
        name (string): parameter name.
        lr_decay_rate (float): base lr decay rate.
        num_layers (int): number of ViT blocks.
    Returns:
        lr decay rate for the given parameter.
    """
    layer_id = num_layers + 1
    if name.startswith("backbone"):
        if ".pos_embed" in name or ".patch_embed" in name:
            layer_id = 0
        elif ".blocks." in name and ".residual." not in name:
            layer_id = int(name[name.find(".blocks.") :].split(".")[2]) + 1

    return lr_decay_rate ** (num_layers + 1 - layer_id)

