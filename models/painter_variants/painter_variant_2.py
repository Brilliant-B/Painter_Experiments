import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

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
            merge_layer=3,
            num_prompts=1,
            cr_depth=12,
            xcr_depth=12,
            use_cr_bank=True,
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
        self.ori_window_size = (img_size[0] // patch_size, img_size[1] // patch_size)
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
        self.merge_layer= merge_layer
        self.encoder_sampling = set([5, 11, 17, 23])
        self.num_prompts = num_prompts
        self.cq = cr_depth
        self.fcq = xcr_depth
        assert self.cq <= self.fcq <= self.depth
        
        self.use_cr_bank = use_cr_bank
        if self.use_cr_bank:
            self.cr_bank = None
            self.latent_bank = dict()
        
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
            if i < self.cq:
                input_size = self.ori_window_size
            elif i < self.fcq:
                input_size = ((self.num_prompts + 1) * self.ori_window_size[0], self.ori_window_size[1])
            else:
                input_size = (2 * self.ori_window_size[0], self.ori_window_size[1])
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
        self.decoder_embed = nn.Linear(embed_dim * 4, patch_size ** 2 * self.decoder_embed_dim, bias=True)
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

    def forward_encoder(self, prompts, query, target, mask):
        qi = self.patch_embed(query.permute(0, 3, 1, 2)) + self.segment_token_x
        qt = self.patch_embed(target.permute(0, 3, 1, 2)) + self.segment_token_y
        B, Hp, Wp, C = qt.shape
        assert mask.shape[1:] == self.ori_window_size == (Hp, Wp)
        assert self.mask_token.shape[-1] == C
        mask_token = self.mask_token.expand(B, Hp, Wp, -1) # (B, Hp, Wp, C)
        w = mask.unsqueeze(-1).type_as(mask_token) # (B, Hp, Wp, 1)
        qt = qt * (1 - w) + mask_token * w # (B, Hp, Hp, C)
        x = torch.stack([qi, qt], dim=1).unsqueeze(2) # (B, 2, 1, Hp, Wp, C)
        assert x.shape == (B, 2, 1, Hp, Wp, C)
        
        if not self.use_cr_bank or self.cr_bank is None:
            # prompts: (B, 2, num_prompts, H, W, 3)
            img, tgt = torch.transpose(prompts, 0, 1)[0].flatten(0, 1), torch.transpose(prompts, 0, 1)[1].flatten(0, 1)
            # img & tgt: (B * num_prompts, H, W, 3); query & target: (B, H, W, 3)
            i = self.patch_embed(img.permute(0, 3, 1, 2)) + self.segment_token_x
            t = self.patch_embed(tgt.permute(0, 3, 1, 2)) + self.segment_token_y
            i = i.reshape(B, self.num_prompts, Hp, Wp, C)
            t = t.reshape(B, self.num_prompts, Hp, Wp, C)
            px = torch.stack([i, t], dim=1) # (B, 2, num_prompts, Hp, Wp, C)
            x = torch.cat([px, x], dim=2) # (B, 2, num_prompts+1, Hp, Wp, C)
            assert x.shape == (B, 2, self.num_prompts + 1, Hp, Wp, C)
        
        if self.pos_embed is not None:
            x = x + get_abs_pos(self.pos_embed, self.pretrain_use_cls_token, (Hp, Wp))
        
        latents = []
        for idx, blk in enumerate(self.blocks):
            # print("blk_idx:", idx)
            # print("ori:", x.shape)
            ori_shape = x.shape
            x = x.reshape(-1, *(blk.input_size), C)
            # print("c:", x.shape)
            x = blk(x)
            x = x.reshape(ori_shape)
            
            if idx == self.merge_layer - 1:
                x = x.mean(1, keepdim=True)
            if idx == self.cq - 1:
                if self.use_cr_bank and self.cr_bank is not None:
                    x = torch.cat([self.cr_bank, x], dim=2) # (B, -1, num_prompts+1, Hp, Wp, C)
                elif self.use_cr_bank:
                    self.cr_bank = x[:, :, :-1] # (B, -1, num_prompts, Hp, Wp, C)
            if idx == self.fcq - 1:
                p, y = x.split((self.num_prompts, 1), dim=2)
                x = torch.cat([torch.mean(p, dim=2, keepdim=True), y], dim=2)
            # print("post:", x.shape)
            if idx in self.encoder_sampling:
                feat = torch.reshape(x, (B, ori_shape[1], -1, Hp, Wp, C)).mean(1) # (B, -1, Hp, Wp, C)
                if self.use_cr_bank and len(self.latent_bank) and idx in self.latent_bank.keys() and idx < self.cr_depth - 1:
                    feat_prompt = self.latent_bank[idx] # (B, Hp, Wp, C)
                    feat_query = feat.squeeze(1) # (B, Hp, Wp, C)
                else:
                    feat_prompts, feat_query = torch.split(feat, (feat.shape[1]-1, 1), dim=1) # (B, 1, Hp, Wp, C)
                    feat_prompt = feat_prompts.mean(dim=1) # (B, Hp, Wp, C)
                    feat_query = feat_query.squeeze(1) # (B, Hp, Wp, C)
                    if self.use_cr_bank and idx < self.cr_depth - 1:
                        self.latent_bank[idx] = feat_prompt
                single_latent = self.norm(torch.cat((feat_prompt, feat_query), dim=1)) # (B, 2 * Hp, Wp, C)
                latents.append(single_latent)
        
        latent = torch.cat(latents, dim=-1)
        assert latent.shape == (B, 2 * Hp, Wp, 4 * C)
        return latent

    def forward_decoder(self, latent):
        # predictor projection
        x = self.decoder_embed(latent)
        ps = self.patch_size
        B, Hl, Wl, _ = x.shape
        x = x.reshape(B, Hl, Wl, ps, ps, self.decoder_embed_dim)
        x = torch.einsum('bhwpqc->bhpwqc', x)
        x = x.reshape(B, Hl * ps, Wl * ps, self.decoder_embed_dim) # (B, 2 * Hd, Wd, D)
        pred = self.decoder_pred(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1) # (B, 2 * H, W, 3)
        pred = pred.chunk(2, dim=1)[-1] # pred: (B, H, W, 3)
        assert pred.shape == (B, *self.img_size, 3)
        return pred

    def forward_loss(self, pred, target, mask, valid):
        b, h, w, p = mask.shape[0], *self.ori_window_size, self.patch_size
        image_mask = mask.unsqueeze(-1).repeat(1, 1, 1, p ** 2 * 3)
        image_mask = image_mask.reshape(b, h, w, p, p, 3)
        image_mask = torch.einsum('bhwpqc->bhpwqc', image_mask)
        image_mask = image_mask.reshape(b, h * p, w * p, 3)
        assert image_mask.shape[1:-1] == self.img_size
        
        imagenet_mean=torch.tensor([0.485, 0.456, 0.406]).to(target.device)[None, None, None, :]
        imagenet_std=torch.tensor([0.229, 0.224, 0.225]).to(target.device)[None, None, None, :]
        inds_ign = ((target * imagenet_std + imagenet_mean) * (1 - 1. * image_mask)).sum((1, 2, 3)) < 100 * 3
        # image_mask: (B, H, W, 3); valid: (B, H, W, 3)
        if inds_ign.sum() > 0:
            valid[inds_ign] = 0.
        image_mask = image_mask * valid
        
        if self.loss_func == "l1l2":
            loss = ((pred - target).abs() + (pred - target) ** 2.) * 0.5
        elif self.loss_func == "l1":
            loss = (pred - target).abs()
        elif self.loss_func == "l2":
            loss = (pred - target) ** 2.
        elif self.loss_func == "smoothl1":
            loss = F.smooth_l1_loss(pred, target, reduction="none", beta=0.01)
        # loss: (B, H, W, 3)
        # print(image_mask.sum()/image_mask.numel())
        Loss = (loss * image_mask).sum() / (image_mask.sum() + 1e-2)  # mean loss on removed patches
        return Loss, image_mask

    def forward(self, prompts, query, target, mask, valid):
        # print(prompts.shape) # (B, 2, NC, H, W, 3)
        # print(query.shape, target.shape) # (B, H, W, 3)
        # print(mask.shape) # (B, Hp, Wp)
        # print(valid.shape) # (B, H, W, 3)
        latent = self.forward_encoder(prompts, query, target, mask)
        pred = self.forward_decoder(latent)
        loss, image_mask = self.forward_loss(pred, target, mask, valid)
        return loss, pred, image_mask


def painter_variant_2_patch16_win_dec64_8glb_sl1(**kwargs):
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
