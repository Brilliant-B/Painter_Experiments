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
    add_decomposed_rel_pos_with_reg,
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
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.use_rel_pos = use_rel_pos
        self.input_size = input_size
        if self.use_rel_pos:
            # initialize relative positional embeddings
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))
            if not rel_pos_zero_init:
                trunc_normal_(self.rel_pos_h, std=0.02)
                trunc_normal_(self.rel_pos_w, std=0.02)

    def forward(self, x, nreg, cait):
        # x: [B, N, C]; N=H*W+n, n=nreg
        B, N, C = x.shape
        # qkv: [3, nHead, B, N, C//nHead]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, -1).permute(2, 3, 0, 1, 4)
        # q: [B*nHead, n, C//nHead]; k, v: [B*nHead, N, C//nHead]
        q, k, v = qkv.reshape(3, self.num_heads * B, N, -1).unbind(0)
        if cait:   q = q[:, -nreg:]
        attn = (q * self.scale) @ k.transpose(-2, -1)
        if self.use_rel_pos and not cait:
            attn = add_decomposed_rel_pos_with_reg(attn, q, self.rel_pos_h, self.rel_pos_w, self.input_size, self.input_size)
        attn = attn.softmax(dim=-1)
        # attn: [B*nHead, N|n, N]
        x = attn @ v
        # x: [B*nHead, N|n, C//nHead]
        x = x.view(self.num_heads, B, -1, C // self.num_heads)
        x = x.permute(1, 2, 0, 3).reshape(B, -1, C)
        x = self.proj(x)
        # out: [B, N|n, C]
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
        use_cait=False,
        ls_init=1e-4, # 1e-5
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
        self.use_cait = use_cait
        self.ls_init = ls_init
        if use_cait:
            self.gamma_1 = nn.Parameter(ls_init * torch.ones((dim)), requires_grad=True)
            self.gamma_2 = nn.Parameter(ls_init * torch.ones((dim)), requires_grad=True)
    
    
    @torch.jit.ignore
    def no_weight_decay(self):
        return {'gamma_1', 'gamma_2'}


    def forward_function(self, x, reg=None):
        # x: [B, H, W, C]; (opt)reg: [B, n, C]
        ori_x = x
        x = x.flatten(1, 2)
        if reg is not None:
            nreg = reg.shape[1]
            x = torch.cat([x, reg], dim=1) 
            shortcut = reg if self.use_cait else x
        else:   
            nreg = 0
            shortcut = x
        # x: [B, H*W+n, C]; shortcut: [B, H*W+n|n, C]
        x = self.norm1(x)
        if not self.use_cait and self.window_size > 0:
            x, pad_hw = window_partition(x, self.window_size)
            x = self.attn(x, nreg, False)
            x = window_unpartition(x, self.window_size, pad_hw, self.input_size)
        else:
            x = self.attn(x, nreg, self.use_cait)
        # x: [B, H*W+n|n, C]
        if self.use_cait:
            x = shortcut + self.drop_path(self.gamma_1 * x)
            reg = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        else:
            x = shortcut + self.drop_path(x)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            x, reg = x.split([x.shape[1]-nreg, nreg], dim=1)
            ori_x = x.reshape(list(ori_x.shape))
            if self.use_residual_block:
                ori_x = self.residual(ori_x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        # ori_x: [B, H, W, C]; reg: [B, n, C]
        return ori_x, reg
    
    def forward(self, x, reg=None):
        # x: [B, ..., H, W, C]; (opt)reg: [B, ..., n, C]
        have_reg = reg is not None
        need_x_transform, need_reg_transform = not x.dim == 4, have_reg and not reg.dim == 3
        if need_x_transform:
            x_shape = list(x.shape)
            x = x.reshape(-1, *x_shape[-3:])
        if need_reg_transform:
            reg_shape = list(reg.shape)
            reg = reg.reshape(-1, *reg_shape[-2:])
        x, reg = self.forward_function(x, reg)
        if need_x_transform:
            x = x.reshape(x_shape)
        if need_reg_transform:
            reg = reg.reshape(reg_shape)
            ret = (x, reg)
        else:   ret = x if not have_reg else (x, reg)
        return ret



class T_Painter(nn.Module):
    def __init__(
            self,
            img_size=(448, 448),
            patch_size=16,
            in_chans=3,
            embed_dim=1024,
            depth=24,
            num_heads=16,
            merge_layer=3,
            
            seed=0,
            datasets_lw=None,
            ni_contexts=1,
            n_contexts=1,
            extract_layers=3,
            collect_depth=12,
            num_registers=64,
            init_layerscale=1e-4,
            is_infer=False,
            use_cache=True,
            
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
        self.seed = seed
        self.is_infer = is_infer
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
        self.nci = ni_contexts
        self.extract_layers = extract_layers
        self.collect_depth = collect_depth
        self.num_register = num_registers
        self.init_layerscale = init_layerscale
        assert merge_layer <= min(self.encoder_sampling)
        assert merge_layer <= collect_depth
        
        if is_infer:
            self.use_cache = use_cache
        else:
            self.need_loss_cal = min(datasets_lw.values()) < max(datasets_lw.values())
        
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, embed_dim))
        self.segment_token_x = nn.Parameter(torch.zeros(1, 1, 1, embed_dim))
        self.segment_token_y = nn.Parameter(torch.zeros(1, 1, 1, embed_dim))
        self.registers = nn.Parameter(torch.zeros(1, 1, num_registers, embed_dim))
        
        if use_abs_pos:
            # Initialize absolute positional embedding with pretrain image size.
            num_patches = (pretrain_img_size // patch_size) * (pretrain_img_size // patch_size)
            num_positions = (num_patches + 1) if pretrain_use_cls_token else num_patches
            self.pos_embed = nn.Parameter(torch.zeros(1, num_positions, embed_dim), requires_grad=True)
            self.reg_pos_embed = nn.Parameter(torch.zeros(1, num_registers, embed_dim))
            self.reg_cxt_embed = nn.Parameter(torch.zeros(1, ni_contexts, 1, 1))
        else:
            self.pos_embed = None
        
        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.depth)]
        
        self.e_blocks = nn.ModuleList()
        self.blocks = nn.ModuleList()
        for i in range(self.extract_layers):
            e_block = Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
                window_size=int(i in window_block_indexes)*window_size,
                use_residual_block=i in residual_block_indexes,
                use_cait=True,
                ls_init=init_layerscale,
                input_size=self.ori_window_size,
            )
            if use_act_checkpoint:
                e_block = checkpoint_wrapper(e_block)
            self.e_blocks.append(e_block)
        
        for i in range(self.depth):
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
                window_size=int(i in window_block_indexes)*window_size,
                use_residual_block=i in residual_block_indexes,
                use_cait=False,
                ls_init=init_layerscale,
                input_size=self.ori_window_size,
            )
            if use_act_checkpoint:
                block = checkpoint_wrapper(block)
            self.blocks.append(block)

        self._out_feature_channels = {out_feature: embed_dim}
        self._out_feature_strides = {out_feature: patch_size}
        self._out_features = [out_feature]
    
        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=0.02)
            trunc_normal_(self.reg_pos_embed, std=0.02)
            trunc_normal_(self.reg_cxt_embed, std=0.02)
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
        torch.nn.init.normal_(self.registers, std=.02)
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
        return {'pos_embed', 'reg_pos_embed', 'reg_cxt_embed', 'cls_token', 'registers'}
    
    # @torch.no_grad()
    # def init_copy_eblocks(self):
    #     for (name_e, param_e), param_q in zip(self.e_blocks.named_parameters(), self.blocks[:self.collect_depth].parameters()):
    #         tmp_q = param_q.data
    #         if "attn.rel_pos_h" in name_e:
    #             tmp_q = F.interpolate(tmp_q.permute(1, 0).unsqueeze(0), size=(0 * 56 + 55), mode='linear')[0].permute(1, 0)
    #         param_e.data.copy_(tmp_q)
    #         param_e.requires_grad = True

    def forward_encoder(self, type, c_query, c_target, query, target, mask):
        x = self.patch_embed(query.permute(0, 3, 1, 2).contiguous()) + self.segment_token_x
        t = self.patch_embed(target.permute(0, 3, 1, 2).contiguous()) + self.segment_token_y
        B, Hp, Wp, C = x.shape
        assert mask.shape[1:] == self.ori_window_size == (Hp, Wp) and self.mask_token.shape[-1] == C
        mask_token = self.mask_token.expand(B, Hp, Wp, -1)
        mask = mask.unsqueeze(-1).type_as(mask_token) # (B, Hp, Wp, 1)
        t = t * (1 - mask) + mask_token * mask # (B, Hp, Hp, C)
        x = torch.stack([x, t], dim=1) # (B, 2, Hp, Wp, C)
        if self.pos_embed is not None:
            x = x + get_abs_pos(self.pos_embed, self.pretrain_use_cls_token, (Hp, Wp))
        
        c, co = c_query.flatten(0, 1), c_target.flatten(0, 1)
        c = self.patch_embed(c.permute(0, 3, 1, 2).contiguous()) + self.segment_token_x
        co = self.patch_embed(co.permute(0, 3, 1, 2).contiguous()) + self.segment_token_x
        c, co = c.reshape(B, -1, Hp, Wp, C), co.reshape(B, -1, Hp, Wp, C)
        c = torch.stack([c, co], dim=1) # (B, 2, nci, Hp, Wp, C)
        if self.pos_embed is not None:
            c = c + get_abs_pos(self.pos_embed, self.pretrain_use_cls_token, (Hp, Wp))
        
        x = torch.cat([x.unsqueeze(2), c], dim=-4)
        reg = self.registers.expand(B, self.nci, -1, -1)
        if self.reg_pos_embed is not None:
            reg = reg + self.reg_pos_embed
        
        # {x c}: [B, 2, 1+nci, Hp, Wp, C]; reg: [B, nci, n, C]
        latents = []
        for idx in range(self.collect_depth):
            # x = self.blocks[idx](x)
            # c = self.blocks[idx](c)
            # if idx + 1 == self.merge_layer:
            #     x, c = x.mean(1), c.mean(1)
            # if idx in self.encoder_sampling:
            #     latents.append(self.norm(x))
            x = self.blocks[idx](x)
            if idx + 1 == self.merge_layer:
                x = x.mean(1)
            if idx in self.encoder_sampling:
                latents.append(self.norm(x[:, 0]))
        
        splited = x.split([1, self.nci], dim=-4)
        x, c = torch.squeeze(splited[0], dim=-4), splited[1]

        # x: [B, Hp, Wp, C]; c: [B, nci, Hp, Wp, C]; reg_pos_embed: [1, nci, 1, 1]
        for idx in range(self.extract_layers):
            c, reg = self.e_blocks[idx](c, reg)
        reg = reg + self.reg_cxt_embed    
        reg = reg.flatten(1, 2)
        
        # reg: [B, nci*n, C]
        for idx in range(self.collect_depth, self.depth):
            x, reg = self.blocks[idx](x, reg)
            if idx in self.encoder_sampling:
                latents.append(self.norm(x))
        
        latents = torch.cat(latents, dim=-1)
        assert latents.shape == (B, Hp, Wp, 4 * C)
        return latents
    
    def forward_decoder(self, latent):
        x = self.decoder_embed(latent) # predictor projection
        ps = self.patch_size
        B, Hl, Wl, _ = x.shape
        x = x.reshape(B, Hl, Wl, ps, ps, self.decoder_embed_dim)
        x = torch.einsum('bhwpqc->bchpwq', x)
        x = x.reshape(B, self.decoder_embed_dim, Hl * ps, Wl * ps) # (B, D, Hd, Wd)
        pred = self.decoder_pred(x)
        pred = pred.permute(0, 2, 3, 1).contiguous() # pred: (B, H, W, 3)
        assert pred.shape == (B, *self.img_size, 3)
        return pred

    def forward_loss(self, type, pred, target, mask, valid):
        b, h, w, p = mask.shape[0], *self.ori_window_size, self.patch_size
        image_mask = mask.unsqueeze(-1).repeat(1, 1, 1, p ** 2 * 3)
        image_mask = image_mask.reshape(b, h, w, p, p, 3)
        image_mask = torch.einsum('bhwpqc->bhpwqc', image_mask).contiguous()
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
        
        if self.need_loss_cal:
            for i in range(b):
                image_mask[i] = image_mask[i] * self.datasets_lw[type[i]]
        
        Loss = (loss * image_mask).sum() / (image_mask.sum() + 1e-2)  # mean loss on removed patches
        return Loss, image_mask

    def forward(self, type, c_query, c_target, query, target, mask, valid):
        # print(c_query.shape, c_target.shape) # (B, nci, H, W, 3)
        # print(query.shape, target.shape) # (B, H, W, 3)
        # print(mask.shape) # (B, Hp, Wp)
        # print(valid.shape) # (B, H, W, 3)
        if self.is_infer:
            latent = self.forward_encoder(type, c_query, c_target, query, target, mask)
            pred = self.forward_decoder(latent)
            return pred
        else:
            latent = self.forward_encoder(type, c_query, c_target, query, target, mask)
            pred = self.forward_decoder(latent)
            loss, image_mask = self.forward_loss(type, pred, target, mask, valid)
            return loss, pred, image_mask


def T1_patch16_win_dec64_8glb_sl1(**kwargs):
    model = T_Painter(
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
