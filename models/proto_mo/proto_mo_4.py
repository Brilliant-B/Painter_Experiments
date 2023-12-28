import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import random

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


class Class_Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, cls, pch):
        B, Hp, Wp, C = cls.shape
        x = torch.cat([cls, pch], dim=1)
        q = self.q(cls).reshape(B, Hp * Wp, self.num_heads, C // self.num_heads).permute(2, 0, 1, 3) * self.scale
        k = self.k(x).reshape(B, -1, self.num_heads, C // self.num_heads).permute(2, 0, 1, 3)
        v = self.v(x).reshape(B, -1, self.num_heads, C // self.num_heads).permute(2, 0, 1, 3)
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x_cls = (attn @ v).view(self.num_heads, B, Hp, Wp, -1).permute(1, 2, 3, 0, 4).reshape(B, Hp, Wp, -1)
        x_cls = self.proj(x_cls)
        x_cls = self.proj_drop(x_cls)
        return x_cls



class CA_Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Class_Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim*mlp_ratio), act_layer=act_layer, drop=drop)
    
    def forward(self, cls, pch):
        cls = cls + self.drop_path(self.attn(self.norm1(cls), self.norm1(pch)))
        cls = cls + self.drop_path(self.mlp(self.norm2(cls)))
        return cls



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

    def forward(self, x, cait=False):
        B, H, W, _ = x.shape
        Ht = W if cait else H
        # qkv with shape (3, nHead, B, H * W, C)
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 3, 0, 1, 4)
        # q, k, v with shape (nHead * B, H * W, C)
        q, k, v = qkv.reshape(3, self.num_heads * B, H * W, -1).unbind(0)
        if cait:    q = q[:, -W*W:]
        attn = (q * self.scale) @ k.transpose(-2, -1)
        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(attn, q, self.rel_pos_h, self.rel_pos_w, (Ht, W), (H, W))
        attn = attn.softmax(dim=-1)
        x = attn @ v
        x = x.view(self.num_heads, B, Ht, W, -1)
        x = x.permute(1, 2, 3, 0, 4).reshape(B, Ht, W, -1)
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
    
    def attn_window(self, x):
        # Window partition
        if self.window_size > 0:
            x, pad_hw = window_partition(x, self.window_size)
        x = self.attn(x)
        # Reverse window partition
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, self.input_size)
        return x

    def forward(self, x, use_cait=False):
        ori_shape = list(x.shape)
        x = x.reshape(-1, *ori_shape[-3:])
        shortcut = x[:, -ori_shape[-2]:] if use_cait else x
        x = self.norm1(x)
        if use_cait:    x = self.attn(x, cait=True)
        else:   x = self.attn_window(x)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        if self.use_residual_block:
            x = self.residual(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        if use_cait:    ori_shape[-3] = ori_shape[-2]
        x = x.reshape(ori_shape)
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
            
            seed=0,
            datasets=None,
            num_contexts_in=1,
            num_contexts=3,
            cq_depth=15,
            p_depth=2,
            insert_pc=True,
            use_kn_cait=True,
            encoder_momentum_weight=0.0,
            context_momentum_weight=0.2,
            query_momentum_weight=1.0,
            skip_query=False,
            use_attn_mean=True,
            use_random_nc=False,
            dataset_loss_weight=None,
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
            decoder_embed_dim=128,
            loss_func="smoothl1",
        ):
        super().__init__()

        # --------------------------------------------------------------------------
        self.seed = seed
        self.is_infer = is_infer
        self.use_cache = use_cache
        self.skip_query = skip_query
        self.insert_pc = insert_pc
        self.use_kn_cait = use_kn_cait
        self.use_attn_mean = use_attn_mean
        self.use_random_nc = use_random_nc
        self.img_size = img_size
        self.ori_window_size = (img_size[0] // patch_size, img_size[1] // patch_size)
        self.pretrain_use_cls_token = pretrain_use_cls_token
        self.patch_size = patch_size
        self.depth = depth
        self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        self.patch_embed.num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)
        self.loss_func = loss_func
        self.merge_layer= merge_layer
        self.encoder_sampling = set([6, 12, 18, 24])
        if not self.is_infer:
            self.dloss_weight = dataset_loss_weight
            self.need_loss_cal = min(self.dloss_weight.values()) < max(self.dloss_weight.values())
        
        self.nc = num_contexts
        self.nci = num_contexts_in
        if self.is_infer:   assert self.nci == self.nc
        else:   assert self.nci <= self.nc and self.nc % self.nci == 0
        self.cq = cq_depth
        self.p = p_depth
        assert self.cq <= self.depth
        if not self.insert_pc:  assert self.cq + self.p <= self.depth
        
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, embed_dim))
        self.segment_token_x = nn.Parameter(torch.zeros(1, 1, 1, embed_dim))
        self.segment_token_y = nn.Parameter(torch.zeros(1, 1, 1, embed_dim))
        if not self.use_attn_mean:
            self.prototype_tokens = nn.Parameter(torch.zeros(1, *self.ori_window_size, self.embed_dim))

        if use_abs_pos:
            # Initialize absolute positional embedding with pretrain image size.
            num_patches = (pretrain_img_size // patch_size) * (pretrain_img_size // patch_size)
            num_positions = (num_patches + 1) if pretrain_use_cls_token else num_patches
            self.pos_embed = nn.Parameter(torch.zeros(1, num_positions, embed_dim), requires_grad=True)
        else:
            self.pos_embed = None

        self.emo = encoder_momentum_weight
        self.cmo = context_momentum_weight
        self.qmo = query_momentum_weight
        assert 0 <= self.emo <= 1 and 0 <= self.cmo <= 1 and 0 <= self.qmo <= 1
        
        self.nl = sum([x <= self.cq for x in self.encoder_sampling])
        
        if self.is_infer:
            self.cache = None
        else:
            self.e_queues = {name: {i: F.normalize(torch.randn(self.nc, *self.ori_window_size, embed_dim), dim=-1) if i >= merge_layer 
                                  else F.normalize(torch.randn(self.nc, 2, *self.ori_window_size, embed_dim), dim=-1)
                                  for i in range(3)} for name in datasets}
            self.queues = {name: F.normalize(torch.randn(self.nc, *self.ori_window_size, (self.nl + 1) * embed_dim), dim=-1) for name in datasets}
            self.ptrs = {name: torch.zeros(1, dtype=int) for name in datasets}
        
        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.depth)]

        self.blocks = nn.ModuleList()
        if self.emo > 0.0:  self.cm_blocks = nn.ModuleList()
        for i in range(self.depth):
            window = window_size if i in window_block_indexes else 0
            input_size = (2 * self.ori_window_size[0], self.ori_window_size[1])
            if i < self.cq:
                input_size = self.ori_window_size
                if self.emo > 0.0:
                    cm_block = Block(
                        dim=embed_dim,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        drop_path=dpr[i],
                        norm_layer=norm_layer,
                        act_layer=act_layer,
                        use_rel_pos=use_rel_pos,
                        rel_pos_zero_init=rel_pos_zero_init,
                        window_size=window,
                        use_residual_block=i in residual_block_indexes,
                        input_size=input_size,
                    )
                    if use_act_checkpoint:
                        cm_block = checkpoint_wrapper(cm_block)
                    self.cm_blocks.append(cm_block)
            elif i < self.cq + self.p:
                if not self.insert_pc:
                    input_size = ((self.nc + 1) * self.ori_window_size[0], self.ori_window_size[1])
                window = 0
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
                window_size=window,
                use_residual_block=i in residual_block_indexes,
                input_size=input_size,
            )
            if use_act_checkpoint:
                block = checkpoint_wrapper(block)
            self.blocks.append(block)
        
        if self.insert_pc:   
            self.p_blocks = nn.ModuleList()
            for i in range(self.p):
                input_size = ((self.nc + 1) * self.ori_window_size[0], self.ori_window_size[1])
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
                    window_size=0,
                    use_residual_block=i in residual_block_indexes,
                    input_size=input_size,
                )
                if use_act_checkpoint:
                    block = checkpoint_wrapper(block)
                self.p_blocks.append(block)
        
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
        '''
        self._out_feature_channels = {out_feature: embed_dim}
        self._out_feature_strides = {out_feature: patch_size}
        self._out_features = [out_feature]
        '''
        # --------------------------------------------------------------------------
        torch.nn.init.normal_(self.mask_token, std=.02)
        torch.nn.init.normal_(self.segment_token_x, std=.02)
        torch.nn.init.normal_(self.segment_token_y, std=.02)
        if not self.use_attn_mean:
            torch.nn.init.normal_(self.prototype_tokens, std=.1)
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
    
    @torch.no_grad()
    def init_cm_encoder(self):
        for param_cm, param_q in zip(self.cm_blocks.parameters(), self.blocks[:self.cq].parameters()):
            param_cm.data.copy_(param_q.data)  # initialize
            param_cm.requires_grad = False
    
    @torch.no_grad()
    def momentum_update_cm_encoder(self):
        # print("momentum update")
        for param_cm, param_q in zip(self.cm_blocks.parameters(), self.blocks[:self.cq].parameters()):
            param_cm.data = param_cm.data * self.emo + param_q.data * (1. - self.emo)
    
    def cm_encoder(self, c):
        c_latent = []
        for idx in range(self.cq):
            c = self.cm_blocks[idx](c)
            if idx + 1 == self.merge_layer:
                c = c.mean(-4) # [B, nci, Hp, Wp, C]
            if idx + 1 in self.encoder_sampling:
                c_latent.append(self.norm(c))
            if idx < 3:
                c_all = self.queue_sample_update(type, idx, c)
        c_latent = None if len(c_latent) == 0 else torch.cat(c_latent, dim=-1)
        return c, c_latent, c_all
    
    def cq_encoder(self, x, extract):
        x_latent = []
        for idx in range(self.cq):
            if idx < 3:
                x = torch.cat([extract, x], dim=-3)
            x = self.blocks[idx](x)
            if idx + 1 == self.merge_layer:
                x = x.mean(-4)
            if idx + 1 in self.encoder_sampling:
                x_latent.append(self.norm(x))
        x_latent = None if len(x_latent) == 0 else torch.cat(x_latent, dim=-1)
        return x, x_latent
    
    def pc_extractor(self, x):
        _, h, w, _ = x.shape
        if self.insert_pc:
            for idx in range(self.p):
                x = self.p_blocks[idx](x, True)
        else:
            for idx in range(self.cq, self.cq + self.p):
                x = self.blocks[idx](x, True)
        x = x[:, h-w:]
        return x
    
    def kn_encoder(self, x):
        latent = []
        start = int(not self.insert_pc) * self.p
        for idx in range(self.cq + start, 24):
            if idx in [17, 20, 23]:
                x = self.blocks[idx](x, self.use_kn_cait)
            else:
                x = self.blocks[idx](x)
            if idx + 1 in self.encoder_sampling:
                latent.append(self.norm(x))
        return latent
    
    @torch.no_grad()
    def cache_storage(self, c, c_latent):
        if c_latent is not None:
            if self.skip_query: c_latent = c_latent.mean(1)
            c = torch.cat([c_latent, c], dim=-1)
        self.cache = c.cpu()
    
    @torch.no_grad()
    def cache_sampling(self, C):
        cache = self.cache.cuda()
        c_latent, c = cache.split([self.nl * C, C], dim=-1)
        c_latent = None if c_latent.shape[-1] == 0 else c_latent if self.skip_query else c_latent.mean(1)
        return c, c_latent  # [B, nc*Hp, Wp, C] [B, Hp, Wp, nl*C]
    
    @torch.no_grad() # cmo_former
    def queue_cmo_update(self, type, c, c_latent):
        # [B, nci, Hp, Wp, (nl+1)*C]
        if c_latent is not None:
            c = torch.cat([c_latent, c], dim=-1).cpu()
        for i in range(len(type)):
            ptr = self.ptrs[type[i]]
            self.queues[type[i]][ptr:ptr + self.nci] = \
                self.cmo * self.queues[type[i]][ptr:ptr + self.nci] + (1 - self.cmo) * c[i]
            self.ptrs[type[i]] = (ptr + self.nci) % self.nc
            
    @torch.no_grad() # qmo_latter
    def queue_qmo_update(self, type, q, q_latent):
        # [B, Hp, Wp, (nl+1)*C]
        if q_latent is not None:
            q = torch.cat([q_latent, q], dim=-1).cpu()
        for i in range(len(type)):
            ptr = self.ptrs[type[i]]
            self.queues[type[i]][ptr:ptr + self.nci] = \
                self.qmo * self.queues[type[i]][ptr:ptr + self.nci] + (1 - self.qmo) * q[i].unsqueeze(0)
    
    @torch.no_grad()
    def context_queue_sampling(self, type, B, C):
        c, c_latent = [], []
        n = random.randint(1, self.nc)
        for i in range(B):
            if self.use_random_nc:
                idx = random.choices(range(self.nc), k=n)
                ic = torch.stack([self.queues[type[i]][j].cuda() for j in idx], dim=0)
            else:   ic = self.queues[type[i]].cuda()
            ic_latent, ic = ic.split([self.nl * C, C], dim=-1)
            c.append(ic)
            if ic_latent.shape[-1] != 0:
                c_latent.append(ic_latent.mean(0))
        c = torch.stack(c, dim=0)
        c_latent = None if len(c_latent) == 0 else torch.stack(c_latent, dim=0)
        return c, c_latent  # [B, nc, Hp, Wp, C] [B, Hp, Wp, nl*C]
    
    
    @torch.no_grad()
    def queue_sample_update(self, type, idx, c):
        c_all = []
        for i in range(len(type)):
            ic = self.e_queues[type[i]][idx]
            c_all.append(ic.cuda())
        c_all = torch.stack(c_all, dim=0)
        if idx >= self.merge_layer:
            for i in range(len(type)):
                ptr = self.ptrs[type[i]]
                self.e_queues[type[i]][idx][ptr:ptr + self.nci] = c[i]
                self.ptrs[type[i]] = (ptr + self.nci) % self.nc
        else:
            c_all = c_all.transpose(1, 2)
            for i in range(len(type)):
                ptr = self.ptrs[type[i]]
                self.e_queues[type[i]][idx][ptr:ptr + self.nci] = c[i].transpose(0, 1)
                self.ptrs[type[i]] = (ptr + self.nci) % self.nc
        return c_all  # [B, nc, Hp, Wp, C] / [B, 2, nc, Hp, Wp, C]
    
    
    def forward_encoder(self, type, c_query, c_target, query, target, mask):
        if (not self.is_infer and self.cmo < 1.0) or (self.is_infer and (not self.use_cache or self.cache is None)):
            c, co = c_query.flatten(0, 1), c_target.flatten(0, 1)
            c = self.patch_embed(c.permute(0, 3, 1, 2).contiguous()) + self.segment_token_x
            co = self.patch_embed(co.permute(0, 3, 1, 2).contiguous()) + self.segment_token_x
            c, co = c.reshape(B, -1, Hp, Wp, C), co.reshape(B, -1, Hp, Wp, C)
            c = torch.stack([c, co], dim=2) # (B, nci, 2, Hp, Wp, C)
            if self.pos_embed is not None:
                c = c + get_abs_pos(self.pos_embed, self.pretrain_use_cls_token, (Hp, Wp))
            c, c_latent, extract = self.cm_encoder(c) if self.emo > 0.0 else self.cq_encoder(c)
            assert c.shape == (B, self.nci, Hp, Wp, C) and len(type) == B # c_latent.shape == (B, self.nci, Hp, Wp, self.nl * C) 
            if not self.is_infer:   self.queue_cmo_update(type, c, c_latent)
            elif not self.skip_query:   self.cache_storage(c, c_latent)
        
        x = self.patch_embed(query.permute(0, 3, 1, 2).contiguous()) + self.segment_token_x
        t = self.patch_embed(target.permute(0, 3, 1, 2).contiguous()) + self.segment_token_y
        B, Hp, Wp, C  = x.shape
        assert mask.shape[1:] == self.ori_window_size == (Hp, Wp) and self.mask_token.shape[-1] == C, "encoder input shape error"
        mask_token = self.mask_token.expand(B, Hp, Wp, -1)
        mask = mask.unsqueeze(-1).type_as(mask_token) # (B, Hp, Wp, 1)
        t = t * (1 - mask) + mask_token * mask # (B, Hp, Hp, C)
        x = torch.stack([x, t], dim=1) # (B, 2, Hp, Wp, C)
        if self.pos_embed is not None:
            x = x + get_abs_pos(self.pos_embed, self.pretrain_use_cls_token, (Hp, Wp))
        x, x_latent = self.cq_encoder(x, extract)
        assert x.shape == (B, Hp, Wp, C) # x_latent.shape == (B, Hp, Wp, self.nl * C)
        
        
        if not self.is_infer:
            if self.qmo < 1.0:  self.queue_qmo_update(type, x, x_latent)
            c, c_latent = self.context_queue_sampling(type, B, C)
        elif not self.skip_query and self.use_cache and self.cache is not None:
            c, c_latent = self.cache_sampling(C)
        
        if self.skip_query:
            if not self.is_infer or not self.use_cache or self.cache is None:
                if self.use_attn_mean:  p = torch.mean(c, dim=1)
                else:   p = self.prototype_tokens.repeat(B, 1, 1, 1)
                c = c.flatten(1, 2)
                p = torch.cat([c, p], dim=1)
                p = self.pc_extractor(p)
                if self.is_infer and self.use_cache:   self.cache_storage(p, c_latent)
            if self.is_infer and self.use_cache:    p, c_latent = self.cache_sampling(C)
        else:
            if self.use_attn_mean:  p = torch.mean(c, dim=1)
            else:   p = self.prototype_tokens.repeat(B, 1, 1, 1)
            c = c.flatten(1, 2).repeat(2, 1, 1, 1) # add query interaction
            x = torch.cat([p, x], dim=0)
            x = torch.cat([c, x], dim=1)
            p, x = self.pc_extractor(x).split([B, B], dim=0)
        
        x = torch.cat([p, x], dim=1)
        assert x.shape == (B, 2 * Hp, Wp, C)
        latent = self.kn_encoder(x)
        if x_latent is not None:
            if self.is_infer and not self.use_cache:    c_latent = c_latent.mean(1)
            x_latent = torch.cat([c_latent, x_latent], dim=1) # [B, 2*Hp, Wp, nl*C]
            latent = torch.cat([x_latent] + latent, dim=-1)
        assert latent.shape == (B, 2 * Hp, Wp, 4 * C)
        return latent

    def forward_decoder(self, latent):
        x = self.decoder_embed(latent) # predictor projection
        ps = self.patch_size
        B, Hl, Wl, _ = x.shape
        x = x.reshape(B, Hl, Wl, ps, ps, self.decoder_embed_dim)
        x = torch.einsum('bhwpqc->bhpwqc', x)
        x = x.reshape(B, Hl * ps, Wl * ps, self.decoder_embed_dim) # (B, 2 * Hd, Wd, D)
        pred = self.decoder_pred(x.permute(0, 3, 1, 2).contiguous())
        pred = pred.permute(0, 2, 3, 1).contiguous() # (B, 2 * H, W, 3)
        pred = pred.chunk(2, dim=1)[-1] # pred: (B, H, W, 3)
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
                image_mask[i] = image_mask[i] * self.dloss_weight[type[i]]
        
        Loss = (loss * image_mask).sum() / (image_mask.sum() + 1e-2)  # mean loss on removed patches
        return Loss, image_mask

    def forward(self, type, c_query, c_target, query, target, mask, valid):
        # print(c_query.shape, c_target.shape) # (B, nci, H, W, 3)
        # print(query.shape, target.shape) # (B, H, W, 3)
        # print(mask.shape) # (B, Hp, Wp)
        # print(valid.shape) # (B, H, W, 3)
        latent = self.forward_encoder(type, c_query, c_target, query, target, mask)
        pred = self.forward_decoder(latent)
        if self.is_infer:
            return pred
        else:
            loss, image_mask = self.forward_loss(type, pred, target, mask, valid)
            return loss, pred, image_mask


def proto_mo_3_patch16_win_dec64_8glb_sl1(**kwargs):
    model = Painter_Varient(
        img_size=(448, 448), patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        drop_path_rate=0.1, window_size=14, qkv_bias=True,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6),
        window_block_indexes=(list(range(0, 2)) + list(range(3, 5)) + list(range(6, 8)) + list(range(9, 11)) + \
                                list(range(12, 14)), list(range(15, 17)), list(range(18, 20)), list(range(21, 23))),
        residual_block_indexes=[], use_rel_pos=True,
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
