from functools import partial
from turtle import forward

import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.models.layers.attn_blocks import CASTBlock, CASTBlockDoubleToken


class SpShLayer(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()

        self.t_fusion = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            nn.GELU()
        )
        self.t_fusion_1 = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            nn.GELU()
        )

        self.fused_sh_fusion_1 = nn.Sequential(nn.Linear(dim * 2, dim),
                                               nn.LayerNorm(dim),
                                               nn.GELU())
        self.fused_sh_fusion_2 = nn.Sequential(nn.Linear(dim * 2, dim),
                                               nn.LayerNorm(dim),
                                               nn.GELU())
        self.fused_t = nn.Sequential(nn.Linear(dim * 2, dim),
                                     nn.LayerNorm(dim),
                                     nn.GELU())

        self.ca_s2t_v2vsp = CASTBlock(
            dim=dim, num_heads=num_heads, mode='s2t', mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop,
            attn_drop=attn_drop, drop_path=drop_path, norm_layer=norm_layer, act_layer=act_layer
        )
        self.ca_s2t_i2isp = CASTBlock(
            dim=dim, num_heads=num_heads, mode='s2t', mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop,
            attn_drop=attn_drop, drop_path=drop_path, norm_layer=norm_layer, act_layer=act_layer
        )
        # self.ca_s2t_f2shv = CASTBlock(
        #     dim=dim, num_heads=num_heads, mode='s2t', mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop,
        #     attn_drop=attn_drop, drop_path=drop_path, norm_layer=norm_layer, act_layer=act_layer
        # )
        # self.ca_s2t_f2shi = CASTBlock(
        #     dim=dim, num_heads=num_heads, mode='s2t', mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop,
        #     attn_drop=attn_drop, drop_path=drop_path, norm_layer=norm_layer, act_layer=act_layer
        # )
        self.ca_s2t_vi2sh = CASTBlockDoubleToken(
            dim=dim, num_heads=num_heads, mode='s2t', mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop,
            attn_drop=attn_drop, drop_path=drop_path, norm_layer=norm_layer, act_layer=act_layer
        )
        self.ca_t2s_vi2sh = CASTBlockDoubleToken(
            dim=dim, num_heads=num_heads, mode='t2s', mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop,
            attn_drop=attn_drop, drop_path=drop_path, norm_layer=norm_layer, act_layer=act_layer
        )
        self.ca_t2t_vi2sh = CASTBlockDoubleToken(
            dim=dim, num_heads=num_heads, mode='t2t', mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop,
            attn_drop=attn_drop, drop_path=drop_path, norm_layer=norm_layer, act_layer=act_layer
        )

        self.ca_t2s_vsp2v = CASTBlock(
            dim=dim, num_heads=num_heads, mode='t2s', mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop,
            attn_drop=attn_drop, drop_path=drop_path, norm_layer=norm_layer, act_layer=act_layer
        )
        self.ca_t2s_isp2i = CASTBlock(
            dim=dim, num_heads=num_heads, mode='t2s', mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop,
            attn_drop=attn_drop, drop_path=drop_path, norm_layer=norm_layer, act_layer=act_layer
        )

        self.ca_t2s_vsh2i = CASTBlock(
            dim=dim, num_heads=num_heads, mode='t2s', mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop,
            attn_drop=attn_drop, drop_path=drop_path, norm_layer=norm_layer, act_layer=act_layer
        )
        self.ca_t2s_ish2v = CASTBlock(
            dim=dim, num_heads=num_heads, mode='t2s', mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop,
            attn_drop=attn_drop, drop_path=drop_path, norm_layer=norm_layer, act_layer=act_layer
        )
        self.ca_t2t_f2v = CASTBlock(
            dim=dim, num_heads=num_heads, mode='t2t', mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop,
            attn_drop=attn_drop, drop_path=drop_path, norm_layer=norm_layer, act_layer=act_layer
        )
        self.ca_t2t_f2i = CASTBlock(
            dim=dim, num_heads=num_heads, mode='t2t', mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop,
            attn_drop=attn_drop, drop_path=drop_path, norm_layer=norm_layer, act_layer=act_layer
        )

    def forward(self, x_v, x_i, lens_z):
        fused_t = torch.cat([x_v[:, :lens_z, :], x_i[:, :lens_z, :]], dim=2)
        fused_t = self.t_fusion(fused_t)  # [B, 64, C]

        z_v_sp = self.ca_s2t_v2vsp(torch.cat([x_v[:, :lens_z, :], x_v[:, lens_z:, :]], dim=1))[:, :lens_z, :]
        z_i_sp = self.ca_s2t_i2isp(torch.cat([x_i[:, :lens_z, :], x_i[:, lens_z:, :]], dim=1))[:, :lens_z, :]

        fused_t = self.ca_s2t_vi2sh(torch.cat([fused_t, x_v[:, lens_z:, :], x_i[:, lens_z:, :]], dim=1))[:, :lens_z, :]
        # fused_t_1 = self.ca_s2t_f2shv(torch.cat([fused_t, x_i[:, lens_z:, :]], dim=1))[:, :lens_z, :]
        # fused_t_2 = self.ca_s2t_f2shi(torch.cat([fused_t, x_v[:, lens_z:, :]], dim=1))[:, :lens_z, :]
        # fused_t = self.t_fusion_1(torch.cat([fused_t_1, fused_t_2], dim=2))

        x_sh = self.ca_t2s_vi2sh(torch.cat([fused_t, x_v[:, lens_z:, :], x_i[:, lens_z:, :]], dim=1))[:, lens_z:, :]
        # x_v_sh = self.ca_t2s_vsh2i(torch.cat([fused_t, x_v[:, lens_z:, :]], dim=1))[:, lens_z:, :]
        # x_i_sh = self.ca_t2s_ish2v(torch.cat([fused_t, x_i[:, lens_z:, :]], dim=1))[:, lens_z:, :]
        # x_sh = self.fused_sh_fusion_2(torch.cat([x_v_sh, x_i_sh], dim=2))

        x_i[:, lens_z:, :] = self.ca_t2s_isp2i(torch.cat([z_v_sp, x_i[:, lens_z:, :]], dim=1))[:, lens_z:, :]
        x_v[:, lens_z:, :] = self.ca_t2s_vsp2v(torch.cat([z_i_sp, x_v[:, lens_z:, :]], dim=1))[:, lens_z:, :]

        t_sh = self.ca_t2t_vi2sh(torch.cat([fused_t, x_i[:, :lens_z, :], x_v[:, :lens_z, :]], dim=1))[:, lens_z:, :]
        x_i[:, :lens_z, :] = self.ca_t2t_f2i(torch.cat([x_i[:, :lens_z, :], fused_t], dim=1))[:, :lens_z, :]
        x_v[:, :lens_z, :] = self.ca_t2t_f2v(torch.cat([x_v[:, :lens_z, :], fused_t], dim=1))[:, :lens_z, :]

        # Next two lines are for abolotion study and we can remove them which will makes model full model.
        # x_sh = self.fused_sh_fusion_1(torch.cat([x_v[:, lens_z:, :], x_i[:, lens_z:, :]], dim=2))
        # t_sh = self.fused_sh_fusion_2(torch.cat([x_v[:, :lens_z, :], x_i[:, :lens_z, :]], dim=2))

        x_sh = torch.cat([t_sh, x_sh], dim=1)

        return x_v, x_i, x_sh
