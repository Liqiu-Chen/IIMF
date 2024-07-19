from functools import partial
from turtle import forward

import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.models.layers.attn_blocks import CASTBlock, CASTBlockDoubleToken


class TBSILayer(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.fused_t = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            nn.GELU()
        )
        # self.fused_t_vi = nn.Sequential(
        #     nn.Linear(dim * 2, dim),
        #     nn.LayerNorm(dim),
        #     nn.GELU()
        # )
        # self.fused_t_1 = nn.Sequential(
        #     nn.Linear(dim * 2, dim),
        #     nn.LayerNorm(dim),
        #     nn.GELU()
        # )
        # self.fused_t_2 = nn.Sequential(
        #     nn.Linear(dim * 2, dim),
        #     nn.LayerNorm(dim),
        #     nn.GELU()
        # )
        # self.fused_t_3 = nn.Sequential(
        #     nn.Linear(dim * 2, dim),
        #     nn.LayerNorm(dim),
        #     nn.GELU()
        # )
        # self.v_fused_sh = nn.Sequential(
        #     nn.Linear(dim * 2, dim)
        # )
        # self.v_fused_sh_norm = nn.Sequential(
        #     nn.LayerNorm(dim),
        #     nn.GELU()
        # )
        # self.i_fused_sh = nn.Sequential(
        #     nn.Linear(dim * 2, dim)
        # )
        # self.i_fused_sh_norm = nn.Sequential(
        #     nn.LayerNorm(dim),
        #     nn.GELU()
        # )

        # self.ca_s2t_v2sh = CASTBlock(
        #     dim=dim, num_heads=num_heads, mode='s2t', mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop,
        #     attn_drop=attn_drop, drop_path=drop_path, norm_layer=norm_layer, act_layer=act_layer
        # )
        # self.ca_s2t_i2sh = CASTBlock(
        #     dim=dim, num_heads=num_heads, mode='s2t', mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop,
        #     attn_drop=attn_drop, drop_path=drop_path, norm_layer=norm_layer, act_layer=act_layer
        # )
        self.ca_s2t_vi2sh = CASTBlockDoubleToken(dim=dim, num_heads=num_heads, mode='s2t', mlp_ratio=mlp_ratio,
                                                 qkv_bias=qkv_bias, drop=drop, attn_drop=attn_drop, drop_path=drop_path,
                                                 norm_layer=norm_layer, act_layer=act_layer)

        self.ca_t2s_f2sh = CASTBlock(
            dim=dim, num_heads=num_heads, mode='t2s', mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop,
            attn_drop=attn_drop, drop_path=drop_path, norm_layer=norm_layer, act_layer=act_layer
        )
        # self.ca_t2t_f2v = CASTBlock(
        #     dim=dim, num_heads=num_heads, mode='t2t', mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop,
        #     attn_drop=attn_drop, drop_path=drop_path, norm_layer=norm_layer, act_layer=act_layer
        # )
        # self.ca_t2t_f2i = CASTBlock(
        #     dim=dim, num_heads=num_heads, mode='t2t', mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop,
        #     attn_drop=attn_drop, drop_path=drop_path, norm_layer=norm_layer, act_layer=act_layer
        # )
        self.ca_t2t_f2f = CASTBlock(
            dim=dim, num_heads=num_heads, mode='t2t', mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop,
            attn_drop=attn_drop, drop_path=drop_path, norm_layer=norm_layer, act_layer=act_layer
        )
        # self.ca_t2t_sh2v = CASTBlock(
        #     dim=dim, num_heads=num_heads, mode='t2t', mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop,
        #     attn_drop=attn_drop, drop_path=drop_path, norm_layer=norm_layer, act_layer=act_layer
        # )
        # self.ca_t2t_sh2i = CASTBlock(
        #     dim=dim, num_heads=num_heads, mode='t2t', mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop,
        #     attn_drop=attn_drop, drop_path=drop_path, norm_layer=norm_layer, act_layer=act_layer
        # )
        # self.ca_s2t_v2sp = CASTBlock(
        #     dim=dim, num_heads=num_heads, mode='s2t', mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop,
        #     attn_drop=attn_drop, drop_path=drop_path, norm_layer=norm_layer, act_layer=act_layer
        # )
        # self.ca_s2t_i2sp = CASTBlock(
        #     dim=dim, num_heads=num_heads, mode='s2t', mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop,
        #     attn_drop=attn_drop, drop_path=drop_path, norm_layer=norm_layer, act_layer=act_layer
        # )
        # self.ca_t2s_sp2i = CASTBlock(
        #     dim=dim, num_heads=num_heads, mode='t2s', mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop,
        #     attn_drop=attn_drop, drop_path=drop_path, norm_layer=norm_layer, act_layer=act_layer
        # )
        # self.ca_t2s_sp2v = CASTBlock(
        #     dim=dim, num_heads=num_heads, mode='t2s', mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop,
        #     attn_drop=attn_drop, drop_path=drop_path, norm_layer=norm_layer, act_layer=act_layer
        # )

    def forward(self, x_v, x_i, x_sh, lens_z):
        # x_v: [B, N, C], N = 320
        # x_i: [B, N, C], N = 320
        fused_t = torch.cat([x_v[:, :lens_z, :], x_i[:, :lens_z, :]], dim=2)
        fused_t = self.fused_t(fused_t)  # [B, 64, C]

        # temp_z_v_sp = self.ca_s2t_v2sp(torch.cat([x_v[:, :lens_z, :], x_v[:, lens_z:, :]], dim=1))[:, :lens_z, :]
        # temp_z_i_sp = self.ca_s2t_i2sp(torch.cat([x_i[:, :lens_z, :], x_i[:, lens_z:, :]], dim=1))[:, :lens_z, :]

        # temp_z_v_sp = self.v_fused_sh_norm(self.v_fused_sh(torch.cat([temp_z_v_sp, temp_t_sh], dim=2)) + temp_z_v_sp)
        # temp_z_i_sp = self.i_fused_sh_norm(self.i_fused_sh(torch.cat([temp_z_i_sp, temp_t_sh], dim=2)) + temp_z_i_sp)

        # temp_t_v_sh = self.ca_s2t_v2sh(torch.cat([fused_t, x_v[:, lens_z:, :]], dim=1))[:, :lens_z, :]
        # temp_t_i_sh = self.ca_s2t_i2sh(torch.cat([fused_t, x_i[:, lens_z:, :]], dim=1))[:, :lens_z, :]
        # temp_t_sh = self.fused_t_1(torch.cat([temp_t_i_sh, temp_t_v_sh], dim=2))
        # temp_x_v = self.ca_t2s_sp2i(torch.cat([temp_z_i_sp, x_v[:, lens_z:, :]], dim=1))[:, lens_z:, :]
        # temp_x_i = self.ca_t2s_sp2v(torch.cat([temp_z_v_sp, x_i[:, lens_z:, :]], dim=1))[:, lens_z:, :]

        temp_t_sh = self.ca_s2t_vi2sh(torch.cat([fused_t, x_v[:, lens_z:, :], x_i[:, lens_z:, :]], dim=1))[:, :lens_z, :]

        temp_x_sh = self.ca_t2s_f2sh(torch.cat([temp_t_sh, x_sh[:, lens_z:, :]], dim=1))[:, lens_z:, :]

        # fused_t_v = self.ca_t2t_sh2v(torch.cat([x_v[:, :lens_z, :], x_sh[:, :lens_z, :]], dim=1))[:, :lens_z, :]
        # fused_t_i = self.ca_t2t_sh2i(torch.cat([x_i[:, :lens_z, :], x_sh[:, :lens_z, :]], dim=1))[:, :lens_z, :]

        # fused_t_sp = self.fused_t_vi(torch.cat([temp_z_v_sp, temp_z_i_sp], dim=2))

        # x_v[:, lens_z:, :] = temp_x_v
        # x_i[:, lens_z:, :] = temp_x_i
        x_sh[:, lens_z:, :] = temp_x_sh

        # x_v[:, :lens_z, :] = self.ca_t2t_f2v(torch.cat([x_v[:, :lens_z, :], fused_t_sp], dim=1))[:, :lens_z, :]
        # x_i[:, :lens_z, :] = self.ca_t2t_f2i(torch.cat([x_i[:, :lens_z, :], fused_t_sp], dim=1))[:, :lens_z, :]
        x_sh[:, :lens_z, :] = self.ca_t2t_f2f(torch.cat([x_sh[:, :lens_z, :], temp_t_sh], dim=1))[:, :lens_z, :]

        return x_v, x_i, x_sh
