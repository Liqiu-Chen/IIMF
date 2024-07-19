import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.models.layers.attn_blocks import CASTBlock


class InformationEnhance(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()

        self.ca_s2t_v2sh = CASTBlock(
            dim=dim, num_heads=num_heads, mode='s2t', mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop,
            attn_drop=attn_drop, drop_path=drop_path, norm_layer=norm_layer, act_layer=act_layer
        )
        self.ca_s2t_i2sh = CASTBlock(
            dim=dim, num_heads=num_heads, mode='s2t', mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop,
            attn_drop=attn_drop, drop_path=drop_path, norm_layer=norm_layer, act_layer=act_layer
        )

        self.ca_s2t_v2sp = CASTBlock(
            dim=dim, num_heads=num_heads, mode='s2t', mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop,
            attn_drop=attn_drop, drop_path=drop_path, norm_layer=norm_layer, act_layer=act_layer
        )
        self.ca_s2t_i2sp = CASTBlock(
            dim=dim, num_heads=num_heads, mode='s2t', mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop,
            attn_drop=attn_drop, drop_path=drop_path, norm_layer=norm_layer, act_layer=act_layer
        )

        self.ca_t2s_f2m = CASTBlock(
            dim=dim, num_heads=num_heads, mode='t2s', mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop,
            attn_drop=attn_drop, drop_path=drop_path, norm_layer=norm_layer, act_layer=act_layer
        )

        self.v_sh_fuse = nn.Sequential(
            nn.Linear(dim * 2, dim),
            norm_layer(dim),
            act_layer()
        )
        self.x_v_fuse = nn.Sequential(
            nn.Linear(dim * 2, dim),
            norm_layer(dim),
            act_layer()
        )
        self.i_sh_fuse = nn.Sequential(
            nn.Linear(dim * 2, dim),
            norm_layer(dim),
            act_layer()
        )
        self.x_i_fuse = nn.Sequential(
            nn.Linear(dim * 2, dim),
            norm_layer(dim),
            act_layer()
        )

        self.x_fuse = nn.Sequential(
            nn.Linear(dim * 2, dim),
            norm_layer(dim),
            act_layer()
        )

    def forward(self, x_v_p, x_v_n, x_i_p, x_i_n, x_m, lens_z):

        z_v_p = x_v_p[:, :lens_z, :]
        x_v_p = x_v_p[:, lens_z:, :]

        z_i_p = x_i_p[:, :lens_z, :]
        x_i_p = x_i_p[:, lens_z:, :]

        z_v_n = x_v_n[:, :lens_z, :]
        x_v_n = x_v_n[:, lens_z:, :]

        z_i_n = x_i_n[:, :lens_z, :]
        x_i_n = x_i_n[:, lens_z:, :]

        x_v_p_sh = self.ca_s2t_v2sh(torch.cat([z_v_p, x_v_p], dim=1))[:, :lens_z, :]
        x_v_n_sh = self.ca_s2t_v2sh(torch.cat([z_v_n, x_v_n], dim=1))[:, :lens_z, :]
        x_v_p_sp = self.ca_s2t_v2sp(torch.cat([z_v_p, x_v_p], dim=1))[:, :lens_z, :]

        x_v_sh = self.v_sh_fuse(torch.cat([x_v_p_sh, x_v_n_sh], dim=2))
        x_v_m = self.x_v_fuse(torch.cat([x_v_p_sp, x_v_sh], dim=2))

        x_i_p_sh = self.ca_s2t_i2sh(torch.cat([z_i_p, x_i_p], dim=1))[:, :lens_z, :]
        x_i_n_sh = self.ca_s2t_i2sh(torch.cat([z_i_n, x_i_n], dim=1))[:, :lens_z, :]
        x_i_p_sp = self.ca_s2t_i2sp(torch.cat([z_i_p, x_i_p], dim=1))[:, :lens_z, :]

        x_i_sh = self.i_sh_fuse(torch.cat([x_i_n_sh, x_i_p_sh], dim=2))
        x_i_m = self.x_i_fuse(torch.cat([x_i_p_sp, x_i_sh], dim=2))

        x = self.x_fuse(torch.cat([x_v_m, x_i_m], dim=2))
        x_m[:, lens_z:, :] = self.ca_t2s_f2m(torch.cat([x, x_m[:, lens_z:, :]], dim=1))[:, lens_z:, :]

        return x_m
