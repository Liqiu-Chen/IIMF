import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_

from lib.models.layers.rpe import generate_2d_concatenated_self_attention_relative_positional_encoding_index


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.,
                 rpe=False, z_size=7, x_size=14):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.rpe =rpe
        if self.rpe:
            relative_position_index = \
                generate_2d_concatenated_self_attention_relative_positional_encoding_index([z_size, z_size],
                                                                                           [x_size, x_size])
            self.register_buffer("relative_position_index", relative_position_index)
            # define a parameter table of relative position bias
            self.relative_position_bias_table = nn.Parameter(torch.empty((num_heads,
                                                                          relative_position_index.max() + 1)))
            trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x, mask=None, return_attention=False):
        # x: B, N, C
        # mask: [B, N, ] torch.bool
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if self.rpe:
            relative_position_bias = self.relative_position_bias_table[:, self.relative_position_index].unsqueeze(0)
            attn += relative_position_bias

        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(1).unsqueeze(2), float('-inf'),)

        split_attn = False
        len_t = 49
        if split_attn:
            attn_t = attn[..., :len_t].softmax(dim=-1)
            attn_s = attn[..., len_t:].softmax(dim=-1)
            attn = torch.cat([attn_t, attn_s], dim=-1)
        else:
            attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        if return_attention:
            return x, attn
        else:
            return x


class Attention_talking_head(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications to add Talking Heads Attention (https://arxiv.org/pdf/2003.02436v1.pdf)
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 rpe=True, z_size=7, x_size=14):
        super().__init__()

        self.num_heads = num_heads

        head_dim = dim // num_heads

        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(dim, dim)

        self.proj_l = nn.Linear(num_heads, num_heads)
        self.proj_w = nn.Linear(num_heads, num_heads)

        self.proj_drop = nn.Dropout(proj_drop)

        self.rpe = rpe
        if self.rpe:
            relative_position_index = \
                generate_2d_concatenated_self_attention_relative_positional_encoding_index([z_size, z_size],
                                                                                           [x_size, x_size])
            self.register_buffer("relative_position_index", relative_position_index)
            # define a parameter table of relative position bias
            self.relative_position_bias_table = nn.Parameter(torch.empty((num_heads,
                                                                          relative_position_index.max() + 1)))
            trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1))

        if self.rpe:
            relative_position_bias = self.relative_position_bias_table[:, self.relative_position_index].unsqueeze(0)
            attn += relative_position_bias

        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(1).unsqueeze(2),
                                    float('-inf'),)

        attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        attn = attn.softmax(dim=-1)

        attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Attention_st(nn.Module):
    def __init__(self, dim, mode, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.,
                 rpe=False, z_size=7, x_size=14):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5  # NOTE: Small scale for attention map normalization

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.mode = mode
        self.rpe =rpe
        if self.rpe:
            relative_position_index = \
                generate_2d_concatenated_self_attention_relative_positional_encoding_index([z_size, z_size],
                                                                                           [x_size, x_size])
            self.register_buffer("relative_position_index", relative_position_index)
            # define a parameter table of relative position bias
            self.relative_position_bias_table = nn.Parameter(torch.empty((num_heads,
                                                                          relative_position_index.max() + 1)))
            trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x, mask=None, return_attention=False):
        # x: B, N, C
        # mask: [B, N, ] torch.bool
        B, N, C = x.shape
        
        lens_z = 64  # Number of template tokens
        lens_x = 256  # Number of search region tokens
        if self.mode == 's2t':  # Search to template
            q = x[:, :lens_z]  # B, lens_z, C
            k = x[:, lens_z:]  # B, lens_x, C
            v = x[:, lens_z:]  # B, lens_x, C
        elif self.mode == 't2s':  # Template to search
            q = x[:, lens_z:]  # B, lens_x, C
            k = x[:, :lens_z]  # B, lens_z, C
            v = x[:, :lens_z]  # B, lens_z, C
        elif self.mode == 't2t':  # Template to template
            q = x[:, :lens_z]  # B, lens_z, C
            k = x[:, lens_z:]  # B, lens_z, C
            v = x[:, lens_z:]  # B, lens_z, C
        elif self.mode == 's2s':  # Search to search
            q = x[:, :lens_x]  # B, lens_x, C
            k = x[:, lens_x:]  # B, lens_x, C
            v = x[:, lens_x:]  # B, lens_x, C
        
        attn = (q @ k.transpose(-2, -1)) * self.scale  # B, lens_z, lens_x; B, lens_x, lens_z

        if self.rpe:
            relative_position_bias = self.relative_position_bias_table[:, self.relative_position_index].unsqueeze(0)
            attn += relative_position_bias

        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(1).unsqueeze(2), float('-inf'),)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = attn @ v  # B, lens_z/x, C
        x = x.transpose(1, 2)  # B, C, lens_z/x
        x = x.reshape(B, -1, C)  # B, lens_z/x, C; NOTE: Rearrange channels, marginal improvement
        x = self.proj(x)
        x = self.proj_drop(x)

        if self.mode == 's2t':
            x = torch.cat([x, k], dim=1)
        elif self.mode == 't2s':
            x = torch.cat([k, x], dim=1)
        elif self.mode == 't2t':
            x = torch.cat([x, k], dim=1)
        elif self.mode == 's2s':
            x = torch.cat([x, k], dim=1)

        if return_attention:
            return x, attn
        else:
            return x


class AttentionDouble(nn.Module):
    def __init__(self, dim, mode, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.,
                 rpe=False, z_size=7, x_size=14):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5  # NOTE: Small scale for attention map normalization

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop_2 = nn.Dropout(attn_drop)
        self.proj_2 = nn.Linear(dim, dim)
        self.proj_3 = nn.Linear(dim * 2, dim)
        self.proj_4 = nn.Linear(dim * 2, dim)
        self.proj_drop_2 = nn.Dropout(proj_drop)
        self.mode = mode
        self.rpe = rpe
        if self.rpe:
            relative_position_index = \
                generate_2d_concatenated_self_attention_relative_positional_encoding_index([z_size, z_size],
                                                                                           [x_size, x_size])
            self.register_buffer("relative_position_index", relative_position_index)
            # define a parameter table of relative position bias
            self.relative_position_bias_table = nn.Parameter(torch.empty((num_heads,
                                                                          relative_position_index.max() + 1)))
            trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x, mask=None, return_attention=False):
        # x: B, N, C
        # mask: [B, N, ] torch.bool
        B, N, C = x.shape

        lens_z = 64  # Number of template tokens
        lens_x = 256  # Number of search region tokens
        if self.mode == 's2t':  # Search to template
            q = x[:, :lens_z]  # B, lens_z, C
            k_1 = x[:, lens_z:lens_z+lens_x]  # B, lens_x, C
            v_1 = x[:, lens_z:lens_z+lens_x]  # B, lens_x, C
            k_2 = x[:, -lens_x:]  # B, lens_x, C
            v_2 = x[:, -lens_x:]
        elif self.mode == 't2s':  # Template to search
            q_1 = x[:, lens_z:lens_z+lens_x]  # B, lens_x, C
            q_2 = x[:, -lens_x:]  # B, lens_x, C
            k = x[:, :lens_z]  # B, lens_z, C
            v = x[:, :lens_z]  # B, lens_z, C
        elif self.mode == 't2t':  # Template to template
            q_1 = x[:, lens_z:lens_z*2]  # B, lens_x, C
            q_2 = x[:, -lens_z:]  # B, lens_x, C
            k = x[:, :lens_z]  # B, lens_z, C
            v = x[:, :lens_z]  # B, lens_z, C
        elif self.mode == 's2s':  # Search to search
            q = x[:, :lens_x]  # B, lens_x, C
            k = x[:, lens_x:]  # B, lens_x, C
            v = x[:, lens_x:]  # B, lens_x, C

        if self.mode == 's2t':
            attn_1 = (q @ k_1.transpose(-2, -1)) * self.scale  # B, lens_z, lens_x; B, lens_x, lens_z
            attn_2 = (q @ k_2.transpose(-2, -1)) * self.scale  # B, lens_z, lens_x; B, lens_x, lens_z

        else:
            attn_1 = (q_1 @ k.transpose(-2, -1)) * self.scale  # B, lens_z, lens_x; B, lens_x, lens_z
            attn_2 = (q_2 @ k.transpose(-2, -1)) * self.scale  # B, lens_z, lens_x; B, lens_x, lens_z

        if self.rpe:
            relative_position_bias = self.relative_position_bias_table[:, self.relative_position_index].unsqueeze(0)
            attn_1 += relative_position_bias
            attn_2 += relative_position_bias

        if mask is not None:
            attn_1 = attn_1.masked_fill(mask.unsqueeze(1).unsqueeze(2), float('-inf'), )
            attn_2 = attn_2.masked_fill(mask.unsqueeze(1).unsqueeze(2), float('-inf'), )

        _, _, lens_attn = attn_1.size()
        attn = torch.cat([attn_1, attn_2], dim=-1)
        attn = attn.softmax(dim=-1)
        attn_1 = attn[:, :, :lens_attn]
        attn_2 = attn[:, :, lens_attn:]
        attn_1 = self.attn_drop(attn_1)
        attn_2 = self.attn_drop_2(attn_2)

        if self.mode == 's2t':
            x_1 = attn_1 @ v_1  # B, lens_z/x, C
            x_2 = attn_2 @ v_2

        else:
            x_1 = attn_1 @ v  # B, lens_z/x, C
            x_2 = attn_2 @ v
            k_1 = k
            k_2 = k
        x_1 = x_1.transpose(1, 2)  # B, C, lens_z/x
        x_1 = x_1.reshape(B, -1, C)  # B, lens_z/x, C; NOTE: Rearrange channels, marginal improvement
        x_1 = self.proj(x_1)
        x_1 = self.proj_drop(x_1)

        x_2 = x_2.transpose(1, 2)  # B, C, lens_z/x
        x_2 = x_2.reshape(B, -1, C)  # B, lens_z/x, C; NOTE: Rearrange channels, marginal improvement
        x_2 = self.proj_2(x_2)
        x_2 = self.proj_drop_2(x_2)

        x = self.proj_3(torch.cat([x_1, x_2], dim=2))
        k = self.proj_4(torch.cat([k_1, k_2], dim=2))
        if self.mode == 's2t':
            x = torch.cat([x, k], dim=1)
        elif self.mode == 't2s':
            x = torch.cat([k, x], dim=1)
        elif self.mode == 't2t':
            x = torch.cat([k, x], dim=1)
        elif self.mode == 's2s':
            x = torch.cat([x, k], dim=1)

        if return_attention:
            return x, attn_1
        else:
            return x
