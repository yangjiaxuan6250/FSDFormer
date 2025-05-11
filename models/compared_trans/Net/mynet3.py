from UDL.Basis.module import PatchMergeModule
from models.base_model import DerainModel
from torch import optim
from UDL.Basis.criterion_metrics import SetCriterion
from .losses import CharbonnierLoss, EdgeLoss, fftLoss

## Restormer: Efficient Transformer for High-Resolution Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
## https://arxiv.org/abs/2111.09881

from einops import rearrange, einops
import torch
import torch.nn as nn
import torch.nn.functional as F

import numbers

from einops import rearrange


##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class Partial_conv3(nn.Module):

    def __init__(self, dim, n_div=4, forward='split_cat'):
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)

        if forward == 'slicing':
            self.forward = self.forward_slicing
        elif forward == 'split_cat':
            self.forward = self.forward_split_cat
        else:
            raise NotImplementedError

    def forward_slicing(self, x):
        # only for inference
        x = x.clone()   # !!! Keep the original input intact for the residual connection later
        x[:, :self.dim_conv3, :, :] = self.partial_conv3(x[:, :self.dim_conv3, :, :])

        return x

    def forward_split_cat(self, x):
        # for training/inference
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        x = torch.cat((x1, x2), 1)

        return x

def block_images_einops(x, patch_size):  #n, h, w, c
  """Image to patches."""
  batch, height, width, channels = x.shape
  grid_height = height // patch_size[0]
  grid_width = width // patch_size[1]
  x = einops.rearrange(
      x, "n (gh fh) (gw fw) c -> n (gh gw) (fh fw) c",
      gh=grid_height, gw=grid_width, fh=patch_size[0], fw=patch_size[1])
  return x

def unblock_images_einops(x, grid_size, patch_size):
  """patches to images."""
  x = einops.rearrange(
      x, "n (gh gw) (fh fw) c -> n (gh fh) (gw fw) c",
      gh=grid_size[0], gw=grid_size[1], fh=patch_size[0], fw=patch_size[1])
  return x


class DFFN(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(DFFN, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)
        self.conv1 = Partial_conv3(dim)
        self.conv2 = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.patch_size = 8
        # self.conv3 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1,
        #                        groups=hidden_features, bias=bias)
        #
        # self.dim = dim
        # # self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        #
        #
        #
        self.fft = nn.Parameter(torch.ones((hidden_features * 2, 1, 1, self.patch_size, self.patch_size // 2 + 1)))

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.conv2(self.conv1(x))
        x_patch = rearrange(x, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,
                            patch2=self.patch_size)
        x_patch_fft = torch.fft.rfft2(x_patch.float())
        x_patch_fft = x_patch_fft * self.fft
        x_patch = torch.fft.irfft2(x_patch_fft, s=(self.patch_size, self.patch_size))
        x = rearrange(x_patch, 'b c h w patch1 patch2 -> b c (h patch1) (w patch2)', patch1=self.patch_size,
                      patch2=self.patch_size)

        x1, x2 = x.chunk(2, dim=1)

        # x = F.gelu(self.conv3(x1)) * x2
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

class FFN(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FFN, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)
        self.conv1 = Partial_conv3(dim)
        self.conv2 = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.conv2(self.conv1(x))
        x1, x2 = x.chunk(2, dim=1)
        x2_fft = torch.fft.fft2(x2, norm='backward')
        x = x1 * x2_fft
        x = torch.fft.ifft2(x, dim=(-2, -1), norm='backward')
        x = torch.abs(x)
        x = x * F.gelu(x2)
        x = self.project_out(x)
        return x


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        # self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        # self.project_in = CheapLP(dim, hidden_features * 2)
        #
        # self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
        #                         groups=hidden_features * 2, bias=bias)
        self.conv1 = Partial_conv3(dim)
        self.conv2 = nn.Conv2d(dim, hidden_features *2, kernel_size=1, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)
        # self.project_out = CheapLP(hidden_features, dim)

    def forward(self, x):
        b, c, h, w = x.shape
        x = self.conv1(x)
        x1, x2 = self.conv2(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        # x = rearrange(x, 'b c h w -> b c (h w)', h=h, w=w)
        # x = 0.5 * x + 0.5 * x.mean(dim=2, keepdim=True)
        # x = rearrange(x, 'b c (h w) -> b c h w', h=h, w=w)
        x = self.project_out(x)
        return x

class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=False, norm=False, relu=True, transpose=False,
                 channel_shuffle_g=0, norm_method=nn.BatchNorm2d, groups=1):
        super(BasicConv, self).__init__()
        self.channel_shuffle_g = channel_shuffle_g
        self.norm = norm
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 - 1
            layers.append(
                nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias,
                                   groups=groups))
        else:
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias,
                          groups=groups))
        if norm:
            layers.append(norm_method(out_channel))
        elif relu:
            layers.append(nn.ReLU(inplace=True))

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)

class Partial_conv5(nn.Module):

    def __init__(self, dim, n_div=4, forward='split_cat'):
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 5, 1, 2, bias=False)

        if forward == 'slicing':
            self.forward = self.forward_slicing
        elif forward == 'split_cat':
            self.forward = self.forward_split_cat
        else:
            raise NotImplementedError

    def forward_slicing(self, x):
        # only for inference
        x = x.clone()   # !!! Keep the original input intact for the residual connection later
        x[:, :self.dim_conv3, :, :] = self.partial_conv3(x[:, :self.dim_conv3, :, :])

        return x

    def forward_split_cat(self, x):
        # for training/inference
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        x = torch.cat((x1, x2), 1)

        return x

# class HighMixer(nn.Module):
#     def __init__(self, dim, kernel_size=3, stride=1, padding=1,
#                  **kwargs, ):
#         super().__init__()
#
#         # self.cnn_in = cnn_in = dim // 2
#         # self.pool_in = pool_in = dim // 2
#         #
#         # self.cnn_dim = cnn_dim = cnn_in * 2
#         # self.pool_dim = pool_dim = pool_in * 2
#
#         self.proj1 = Partial_conv3(dim)
#         self.conv1 = nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0, bias=False)
#
#         self.mid_gelu1 = nn.GELU()
#
#         self.proj2 = nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0)
#         self.mid_gelu2 = nn.GELU()
#
#         self.proj3 = Partial_conv5(dim)
#         self.conv2 = nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0, bias=False)
#         self.mid_gelu3 = nn.GELU()
#     def forward(self, x):
#         # B, C H, W
#         cx = x
#         cx = self.proj1(cx)
#         cx = self.conv1(cx)
#         cx = self.mid_gelu1(cx)
#
#         rx = x
#         rx = self.proj3(rx)
#         rx = self.conv2(rx)
#         rx = self.mid_gelu3(rx)
#
#         px = x
#         px = self.proj2(px)
#         px = self.mid_gelu2(px)
#
#
#         hx = torch.cat((cx, px, rx), dim=1)
#         return hx
#
# class Attention(nn.Module):
#     def __init__(self, dim, num_heads, bias, patch_size, thres=0.1, suppress=0.75, pool_size=2, cut_num=4, cut_low=2):
#         super(Attention, self).__init__()
#         self.__class__.__name__ = 'XCTEB'
#         self.num_heads = num_heads
#         self.temperature1 = nn.Parameter(torch.ones(num_heads, 1, 1))
#         self.temperature2 = nn.Parameter(torch.ones(num_heads, 1, 1))
#         self.max_relative_position = 2
#         per_dim = dim // cut_num
#         self.atten_dim = atten_dim = cut_low * per_dim
#         self.high_dim = high_dim = (cut_num - cut_low) * per_dim
#         self.high_mixer = HighMixer(high_dim)
#         self.conv_fuse = nn.Conv2d(atten_dim + high_dim * 3, atten_dim + high_dim * 3, kernel_size=3, stride=1,
#                                    padding=1,
#                                    bias=False, groups=atten_dim + high_dim * 3)
#         self.pool_size = pool_size
#         # self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
#         # self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
#         self.qkv = Partial_conv3(atten_dim)
#         self.qkv_conv = nn.Conv2d(atten_dim, atten_dim * 3, kernel_size=1, bias=bias)
#         self.project_out = nn.Conv2d(atten_dim+high_dim*3, dim, kernel_size=1, stride=1, padding=0)
#         self.ap_q = nn.AdaptiveAvgPool2d(1)
#         self.ap_k = nn.AdaptiveAvgPool2d(1)
#
#         self.conv = nn.Conv2d(atten_dim*2, atten_dim, kernel_size=1, bias=bias)
#
#         self.time_weighting1 = nn.Parameter(torch.ones(self.num_heads, (atten_dim // self.num_heads)//2+1, (atten_dim // self.num_heads)//2+1))
#         self.time_weighting2 = nn.Parameter(torch.ones(self.num_heads, (atten_dim // self.num_heads), (atten_dim // self.num_heads)))
#     def forward(self, x):
#         # q, k, v = self.qkv(x, chunk=3)
#         # qkv = self.qkv_cheap(torch.cat([q, k, v], dim=1))
#         b, c, h, w = x.shape
#         hx = x[:, :self.high_dim, :, :].contiguous()
#         hx = self.high_mixer(hx)
#
#         lx = x[:, self.high_dim:, :, :].contiguous()
#         qkv = self.qkv_conv(self.qkv(lx))
#         q, k, v = qkv.chunk(3, dim=1)
#         # q = self.ap_q(q)
#         # k = self.ap_k(k)
#         # v = self.ap_v(v)
#
#         # v = torch.fft.rfft(v, dim=1)
#         q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
#         k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
#         v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
#         q1 = torch.nn.functional.normalize(q, dim=-1)
#         k1 = torch.nn.functional.normalize(k, dim=-1)
#
#         attn1 = (q1 @ k1.transpose(-2, -1)) * self.temperature1
#         attn1 = attn1.softmax(dim=-1) * self.time_weighting2
#
#         out1 = (attn1 @ v)
#         out1 = rearrange(out1, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
#         q = torch.fft.rfft(q, dim=2)
#         k = torch.fft.rfft(k, dim=2)
#         v = torch.fft.rfft(v, dim=2)
#         q = torch.nn.functional.normalize(q, dim=-1)
#         k = torch.nn.functional.normalize(k, dim=-1)
#         attn = (q @ k.transpose(-2, -1)) * self.temperature2
#         real = torch.real(attn)
#         imag = torch.imag(attn)
#         real = real.softmax(dim=-1)
#         imag = imag.softmax(dim=-1)
#         attn = torch.complex(real, imag)
#         # attn = attn.softmax(dim=-1) * self.time_weighting
#         # if self.suppress_status:
#         #     row_max = torch.amax(attn, dim=-1).unsqueeze(-1).expand(attn.shape)
#         #     row_mask = torch.le(attn, row_max * self.thres)
#         #     row_mask_after = torch.mul(attn, row_mask)
#         #     row_sum_thres = torch.sum(row_mask_after, dim=-1).unsqueeze(-1).expand(attn.shape)
#         #     ele_coefs = self.suppress * torch.div(attn, row_sum_thres + 0.000001)
#         #     attn = ~row_mask * attn + row_mask * ele_coefs * attn
#         # attn = torch.fft.irfft2(attn, dim=(-2, -1), s=(v.shape[-2], v.shape[-2]))
#         attn = attn * self.time_weighting1
#         # print(attn.shape)
#         lx = (attn @ v)
#         lx = torch.fft.irfft(lx, dim=2)
#
#         lx = rearrange(lx, 'b head c (h w) -> b (head c) h w', head=self.num_heads,h=h, w=w)
#         lx = torch.cat((lx, out1), dim=1)
#         lx = self.conv(lx)
#         out = torch.cat((hx, lx), dim=1)
#         out = out + self.conv_fuse(out)
#         out = self.project_out(out)
#
#         return out
class HighMixer(nn.Module):
    def __init__(self, dim, kernel_size=3, stride=1, padding=1,
                 **kwargs, ):
        super().__init__()

        # self.cnn_in = cnn_in = dim // 2
        # self.pool_in = pool_in = dim // 2
        #
        # self.cnn_dim = cnn_dim = cnn_in * 2
        # self.pool_dim = pool_dim = pool_in * 2

        self.proj1 = Partial_conv3(dim)
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0, bias=False)

        self.mid_gelu1 = nn.GELU()

        self.proj2 = nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0)
        self.mid_gelu2 = nn.GELU()

        self.proj3 = Partial_conv5(dim)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0, bias=False)
        self.mid_gelu3 = nn.GELU()
        self.fuse = nn.Conv2d(dim*3, dim, 1, 1, 0, bias=False)
    def forward(self, x):
        b, c, h, w = x.shape
        # B, C H, W
        cx = x
        cx = self.proj1(cx)
        cx = self.conv1(cx)
        cx = self.mid_gelu1(cx)

        rx = x
        rx = self.proj3(rx)
        rx = self.conv2(rx)
        rx = self.mid_gelu3(rx)

        px = x
        px = self.proj2(px)
        px = self.mid_gelu2(px)


        hx = torch.cat((cx, px, rx), dim=1)
        hx = self.fuse(hx)

        hx = hx + x
        return hx

class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias, patch_size, thres=0.1, suppress=0.75, pool_size=2, cut_num=4, cut_low=2):
        super(Attention, self).__init__()
        self.__class__.__name__ = 'XCTEB'
        self.num_heads = num_heads
        self.temperature1 = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.temperature2 = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.max_relative_position = 2
        per_dim = dim // cut_num
        self.atten_dim = atten_dim = cut_low * per_dim
        self.high_dim = high_dim = (cut_num - cut_low) * per_dim
        self.high_mixer = HighMixer(dim)
        self.pool_size = pool_size
        # self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        # self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.qkv = Partial_conv3(dim)
        self.qkv_conv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0)
        self.ap_q = nn.AdaptiveAvgPool2d(1)
        self.ap_k = nn.AdaptiveAvgPool2d(1)

        # self.conv = nn.Conv2d(dim*2, dim, kernel_size=1, bias=bias)

        self.time_weighting1 = nn.Parameter(torch.ones(self.num_heads, (dim // 2 // self.num_heads)//2+1, (dim // 2 // self.num_heads)//2+1))
        self.time_weighting2 = nn.Parameter(torch.ones(self.num_heads, (dim // 2 // self.num_heads), (dim // 2 // self.num_heads)))
    def forward(self, x):
        # q, k, v = self.qkv(x, chunk=3)
        # qkv = self.qkv_cheap(torch.cat([q, k, v], dim=1))
        b, c, h, w = x.shape
        hx = x
        hx = self.high_mixer(hx)


        qkv = self.qkv_conv(self.qkv(hx))
        q, k, v = qkv.chunk(3, dim=1)
        # q = self.ap_q(q)
        # k = self.ap_k(k)
        # v = self.ap_v(v)

        # v = torch.fft.rfft(v, dim=1)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q1, q = q.chunk(2, dim=2)
        k1, k = k.chunk(2, dim=2)
        v1, v = v.chunk(2, dim=2)
        q1 = torch.nn.functional.normalize(q, dim=-1)
        k1 = torch.nn.functional.normalize(k, dim=-1)

        attn1 = (q1 @ k1.transpose(-2, -1)) * self.temperature1
        attn1 = attn1.softmax(dim=-1) * self.time_weighting2

        out1 = (attn1 @ v)
        out1 = rearrange(out1, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        q = torch.fft.rfft(q, dim=2)
        k = torch.fft.rfft(k, dim=2)
        v = torch.fft.rfft(v, dim=2)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature2
        real = torch.real(attn)
        imag = torch.imag(attn)
        real = real.softmax(dim=-1)
        imag = imag.softmax(dim=-1)
        attn = torch.complex(real, imag)
        # attn = attn.softmax(dim=-1) * self.time_weighting
        # if self.suppress_status:
        #     row_max = torch.amax(attn, dim=-1).unsqueeze(-1).expand(attn.shape)
        #     row_mask = torch.le(attn, row_max * self.thres)
        #     row_mask_after = torch.mul(attn, row_mask)
        #     row_sum_thres = torch.sum(row_mask_after, dim=-1).unsqueeze(-1).expand(attn.shape)
        #     ele_coefs = self.suppress * torch.div(attn, row_sum_thres + 0.000001)
        #     attn = ~row_mask * attn + row_mask * ele_coefs * attn
        # attn = torch.fft.irfft2(attn, dim=(-2, -1), s=(v.shape[-2], v.shape[-2]))
        attn = attn * self.time_weighting1
        # print(attn.shape)
        lx = (attn @ v)
        lx = torch.fft.irfft(lx, dim=2)

        lx = rearrange(lx, 'b head c (h w) -> b (head c) h w', head=self.num_heads,h=h, w=w)
        lx = torch.cat((lx, out1), dim=1)
        # lx = self.conv(lx)
        out = self.project_out(lx)

        return out




##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type, patch_size=1, cut_num=4, cut_low=2):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias, patch_size, cut_num=cut_num, cut_low=cut_low)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = DFFN(dim, ffn_expansion_factor, bias)
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x

##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x


##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)

def get_residue(tensor , r_dim = 1):
    """
    return residue_channle (RGB)
    """
    # res_channel = []
    max_channel = torch.max(tensor, dim=r_dim, keepdim=True)  # keepdim
    min_channel = torch.min(tensor, dim=r_dim, keepdim=True)
    res_channel = max_channel[0] - min_channel[0]
    return res_channel

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class RB(nn.Module):
    def __init__(self, n_feats, nm='in'):
        super(RB, self).__init__()
        module_body = []
        for i in range(2):
            module_body.append(nn.Conv2d(n_feats, n_feats, kernel_size=3, stride=1, padding=1, bias=False))
            module_body.append(nn.ReLU())
        self.module_body = nn.Sequential(*module_body)
        self.relu = nn.ReLU()
        self.se = SELayer(n_feats, 1)

    def forward(self, x):
        res = self.module_body(x)
        res = self.se(res)
        res += x
        return res

class RIR(nn.Module):
    def __init__(self, n_feats, n_blocks, nm='in'):
        super(RIR, self).__init__()
        module_body = [
            RB(n_feats) for _ in range(n_blocks)
        ]
        module_body.append(nn.Conv2d(n_feats, n_feats, kernel_size=3, stride=1, padding=1, bias=False))
        self.module_body = nn.Sequential(*module_body)
        self.relu = nn.ReLU()

    def forward(self, x):
        res = self.module_body(x)
        res += x
        return self.relu(res)

class Prior_Sp(nn.Module):
    """ Channel attention module"""
    def __init__(self, dim, num_heads=2, bias=False):
        super(Prior_Sp, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qv = Partial_conv3(dim)
        self.qv_conv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.k = Partial_conv3(dim)
        self.k_conv = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)



    def forward(self, x, res):
        b, c, h, w = x.shape

        qv = self.qv_conv(self.qv(x))
        q, v = qv.chunk(2, dim=1)
        k = self.k_conv(self.k(res))

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

##########################################################################
##---------- Restormer -----------------------
class Priorformer(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=1, bias=False, LayerNorm_type='WithBias', patch_size=1, cut_num=4, cut_low=2):
        super(Priorformer, self).__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.norm_res = LayerNorm(dim, LayerNorm_type)
        self.attn = Prior_Sp(dim)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)
    def forward(self, x, res):
        res = self.norm_res(res)
        x = x + self.attn(self.norm1(x), res)
        x = x + self.ffn(self.norm2(x))

        return x


# class Net(PatchMergeModule):
#     def __init__(self,
#                 args,
#                  patch_size=[64, 32, 16, 8],
#                  inp_channels=3,
#                  out_channels=3,
#                  dim=32,
#                  num_blocks=[4, 6, 6, 4],
#                  num_refinement_blocks=4,
#                  heads=[1, 2, 4, 8],
#                  ffn_expansion_factor=2.66,
#                  bias=False,
#                  n_embedding=64,
#                  LayerNorm_type='WithBias',  ## Other option 'BiasFree'
#                  dual_pixel_task=False,  ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
#                  cut_nums=[8, 8, 8, 8],
#                  cut_low=[4, 4, 4, 4],
#                  ):
#
#         super(Net, self).__init__()
#         self.args = args
#         self.patch_embed = OverlapPatchEmbed(inp_channels, dim)
#         self.patch_embed2 = OverlapPatchEmbed(inp_channels, dim)
#         self.rir = RIR(dim, 3)
#         self.fuse = Priorformer(dim)
#
#         # self.prompt1 = PromptGenBlock(prompt_dim=32, prompt_len=5, prompt_size=64, lin_dim=96)
#         # self.prompt2 = PromptGenBlock(prompt_dim=64, prompt_len=5, prompt_size=32, lin_dim=192)
#         # self.prompt3 = PromptGenBlock(prompt_dim=128, prompt_len=5, prompt_size=16, lin_dim=384)
#
#         self.encoder_level1 = nn.Sequential(
#             *[TransformerBlock(patch_size=patch_size[0], dim=dim, num_heads=heads[0],
#                                ffn_expansion_factor=ffn_expansion_factor, bias=bias,
#                                LayerNorm_type=LayerNorm_type, cut_num=cut_nums[0], cut_low=cut_low[0]) for i in
#               range(num_blocks[0])])
#
#         self.down1_2 = Downsample(dim)  ## From Level 1 to Level 2
#         self.encoder_level2 = nn.Sequential(*[
#             TransformerBlock(patch_size=patch_size[1], dim=int(dim * 2 ** 1), num_heads=heads[1],
#                              ffn_expansion_factor=ffn_expansion_factor,
#                              bias=bias, LayerNorm_type=LayerNorm_type, cut_num=cut_nums[1], cut_low=cut_low[1]) for
#             i in
#             range(num_blocks[1])])
#
#         self.down2_3 = Downsample(int(dim * 2 ** 1))  ## From Level 2 to Level 3
#         self.encoder_level3 = nn.Sequential(*[
#             TransformerBlock(patch_size=patch_size[2], dim=int(dim * 2 ** 2), num_heads=heads[2],
#                              ffn_expansion_factor=ffn_expansion_factor,
#                              bias=bias, LayerNorm_type=LayerNorm_type, cut_num=cut_nums[2], cut_low=cut_low[2]) for
#             i in
#             range(num_blocks[2])])
#
#         self.down3_4 = Downsample(int(dim * 2 ** 2))  ## From Level 3 to Level 4
#         self.latent = nn.Sequential(*[
#             TransformerBlock(patch_size=patch_size[3], dim=int(dim * 2 ** 3), num_heads=heads[3],
#                              ffn_expansion_factor=ffn_expansion_factor,
#                              bias=bias, LayerNorm_type=LayerNorm_type, cut_num=cut_nums[3], cut_low=cut_low[3]) for
#             i in
#             range(num_blocks[3])])
#
#         self.up4_3 = Upsample(int(dim * 2 ** 3))  ## From Level 4 to Level 3
#         # self.together3 = Together_3(int(dim * 2 ** 2))
#
#         self.reduce_chan_level3 = nn.Conv2d(int(dim * 2 ** 3), int(dim * 2 ** 2), kernel_size=1, bias=bias)
#         self.decoder_level3 = nn.Sequential(*[
#             TransformerBlock(patch_size=patch_size[2], dim=int(dim * 2 ** 2), num_heads=heads[2],
#                              ffn_expansion_factor=ffn_expansion_factor,
#                              bias=bias, LayerNorm_type=LayerNorm_type, cut_num=cut_nums[2], cut_low=cut_low[2]) for
#             i in
#             range(num_blocks[2])])
#
#         self.up3_2 = Upsample(int(dim * 2 ** 2))  ## From Level 3 to Level 2
#         self.reduce_chan_level2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
#         # self.together2 = Together_2(int(dim * 2 ** 1))
#
#         # self.noise_level2 = TransformerBlock(patch_size=patch_size[1], dim=int(dim * 2 ** 2) + 64, num_heads=heads[1],
#         #                                      ffn_expansion_factor=ffn_expansion_factor,
#         #                                      bias=bias, LayerNorm_type=LayerNorm_type, cut_num=cut_nums[1],
#         #                                      cut_low=cut_low[1])
#         # self.reduce_noise_level2 = nn.Conv2d(int(dim * 2 ** 2) + 64, int(dim * 2 ** 2), kernel_size=1, bias=bias)
#
#         self.decoder_level2 = nn.Sequential(*[
#             TransformerBlock(patch_size=patch_size[1], dim=int(dim * 2 ** 1), num_heads=heads[1],
#                              ffn_expansion_factor=ffn_expansion_factor,
#                              bias=bias, LayerNorm_type=LayerNorm_type, cut_num=cut_nums[1], cut_low=cut_low[1]) for
#             i in
#             range(num_blocks[1])])
#
#         # self.noise_level1 = TransformerBlock(patch_size=patch_size[0], dim=int(dim * 2 ** 1) + 32, num_heads=heads[0],
#         #                                      ffn_expansion_factor=ffn_expansion_factor,
#         #                                      bias=bias, LayerNorm_type=LayerNorm_type, cut_num=cut_nums[0],
#         #                                      cut_low=cut_low[0])
#         # self.reduce_noise_level1 = nn.Conv2d(int(dim * 2 ** 1) + 32, int(dim * 2 ** 1), kernel_size=1, bias=bias)
#
#         self.up2_1 = Upsample(int(dim * 2 ** 1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)
#         # self.together1 = Together_1(dim)
#         self.reduce_chan_level1 = nn.Conv2d(int(dim * 2 ** 1), dim, kernel_size=1, bias=bias)
#         self.decoder_level1 = nn.Sequential(*[
#             TransformerBlock(patch_size=patch_size[0], dim=dim, num_heads=heads[0],
#                              ffn_expansion_factor=ffn_expansion_factor,
#                              bias=bias, LayerNorm_type=LayerNorm_type, cut_num=cut_nums[0], cut_low=cut_low[0]) for
#             i in
#             range(num_blocks[0])])
#
#         self.refinement = nn.Sequential(*[
#             TransformerBlock(patch_size=patch_size[0], dim=dim, num_heads=heads[0],
#                              ffn_expansion_factor=ffn_expansion_factor,
#                              bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])
#
#         #### For Dual-Pixel Defocus Deblurring Task ####
#         self.dual_pixel_task = dual_pixel_task
#         if self.dual_pixel_task:
#             self.skip_conv = nn.Conv2d(dim, int(dim * 2 ** 1), kernel_size=1, bias=bias)
#         ###########################
#
#         self.output = nn.Conv2d(dim, out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
#
#     def forward(self, inp_img):
#
#         inp_enc_level1 = self.patch_embed(inp_img)
#         res = get_residue(inp_img)
#         res = torch.cat([res, res, res], dim=1)
#         res = self.patch_embed2(res)
#         res = self.rir(res)
#         inp_enc_level1 = self.fuse(inp_enc_level1, res)
#         out_enc_level1 = self.encoder_level1(inp_enc_level1)
#         # res = self.res(out_enc_level1)
#         inp_enc_level2 = self.down1_2(out_enc_level1)
#         out_enc_level2 = self.encoder_level2(inp_enc_level2)
#
#         inp_enc_level3 = self.down2_3(out_enc_level2)
#         out_enc_level3 = self.encoder_level3(inp_enc_level3)
#
#         inp_enc_level4 = self.down3_4(out_enc_level3)
#         latent = self.latent(inp_enc_level4)
#         # latent += res
#         # dec3_param = self.prompt3(latent)
#         # latent = torch.cat([latent, dec3_param], dim=1)
#         # latent = self.noise_level3(latent)
#         # latent = self.reduce_noise_level3(latent)
#
#         inp_dec_level3 = self.up4_3(latent)
#         inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
#         inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
#         # inp_dec_level3 = self.together3(out_enc_level1, out_enc_level2, out_enc_level3, inp_dec_level3)
#         out_dec_level3 = self.decoder_level3(inp_dec_level3)
#
#         # dec2_param = self.prompt2(out_dec_level3)
#         # out_dec_level3 = torch.cat([out_dec_level3, dec2_param], dim=1)
#         # out_dec_level3 = self.noise_level2(out_dec_level3)
#         # out_dec_level3 = self.reduce_noise_level2(out_dec_level3)
#
#         inp_dec_level2 = self.up3_2(out_dec_level3)
#         inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
#         inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
#         # inp_dec_level2 = self.together2(out_enc_level1, out_enc_level2, out_enc_level3, inp_dec_level2)
#         out_dec_level2 = self.decoder_level2(inp_dec_level2)
#
#         # dec1_param = self.prompt1(out_dec_level2)
#         # out_dec_level2 = torch.cat([out_dec_level2, dec1_param], dim=1)
#         # out_dec_level2 = self.noise_level1(out_dec_level2)
#         # out_dec_level2 = self.reduce_noise_level1(out_dec_level2)
#
#         inp_dec_level1 = self.up2_1(out_dec_level2)
#         inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
#         # inp_dec_level1 = self.together1(out_enc_level1, out_enc_level2, out_enc_level3, inp_dec_level1)
#         inp_dec_level1 = self.reduce_chan_level1(inp_dec_level1)
#         out_dec_level1 = self.decoder_level1(inp_dec_level1)
#
#         out_dec_level1 = self.refinement(out_dec_level1)
#
#         #### For Dual-Pixel Defocus Deblurring Task ####
#         if self.dual_pixel_task:
#             out_dec_level1 = out_dec_level1 + self.skip_conv(inp_enc_level1)
#             out_dec_level1 = self.output(out_dec_level1)
#         ###########################
#         else:
#             out_dec_level1 = self.output(out_dec_level1) + inp_img
#
#         return [out_dec_level1]










class SubNet(nn.Module):
    def __init__(self,
                 patch_size=[64, 32, 16, 8],
                 dim=32,
                 num_blocks=[1, 1, 1, 1],
                 num_refinement_blocks=1,
                 heads=[1, 2, 4, 8],
                 ffn_expansion_factor=2.66,
                 bias=False,
                 n_embedding=64,
                 LayerNorm_type='WithBias',  ## Other option 'BiasFree'
                 dual_pixel_task=False,  ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
                 cut_nums=[8, 8, 8, 8],
                 cut_low=[4, 4, 4, 4],
                 ):

        super(SubNet, self).__init__()



        self.fuse = Priorformer(dim)


        # self.output1 = nn.Conv2d(dim * 2, 3, 3, 1, 1, bias=bias)
        # self.output2 = nn.Conv2d(dim * 4, 3, 3, 1, 1, bias=bias)


        self.encoder_level1 = nn.Sequential(*[TransformerBlock(patch_size=patch_size[0], dim=dim, num_heads=heads[0],
                             ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                             LayerNorm_type=LayerNorm_type, cut_num=cut_nums[0], cut_low=cut_low[0]) for i in
            range(num_blocks[0])])

        self.down1_2 = Downsample(dim)  ## From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential(*[
            TransformerBlock(patch_size=patch_size[1], dim=int(dim * 2 ** 1), num_heads=heads[1],
                             ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type, cut_num=cut_nums[1], cut_low=cut_low[1]) for i in
            range(num_blocks[1])])

        self.down2_3 = Downsample(int(dim * 2 ** 1))  ## From Level 2 to Level 3
        self.encoder_level3 = nn.Sequential(*[
            TransformerBlock(patch_size=patch_size[2], dim=int(dim * 2 ** 2), num_heads=heads[2],
                             ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type, cut_num=cut_nums[2], cut_low=cut_low[2]) for i in
            range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim * 2 ** 2))  ## From Level 3 to Level 4
        self.latent = nn.Sequential(*[
            TransformerBlock(patch_size=patch_size[3], dim=int(dim * 2 ** 3), num_heads=heads[3],
                             ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type, cut_num=cut_nums[3], cut_low=cut_low[3]) for i in
            range(num_blocks[3])])


        self.up4_3 = Upsample(int(dim * 2 ** 3))  ## From Level 4 to Level 3
        # self.together3 = Together_3(int(dim * 2 ** 2))



        self.reduce_chan_level3 = nn.Conv2d(int(dim * 2 ** 3), int(dim * 2 ** 2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[
            TransformerBlock(patch_size=patch_size[2], dim=int(dim * 2 ** 2), num_heads=heads[2],
                             ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type, cut_num=cut_nums[2], cut_low=cut_low[2]) for i in
            range(num_blocks[2])])

        self.up3_2 = Upsample(int(dim * 2 ** 2))  ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        # self.together2 = Together_2(int(dim * 2 ** 1))

        # self.noise_level2 = TransformerBlock(patch_size=patch_size[1], dim=int(dim * 2 ** 2) + 64, num_heads=heads[1],
        #                                      ffn_expansion_factor=ffn_expansion_factor,
        #                                      bias=bias, LayerNorm_type=LayerNorm_type, cut_num=cut_nums[1],
        #                                      cut_low=cut_low[1])
        # self.reduce_noise_level2 = nn.Conv2d(int(dim * 2 ** 2) + 64, int(dim * 2 ** 2), kernel_size=1, bias=bias)

        self.decoder_level2 = nn.Sequential(*[
            TransformerBlock(patch_size=patch_size[1], dim=int(dim * 2 ** 1), num_heads=heads[1],
                             ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type, cut_num=cut_nums[1], cut_low=cut_low[1]) for i in
            range(num_blocks[1])])

        # self.noise_level1 = TransformerBlock(patch_size=patch_size[0], dim=int(dim * 2 ** 1) + 32, num_heads=heads[0],
        #                                      ffn_expansion_factor=ffn_expansion_factor,
        #                                      bias=bias, LayerNorm_type=LayerNorm_type, cut_num=cut_nums[0],
        #                                      cut_low=cut_low[0])
        # self.reduce_noise_level1 = nn.Conv2d(int(dim * 2 ** 1) + 32, int(dim * 2 ** 1), kernel_size=1, bias=bias)

        self.up2_1 = Upsample(int(dim * 2 ** 1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)
        # self.together1 = Together_1(dim)
        self.reduce_chan_level1 = nn.Conv2d(int(dim * 2 ** 1), dim, kernel_size=1, bias=bias)
        self.decoder_level1 = nn.Sequential(*[
            TransformerBlock(patch_size=patch_size[0], dim=dim, num_heads=heads[0],
                             ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type, cut_num=cut_nums[0], cut_low=cut_low[0]) for i in
            range(num_blocks[0])])

        self.refinement = nn.Sequential(*[
            TransformerBlock(patch_size=patch_size[0], dim=dim, num_heads=heads[0],
                             ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])


        ###########################


    def forward(self, x, res):


        inp_enc_level1 = self.fuse(x, res)



        out_enc_level1 = self.encoder_level1(inp_enc_level1)
        # res = self.res(out_enc_level1)
        inp_enc_level2 = self.down1_2(out_enc_level1)


        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)




        out_enc_level3 = self.encoder_level3(inp_enc_level3)

        inp_enc_level4 = self.down3_4(out_enc_level3)
        latent = self.latent(inp_enc_level4)
        # latent += res
        # dec3_param = self.prompt3(latent)
        # latent = torch.cat([latent, dec3_param], dim=1)
        # latent = self.noise_level3(latent)
        # latent = self.reduce_noise_level3(latent)

        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        # inp_dec_level3 = self.together3(out_enc_level1, out_enc_level2, out_enc_level3, inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)
        # out_2 = self.output2(out_dec_level3) + inp_img_2

        # dec2_param = self.prompt2(out_dec_level3)
        # out_dec_level3 = torch.cat([out_dec_level3, dec2_param], dim=1)
        # out_dec_level3 = self.noise_level2(out_dec_level3)
        # out_dec_level3 = self.reduce_noise_level2(out_dec_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        # inp_dec_level2 = self.together2(out_enc_level1, out_enc_level2, out_enc_level3, inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)
        # out_1 = self.output1(out_dec_level2) + inp_img_1

        # dec1_param = self.prompt1(out_dec_level2)
        # out_dec_level2 = torch.cat([out_dec_level2, dec1_param], dim=1)
        # out_dec_level2 = self.noise_level1(out_dec_level2)
        # out_dec_level2 = self.reduce_noise_level1(out_dec_level2)


        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        # inp_dec_level1 = self.together1(out_enc_level1, out_enc_level2, out_enc_level3, inp_dec_level1)
        inp_dec_level1 = self.reduce_chan_level1(inp_dec_level1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        out_dec_level1 = self.refinement(out_dec_level1)

        # #### For Dual-Pixel Defocus Deblurring Task ####
        # if self.dual_pixel_task:
        #     out_dec_level1 = out_dec_level1 + self.skip_conv(inp_enc_level1)
        #     out_dec_level1 = self.output(out_dec_level1)
        # ###########################
        # else:
        #     out_dec_level1 = self.output(out_dec_level1) + inp_img

        return out_dec_level1

class res_ch(nn.Module):
    def __init__(self, n_feats, blocks=2):
        super(res_ch,self).__init__()
        self.conv_init1 = nn.Conv2d(3, n_feats, 3, 1, 1)
        self.extra = RIR(n_feats, n_blocks=blocks)

    def forward(self,x):
        x = self.conv_init1(x)
        x = self.extra(x)
        return x


class Net(PatchMergeModule):
    def __init__(self, args):
        super(Net, self).__init__()
        self.args = args
        n_feats = 32
        blocks = 3

        self.conv_init1 = nn.Conv2d(3, n_feats, 3, 1, 1)
        self.res_extra1 = res_ch(n_feats, blocks)
        self.sub1 = SubNet(dim=n_feats)
        self.res_extra2 = res_ch(n_feats, blocks)
        self.sub2 = SubNet(dim=n_feats)
        self.res_extra3 = res_ch(n_feats, blocks)
        self.sub3 = SubNet(dim=n_feats)

        self.ag1 = nn.Conv2d(n_feats * 2, n_feats, 3, 1, 1)
        self.ag2 = nn.Conv2d(n_feats * 3, n_feats, 3, 1, 1)
        self.ag2_en = nn.Conv2d(n_feats * 2, n_feats, 3, 1, 1)
        self.ag_en = nn.Conv2d(n_feats * 3, n_feats, 3, 1, 1)

        self.output1 = nn.Conv2d(n_feats, 3, 3, 1, padding=1)
        self.output2 = nn.Conv2d(n_feats, 3, 3, 1, padding=1)
        self.output3 = nn.Conv2d(n_feats, 3, 3, 1, padding=1)

        # self._initialize_weights()

    def forward(self, x):
        res_x = get_residue(x)
        x_init = self.conv_init1(x)
        x1 = self.sub1(x_init, self.res_extra1(torch.cat((res_x, res_x, res_x), dim=1)))  # + x   # 1
        out1 = self.output1(x1) + x
        res_out1 = get_residue(out1)
        x2 = self.sub2(self.ag1(torch.cat((x1, x_init), dim=1)),
                       self.res_extra2(torch.cat((res_out1, res_out1, res_out1), dim=1)))  # + x1 # 2
        x2_ = self.ag2_en(torch.cat([x2, x1], dim=1))
        out2 = self.output2(x2_) + x
        res_out2 = get_residue(out2)
        x3 = self.sub3(self.ag2(torch.cat((x2, x1, x_init), dim=1)),
                       self.res_extra3(torch.cat((res_out2, res_out2, res_out2), dim=1)))  # + x2 # 3
        x3 = self.ag_en(torch.cat([x3, x2, x1], dim=1))
        out3 = self.output3(x3) + x

        return [out3, out2, out1]


    def train_step(self, *args, **kwargs):

        return self(*args, **kwargs)

    def val_step(self, *args, **kwargs):

        return self.forward_chop(*args, **kwargs)

class build_Net(DerainModel, name='Net_3'):
    def __call__(self, args):
        scheduler = dict(type='CosineAnnealingLrUpdaterHook', policy='CosineAnnealing', min_lr=1e-4,by_epoch=True)
        # scheduler = None
        m_loss = nn.L1Loss()
        l1_loss = CharbonnierLoss()
        edge_loss = EdgeLoss()
        f_loss = fftLoss()
        # S3imLoss = S3IM().cuda()
        # loss = BCMSLoss().cuda()
        weight_dict = {'loss': 1}
        losses = {'l1_loss': l1_loss, 'edge_loss': edge_loss, 'f_loss': f_loss, 'm_loss': m_loss}
        criterion = SetCriterion(losses, weight_dict)
        model = Net(args).cuda()
        # optimizer = optim.Adam(model.parameters(), lr=args.lr)  ## optimizer 1: Adam
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-4)
        # lrs.set_optimizer(optimizer, optim.lr_scheduler.CosineAnnealingLR)
        # lrs.get_lr_map("CosineAnnealingLR")
        # model.set_metrics(criterion)
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('模型参数为：{}'.format(total_params))

        return model, criterion, optimizer, scheduler