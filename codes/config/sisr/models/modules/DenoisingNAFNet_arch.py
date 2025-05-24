import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
import os

from .module_util import SinusoidalPosEmb, LayerNorm, exists
from torchvision.utils import save_image

import sys
sys.path.append('/home/yuki/EDiffSR/external/UNO')
from navier_stokes_uno2d import UNO, UNO_S256

sys.path.append('/home/yuki/EDiffSR/external/galerkin_transformer/libs')
from model import SimpleTransformerEncorderOnly

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    def __init__(self, c, time_emb_dim=None, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        self.mlp = nn.Sequential(
            SimpleGate(), nn.Linear(time_emb_dim // 2, c * 4)
        ) if time_emb_dim else None

        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm(c)
        self.norm2 = LayerNorm(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def time_forward(self, time, mlp):
        time_emb = mlp(time)
        time_emb = rearrange(time_emb, 'b c -> b c 1 1')
        return time_emb.chunk(4, dim=1)

    def forward(self, x):
        inp, time = x
        shift_att, scale_att, shift_ffn, scale_ffn = self.time_forward(time, self.mlp)

        x = inp

        x = self.norm1(x)
        x = x * (scale_att + 1) + shift_att
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.norm2(y)
        x = x * (scale_ffn + 1) + shift_ffn
        x = self.conv4(x)
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        x = y + x * self.gamma

        return x, time

# --------------------------------------- RCAB modules-----------------------------------------------------------
def make_layer(basic_block, num_basic_block, **kwarg):
    """Make layers by stacking the same blocks.

    Args:
        basic_block (nn.module): nn.module class for basic block.
        num_basic_block (int): number of blocks.

    Returns:
        nn.Sequential: Stacked blocks in nn.Sequential.
    """
    layers = []
    for _ in range(num_basic_block):
        layers.append(basic_block(**kwarg))
    return nn.Sequential(*layers)
class ChannelAttention(nn.Module):
    """Channel attention used in RCAN.
    Args:
        num_feat (int): Channel number of intermediate features.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
    """

    def __init__(self, num_feat, squeeze_factor=16):
        super(ChannelAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0),
            nn.Sigmoid())

    def forward(self, x):
        y = self.attention(x)
        return x * y

class RCAB(nn.Module):
    """Residual Channel Attention Block (RCAB) used in RCAN.

    Args:
        num_feat (int): Channel number of intermediate features.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
        res_scale (float): Scale the residual. Default: 1.
    """

    def __init__(self, num_feat, squeeze_factor=16, res_scale=1):
        super(RCAB, self).__init__()
        self.res_scale = res_scale

        self.rcab = nn.Sequential(
            nn.Conv2d(num_feat, num_feat, 3, 1, 1), nn.ReLU(True), nn.Conv2d(num_feat, num_feat, 3, 1, 1),
            ChannelAttention(num_feat, squeeze_factor))

    def forward(self, x):
        res = self.rcab(x) * self.res_scale
        return res + x

class ResidualGroup(nn.Module):
    """Residual Group of RCAB.

    Args:
        num_feat (int): Channel number of intermediate features.
        num_block (int): Block number in the body network.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
        res_scale (float): Scale the residual. Default: 1.
    """

    def __init__(self, num_feat, num_block, squeeze_factor=16, res_scale=1):
        super(ResidualGroup, self).__init__()

        self.residual_group = make_layer(
            RCAB, num_block, num_feat=num_feat, squeeze_factor=squeeze_factor, res_scale=res_scale)
        self.conv = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

    def forward(self, x):
        res = self.conv(self.residual_group(x))
        return res + x

# ---------------------------------------- -----------------------------------------------------------------------

class ConditionalNAFNet(nn.Module):

    def __init__(self, img_channel=3, width=16, middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[], upscale=1, uno_whole_image=None,uno_patch=None,galerkin_encorder=None):
        super().__init__()
        self.upscale = upscale
        fourier_dim = width
        sinu_pos_emb = SinusoidalPosEmb(fourier_dim)
        time_dim = width * 4
        self.uno_whole_image = uno_whole_image
        self.uno_patch = uno_patch
        self.galerkin_encorder = galerkin_encorder

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim*2),
            SimpleGate(),
            nn.Linear(time_dim, time_dim)
        )

        self.intro = nn.Conv2d(in_channels=img_channel*2, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        self.enhance = RCAB(num_feat=width)
        
        ################## fix input size
        # self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1, groups=1,
        #                       bias=True)
        self.ending = nn.Conv2d(in_channels=width * 2, out_channels=img_channel, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(
                    *[NAFBlock(chan, time_dim) for _ in range(num)]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2*chan, 2, 2)
            )
            chan = chan * 2

        self.middle_blks = \
            nn.Sequential(
                *[NAFBlock(chan, time_dim) for _ in range(middle_blk_num)]
            )

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[NAFBlock(chan, time_dim) for _ in range(num)]
                )
            )

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp, cond, time):
        inp_res = inp.clone()
                
        if isinstance(time, int) or isinstance(time, float):
            time = torch.tensor([time]).to(inp.device)
    
        x = inp - cond
    
        x = torch.cat([x, cond], dim=1)

        t = self.time_mlp(time)
        
        B, C, H, W = x.shape
        x = self.check_image_size(x)

        x = self.intro(x)

        # RCAB enhance
        x = x + self.enhance(x)

        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x, _ = encoder([x, t])
            encs.append(x)
            x = down(x)

        x, _ = self.middle_blks([x, t])
        

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x, _ = decoder([x, t])
            
        x_feature = x.clone()
        print(f" - Output from decoder: {x.shape}")

        # print(f" - Output from ending: {x.shape}")
        
        if self.uno_whole_image is not None:
            # for linear
            # High frequency whole image
            cond_input = cond.permute(0, 2, 3, 1)
            uno_whole_image_out = self.uno_whole_image(cond_input).permute(0, 3, 1, 2) # use nn.linear
            
            # separate the cond into four parts
            uno_patch_each_output = []
            cond_patch = cond_input.chunk(4, dim=1)
            for i in range(4):
                uno_patch_each_output.append(self.uno_patch(cond_patch[i]))
            uno_patch_out = torch.cat(uno_patch_each_output, dim=1).permute(0, 3, 1, 2)
            print(f" - Output from UNO patch: {uno_patch_out.shape}")
            uno_last_feature = torch.cat([uno_whole_image_out, uno_patch_out], dim=1)
            print(f" - Output from UNO combine image: {uno_patch_out.shape}")
            uno_last_feature_gelu = F.gelu(uno_last_feature)
            print(f" - Output after gelu: {uno_last_feature_gelu.shape}")
            
            B, C, H, W = uno_last_feature_gelu.shape
            uno_galerkin_input = uno_last_feature_gelu.permute(0, 2, 3, 1).reshape(B, H * W, C)
            galerkin_out = self.galerkin_encorder(node=uno_galerkin_input, edge=None, pos=None)
            print(f" - Output from galerkin: {galerkin_out.shape}")
            B, N, C = galerkin_out.shape
            H = W = int(N ** 0.5)
            galerkin_output_reshape = galerkin_out.permute(0, 2, 1).reshape(B, C, H, W)
            print(f" - Output from galerkin reshape: {galerkin_output_reshape.shape}")
            
            # concate EDiffSR and Galerkin feature after changing them frequency domain
            x_fft = torch.fft.fft2(x_feature, norm="ortho")
            galerkin_fft = torch.fft.fft2(galerkin_output_reshape, norm="ortho")
            fused_fft = torch.cat([x_fft, galerkin_fft], dim=1)
            fused_feature = torch.fft.ifft2(fused_fft, norm="ortho").real # for conv
            print(f" - Output from fused feature: {fused_feature.shape}")

            x = self.ending(fused_feature)
        else:
            x = self.ending(x)
        x = x[..., :H, :W]
        
        # print(f" - Output from ending: {x.shape}")
        # # Save x as an RGB image in the tmp directory
        # image_rgb = x[:, 0:3, :, :]  # Convert to [B, H, W, C]
        # if image_rgb.dtype == torch.float32:
        #     image_rgb = (image_rgb + 1) / 2.0
        # for idx, img in enumerate(image_rgb):
        #     save_image(img, f"/home/yuki/EDiffSR/tmp/output_{idx}.png", normalize=False)
        
        return x

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x
    
def add_coord_channels(x):
    """
    add x and y coordinates as additional channels to the input tensor
    Input:
        x: Tensor of shape [B, C, H, W]
    Output:
        Tensor of shape [B, C+2, H, W]
    """
    B, C, H, W = x.size()

    # Create normalized coordinate grid from -1 to 1
    x_coords = torch.linspace(-1, 1, W, device=x.device).view(1, 1, 1, W).expand(B, 1, H, W)
    y_coords = torch.linspace(-1, 1, H, device=x.device).view(1, 1, H, 1).expand(B, 1, H, W)

    # Concatenate as new channels
    x_with_coords = torch.cat([x, x_coords, y_coords], dim=1)
    return x_with_coords


