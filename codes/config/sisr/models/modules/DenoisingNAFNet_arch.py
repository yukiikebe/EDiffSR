# Add frequecy domain feature fusion with Galerkin Transformer and UNO
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
import os

from .module_util import SinusoidalPosEmb, LayerNorm, exists
from torchvision.utils import save_image
from .fuse_block import FuseBlock
from .AdaLN import FuseFreqSpatialAdaLN

# import sys
# sys.path.append('/home/yuki/research/EDiffSR/external/UNO')
# from navier_stokes_uno2d import UNO, UNO_S256

# sys.path.append('/home/yuki/research/EDiffSR/external/galerkin_transformer/libs')

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

    def __init__(self, img_channel=3, width=16, middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[], upscale=1, uno=None, enc_plus_dec_uno=False, input_fuse=False):
        super().__init__()
        self.upscale = upscale
        fourier_dim = width
        sinu_pos_emb = SinusoidalPosEmb(fourier_dim)
        time_dim = width * 4
        self.uno = uno
        self.enc_plus_dec_uno = enc_plus_dec_uno
        self.input_fuse = input_fuse

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim*2),
            SimpleGate(),
            nn.Linear(time_dim, time_dim)
        )

        self.intro = nn.Conv2d(in_channels=img_channel*2, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        # self.intro = DoubleConv(in_ch=img_channel * 2, out_ch=width, mid_ch=width)
        self.enhance = RCAB(num_feat=width)
        
        ################## fix input size
        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        
        chan = width
        
        if self.input_fuse:
            self.first_fuse_block = FuseFreqSpatialAdaLN(freq_channels=2304, spatial_channels=chan, out_channels=chan, hidden_dim=chan * 4)
        
        # for only encoder integeration
        # if self.input_fuse is False:
        self.FuseBlocks = nn.ModuleList([
            FuseBlock(spacial_channels=chan * 2, frequency_channels=144, num_heads=8, fourier_dim=(-2, -1)),
            FuseBlock(spacial_channels=chan * 4, frequency_channels=288, num_heads=8, fourier_dim=(-2, -1)),
            FuseBlock(spacial_channels=chan * 8, frequency_channels=576, num_heads=8, fourier_dim=(-2, -1)),
            FuseBlock(spacial_channels=chan * 16, frequency_channels=1152, num_heads=8, fourier_dim=(-2, -1)),
        ])
        # else:
        #     # for encoder and decpder integeration and initial fuse block
        #     self.FuseBlocks = nn.ModuleList([
        #         FuseBlock(spacial_channels=chan * 2, frequency_channels=72, num_heads=8, fourier_dim=(-2, -1)),
        #         FuseBlock(spacial_channels=chan * 4, frequency_channels=144, num_heads=8, fourier_dim=(-2, -1)),
        #         FuseBlock(spacial_channels=chan * 8, frequency_channels=288, num_heads=8, fourier_dim=(-2, -1)),
        #         FuseBlock(spacial_channels=chan * 16, frequency_channels=576, num_heads=8, fourier_dim=(-2, -1)),
        #     ])
        
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(
                    *[NAFBlock(chan, time_dim) for _ in range(num)]
                )
            )
            # self.downs.append(
            #     nn.Conv2d(chan, 2*chan, 2, 2)
            # )
            
            
            self.downs.append(
                DownConv(chan, chan * 2)
            )
            
            chan = chan * 2

        self.middle_blks = \
            nn.Sequential(
                *[NAFBlock(chan, time_dim) for _ in range(middle_blk_num)]
            )

        for num in dec_blk_nums:
            # self.ups.append(
            #     nn.Sequential(
            #         nn.Conv2d(chan, chan * 2, 1, bias=False),
            #         nn.PixelShuffle(2)
            #     )
            # )
            
            self.ups.append(
                UpConv(chan, chan // 2)
            )
            
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[NAFBlock(chan, time_dim) for _ in range(num)]
                )
            )

        self.padder_size = 2 ** len(self.encoders)
        self.uno_features = None
        self.last_layer_feature = None

    def forward(self, inp, cond, time, freq_features=None):
        inp_res = inp.clone()
                
        if isinstance(time, int) or isinstance(time, float):
            time = torch.tensor([time]).to(inp.device)
    
        x = inp - cond
    
        x = torch.cat([x, cond], dim=1)

        
        t = self.time_mlp(time)
        
        B, C, H, W = x.shape
        x = self.check_image_size(x)
        cond = self.check_image_size(cond)

        x = self.intro(x)
        if freq_features is not None:
            self.uno_features, self.last_layer_feature = freq_features
            
        # RCAB enhance
        x = x + self.enhance(x)
        
        if self.input_fuse:
            x = self.first_fuse_block(self.last_layer_feature, x)
        encs = [x]
        
        for idx, (encoder, down) in enumerate(zip(self.encoders, self.downs)):
            x, _ = encoder([x, t])
            encs.append(x)
            
            if self.uno is not None:
                uno_feature = self.uno_features[idx]
                x = down(x)
                x = self.FuseBlocks[idx](x, uno_feature)
            else:
                x = down(x)        
        
        x, _ = self.middle_blks([x, t])
        
        if self.enc_plus_dec_uno:
            if self.uno is not None:
                for decoder, up, enc_skip, uno_feature, fuse_block in zip(self.decoders, self.ups, encs[::-1], self.uno_features[::-1], self.FuseBlocks[::-1]):
                    # print(f"x {x.shape} Enc Skip {enc_skip.shape}")
                    x = up(x, enc_skip, x_freq=uno_feature, fuse_block=fuse_block)
                    x, _ = decoder([x, t])
            else:
                raise ValueError("When enc_plus_dec_uno is True, uno must be provided.")
        else:
            for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
                # print(f"no fusion x {x.shape} Enc Skip {enc_skip.shape}")
                x = up(x, enc_skip)
                x = x + enc_skip
                x, _ = decoder([x, t])

        x = self.ending(x)
        
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

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, mid_ch=None):
        super().__init__()
        if mid_ch is None:
            mid_ch = out_ch
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, mid_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class DownConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            DoubleConv(in_ch, out_ch)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class UpConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2, x_freq=None, fuse_block=None):
        if x_freq is not None and fuse_block is not None:
            x1 = fuse_block(x1, x_freq)
        x1 = self.up(x1)
        
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        '''
           if you have padding issues, see
           https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
           https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        '''
        # print(f"UpConv x1 {x1.shape} x2 {x2.shape}")
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


