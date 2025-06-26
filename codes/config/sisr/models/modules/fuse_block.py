import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

class Attention(nn.Module):
    def __init__(self, dim=64, num_heads=8, bias=False):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias)
        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, y):
        b, c, h, w = x.shape

        if y.shape[2:] != (h, w):
            y = F.interpolate(y, size=(h, w), mode='bilinear', align_corners=False)
        
        assert x.shape[1:] == y.shape[1:], "spatial and frequency tensors must have the same shape"
        
        kv = self.kv_dwconv(self.kv(y))
        k, v = kv.chunk(2, dim=1)
        q = self.q_dwconv(self.q(x))

        # Removed 'd' dimension from rearrange
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        # Removed 'd' dimension from rearrange
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

class FuseBlock(nn.Module):
    def __init__(self, spacial_channels, freuency_channels, num_heads=8, fourier_dim=(-2, -1)):
        super(FuseBlock, self).__init__()
        self.spatial_channels = spacial_channels
        self.frequency_channels = freuency_channels
        self.num_heads = num_heads
        self.fourier_dim = fourier_dim
        print("spatial_channels:", self.spatial_channels, "frequency_channels:", self.frequency_channels, "num_heads:", self.num_heads, "fourier_dim:", self.fourier_dim)
        # Convolutional preprocessing for spatial_fature and frequemcy_feature
        self.frequency_conv = nn.Conv2d(self.frequency_channels, self.spatial_channels, kernel_size=3, stride=1, padding=1)
        self.spatial_conv = nn.Conv2d(self.spatial_channels, self.spatial_channels, kernel_size=3, stride=1, padding=1)
        
        # Attention modules for real and imaginary components
        self.real_spatial_att = Attention(dim=self.spatial_channels, num_heads=num_heads)
        self.imag_spatial_att = Attention(dim=self.spatial_channels, num_heads=num_heads)

    def forward(self, spatial_feature, frequemcy_feature):
        skip_connnection = spatial_feature
        
        # Preprocess spatial_feature and frequemcy_feature with convolutions
        spatial_feature = self.spatial_conv(spatial_feature)
        frequemcy_feature = self.frequency_conv(frequemcy_feature)
        
        # FFT both spatial_feature and frequemcy_feature
        spatial_fft = torch.fft.fftn(spatial_feature, dim=self.fourier_dim)
        frequency_fft = torch.fft.fftn(frequemcy_feature, dim=self.fourier_dim)
        
        # Extract real and imaginary parts
        real_spatial = spatial_fft.real.contiguous()
        imag_spatial = spatial_fft.imag.contiguous()
        real_frequency = frequency_fft.real.contiguous()
        imag_frequency = frequency_fft.imag.contiguous()
        
        # Update real_spatial with attention
        new_real_spatial = real_spatial + self.real_spatial_att(real_spatial, real_frequency)
        
        # Update imag_spatial with attention
        new_imag_spatial = imag_spatial + self.imag_spatial_att(imag_spatial, imag_frequency)
        
        # Reconstruct complex tensor and apply IFFT
        spatial_updated = torch.complex(new_real_spatial, new_imag_spatial)
        result = torch.fft.ifftn(spatial_updated, dim=self.fourier_dim)
        
        # Take real part of IFFT result for compatibility
        result = result.real
        # Add residual connection
        return result + skip_connnection