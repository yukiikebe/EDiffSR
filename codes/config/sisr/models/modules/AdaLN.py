import torch
import torch.nn as nn
import torch.nn.functional as F

#frequency feature used as conditioning vector
#spatial feature modulated by AdaLN using the conditioning vector
class AdaLN(nn.Module):
    def __init__(self, num_channels, cond_channels):
        super().__init__()
        self.norm = nn.LayerNorm(num_channels)
        self.gamma_fc = nn.Linear(cond_channels, num_channels)
        self.beta_fc = nn.Linear(cond_channels, num_channels)

    def forward(self, x, cond):
        # x: (B, C, H, W) — the modulated feature
        # cond: (B, cond_channels) — the conditioning vector
        B, C, H, W = x.shape
        x_ = x.permute(0, 2, 3, 1)  # (B, H, W, C)
        x_ = self.norm(x_)
        gamma = self.gamma_fc(cond).unsqueeze(1).unsqueeze(1) 
        beta = self.beta_fc(cond).unsqueeze(1).unsqueeze(1)
        out = gamma * x_ + beta
        out = out.permute(0, 3, 1, 2)  # (B, C, H, W)
        return out

class FuseFreqSpatialAdaLN(nn.Module):
    def __init__(self, freq_channels=1152, spatial_channels=32, out_channels=32, hidden_dim=128):
        super().__init__()

        self.freq_channels = freq_channels
        
        self.freq_encoder = nn.Sequential(
            nn.AdaptiveAvgPool2d((8, 8)),  # (B, 1152, 8, 8)
            nn.Flatten(),                 # (B, 1152*8*8)
            nn.Linear(self.freq_channels * 8 * 8, hidden_dim),
            nn.ReLU()
        )

        # Apply AdaLN to spatial features, modulated by frequency
        self.adaln = AdaLN(spatial_channels, hidden_dim)

        # Fuse (optional) — or return spatial_feat modulated
        self.project = nn.Conv2d(spatial_channels, out_channels, kernel_size=1)

    def forward(self, freq_feat, spatial_feat):
        # freq_feat: (B, 1152, 30, 30)
        # spatial_feat: (B, 32, 256, 256)

        # Encode frequency features to vector
        cond = self.freq_encoder(freq_feat)  # (B, hidden_dim)

        # Apply AdaLN to spatial feature
        spatial_feat_norm = self.adaln(spatial_feat, cond)  # (B, 32, 256, 256)

        # Project to output channels
        output = self.project(spatial_feat_norm)  # (B, 32, 256, 256)

        return output
