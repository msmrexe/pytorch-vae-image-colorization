"""
Core model components: Encoder, Decoder, and supporting blocks
(SEBlock, ResidualBlock, SelfAttention).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block."""
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ResidualBlock(nn.Module):
    """Residual Block with optional Squeeze-and-Excitation."""
    def __init__(self, in_channels, out_channels, use_se=False):
        super(ResidualBlock, self).__init__()
        self.use_se = use_se
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.gn1 = nn.GroupNorm(num_groups=8, num_channels=out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.gn2 = nn.GroupNorm(num_groups=8, num_channels=out_channels)
        
        if self.use_se:
            self.se = SEBlock(out_channels)
            
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
                nn.GroupNorm(num_groups=8, num_channels=out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.gn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.gn2(out)
        if self.use_se:
            out = self.se(out)
        out += identity
        out = self.relu(out)
        return out

class SelfAttention(nn.Module):
    """Self-Attention Block."""
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)
        
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)
        out = self.gamma * out + x
        return out

class Encoder(nn.Module):
    """VAE Encoder with U-Net skip connections."""
    def __init__(self, latent_dim=128):
        super(Encoder, self).__init__()
        self.block1 = self.inner_block(1, 32)
        self.block2 = self.inner_block(32, 64)
        self.block3 = self.inner_block(64, 128)
        self.block4 = self.inner_block(128, 256)
        self.self_attention = SelfAttention(256)
        
        # Calculate flattened size based on IMG_SIZE
        final_spatial_dim = 160 // (2**4) # 160 -> 80 -> 40 -> 20 -> 10
        flattened_size = 256 * final_spatial_dim * final_spatial_dim
        
        self.fc_mu = nn.Linear(flattened_size, latent_dim)
        self.fc_logvar = nn.Linear(flattened_size, latent_dim)

    def inner_block(self, in_c, out_c, maxpool=2):
        layers = [
            ResidualBlock(in_c, out_c, use_se=True),
            nn.MaxPool2d(kernel_size=maxpool, stride=maxpool),
            nn.Dropout(0.1)
        ]
        return nn.Sequential(*layers)

    def forward(self, x):
        features = []
        x = self.block1(x)  # (B, 32, 80, 80)
        features.append(x)
        x = self.block2(x)  # (B, 64, 40, 40)
        features.append(x)
        x = self.block3(x)  # (B, 128, 20, 20)
        features.append(x)
        x = self.block4(x)  # (B, 256, 10, 10)
        x = self.self_attention(x)
        features.append(x)
        
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar, features

class Decoder(nn.Module):
    """VAE Decoder with U-Net skip connections."""
    def __init__(self, latent_dim=128):
        super(Decoder, self).__init__()
        # Calculate spatial dim and flattened size
        final_spatial_dim = 160 // (2**4) # 10
        flattened_size = 256 * final_spatial_dim * final_spatial_dim

        self.fc = nn.Linear(latent_dim, flattened_size)
        self.self_attention = SelfAttention(256)
        self.up1 = self.inner_block(256, 128)
        self.up2 = self.inner_block(128 + 128, 64)  # Skip connection from encoder block3
        self.up3 = self.inner_block(64 + 64, 32)    # Skip connection from encoder block2
        self.up4 = self.inner_block(32 + 32, 16)    # Skip connection from encoder block1
        self.final_conv = nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()

    def inner_block(self, in_c, out_c):
        layers = [
            nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2),
            nn.GroupNorm(num_groups=8, num_channels=out_c),
            nn.ReLU(inplace=True)
        ]
        return nn.Sequential(*layers)

    def forward(self, z, skip_features):
        final_spatial_dim = 160 // (2**4) # 10
        h = self.fc(z)
        h = h.view(h.size(0), 256, final_spatial_dim, final_spatial_dim)
        h = self.self_attention(h)
        
        h = self.up1(h)
        h = torch.cat([h, skip_features[2]], dim=1) # (B, 128+128, 20, 20)
        
        h = self.up2(h)
        h = torch.cat([h, skip_features[1]], dim=1) # (B, 64+64, 40, 40)
        
        h = self.up3(h)
        h = torch.cat([h, skip_features[0]], dim=1) # (B, 32+32, 80, 80)
        
        h = self.up4(h)
        h = self.final_conv(h)
        h = self.sigmoid(h)
        return h
