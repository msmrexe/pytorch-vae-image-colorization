"""
Combines the model, perceptual loss, and VAE logic into a single system.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import logging

from .model import Encoder, Decoder
from .config import VGG_FEATURE_LAYERS, BETA_KLD, LAMBDA_PERCEPTUAL

class PerceptualLoss(nn.Module):
    """VGG19-based Perceptual Loss."""
    def __init__(self, feature_layers=None):
        super(PerceptualLoss, self).__init__()
        try:
            vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        except Exception as e:
            logging.warning(f"Could not load pretrained VGG19 weights, downloading: {e}")
            vgg = models.vgg19(weights=None) # Fallback, though not ideal
            # In a real scenario, you might want to handle this better
            
        self.layers = nn.ModuleList()
        if feature_layers is None:
            feature_layers = VGG_FEATURE_LAYERS
        self.feature_layers = feature_layers
        
        max_layer = max(feature_layers) + 1
        for i in range(max_layer):
            self.layers.append(vgg[i])
            
        for param in self.layers.parameters():
            param.requires_grad = False
            
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        logging.info(f"PerceptualLoss initialized using VGG19 layers: {feature_layers}")

    def forward(self, x, y):
        # Normalize for VGG
        self.mean = self.mean.to(x.device)
        self.std = self.std.to(x.device)
        x = (x - self.mean) / self.std
        y = (y - self.mean) / self.std
        
        loss = 0
        for i, layer in enumerate(self.layers):
            x = layer(x)
            y = layer(y)
            if i in self.feature_layers:
                loss += F.l1_loss(x, y)
        return loss

class VAEColorizer(nn.Module):
    """The complete VAE Colorization system."""
    def __init__(self, latent_dim=128, use_perceptual_loss=True):
        super(VAEColorizer, self).__init__()
        self.encoder = Encoder(latent_dim=latent_dim)
        self.decoder = Decoder(latent_dim=latent_dim)
        
        self.use_perceptual_loss = use_perceptual_loss
        if use_perceptual_loss:
            self.perceptual_loss = PerceptualLoss()
        
        logging.info(f"VAEColorizer model initialized with latent_dim={latent_dim}")

    def reparameterize(self, mu, logvar):
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar, skip_features = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z, skip_features)
        return recon, mu, logvar

    def compute_loss(self, pred, target, mu, logvar):
        """
        Computes the hybrid loss:
        L_total = (L_MSE + lambda * L_perceptual) + beta * L_KLD
        """
        # Reconstruction Loss (L1)
        recon_loss_mse = F.mse_loss(pred, target)
        
        if self.use_perceptual_loss:
            perceptual = self.perceptual_loss(pred, target)
            recon_loss = recon_loss_mse + LAMBDA_PERCEPTUAL * perceptual
        else:
            recon_loss = recon_loss_mse
            
        # KL Divergence
        kld_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        
        # Total Loss
        total_loss = recon_loss + BETA_KLD * kld_loss
        
        return total_loss, recon_loss_mse, kld_loss
