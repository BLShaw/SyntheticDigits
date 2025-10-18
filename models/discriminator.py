"""
Discriminator network module for SyntheticDigits GAN implementation.

This module contains the Discriminator class that classifies real vs fake images.
"""

import torch
import torch.nn as nn


class Discriminator(nn.Module):
    """Discriminator network that classifies real vs fake images."""
    
    def __init__(self, img_channels: int = 1, img_size: int = 28):
        super(Discriminator, self).__init__()
        
        def discriminator_block(in_filters: int, out_filters: int, bn: bool = True) -> nn.Sequential:
            block = [
                nn.Conv2d(in_filters, out_filters, 4, 2, 1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout2d(0.25)
            ]
            if bn:
                block.insert(1, nn.BatchNorm2d(out_filters))
            return nn.Sequential(*block)
        
        self.model = nn.Sequential(
            *discriminator_block(img_channels, 32, bn=False),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128), 
        )
        
        # Calculate the size after convolution layers
        self.adv_layer = nn.Sequential(
            nn.AdaptiveAvgPool2d((2, 2)),  
            nn.Flatten(),
            nn.Linear(128 * 2 * 2, 1)
        )
        
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        out = self.model(img)
        validity = self.adv_layer(out)
        return validity