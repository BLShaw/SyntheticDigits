"""
Generator network module for SyntheticDigits GAN implementation.

This module contains the Generator class that transforms noise into images.
"""

import torch
import torch.nn as nn
from typing import Tuple


class Generator(nn.Module):
    """Generator network that transforms noise into images."""
    
    def __init__(self, noise_dim: int = 100, img_channels: int = 1, img_size: int = 28):
        super(Generator, self).__init__()
        self.img_size = img_size
        self.img_channels = img_channels
        self.noise_dim = noise_dim
        
        # Calculate dimensions after each layer
        self.init_size = img_size // 4
        self.fc_input_dim = 128 * self.init_size * self.init_size
        
        self.fc = nn.Sequential(
            nn.Linear(noise_dim, self.fc_input_dim),
            nn.BatchNorm1d(self.fc_input_dim),
            nn.ReLU(True)
        )
        
        self.conv_layers = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), 
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), 
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            
            nn.Conv2d(32, img_channels, 3, stride=1, padding=1),
            nn.Tanh()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x = x.view(x.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_layers(x)
        return img