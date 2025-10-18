"""
Data loading module for SyntheticDigits GAN implementation.

This module handles data loading and preprocessing for the GAN training.
"""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_data_loader(batch_size: int = 128, data_dir: str = "./data") -> DataLoader:
    """Create and return MNIST data loader."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
    ])
    
    dataset = datasets.MNIST(
        root=data_dir, 
        train=True, 
        download=True, 
        transform=transform
    )
    
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True,
        drop_last=True,
        num_workers=2,
        pin_memory=True
    )


def get_test_data_loader(batch_size: int = 128, data_dir: str = "./data") -> DataLoader:
    """Create and return MNIST test data loader."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
    ])
    
    dataset = datasets.MNIST(
        root=data_dir, 
        train=False, 
        download=True, 
        transform=transform
    )
    
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )