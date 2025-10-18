"""
Visualization utilities for SyntheticDigits GAN implementation.

This module contains functions for visualizing GAN training results,
generated images, and loss metrics.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.utils import save_image


def save_generated_images(generator, fixed_noise, epoch, output_dir, device):
    """Generate and save sample images from the generator."""
    generator.eval()
    with torch.no_grad():
        gen_imgs = generator(fixed_noise)
        # Denormalize images for saving
        gen_imgs = (gen_imgs + 1) / 2.0
        gen_imgs = torch.clamp(gen_imgs, 0, 1)
        
        # Save image grid
        image_path = os.path.join(output_dir, f"generated_epoch_{epoch}.png")
        save_image(gen_imgs, image_path, nrow=8, padding=2, normalize=False)
        
    generator.train()


def plot_training_losses(g_losses, d_losses, output_dir):
    """Plot generator and discriminator losses."""
    if not g_losses or not d_losses:
        return
        
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(g_losses, label="G")
    plt.plot(d_losses, label="D")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_metrics.png"))
    plt.close()


def visualize_generated_samples(generator, fixed_noise, device, n_samples=64, nrow=8):
    """Visualize a grid of generated samples."""
    generator.eval()
    with torch.no_grad():
        gen_imgs = generator(fixed_noise[:n_samples])
        # Denormalize images
        gen_imgs = (gen_imgs + 1) / 2.0
        gen_imgs = torch.clamp(gen_imgs, 0, 1)
        
    # Convert to numpy for visualization
    gen_imgs_np = gen_imgs.cpu().numpy()
    
    # Plot the images
    fig, axes = plt.subplots(nrow, nrow, figsize=(10, 10))
    for i, ax in enumerate(axes.flat):
        ax.imshow(gen_imgs_np[i].squeeze(), cmap='gray')
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()
    generator.train()