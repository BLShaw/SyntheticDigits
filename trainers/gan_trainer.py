"""
GAN Trainer module for training Generative Adversarial Networks.

This module contains the GANTrainer class which handles the training process
for the SyntheticDigits GAN, including model initialization, training loops, and visualization.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import logging
from datetime import datetime
import pickle
from typing import Tuple

from models.generator import Generator
from models.discriminator import Discriminator
from data.dataloader import get_data_loader


class GANTrainer:
    """Trainer class for the GAN with visualization capabilities."""
    
    def __init__(self, 
                 config: dict):
        
        # Extract configuration parameters
        self.noise_dim = config.get('noise_dim', 100)
        self.img_channels = config.get('img_channels', 1)
        self.img_size = config.get('img_size', 28)
        self.batch_size = config.get('batch_size', 128)
        self.epochs = config.get('epochs', 100)
        self.lr = config.get('lr', 0.0002)
        self.beta1 = config.get('beta1', 0.5)
        self.output_dir = config.get('output_dir', 'results')
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Using device: {self.device}")
        
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            self.logger.info(f"CUDA: {torch.cuda.get_device_name(0)}")
        
        # Initialize models
        self.generator = Generator(self.noise_dim, self.img_channels, self.img_size).to(self.device)
        self.discriminator = Discriminator(self.img_channels, self.img_size).to(self.device)
        
        # Log model parameters
        g_params = sum(p.numel() for p in self.generator.parameters())
        d_params = sum(p.numel() for p in self.discriminator.parameters())
        self.logger.info(f"Generator parameters: {g_params:,}")
        self.logger.info(f"Discriminator parameters: {d_params:,}")
        
        # Initialize optimizers
        self.optimizer_G = optim.Adam(self.generator.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
        self.optimizer_D = optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
        
        # Loss function
        self.adversarial_loss = nn.BCEWithLogitsLoss()
        
        # Initialize data loader
        self.dataloader = get_data_loader(self.batch_size, config.get('data_dir', './data'))
        
        # Fixed noise for visualization
        self.sample_noise = torch.randn(64, self.noise_dim).to(self.device)
        
        # Tracking variables
        self.G_losses = []
        self.D_losses = []
        
        # Save configuration
        self.config = config

    def setup_logging(self):
        """Set up logging configuration."""
        log_filename = f"gan_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(self.output_dir, log_filename)),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def train_step(self, real_imgs: torch.Tensor) -> Tuple[float, float]:
        """Perform one training step for both generator and discriminator."""
        batch_size = real_imgs.size(0)
        
        # Create labels
        real_labels = torch.ones(batch_size, 1).to(self.device)
        fake_labels = torch.zeros(batch_size, 1).to(self.device)
        
        # -----------------
        # Train Discriminator
        # -----------------
        self.optimizer_D.zero_grad()

        # Real images
        real_output = self.discriminator(real_imgs)
        d_loss_real = self.adversarial_loss(real_output, real_labels)

        # Fake images
        z = torch.randn(batch_size, self.noise_dim).to(self.device)
        fake_imgs = self.generator(z)
        fake_output = self.discriminator(fake_imgs.detach())
        d_loss_fake = self.adversarial_loss(fake_output, fake_labels)

        # Total discriminator loss
        d_loss = (d_loss_real + d_loss_fake) / 2
        d_loss.backward()
        self.optimizer_D.step()

        # -----------------
        # Train Generator
        # -----------------
        self.optimizer_G.zero_grad()

        # Generate fake images
        z = torch.randn(batch_size, self.noise_dim).to(self.device)
        fake_imgs = self.generator(z)

        # Calculate generator loss
        fake_output = self.discriminator(fake_imgs)
        g_loss = self.adversarial_loss(fake_output, real_labels)

        g_loss.backward()
        self.optimizer_G.step()

        return g_loss.item(), d_loss.item()
    
    def save_sample_images(self, epoch: int):
        """Generate and save sample images."""
        self.generator.eval()
        with torch.no_grad():
            gen_imgs = self.generator(self.sample_noise)
            # Denormalize images for saving
            gen_imgs = (gen_imgs + 1) / 2.0
            gen_imgs = torch.clamp(gen_imgs, 0, 1)
            
            # Save image grid
            image_path = os.path.join(self.output_dir, f"generated_epoch_{epoch}.png")
            from torchvision.utils import save_image  # Import here to avoid circular import issues
            save_image(gen_imgs, image_path, nrow=8, padding=2, normalize=False)
            
        self.generator.train()
    
    def plot_losses(self):
        """Plot generator and discriminator losses."""
        if not self.G_losses or not self.D_losses:
            return
            
        plt.figure(figsize=(10, 5))
        plt.title("Generator and Discriminator Loss During Training")
        plt.plot(self.G_losses, label="G")
        plt.plot(self.D_losses, label="D")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "training_metrics.png"))
        plt.close()
        
        # Save metrics to pickle file for later analysis
        metrics = {
            'G_losses': self.G_losses,
            'D_losses': self.D_losses,
        }
        with open(os.path.join(self.output_dir, "training_metrics.pkl"), 'wb') as f:
            pickle.dump(metrics, f)
    
    def train(self):
        """Main training loop."""
        self.logger.info("Starting training...")
        
        for epoch in range(1, self.epochs + 1):
            epoch_g_losses = []
            epoch_d_losses = []
            
            for i, (real_imgs, _) in enumerate(self.dataloader):
                real_imgs = real_imgs.to(self.device)
                
                g_loss, d_loss = self.train_step(real_imgs)
                epoch_g_losses.append(g_loss)
                epoch_d_losses.append(d_loss)
            
            # Calculate average losses for the epoch
            avg_g_loss = sum(epoch_g_losses) / len(epoch_g_losses)
            avg_d_loss = sum(epoch_d_losses) / len(epoch_d_losses)
            
            self.G_losses.append(avg_g_loss)
            self.D_losses.append(avg_d_loss)
            
            # Print metrics
            print(f"Epoch [{epoch}/{self.epochs}] - G: {avg_g_loss:.4f}, D: {avg_d_loss:.4f}, "
                  f"Time: {datetime.now().strftime('%H:%M:%S')}")
            
            # Log progress to file
            self.logger.info(f"Epoch [{epoch}/{self.epochs}] - "
                             f"G Loss: {avg_g_loss:.4f}, D Loss: {avg_d_loss:.4f}")
            
            # Save images at specified epochs
            if epoch in [1, 5, 10, 25, 50, 100] or epoch % 10 == 0:
                self.save_sample_images(epoch)
        
        # Final visualization
        self.plot_losses()
        self.save_sample_images(self.epochs)
        
        self.logger.info("Training completed!")
        print(f"Training completed! Results saved to {self.output_dir}")
        
        # Save final models
        torch.save(self.generator.state_dict(), os.path.join(self.output_dir, "generator_final.pth"))
        torch.save(self.discriminator.state_dict(), os.path.join(self.output_dir, "discriminator_final.pth"))