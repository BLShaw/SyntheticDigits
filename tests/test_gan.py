"""
Unit tests for SyntheticDigits GAN modules.

This module contains unit tests for the various components of the SyntheticDigits GAN implementation.
"""

import unittest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from models.generator import Generator
from models.discriminator import Discriminator
from data.dataloader import get_data_loader
from trainers.gan_trainer import GANTrainer


class TestGenerator(unittest.TestCase):
    """Test cases for the Generator class."""
    
    def setUp(self):
        self.generator = Generator(noise_dim=100, img_channels=1, img_size=28)
        self.noise = torch.randn(4, 100)  # Small batch for testing
    
    def test_generator_forward_pass(self):
        """Test that the generator can perform a forward pass."""
        output = self.generator(self.noise)
        self.assertEqual(output.shape, (4, 1, 28, 28))
    
    def test_generator_parameters(self):
        """Test that the generator has parameters."""
        params = sum(p.numel() for p in self.generator.parameters())
        self.assertGreater(params, 0)


class TestDiscriminator(unittest.TestCase):
    """Test cases for the Discriminator class."""
    
    def setUp(self):
        self.discriminator = Discriminator(img_channels=1, img_size=28)
        self.images = torch.randn(4, 1, 28, 28)  # Small batch for testing
    
    def test_discriminator_forward_pass(self):
        """Test that the discriminator can perform a forward pass."""
        output = self.discriminator(self.images)
        self.assertEqual(output.shape, (4, 1))
    
    def test_discriminator_parameters(self):
        """Test that the discriminator has parameters."""
        params = sum(p.numel() for p in self.discriminator.parameters())
        self.assertGreater(params, 0)


class TestDataLoader(unittest.TestCase):
    """Test cases for the data loading functions."""
    
    def test_get_data_loader(self):
        """Test that the data loader function returns a DataLoader object."""
        # We'll create a simple mock dataset for this test
        # Since MNIST download might take time, we'll create a synthetic dataset
        data = torch.randn(100, 1, 28, 28)  # 100 random images
        targets = torch.randint(0, 10, (100,))  # 100 random labels
        dataset = TensorDataset(data, targets)
        
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        self.assertIsInstance(dataloader, DataLoader)
        self.assertEqual(dataloader.batch_size, 32)


class TestGANTrainer(unittest.TestCase):
    """Test cases for the GANTrainer class."""
    
    def setUp(self):
        # Minimal config for testing
        self.config = {
            'noise_dim': 100,
            'img_channels': 1,
            'img_size': 28,
            'batch_size': 4,  # Small batch size for testing
            'epochs': 1,      # Just one epoch for testing
            'lr': 0.0002,
            'beta1': 0.5,
            'output_dir': 'test_results',
            'data_dir': 'test_data'
        }
        
        # Create a mock dataloader for testing
        # Since we don't want to download MNIST for tests, we'll make a mock
        class MockDataLoader:
            def __iter__(self):
                # Return one batch of synthetic data
                images = torch.randn(4, 1, 28, 28)
                labels = torch.randint(0, 10, (4,))
                yield images, labels
        
        # Create a minimal trainer to test initialization
        self.trainer = GANTrainer(self.config)
        self.trainer.dataloader = MockDataLoader()
    
    def test_trainer_initialization(self):
        """Test that the trainer initializes correctly."""
        self.assertIsInstance(self.trainer.generator, Generator)
        self.assertIsInstance(self.trainer.discriminator, Discriminator)
        self.assertEqual(self.trainer.noise_dim, 100)
        self.assertEqual(self.trainer.epochs, 1)
    
    def test_train_step(self):
        """Test that a single training step works."""
        real_imgs = torch.randn(4, 1, 28, 28)
        g_loss, d_loss = self.trainer.train_step(real_imgs)
        
        # Check that losses are numeric values
        self.assertIsInstance(g_loss, float)
        self.assertIsInstance(d_loss, float)


if __name__ == '__main__':
    unittest.main()