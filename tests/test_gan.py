import unittest
import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

from syntheticdigits.models.networks import Generator, Discriminator
from syntheticdigits.config.schema import AppConfig

class TestGAN(unittest.TestCase):
    def setUp(self):
        self.config = AppConfig()
        self.gen = Generator(
            self.config.model.noise_dim, 
            self.config.model.img_channels, 
            self.config.model.features_g,
            self.config.model.num_classes,
            self.config.model.embed_size
        )
        self.disc = Discriminator(
            self.config.model.img_channels, 
            self.config.model.features_d,
            self.config.model.num_classes,
            self.config.model.img_size
        )

    def test_generator_output_shape(self):
        batch_size = 4
        z = torch.randn(batch_size, self.config.model.noise_dim)
        labels = torch.randint(0, self.config.model.num_classes, (batch_size,))
        output = self.gen(z, labels)
        self.assertEqual(output.shape, (batch_size, self.config.model.img_channels, 28, 28))

    def test_discriminator_output_shape(self):
        batch_size = 4
        img = torch.randn(batch_size, self.config.model.img_channels, 28, 28)
        labels = torch.randint(0, self.config.model.num_classes, (batch_size,))
        output = self.disc(img, labels)
        self.assertEqual(output.shape, (batch_size, 1, 1, 1))

if __name__ == '__main__':
    unittest.main()
