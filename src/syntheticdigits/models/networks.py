import torch
import torch.nn as nn

class Generator(nn.Module):
    """
    Generator Network.
    Upsamples a latent vector (z) and class label into an image.
    Uses a Linear layer to project to 7x7 spatial dim, then upsamples.
    """
    def __init__(self, z_dim: int, channels_img: int, features_g: int, num_classes: int, embed_size: int):
        super(Generator, self).__init__()
        
        self.features_g = features_g
        
        # Conditional Embedding
        self.embed = nn.Embedding(num_classes, embed_size)
        
        # Project and reshape (Input is z + embedding)
        self.linear = nn.Sequential(
            nn.Linear(z_dim + embed_size, features_g * 4 * 7 * 7),
            nn.BatchNorm1d(features_g * 4 * 7 * 7),
            nn.ReLU(),
        )
        
        self.net = nn.Sequential(
            self._block(features_g * 4, features_g * 2, 4, 2, 1),
            nn.ConvTranspose2d(features_g * 2, channels_img, 4, 2, 1),
            nn.Tanh(),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size, stride, padding, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x, labels):
        embedding = self.embed(labels)
        x = torch.cat([x, embedding], dim=1)
        
        x = self.linear(x)
        x = x.view(x.shape[0], self.features_g * 4, 7, 7)
        return self.net(x)

class Discriminator(nn.Module):
    """
    Discriminator Network.
    Classifies images as real or fake, conditioned on labels.
    Uses Spectral Normalization for stability.
    """
    def __init__(self, channels_img: int, features_d: int, num_classes: int, img_size: int):
        super(Discriminator, self).__init__()
        self.img_size = img_size
        
        self.embed = nn.Embedding(num_classes, img_size * img_size)
        
        self.disc = nn.Sequential(
            nn.utils.spectral_norm(
                nn.Conv2d(channels_img + 1, features_d, 4, 2, 1)
            ),
            nn.LeakyReLU(0.2),
            
            self._block(features_d, features_d * 2, 4, 2, 1),
            self._block(features_d * 2, features_d * 4, 3, 2, 0),
            
            nn.utils.spectral_norm(
                nn.Conv2d(features_d * 4, 1, 3, 1, 0)
            ),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.utils.spectral_norm(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size, stride, padding, bias=False
                )
            ),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x, labels):
        embedding = self.embed(labels).view(labels.shape[0], 1, self.img_size, self.img_size)
        x = torch.cat([x, embedding], dim=1)
        return self.disc(x)

def initialize_weights(model):
    """
    Initializes weights according to the DCGAN paper.
    Random normal with mean 0 and std 0.02.
    """
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d, nn.InstanceNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
