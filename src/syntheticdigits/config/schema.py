from pydantic import BaseModel, Field
from typing import List, Optional

class ModelConfig(BaseModel):
    noise_dim: int = Field(100, description="Dimension of the latent noise vector")
    img_channels: int = Field(1, description="Number of image channels (1 for grayscale)")
    img_size: int = Field(28, description="Size of the image (H=W)")
    features_g: int = Field(64, description="Base feature map depth for Generator")
    features_d: int = Field(64, description="Base feature map depth for Discriminator")
    num_classes: int = Field(10, description="Number of classes for conditional generation (MNIST=10)")
    embed_size: int = Field(50, description="Dimension of the class embedding vector")

class TrainingConfig(BaseModel):
    batch_size: int = Field(128, description="Batch size for training")
    epochs: int = Field(50, description="Number of training epochs")
    lr: float = Field(0.0002, description="Learning rate")
    beta1: float = Field(0.5, description="Beta1 for Adam optimizer")
    beta2: float = Field(0.999, description="Beta2 for Adam optimizer")
    critic_iterations: int = Field(5, description="Number of discriminator steps per generator step (useful for WGAN)")
    lambda_gp: float = Field(10.0, description="Gradient penalty coefficient for WGAN-GP")
    device: str = Field("auto", description="Device to use: 'cuda', 'cpu', or 'auto'")

class DataConfig(BaseModel):
    data_dir: str = Field("./data", description="Directory to store dataset")
    num_workers: int = Field(4, description="Number of dataloader workers")
    download: bool = Field(True, description="Whether to download the dataset")

class OutputConfig(BaseModel):
    output_dir: str = Field("results", description="Root directory for outputs")
    save_image_interval: int = Field(5, description="Save sample images every N epochs")
    save_model_interval: int = Field(10, description="Save model checkpoints every N epochs")

class AppConfig(BaseModel):
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
