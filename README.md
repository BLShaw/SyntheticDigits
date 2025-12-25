# SyntheticDigits - Conditional WGAN-GP Implementation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)](https://pytorch.org/)

SyntheticDigits is a robust, research-grade implementation of a **Conditional Wasserstein GAN with Gradient Penalty (cWGAN-GP)**. It generates high-quality synthetic MNIST digit images and allows for controlled generation of specific digits.

## Key Features

*   **Conditional Generation**: Request specific digits (e.g., "generate a 7") using class-conditional embeddings.
*   **WGAN-GP Architecture**: Uses Wasserstein Loss with Gradient Penalty and Spectral Normalization for superior training stability and resistance to mode collapse.
*   **Experiment Tracking**: Integrated **TensorBoard** support for real-time monitoring of losses and generated image grids.
*   **Robust Engineering**:
    *   Modular Pydantic-based configuration.
    *   Automatic device selection (CPU/CUDA).
    *   Checkpointing and **Resume Training** capability.
    *   Clean, packaged source structure.

## Project Structure

```
├── configs/                 
│   └── default_config.yaml  # Hyperparameters (WGAN-GP settings, etc.)
├── src/                     
│   └── syntheticdigits/     
│       ├── config/          # Pydantic configuration schemas
│       ├── data/            # Data loading (MNIST)
│       ├── models/          # Conditional Generator & Discriminator
│       ├── training/        # Trainer logic (WGAN loop, GP, Logging)
│       └── utils/           # Utilities
├── tests/                   # Unit tests
├── results/                 # Output (Checkpoints, Logs, Images)
├── train.py                 # Training entry point
├── generate.py              # Inference entry point
└── requirements.txt         
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/BLShaw/SyntheticDigits.git
   cd SyntheticDigits
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Training

Train the model using the default configuration (Conditional WGAN-GP):

```bash
python train.py --epochs 50
```

**Resume from a checkpoint** (useful if training was interrupted):

```bash
python train.py --resume results/checkpoints/checkpoint_epoch_20.pth
```

**Monitor with TensorBoard**:

```bash
tensorboard --logdir results/logs
```

### 2. Generation (Inference)

Generate images using a trained checkpoint.

**Generate a random grid of digits:**
```bash
python generate.py --checkpoint results/checkpoints/checkpoint_epoch_49.pth --output_dir my_results
```

**Generate a specific digit (e.g., only 5s):**
```bash
python generate.py --checkpoint results/checkpoints/checkpoint_epoch_49.pth --output_dir my_fives --digit 5
```

### 3. Configuration

Hyperparameters are managed in `configs/default_config.yaml`. Key WGAN-GP parameters include:

```yaml
training:
  critic_iterations: 5   # Number of discriminator steps per generator step
  lambda_gp: 10.0        # Gradient penalty coefficient
  lr: 0.0002
  beta1: 0.5
  beta2: 0.999
```

## Architecture Details

*   **Generator**: Hybrid architecture using a Linear projection to 7x7 followed by `ConvTranspose2d` upsampling layers. Receives noise $z$ concatenated with a class embedding.
*   **Discriminator**: Deep Convolutional network with **Spectral Normalization** and **Instance Normalization** (no Batch Norm, as required for WGAN-GP). Receives the image concatenated with a spatial class embedding.
*   **Loss**: Wasserstein Loss $-(E[D(x)] - E[D(G(z))])$ + Gradient Penalty.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.