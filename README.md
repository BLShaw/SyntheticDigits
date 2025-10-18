# SyntheticDigits - MNIST GAN PyTorch Implementation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.7%2B-blue)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)](https://pytorch.org/)

This project implements a Generative Adversarial Network (GAN) for generating synthetic MNIST digit images using PyTorch. The implementation uses a Deep Convolutional GAN (DCGAN) architecture.

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Overview

This project, named SyntheticDigits, implements a Generative Adversarial Network (GAN) to generate synthetic MNIST digit images. The implementation includes:

- Generator and Discriminator networks with convolutional layers
- Configurable training parameters
- Visualization tools for generated images
- Loss tracking and plotting
- Proper logging of training progress

The GAN architecture is based on the DCGAN (Deep Convolutional GAN) approach, with:

- Generator: Takes random noise and produces image-like outputs
- Discriminator: Distinguishes between real and generated images

## Project Structure

```
├── configs/                 # Configuration files
│   └── default_config.yaml  # Default configuration
├── data/                    # Data loading utilities
│   └── dataloader.py        # Data loading functions
├── models/                  # Neural network architectures
│   ├── generator.py         # Generator model
│   └── discriminator.py     # Discriminator model
├── trainers/                # Training logic
│   └── gan_trainer.py       # GAN trainer class
├── utils/                   # Utility functions
│   ├── visualization.py     # Visualization utilities
│   ├── logging.py           # Logging utilities
│   └── config.py            # Configuration utilities
├── tests/                   # Unit tests
│   └── test_gan.py          # Tests for GAN components
├── src/                     # Source package
│   └── syntheticdigits/     # Main package
├── results/                 # Output directory for results
├── train.py                 # Main training script
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/BLShaw/SyntheticDigits.git
cd SyntheticDigits
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training the GAN

To train the GAN with default settings:

```bash
python train.py
```

To train with a custom configuration file:

```bash
python train.py --config configs/default_config.yaml
```

To override specific parameters:

```bash
python train.py --epochs 50 --output_dir custom_results
```

### Running Tests

To run the unit tests:

```bash
python -m pytest tests/
```

## Configuration

The project uses YAML configuration files located in the `configs/` directory. The default configuration (`configs/default_config.yaml`) contains:

- Model parameters (noise dimension, image channels, image size)
- Training parameters (batch size, epochs, learning rate)
- Data parameters (data directory)
- Output parameters (results directory)

You can create custom configurations by copying the default config and modifying the values.

## Results

Training results are saved in the `results/` directory by default and include:

- Generated images at specified epochs
- Training loss plots
- Saved model weights
- Training logs

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct, and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
