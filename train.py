"""
Main entry point for the SyntheticDigits GAN training project.

This script provides the main interface for training the GAN model.
"""

import argparse
import os
import sys

# Add the src directory to the path so we can import our SyntheticDigits modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from trainers.gan_trainer import GANTrainer
from utils.config import load_config
from utils.logging import setup_logging


def main():
    parser = argparse.ArgumentParser(description="Train a SyntheticDigits GAN model")
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/default_config.yaml",
        help="Path to the configuration file"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="Number of epochs to train (overrides config value)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Output directory for results (overrides config value)"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config values if provided as command line arguments
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.output_dir:
        config['output']['output_dir'] = args.output_dir
    
    # Set up logging
    logger = setup_logging(config['output']['output_dir'])
    
    # Initialize and run the trainer
    trainer = GANTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()