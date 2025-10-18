"""
Logging utilities for SyntheticDigits GAN implementation.

This module contains functions for setting up logging configuration.
"""

import logging
import os
from datetime import datetime


def setup_logging(output_dir: str = "results", name: str = __name__):
    """Set up logging configuration."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create log filename with timestamp
    log_filename = f"gan_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(output_dir, log_filename)),
            logging.StreamHandler()
        ]
    )
    
    # Return logger instance
    return logging.getLogger(name)