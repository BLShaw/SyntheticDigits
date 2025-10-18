"""
Configuration loader for SyntheticDigits GAN implementation.

This module loads configuration from YAML files and provides a configuration object.
"""

import yaml
import os
from typing import Dict, Any


def load_config(config_path: str = "configs/default_config.yaml") -> Dict[str, Any]:
    """
    Load configuration from a YAML file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Dictionary containing configuration parameters
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
    
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    return config


def get_config_value(config: Dict[str, Any], key_path: str, default=None):
    """
    Get a configuration value using a dot-separated key path.
    
    Args:
        config: Configuration dictionary
        key_path: Dot-separated path to the desired value (e.g., "training.epochs")
        default: Default value to return if key is not found
        
    Returns:
        Configuration value or default
    """
    keys = key_path.split('.')
    value = config
    
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default
    
    return value