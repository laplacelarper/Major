"""Configuration utilities for loading, saving, and setup"""

import logging
import random
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import os

from .config import Config


def load_config(config_path: Path) -> Config:
    """Load configuration from YAML file
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Config object with loaded parameters
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If YAML parsing fails
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    try:
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Create config object from dictionary
        config = _dict_to_config(config_dict)
        return config
        
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing YAML configuration: {e}")


def save_config(config: Config, config_path: Path) -> None:
    """Save configuration to YAML file
    
    Args:
        config: Configuration object to save
        config_path: Path where to save the configuration
    """
    # Ensure directory exists
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert config to dictionary
    config_dict = config.to_dict()
    
    try:
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    except Exception as e:
        raise RuntimeError(f"Error saving configuration: {e}")


def _dict_to_config(config_dict: Dict[str, Any]) -> Config:
    """Convert dictionary to Config object
    
    Args:
        config_dict: Dictionary with configuration parameters
        
    Returns:
        Config object
    """
    # Handle nested dictionaries by creating appropriate dataclass instances
    from .config import PhysicsConfig, ModelConfig, TrainingConfig, DataConfig
    
    # Create sub-configurations
    physics_config = PhysicsConfig()
    model_config = ModelConfig()
    training_config = TrainingConfig()
    data_config = DataConfig()
    
    # Update sub-configurations if present in dictionary
    if 'physics' in config_dict:
        physics_config = PhysicsConfig(**config_dict['physics'])
    
    if 'model' in config_dict:
        model_config = ModelConfig(**config_dict['model'])
    
    if 'training' in config_dict:
        training_config = TrainingConfig(**config_dict['training'])
    
    if 'data' in config_dict:
        data_config = DataConfig(**config_dict['data'])
    
    # Create main config with sub-configurations
    main_config_dict = {k: v for k, v in config_dict.items() 
                       if k not in ['physics', 'model', 'training', 'data']}
    
    config = Config(
        physics=physics_config,
        model=model_config,
        training=training_config,
        data=data_config,
        **main_config_dict
    )
    
    return config


def setup_logging(config: Config) -> logging.Logger:
    """Setup logging configuration
    
    Args:
        config: Configuration object with logging parameters
        
    Returns:
        Configured logger instance
    """
    # Create logs directory
    config.logs_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure logging format
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    
    # Set up root logger
    logger = logging.getLogger("sonar_detection")
    logger.setLevel(getattr(logging, config.log_level.upper()))
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Console handler
    if config.log_to_console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, config.log_level.upper()))
        console_formatter = logging.Formatter(log_format, date_format)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if config.log_to_file:
        run_name = config.run_name or f"run_{config.random_seed}"
        log_file = config.logs_dir / f"{config.experiment_name}_{run_name}.log"
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, config.log_level.upper()))
        file_formatter = logging.Formatter(log_format, date_format)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    # Log initial setup message
    logger.info(f"Logging initialized - Level: {config.log_level}")
    logger.info(f"Log file: {log_file if config.log_to_file else 'Console only'}")
    
    return logger


def set_random_seeds(seed: int, deterministic: bool = True) -> None:
    """Set random seeds for reproducibility
    
    Args:
        seed: Random seed value
        deterministic: Whether to enable deterministic operations
    """
    # Set Python random seed
    random.seed(seed)
    
    # Set NumPy random seed
    np.random.seed(seed)
    
    # Set environment variables for deterministic behavior
    if deterministic:
        os.environ['PYTHONHASHSEED'] = str(seed)
        
        # Note: PyTorch seeds would be set here if PyTorch was available
        # torch.manual_seed(seed)
        # torch.cuda.manual_seed_all(seed)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
    
    logging.getLogger("sonar_detection").info(f"Random seeds set to {seed}, deterministic={deterministic}")


def create_default_config() -> Config:
    """Create a default configuration object
    
    Returns:
        Default Config object
    """
    return Config()


def validate_config_file(config_path: Path) -> bool:
    """Validate that a configuration file is properly formatted
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        True if valid, False otherwise
    """
    try:
        config = load_config(config_path)
        # If we can load and create the config, it's valid
        return True
    except Exception as e:
        logging.getLogger("sonar_detection").error(f"Configuration validation failed: {e}")
        return False


def merge_configs(base_config: Config, override_dict: Dict[str, Any]) -> Config:
    """Merge configuration with override parameters
    
    Args:
        base_config: Base configuration object
        override_dict: Dictionary with parameters to override
        
    Returns:
        New Config object with merged parameters
    """
    # Convert base config to dictionary
    base_dict = base_config.to_dict()
    
    # Deep merge override parameters
    merged_dict = _deep_merge_dicts(base_dict, override_dict)
    
    # Create new config from merged dictionary
    return _dict_to_config(merged_dict)


def _deep_merge_dicts(base_dict: Dict[str, Any], override_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge two dictionaries
    
    Args:
        base_dict: Base dictionary
        override_dict: Dictionary with override values
        
    Returns:
        Merged dictionary
    """
    merged = base_dict.copy()
    
    for key, value in override_dict.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge_dicts(merged[key], value)
        else:
            merged[key] = value
    
    return merged