"""
Logging utilities for DPO training.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path


def setup_logger(name: str = "DPO", log_file: str = None, level=logging.INFO):
    """
    Set up a logger with console and file handlers.
    
    Args:
        name: Logger name
        log_file: Path to log file (optional)
        level: Logging level
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers
    logger.handlers = []
    
    # Format
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def log_experiment_info(logger, config: dict):
    """
    Log experiment configuration information.
    
    Args:
        logger: Logger instance
        config: Configuration dictionary
    """
    logger.info("="*80)
    logger.info("Experiment Configuration")
    logger.info("="*80)
    
    logger.info(f"Model: {config.get('model', {}).get('name', 'N/A')}")
    logger.info(f"Dataset: {config.get('dataset', {}).get('name', 'N/A')}")
    logger.info(f"Beta (DPO temperature): {config.get('training', {}).get('beta', 'N/A')}")
    logger.info(f"Learning rate: {config.get('training', {}).get('learning_rate', 'N/A')}")
    logger.info(f"Epochs: {config.get('training', {}).get('num_train_epochs', 'N/A')}")
    logger.info(f"Batch size: {config.get('training', {}).get('per_device_train_batch_size', 'N/A')}")
    
    logger.info("="*80)
