import sys
from loguru import logger
import os

def setup_logger(log_file="logs/ingestion.log"):
    # Clear default logger
    logger.remove()
    
    # Add console logger with rich formatting
    logger.add(
        sys.stderr, 
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO"
    )
    
    # Add file logger for detailed debugging
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger.add(
        log_file,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level="DEBUG",
        rotation="10 MB"
    )
    
    return logger

# Example usage
# from src.logging_config import setup_logger
# logger = setup_logger()
