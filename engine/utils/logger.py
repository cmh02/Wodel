"""Wodle - Logger Utility
Author: Chris Hinkson (@cmh02)

The Logger Utility provides a clean, central, and standardized logger for all
logging that is completed in the Wodle Engine.
"""

# Library Imports
import logging
import os
import sys


def get_logger(name: str, log_file: str = "wodle.log", level: int = logging.DEBUG) -> logging.Logger:
    """Retrieves or creates a configured logger.
    
    Args:
        name (str): The name of the module/logger (e.g., __name__).
        log_file (str): The destination file path for logs.
        level (int): The logging level (e.g., logging.INFO).
        
    Returns:
        logging.Logger: A configured logger instance.
    """
    # Grab the base logger and setup
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    # Check if handlers have already been configured to prevent duplicates
    if not logger.handlers:

        # Format: [Timestamp] [Wodle] [ModuleName] [Level] <message>
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [Wodle] [%(name)s] [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        # Stdout/Console handler
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setFormatter(formatter)
        logger.addHandler(stdout_handler)

        # File handler
        try:
            log_dir = os.path.dirname(log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)

            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except OSError as e:
            stdout_handler.flush()
            print(f"Warning: Could not configure file logger: {e}", file=sys.stderr)

    # Return logger
    return logger
