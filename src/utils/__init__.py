# src/utils/__init__.py

# This file makes the 'utils' directory a Python package.
# We can use it to make imports from this package cleaner.

# For example, you can import functions/classes directly here
# so they can be accessed like `from src.utils import function_name`
# instead of `from src.utils.module_name import function_name`.

from .config_loader import load_config
from .logger_setup import setup_logging
from .file_handler import ensure_dir_exists, save_json, load_json

# You can add other utility functions here as they are created,
# for example, functions for common text processing, time formatting, etc.

__all__ = [
    "load_config",
    "setup_logging",
    "ensure_dir_exists",
    "save_json",
    "load_json",
]

print("AISmartMixer 'utils' package initialized.")