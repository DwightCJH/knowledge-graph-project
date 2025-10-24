"""
general-purpose utility functions for file I/O and path handling.
"""

import json
import os


def ensure_dir(path: str):
    """Create a directory if it doesn't already exist."""
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def save_json(data, path: str):
    """Save a Python object (dict or list) as a JSON file."""
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(path: str):
    """Load and return a JSON file as a Python object."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
