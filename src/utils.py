#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Common utility functions for patent data preprocessing pipeline.
"""
import os
import json
import logging
from pathlib import Path
from datetime import datetime
import numpy as np


def setup_logging(log_dir=None, log_level=logging.INFO):
    """
    Setup logging configuration.
    
    Args:
        log_dir: Directory to save log files (default: ./logs)
        log_level: Logging level (default: INFO)
    
    Returns:
        logger: Configured logger instance
    """
    if log_dir is None:
        log_dir = Path("./logs")
    else:
        log_dir = Path(log_dir)
    
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'preprocessing_{timestamp}.log'
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    logger = logging.getLogger()
    logger.setLevel(log_level)
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # File handler
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def load_json(filepath):
    """
    Load JSON file.
    
    Args:
        filepath: Path to JSON file
    
    Returns:
        dict: Loaded JSON data
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data, filepath, indent=4):
    """
    Save data to JSON file with NumPy type conversion.
    
    Args:
        data: Data to save
        filepath: Output file path
        indent: JSON indentation (default: 4)
    """
    def numpy_converter(obj):
        """Convert NumPy types to Python native types for JSON serialization."""
        if hasattr(obj, "item"):
            return obj.item()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
    
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False, default=numpy_converter)


def find_parquet_files(directory, pattern="patents_*.parquet", year_range=None):
    """
    Find parquet files matching the pattern in the directory.
    
    Args:
        directory: Directory to search
        pattern: File pattern to match (default: "patents_*.parquet")
        year_range: Tuple of (start_year, end_year) to filter files (optional)
    
    Returns:
        list: Sorted list of matching file paths
    """
    directory = Path(directory)
    files = []
    
    for f in os.listdir(directory):
        if f.startswith("patents_") and f.endswith(".parquet"):
            if year_range is not None:
                start_year, end_year = year_range
                try:
                    # Extract year from filename: patents_YYYY_*.parquet
                    parts = f.split('_')
                    if len(parts) >= 2:
                        year = int(parts[1])
                        if start_year <= year <= end_year:
                            files.append(directory / f)
                except (ValueError, IndexError):
                    continue
            else:
                files.append(directory / f)
    
    return sorted(files)


def get_year_from_filename(filename):
    """
    Extract year from patent filename.
    
    Args:
        filename: Filename in format patents_YYYY_*.parquet
    
    Returns:
        int: Extracted year, or None if extraction fails
    """
    try:
        parts = Path(filename).stem.split('_')
        if len(parts) >= 2:
            return int(parts[1])
    except (ValueError, IndexError):
        pass
    return None


def min_max_scale(series, min_val, max_val):
    """
    Apply min-max scaling to a pandas Series.
    
    Args:
        series: Input pandas Series
        min_val: Minimum value for scaling
        max_val: Maximum value for scaling
    
    Returns:
        pandas.Series: Scaled series in range [0, 1]
    """
    return (series - min_val) / (max_val - min_val + 1e-6)


def parse_ids_string(ids_str, max_len=None):
    """
    Parse comma-separated ID string and return list of integers.
    
    Args:
        ids_str: Comma-separated ID string (e.g., "1,2,3")
        max_len: Maximum number of IDs to return (optional)
    
    Returns:
        list: List of unique integer IDs
    """
    if not isinstance(ids_str, str) or not ids_str:
        return []
    
    id_list = [int(x) for x in ids_str.split(',') if x.strip()]
    # Remove duplicates while preserving order
    id_list = list(dict.fromkeys(id_list))
    
    if max_len is not None:
        id_list = id_list[:max_len]
    
    return id_list


def timestamp_to_days(timestamp_ms):
    """
    Convert timestamp in milliseconds to days since epoch.
    
    Args:
        timestamp_ms: Timestamp in milliseconds
    
    Returns:
        int: Number of days since epoch
    """
    return int(timestamp_ms / (1000 * 60 * 60 * 24))
