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


# ============================================================
# Data Loading for Training
# ============================================================

def load_preprocessed_data(data_dir, start_year, end_year, logger=None):
    """
    Load preprocessed patent data for training.
    
    Args:
        data_dir: Directory containing patents_*.parquet files
        start_year: Start year for data filtering
        end_year: End year for data filtering
        logger: Optional logger instance (if None, uses print)
    
    Returns:
        pandas.DataFrame: Combined DataFrame with all records
    """
    import pandas as pd
    
    data_dir = Path(data_dir)
    
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    log_fn = logger.info if logger else print
    
    log_fn(f"Loading data from: {data_dir}")
    log_fn(f"Year range: {start_year}-{end_year}")
    
    # Find parquet files
    parquet_files = list(data_dir.glob("*.parquet"))
    
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in: {data_dir}")
    
    log_fn(f"Found {len(parquet_files)} parquet files")
    
    # Filter by year
    files = sorted([
        f for f in parquet_files 
        if f.name.startswith("patents_") and f.name.endswith(".parquet")
    ])
    
    selected_files = []
    for f in files:
        try:
            # Extract year from filename: patents_YYYY_*.parquet
            year = int(f.name.split('_')[1])
            if start_year <= year <= end_year:
                selected_files.append(f)
        except (IndexError, ValueError):
            log_fn(f"Warning: Could not extract year from filename: {f.name}")
            continue
    
    if not selected_files:
        raise ValueError(
            f"No files found for the specified period ({start_year}-{end_year})"
        )
    
    log_fn(f"Selected {len(selected_files)} files for processing")
    
    # Load and concatenate dataframes
    dataframes = []
    total_records = 0
    
    for file_path in selected_files:
        try:
            log_fn(f"  Loading: {file_path.name}")
            
            # Check file size
            if file_path.stat().st_size == 0:
                log_fn(f"    Warning: Empty file - skipping")
                continue
            
            # Load parquet file
            df_chunk = pd.read_parquet(file_path)
            
            if len(df_chunk) == 0:
                log_fn(f"    Warning: No data - skipping")
                continue
            
            record_count = len(df_chunk)
            total_records += record_count
            
            log_fn(f"    Loaded: {record_count} records")
            dataframes.append(df_chunk)
                
        except Exception as e:
            log_fn(f"    Error loading file: {e}")
            continue
    
    if not dataframes:
        raise ValueError(f"No valid data could be loaded")
    
    # Concatenate all dataframes
    result_df = pd.concat(dataframes, ignore_index=True)
    
    log_fn(f"\n=== Data Loading Summary ===")
    log_fn(f"Total records: {len(result_df)}")
    log_fn(f"Event rate: {(result_df['s']==1).sum()} / {len(result_df)} "
           f"({(result_df['s']==1).mean()*100:.2f}%)")
    
    return result_df


# ============================================================
# PyTorch Utilities
# ============================================================

def custom_collate_fn(batch):
    """
    Custom collate function for variable-length data (7 elements).
    
    Used for survival analysis models (DeepSurv, Cure).
    
    Args:
        batch: List of tuples (num_claims, backward_cites, embeddings, 
                              cpc_ids, app_ids, t_normalized, s)
    
    Returns:
        Tuple of padded tensors
    """
    import torch
    
    num_claims, backward_cites, embeddings, cpc_ids, app_ids, t_normalized, s = zip(*batch)
    
    # Fixed-length data
    num_claims = torch.stack(num_claims)
    backward_cites = torch.stack(backward_cites)
    embeddings = torch.stack(embeddings)
    t_normalized = torch.stack(t_normalized)
    s = torch.stack(s)
    
    # Variable-length data padding
    max_cpc_len = max(len(ids) for ids in cpc_ids)
    max_app_len = max(len(ids) for ids in app_ids)
    
    # CPC padding (0 is padding value)
    padded_cpc = torch.zeros((len(cpc_ids), max_cpc_len), dtype=torch.long)
    for i, ids in enumerate(cpc_ids):
        if len(ids) > 0:
            padded_cpc[i, :len(ids)] = ids
    
    # Applicant padding (0 is padding value)
    padded_app = torch.zeros((len(app_ids), max_app_len), dtype=torch.long)
    for i, ids in enumerate(app_ids):
        if len(ids) > 0:
            padded_app[i, :len(ids)] = ids
    
    return num_claims, backward_cites, embeddings, padded_cpc, padded_app, t_normalized, s


def custom_collate_fn_6(batch):
    """
    Custom collate function for variable-length data (6 elements).
    
    Used for binary classification models.
    
    Args:
        batch: List of tuples (num_claims, backward_cites, embeddings, 
                              cpc_ids, app_ids, label)
    
    Returns:
        Tuple of padded tensors
    """
    import torch
    
    num_claims, backward_cites, embeddings, cpc_ids, app_ids, labels = zip(*batch)
    
    # Fixed-length data
    num_claims = torch.stack(num_claims)
    backward_cites = torch.stack(backward_cites)
    embeddings = torch.stack(embeddings)
    labels = torch.stack(labels)
    
    # Variable-length data padding
    max_cpc_len = max(len(ids) for ids in cpc_ids)
    max_app_len = max(len(ids) for ids in app_ids)
    
    # CPC padding
    padded_cpc = torch.zeros((len(cpc_ids), max_cpc_len), dtype=torch.long)
    for i, ids in enumerate(cpc_ids):
        if len(ids) > 0:
            padded_cpc[i, :len(ids)] = ids
    
    # Applicant padding
    padded_app = torch.zeros((len(app_ids), max_app_len), dtype=torch.long)
    for i, ids in enumerate(app_ids):
        if len(ids) > 0:
            padded_app[i, :len(ids)] = ids
    
    return num_claims, backward_cites, embeddings, padded_cpc, padded_app, labels
