#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scaling Parameter and Vocabulary Computation Script

This script computes scaling parameters and vocabularies needed for data transformation:
1. Min/Max values for numeric features (num_claims, backward_citation_count, etc.)
2. Global Min/Max for text embeddings (text_emb_0 ~ text_emb_767)
3. Vocabularies for categorical features (CPC codes, applicant names)

The script streams through all parquet files without loading the entire dataset
into memory, making it efficient for large datasets.

Usage:
    python scaler.py --input_dir data/intermediate --output_dir data/stats \\
        --start_year 2015 --end_year 2020
"""
import argparse
import pandas as pd
import numpy as np
from collections import defaultdict

from utils import setup_logging, find_parquet_files, save_json


def build_argparser():
    """Build argument parser."""
    parser = argparse.ArgumentParser(
        description="Compute scaling parameters and vocabularies for patent data"
    )
    parser.add_argument(
        "--input_dir", type=str, required=True,
        help="Directory containing patents_*.parquet files"
    )
    parser.add_argument(
        "--output_dir", type=str, default="data/stats",
        help="Output directory for scaling parameters and vocabularies"
    )
    parser.add_argument(
        "--start_year", type=int, required=True,
        help="Start year for processing"
    )
    parser.add_argument(
        "--end_year", type=int, required=True,
        help="End year for processing"
    )
    parser.add_argument(
        "--max_files", type=int, default=None,
        help="Maximum number of files to process (for testing)"
    )
    parser.add_argument(
        "--log_dir", type=str, default="logs",
        help="Directory for log files"
    )
    return parser


def compute_scaling_params(files, logger):
    """
    Compute min/max scaling parameters for numeric features and embeddings.
    
    Args:
        files: List of parquet file paths
        logger: Logger instance
    
    Returns:
        dict: Scaling parameters with structure:
            {
                'num_claims': {'min': float, 'max': float},
                'backward_citation_count': {'min': float, 'max': float},
                'embedding': {'min': float, 'max': float}
            }
    """
    # Numeric feature columns
    numeric_cols = [
        "num_claims",
        "backward_citation_count",
    ]
    
    # Initialize: min=+inf, max=-inf
    scaling = {
        col: {"min": float("inf"), "max": float("-inf")}
        for col in numeric_cols
    }
    
    # Embedding global min/max
    emb_min, emb_max = float("inf"), float("-inf")
    
    # Process each file
    processed_files = 0
    for path in files:
        try:
            df = pd.read_parquet(path)
            processed_files += 1
            logger.info(
                f"Processing file {processed_files}/{len(files)}: "
                f"{path.name} ({len(df)} rows)"
            )
        except Exception as e:
            logger.error(f"Error reading {path}: {e}")
            continue
        
        # Update numeric feature min/max
        for col in numeric_cols:
            if col in df.columns:
                col_data = df[col].dropna()
                if len(col_data) > 0:
                    col_min = col_data.min()
                    col_max = col_data.max()
                    scaling[col]["min"] = min(scaling[col]["min"], float(col_min))
                    scaling[col]["max"] = max(scaling[col]["max"], float(col_max))
        
        # Update embedding global min/max
        emb_cols = [c for c in df.columns if c.startswith("text_emb_")]
        if emb_cols:
            emb_data = df[emb_cols].values
            # Remove NaN values
            emb_data_flat = emb_data[~np.isnan(emb_data)]
            if len(emb_data_flat) > 0:
                local_min = np.min(emb_data_flat)
                local_max = np.max(emb_data_flat)
                emb_min = min(emb_min, float(local_min))
                emb_max = max(emb_max, float(local_max))
    
    # Remove features with no valid data
    final_scaling = {}
    for col, vals in scaling.items():
        if vals["min"] != float("inf") and vals["max"] != float("-inf"):
            final_scaling[col] = vals
        else:
            logger.warning(f"No valid data found for column '{col}'")
    
    # Add embedding parameters
    if emb_min != float("inf") and emb_max != float("-inf"):
        final_scaling["embedding"] = {"min": emb_min, "max": emb_max}
        logger.info(f"Text embedding global range: [{emb_min:.6f}, {emb_max:.6f}]")
    else:
        logger.warning("No valid embedding data found")
    
    return final_scaling


def build_vocabularies(files, logger):
    """
    Build vocabularies for categorical features (CPC codes, applicant names).
    
    Args:
        files: List of parquet file paths
        logger: Logger instance
    
    Returns:
        dict: Vocabularies with structure:
            {
                'cpc_to_id': {code: id, ...},
                'id_to_cpc': {id: code, ...},
                'applicant_to_id': {name: id, ...},
                'id_to_applicant': {id: name, ...}
            }
    """
    cpc_counter = defaultdict(int)
    applicant_counter = defaultdict(int)
    
    processed_files = 0
    for path in files:
        try:
            df = pd.read_parquet(path)
            processed_files += 1
            logger.info(
                f"Building vocab - file {processed_files}/{len(files)}: "
                f"{path.name}"
            )
        except Exception as e:
            logger.error(f"Error reading {path}: {e}")
            continue
        
        # Collect CPC codes (first 3 characters)
        if "cpc_codes" in df.columns:
            for codes_str in df["cpc_codes"].dropna():
                if isinstance(codes_str, str) and codes_str:
                    for code in codes_str.split('|'):
                        code = code.strip()
                        if len(code) >= 3:
                            cpc_3char = code[:3]
                            cpc_counter[cpc_3char] += 1
        
        # Collect applicant names
        if "applicant_names" in df.columns:
            for names_str in df["applicant_names"].dropna():
                if isinstance(names_str, str) and names_str:
                    for name in names_str.split('|'):
                        name = name.strip()
                        if name:
                            applicant_counter[name] += 1
    
    # Build CPC vocabulary (sorted by code)
    cpc_codes = sorted(cpc_counter.keys())
    cpc_to_id = {code: idx + 1 for idx, code in enumerate(cpc_codes)}  # Start from 1 (0 is padding)
    id_to_cpc = {idx: code for code, idx in cpc_to_id.items()}
    
    # Build applicant vocabulary (sorted by name)
    applicant_names = sorted(applicant_counter.keys())
    applicant_to_id = {name: idx + 1 for idx, name in enumerate(applicant_names)}  # Start from 1
    id_to_applicant = {idx: name for name, idx in applicant_to_id.items()}
    
    logger.info(f"CPC vocabulary size: {len(cpc_to_id)} unique codes")
    logger.info(f"Applicant vocabulary size: {len(applicant_to_id)} unique names")
    
    return {
        'cpc_to_id': cpc_to_id,
        'id_to_cpc': id_to_cpc,
        'applicant_to_id': applicant_to_id,
        'id_to_applicant': id_to_applicant
    }


def main():
    """Main execution function."""
    args = build_argparser().parse_args()
    
    # Setup logging
    logger = setup_logging(log_dir=args.log_dir)
    logger.info("=== Scaling Parameters and Vocabulary Computation ===")
    logger.info(f"Input directory: {args.input_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Year range: {args.start_year}-{args.end_year}")
    
    # Find target files
    files = find_parquet_files(
        args.input_dir,
        year_range=(args.start_year, args.end_year)
    )
    
    if args.max_files is not None:
        files = files[:args.max_files]
    
    if not files:
        logger.error(
            f"No parquet files found matching patents_*.parquet pattern "
            f"for years {args.start_year}-{args.end_year}"
        )
        return
    
    logger.info(f"Found {len(files)} parquet files to process")
    
    # Compute scaling parameters
    logger.info("\n--- Computing scaling parameters ---")
    scaling_params = compute_scaling_params(files, logger)
    
    logger.info("\nScaling parameters computed:")
    for col, vals in scaling_params.items():
        logger.info(f"  {col}: min={vals['min']:.6f}, max={vals['max']:.6f}")
    
    # Build vocabularies
    logger.info("\n--- Building vocabularies ---")
    vocabularies = build_vocabularies(files, logger)
    
    # Save results
    scaling_output = f"{args.output_dir}/scaling_params.json"
    vocab_output = f"{args.output_dir}/vocabularies.json"
    
    save_json(scaling_params, scaling_output)
    logger.info(f"\nScaling parameters saved to {scaling_output}")
    
    save_json(vocabularies, vocab_output)
    logger.info(f"Vocabularies saved to {vocab_output}")
    
    logger.info("\n=== Computation Complete ===")
    logger.info(f"Processed {len(files)} files successfully")


if __name__ == "__main__":
    main()
