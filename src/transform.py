#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Transformation and Feature Engineering Script

This script applies the final transformations to prepare data for model training:
1. Load scaling parameters and vocabularies
2. Apply min-max scaling to numeric features and embeddings
3. Convert categorical features (CPC, applicants) to IDs
4. Handle missing values in target variables (s, t)
5. Normalize survival time (t)
6. Select only required columns for training

Usage:
    python transform.py --input_dir data/intermediate --output_dir data/processed \\
        --stats_dir data/stats --start_year 2015 --end_year 2020 \\
        --observation_end_date 2022-01-01
"""
import argparse
import pandas as pd
from datetime import datetime
from pathlib import Path

from utils import setup_logging, find_parquet_files, load_json, min_max_scale


def build_argparser():
    """Build argument parser."""
    parser = argparse.ArgumentParser(
        description="Transform patent data for model training"
    )
    parser.add_argument(
        "--input_dir", type=str, required=True,
        help="Input directory with intermediate data"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Output directory for processed data"
    )
    parser.add_argument(
        "--stats_dir", type=str, required=True,
        help="Directory containing scaling_params.json and vocabularies.json"
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
        "--observation_end_date", type=str, default="2022-01-01",
        help="Observation end date for censoring (YYYY-MM-DD format)"
    )
    parser.add_argument(
        "--log_dir", type=str, default="logs",
        help="Directory for log files"
    )
    return parser


def handle_missing_values(df, observation_end_date_ms, logger):
    """
    Handle missing values in target variables (s, t).
    
    Strategy:
    - If s=1 (event occurred) and t is NaN: Remove the record
    - If s=0 (censored) and t is NaN: Impute with observation period
    
    Args:
        df: DataFrame with 's' and 't' columns
        observation_end_date_ms: Observation end date in milliseconds
        logger: Logger instance
    
    Returns:
        pandas.DataFrame: DataFrame with missing values handled
    """
    df = df.copy()
    
    # Check if there are any NaN values in 't'
    nan_mask = df['t'].isna()
    if not nan_mask.any():
        logger.info("No missing values in 't' column")
        return df
    
    initial_rows = len(df)
    
    # Remove records where s=1 and t=NaN (invalid event records)
    invalid_event_mask = (df['s'] == 1) & nan_mask
    n_invalid_events = invalid_event_mask.sum()
    if n_invalid_events > 0:
        df = df[~invalid_event_mask].copy()
        logger.info(
            f"Removed {n_invalid_events} records with s=1 and t=NaN "
            "(invalid event records)"
        )
    
    # Impute t for censored records (s=0 and t=NaN)
    censored_nan_mask = (df['s'] == 0) & df['t'].isna()
    n_censored_nan = censored_nan_mask.sum()
    
    if n_censored_nan > 0:
        logger.info(f"Imputing {n_censored_nan} censored records with missing 't'")
        
        # Convert observation end date to datetime
        censor_date = datetime.fromtimestamp(observation_end_date_ms / 1000)
        
        # Temporary column for application date
        df['application_date_dt'] = pd.to_datetime(df['application_date'], errors='coerce')
        
        # Impute missing values
        for idx in df[censored_nan_mask].index:
            app_date = df.loc[idx, 'application_date_dt']
            if pd.notna(app_date):
                observation_period = max(1, (censor_date - app_date).days)
                df.loc[idx, 't'] = observation_period
        
        # Drop temporary column
        df = df.drop(columns=['application_date_dt'])
        
        logger.info(f"Imputed {n_censored_nan} censored records with observation period")
    
    final_rows = len(df)
    logger.info(
        f"Missing value handling complete: {initial_rows} -> {final_rows} rows "
        f"({initial_rows - final_rows} removed)"
    )
    
    return df


def apply_scaling(df, scaling_params, logger):
    """
    Apply min-max scaling to numeric features and embeddings.
    
    Args:
        df: Input DataFrame
        scaling_params: Dictionary with scaling parameters
        logger: Logger instance
    
    Returns:
        pandas.DataFrame: DataFrame with scaled features
    """
    df = df.copy()
    
    # Scale numeric features
    if 'num_claims' in df.columns and 'num_claims' in scaling_params:
        params = scaling_params['num_claims']
        df['num_claims_scaled'] = min_max_scale(
            df['num_claims'], params['min'], params['max']
        )
        logger.info(f"Scaled num_claims: [{params['min']:.2f}, {params['max']:.2f}] -> [0, 1]")
    
    if 'backward_citation_count' in df.columns and 'backward_citation_count' in scaling_params:
        params = scaling_params['backward_citation_count']
        df['backward_citation_count_scaled'] = min_max_scale(
            df['backward_citation_count'], params['min'], params['max']
        )
        logger.info(
            f"Scaled backward_citation_count: "
            f"[{params['min']:.2f}, {params['max']:.2f}] -> [0, 1]"
        )
    
    # Scale text embeddings
    if 'embedding' in scaling_params:
        emb_min = scaling_params['embedding']['min']
        emb_max = scaling_params['embedding']['max']
        
        for i in range(768):
            emb_col = f'text_emb_{i}'
            if emb_col in df.columns:
                df[f'{emb_col}_scaled'] = min_max_scale(
                    df[emb_col], emb_min, emb_max
                )
        
        logger.info(f"Scaled 768 embedding dimensions: [{emb_min:.6f}, {emb_max:.6f}] -> [0, 1]")
    
    return df


def convert_categorical_features(df, vocabularies, logger):
    """
    Convert categorical features (CPC codes, applicant names) to IDs.
    
    Args:
        df: Input DataFrame
        vocabularies: Dictionary with CPC and applicant vocabularies
        logger: Logger instance
    
    Returns:
        pandas.DataFrame: DataFrame with categorical features converted to IDs
    """
    df = df.copy()
    
    cpc_to_id = vocabularies['cpc_to_id']
    applicant_to_id = vocabularies['applicant_to_id']
    
    # Convert CPC codes to IDs
    if 'cpc_codes' in df.columns:
        cpc_ids_list = []
        for cpc_codes in df['cpc_codes']:
            if isinstance(cpc_codes, str) and cpc_codes:
                # Extract first 3 characters and map to ID
                cpc_id_list = [
                    str(cpc_to_id[code[:3]])
                    for code in cpc_codes.split('|')
                    if code.strip() and len(code) >= 3 and code[:3] in cpc_to_id
                ]
                cpc_ids_list.append(','.join(cpc_id_list))
            else:
                cpc_ids_list.append('')
        
        df['cpc_ids'] = cpc_ids_list
        logger.info(f"Converted CPC codes to IDs (vocab size: {len(cpc_to_id)})")
    
    # Convert applicant names to IDs
    if 'applicant_names' in df.columns:
        app_ids_list = []
        for applicant_names in df['applicant_names']:
            if isinstance(applicant_names, str) and applicant_names:
                # Map names to IDs
                app_id_list = [
                    str(applicant_to_id[name])
                    for name in applicant_names.split('|')
                    if name.strip() and name in applicant_to_id
                ]
                app_ids_list.append(','.join(app_id_list))
            else:
                app_ids_list.append('')
        
        df['app_ids'] = app_ids_list
        logger.info(f"Converted applicant names to IDs (vocab size: {len(applicant_to_id)})")
    
    return df


def normalize_survival_time(df, logger):
    """
    Normalize survival time 't' by dividing by 10000.
    
    Args:
        df: Input DataFrame
        logger: Logger instance
    
    Returns:
        pandas.DataFrame: DataFrame with normalized survival time
    """
    df = df.copy()
    
    if 't' in df.columns:
        df['t_normalized'] = df['t'] / 10000
        logger.info(f"Normalized survival time 't': divided by 10000")
    
    return df


def select_output_columns(df, logger):
    """
    Select only required columns for training.
    
    Required columns:
    - publication_number (ID)
    - s (event indicator)
    - t_normalized (normalized survival time)
    - num_claims_scaled
    - backward_citation_count_scaled
    - cpc_ids
    - app_ids
    - text_emb_0_scaled ~ text_emb_767_scaled (768 dimensions)
    
    Args:
        df: Input DataFrame
        logger: Logger instance
    
    Returns:
        pandas.DataFrame: DataFrame with selected columns
    """
    required_cols = [
        'publication_number',
        's',
        't_normalized',
        'num_claims_scaled',
        'backward_citation_count_scaled',
        'cpc_ids',
        'app_ids'
    ]
    
    # Add scaled embedding columns
    scaled_emb_cols = [f'text_emb_{i}_scaled' for i in range(768)]
    output_cols = required_cols + scaled_emb_cols
    
    # Filter to available columns
    available_cols = [col for col in output_cols if col in df.columns]
    missing_cols = [col for col in output_cols if col not in df.columns]
    
    if missing_cols:
        logger.warning(f"Missing columns: {missing_cols}")
    
    logger.info(f"Selected {len(available_cols)} output columns")
    
    return df[available_cols]


def process_file(input_path, output_path, scaling_params, vocabularies,
                observation_end_date_ms, logger):
    """
    Process a single parquet file through the transformation pipeline.
    
    Args:
        input_path: Path to input parquet file
        output_path: Path to output parquet file
        scaling_params: Scaling parameters dictionary
        vocabularies: Vocabularies dictionary
        observation_end_date_ms: Observation end date in milliseconds
        logger: Logger instance
    """
    try:
        # Load data
        df = pd.read_parquet(input_path)
        logger.info(f"Processing {input_path.name} ({len(df)} rows)")
        
        # Convert boolean 's' column to int if necessary (from BigQuery output)
        if 's' in df.columns and df['s'].dtype == bool:
            df['s'] = df['s'].astype(int)
            logger.info("Converted 's' column from boolean to int")
        
        # Transformation pipeline
        df = handle_missing_values(df, observation_end_date_ms, logger)
        df = apply_scaling(df, scaling_params, logger)
        df = convert_categorical_features(df, vocabularies, logger)
        df = normalize_survival_time(df, logger)
        df_output = select_output_columns(df, logger)
        
        # Save output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_output.to_parquet(output_path, index=False)
        
        logger.info(
            f"Successfully processed and saved to {output_path.name} "
            f"({len(df_output)} rows)\n"
        )
        
    except Exception as e:
        logger.error(f"Error processing file {input_path.name}: {e}")
        raise


def main():
    """Main execution function."""
    args = build_argparser().parse_args()
    
    # Setup logging
    logger = setup_logging(log_dir=args.log_dir)
    logger.info("=== Patent Data Transformation ===")
    logger.info(f"Input directory: {args.input_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Stats directory: {args.stats_dir}")
    logger.info(f"Year range: {args.start_year}-{args.end_year}")
    logger.info(f"Observation end date: {args.observation_end_date}")
    
    # Load scaling parameters and vocabularies
    scaling_path = Path(args.stats_dir) / "scaling_params.json"
    vocab_path = Path(args.stats_dir) / "vocabularies.json"
    
    logger.info(f"\nLoading scaling parameters from {scaling_path}")
    scaling_params = load_json(scaling_path)
    
    logger.info(f"Loading vocabularies from {vocab_path}")
    vocabularies = load_json(vocab_path)
    
    # Convert observation end date to milliseconds
    observation_end_date = datetime.strptime(args.observation_end_date, '%Y-%m-%d')
    observation_end_date_ms = int(observation_end_date.timestamp() * 1000)
    
    # Find files to process
    files = find_parquet_files(
        args.input_dir,
        year_range=(args.start_year, args.end_year)
    )
    
    if not files:
        logger.error(
            f"No parquet files found for years {args.start_year}-{args.end_year}"
        )
        return
    
    logger.info(f"\nFound {len(files)} files to process\n")
    
    # Process each file
    for file_path in files:
        output_path = Path(args.output_dir) / file_path.name
        process_file(
            file_path, output_path, scaling_params, vocabularies,
            observation_end_date_ms, logger
        )
    
    logger.info("=== Transformation Complete ===")
    logger.info(f"Processed {len(files)} files successfully")


if __name__ == "__main__":
    main()
