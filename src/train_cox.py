#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Classical Cox Proportional Hazards Model Training

This script trains a classical Cox proportional hazards model using scikit-survival.
Features:
- Linear Cox model with L2 regularization (Ridge)
- Standard survival analysis implementation
- Baseline model for comparison with deep learning approaches

Usage:
    python train_cox.py --data_dir data/processed \\
        --model_path models/cox.pkl --results_path results/cox.json \\
        --train_start 2000 --train_end 2015 --val_start 2016 --val_end 2017
"""
import argparse
import pickle
import numpy as np
from datetime import datetime
from pathlib import Path
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.util import Surv

from utils import setup_logging, load_preprocessed_data, save_json


def prepare_features(df, logger):
    """
    Prepare features for classical Cox regression.
    
    Args:
        df: Input DataFrame
        logger: Logger instance
    
    Returns:
        X: Feature matrix (n_samples, n_features)
        y: Structured array (event, time)
        feature_names: List of feature names
    """
    logger.info("=== Feature Preparation ===")
    
    features = []
    feature_names = []
    
    # 1. Scaled numeric features
    features.append(df['num_claims_scaled'].values.reshape(-1, 1))
    feature_names.append('num_claims_scaled')
    
    features.append(df['backward_citation_count_scaled'].values.reshape(-1, 1))
    feature_names.append('backward_citation_count_scaled')
    
    # 2. Text embeddings (768 dimensions)
    emb_cols = [f'text_emb_{i}_scaled' for i in range(768)]
    embeddings_data = []
    
    for col in emb_cols:
        if col in df.columns:
            embeddings_data.append(df[col].values)
        else:
            logger.warning(f"Embedding column {col} not found. Filling with zeros.")
            embeddings_data.append(np.zeros(len(df)))
    
    embeddings = np.column_stack(embeddings_data)
    features.append(embeddings)
    feature_names.extend(emb_cols)
    
    # 3. CPC code features (one-hot style)
    logger.info("Creating CPC code features...")
    cpc_counts = np.zeros((len(df), 129))  # CPC vocab size
    
    for idx in range(len(df)):
        cpc_ids_str = df.iloc[idx]['cpc_ids']
        if isinstance(cpc_ids_str, str) and cpc_ids_str:
            cpc_id_list = [int(x) for x in cpc_ids_str.split(',') if x.strip()]
            for cpc_id in cpc_id_list:
                if 0 <= cpc_id < 129:
                    cpc_counts[idx, cpc_id] = 1
    
    features.append(cpc_counts)
    feature_names.extend([f'cpc_{i}' for i in range(129)])
    
    # 4. Applicant features (top 100 by frequency)
    logger.info("Creating applicant features...")
    applicant_freq = {}
    for idx in range(len(df)):
        app_ids_str = df.iloc[idx]['app_ids']
        if isinstance(app_ids_str, str) and app_ids_str:
            app_id_list = [int(x) for x in app_ids_str.split(',') if x.strip()]
            for app_id in app_id_list:
                applicant_freq[app_id] = applicant_freq.get(app_id, 0) + 1
    
    top_applicants = sorted(applicant_freq.items(), key=lambda x: x[1], reverse=True)[:100]
    top_app_ids = [app_id for app_id, _ in top_applicants]
    logger.info(f"  Using top 100 applicants by frequency")
    
    app_counts = np.zeros((len(df), len(top_app_ids)))
    for idx in range(len(df)):
        app_ids_str = df.iloc[idx]['app_ids']
        if isinstance(app_ids_str, str) and app_ids_str:
            app_id_list = [int(x) for x in app_ids_str.split(',') if x.strip()]
            for app_id in app_id_list:
                if app_id in top_app_ids:
                    app_idx = top_app_ids.index(app_id)
                    app_counts[idx, app_idx] = 1
    
    features.append(app_counts)
    feature_names.extend([f'app_{i}' for i in range(len(top_app_ids))])
    
    # Concatenate all features
    X = np.hstack(features)
    
    # Create structured array for survival data
    # scikit-survival uses (event, time) format
    # event: True if event occurred, False if censored
    # time: observation time (normalized)
    
    time_values = df['t_normalized'].values.copy()
    
    # Handle negative values (pre-grant licenses)
    negative_mask = time_values < 0
    if negative_mask.any():
        n_negative = negative_mask.sum()
        logger.info(
            f"  Pre-grant licenses (t_normalized < 0) set to near-zero: "
            f"{n_negative} cases ({n_negative/len(time_values)*100:.2f}%)"
        )
        time_values[negative_mask] = 1e-6  # Small positive value for numerical stability
    
    # Handle zero values
    zero_mask = time_values <= 0
    if zero_mask.any():
        n_zero = zero_mask.sum()
        logger.warning(
            f"  Warning: t_normalized == 0 cases converted to near-zero: {n_zero}"
        )
        time_values[zero_mask] = 1e-6
    
    event_values = df['s'].values
    
    # Create structured array
    y = Surv.from_arrays(
        event=event_values.astype(bool),
        time=time_values
    )
    
    logger.info(f"Feature matrix shape: {X.shape}")
    logger.info(f"Number of features: {X.shape[1]}")
    logger.info(f"Number of samples: {X.shape[0]}")
    logger.info(f"Number of events: {np.sum(event_values)}")
    
    return X, y, feature_names


def train_cox_model(X_train, y_train, X_val, y_val, config, logger):
    """
    Train Cox proportional hazards model.
    
    Args:
        X_train: Training features
        y_train: Training survival data
        X_val: Validation features
        y_val: Validation survival data
        config: Configuration dictionary
        logger: Logger instance
    
    Returns:
        model: Trained model
        result: Training results dictionary
    """
    logger.info("=== Cox Proportional Hazards Model Training ===")
    
    # Initialize model
    # alpha: L2 regularization parameter (equivalent to weight_decay)
    model = CoxPHSurvivalAnalysis(
        alpha=config['weight_decay'],
        ties='efron',
        n_iter=config['epochs'],
        tol=1e-9,
        verbose=1
    )
    
    # Train
    logger.info("Training started...")
    start_time = datetime.now()
    
    model.fit(X_train, y_train)
    
    end_time = datetime.now()
    training_time = end_time - start_time
    
    logger.info(f"Training completed: {training_time}")
    
    # Record results
    result = {
        'model_type': 'cox_baseline',
        'training_time': str(training_time),
        'n_features': X_train.shape[1],
        'n_train_samples': X_train.shape[0],
        'n_val_samples': X_val.shape[0],
        'alpha': config['weight_decay'],
        'n_iter': config['epochs']
    }
    
    return model, result


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Train classical Cox proportional hazards model'
    )
    parser.add_argument(
        '--data_dir', type=str, required=True,
        help='Directory with preprocessed data'
    )
    parser.add_argument(
        '--model_path', type=str, required=True,
        help='Path to save trained model'
    )
    parser.add_argument(
        '--results_path', type=str, required=True,
        help='Path to save results JSON'
    )
    parser.add_argument(
        '--train_start', type=int, required=True,
        help='Training data start year'
    )
    parser.add_argument(
        '--train_end', type=int, required=True,
        help='Training data end year'
    )
    parser.add_argument(
        '--val_start', type=int, required=True,
        help='Validation data start year'
    )
    parser.add_argument(
        '--val_end', type=int, required=True,
        help='Validation data end year'
    )
    parser.add_argument(
        '--epochs', type=int, default=100,
        help='Maximum number of iterations'
    )
    parser.add_argument(
        '--weight_decay', type=float, default=0.1,
        help='L2 regularization parameter (alpha)'
    )
    parser.add_argument(
        '--log_dir', type=str, default='logs',
        help='Directory for log files'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(log_dir=args.log_dir)
    logger.info("=== Classical Cox Proportional Hazards Model Training ===")
    
    # Configuration
    config = vars(args)
    
    logger.info("Configuration:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")
    
    # Create output directories
    Path(config['model_path']).parent.mkdir(parents=True, exist_ok=True)
    Path(config['results_path']).parent.mkdir(parents=True, exist_ok=True)
    
    # Load data
    logger.info("\n=== Data Loading ===")
    logger.info("Loading training data...")
    train_df = load_preprocessed_data(
        config['data_dir'], config['train_start'], config['train_end'], logger
    )
    
    logger.info("\nLoading validation data...")
    val_df = load_preprocessed_data(
        config['data_dir'], config['val_start'], config['val_end'], logger
    )
    
    # Prepare features
    logger.info("\nPreparing training features...")
    X_train, y_train, feature_names = prepare_features(train_df, logger)
    
    logger.info("\nPreparing validation features...")
    X_val, y_val, _ = prepare_features(val_df, logger)
    
    # Train model
    start_time = datetime.now()
    
    try:
        model, result = train_cox_model(X_train, y_train, X_val, y_val, config, logger)
        
        # Save model
        logger.info("\nSaving model...")
        with open(config['model_path'], 'wb') as f:
            pickle.dump({
                'model': model,
                'feature_names': feature_names,
                'metadata': result
            }, f)
        logger.info(f"Model saved: {config['model_path']}")
        
        # Save results
        logger.info("Saving results...")
        save_json(result, config['results_path'])
        logger.info(f"Results saved: {config['results_path']}")
        
        # Final summary
        end_time = datetime.now()
        duration = end_time - start_time
        
        logger.info("\n=== Training Complete ===")
        logger.info(f"Total time: {duration}")
        logger.info(f"Model saved: {config['model_path']}")
        logger.info(f"Results saved: {config['results_path']}")
        
    except Exception as e:
        logger.error(f"Training error: {e}")
        import traceback
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        raise


if __name__ == '__main__':
    main()
