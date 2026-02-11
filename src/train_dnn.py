#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Binary Classification DNN Training

This script trains simple feedforward neural networks for binary classification
at different time horizons (e.g., 1 year, 3 years, 5 years).

Features:
- Treats survival analysis as binary classification problem
- Independent models for each time horizon
- Simple DNN architecture with sigmoid output
- Does not use survival-specific loss functions

Usage:
    python train_dnn.py --data_dir data/processed \\
        --model_path models/dnn.pt --results_path results/dnn.json \\
        --train_start 2000 --train_end 2015 --val_start 2016 --val_end 2017 \\
        --horizon_years 1,3,5
"""
import argparse
import torch
import pandas as pd
import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from datetime import datetime
from pathlib import Path

from utils import setup_logging, load_preprocessed_data, save_json, custom_collate_fn_6


# ============================================================
# Label Creation
# ============================================================

def create_binary_labels(df, horizon_days):
    """
    Create binary labels for specified time horizon.
    
    Args:
        df: DataFrame with 's' and 't_normalized' columns
        horizon_days: Time horizon in days
    
    Returns:
        Binary labels: 1 = licensed within horizon, 0 = otherwise
    """
    actual_days = df['t_normalized'].values * 10000
    
    # Licensed within horizon (s==1 AND t <= horizon)
    labels = (
        (df['s'].values == 1) & (actual_days <= horizon_days)
    ).astype(np.float32)
    
    return labels


# ============================================================
# Model Definition
# ============================================================

class SimpleDNN(nn.Module):
    """
    Simple deep neural network for binary classification.
    
    Architecture:
    - Embedding layers for categorical features
    - Feedforward network with ReLU activations
    - Sigmoid output for binary classification
    """
    
    def __init__(self, emb_dim=768, cpc_vocab_size=129, app_vocab_size=207499,
                 cpc_emb_dim=16, app_emb_dim=256, cpc_pad_idx=0, app_pad_idx=0):
        super().__init__()
        
        self.cpc_pad_idx = cpc_pad_idx
        self.app_pad_idx = app_pad_idx
        
        # Embedding layers
        self.cpc_emb = nn.EmbeddingBag(
            cpc_vocab_size + 1, cpc_emb_dim, mode='mean', padding_idx=cpc_pad_idx
        )
        self.app_emb = nn.EmbeddingBag(
            app_vocab_size + 1, app_emb_dim, mode='mean', padding_idx=app_pad_idx
        )
        
        # Feedforward network
        feature_dim = emb_dim + cpc_emb_dim + app_emb_dim + 2  # 768 + 16 + 256 + 2
        
        self.network = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, num_claims, backward_cites, embeddings, cpc_ids, app_ids):
        batch_size = embeddings.shape[0]
        device = embeddings.device
        
        # Process CPC codes
        if cpc_ids.dim() == 2:
            cpc_mask = (cpc_ids != self.cpc_pad_idx)
            if cpc_mask.any():
                flat_cpc = cpc_ids[cpc_mask]
                cpc_lengths = cpc_mask.sum(dim=1)
                cpc_offsets = torch.cat([
                    torch.tensor([0], device=device),
                    cpc_lengths.cumsum(dim=0)[:-1]
                ])
                cpc_vec = self.cpc_emb(flat_cpc, cpc_offsets)
            else:
                cpc_vec = torch.zeros(batch_size, self.cpc_emb.embedding_dim, device=device)
        else:
            if len(cpc_ids) > 0:
                offsets = torch.tensor([0], dtype=torch.long, device=device)
                cpc_vec = self.cpc_emb(cpc_ids, offsets)
            else:
                cpc_vec = torch.zeros(1, self.cpc_emb.embedding_dim, device=device)
        
        # Process applicants
        if app_ids.dim() == 2:
            app_mask = (app_ids != self.app_pad_idx)
            if app_mask.any():
                flat_app = app_ids[app_mask]
                app_lengths = app_mask.sum(dim=1)
                app_offsets = torch.cat([
                    torch.tensor([0], device=device),
                    app_lengths.cumsum(dim=0)[:-1]
                ])
                app_vec = self.app_emb(flat_app, app_offsets)
            else:
                app_vec = torch.zeros(batch_size, self.app_emb.embedding_dim, device=device)
        else:
            if len(app_ids) > 0:
                offsets = torch.tensor([0], dtype=torch.long, device=device)
                app_vec = self.app_emb(app_ids, offsets)
            else:
                app_vec = torch.zeros(1, self.app_emb.embedding_dim, device=device)
        
        # Concatenate features
        x = torch.cat([
            embeddings, cpc_vec, app_vec,
            num_claims.unsqueeze(1),
            backward_cites.unsqueeze(1)
        ], dim=1)
        
        # Binary classification output
        output = self.network(x).squeeze(1)
        
        return output


# ============================================================
# Dataset Definition
# ============================================================

class BinaryDataset(Dataset):
    """Dataset for binary classification."""
    
    def __init__(self, df, labels, max_cpc_len=10, max_app_len=5):
        self.df = df.reset_index(drop=True)
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.max_cpc_len = max_cpc_len
        self.max_app_len = max_app_len
        self._prepare()
    
    def _prepare(self):
        # Numeric features
        self.num_claims = torch.tensor(
            self.df['num_claims_scaled'].values, dtype=torch.float32
        )
        self.backward_cites = torch.tensor(
            self.df['backward_citation_count_scaled'].values, dtype=torch.float32
        )
        
        # Text embeddings
        emb_cols = [f'text_emb_{i}_scaled' for i in range(768)]
        embeddings_data = []
        for col in emb_cols:
            if col in self.df.columns:
                embeddings_data.append(self.df[col].values)
            else:
                embeddings_data.append(np.zeros(len(self.df)))
        self.embeddings = np.column_stack(embeddings_data).astype(np.float32)
        
        # CPC codes
        self.cpc_ids = []
        for idx in range(len(self.df)):
            cpc_ids_str = self.df.iloc[idx]['cpc_ids']
            if isinstance(cpc_ids_str, str) and cpc_ids_str:
                cpc_id_list = [int(x) for x in cpc_ids_str.split(',') if x.strip()]
                cpc_id_list = list(dict.fromkeys(cpc_id_list))
            else:
                cpc_id_list = []
            cpc_id_list = cpc_id_list[:self.max_cpc_len]
            self.cpc_ids.append(torch.tensor(cpc_id_list, dtype=torch.long))
        
        # Applicant IDs
        self.applicant_ids = []
        for idx in range(len(self.df)):
            app_ids_str = self.df.iloc[idx]['app_ids']
            if isinstance(app_ids_str, str) and app_ids_str:
                app_id_list = [int(x) for x in app_ids_str.split(',') if x.strip()]
                app_id_list = list(dict.fromkeys(app_id_list))
            else:
                app_id_list = []
            app_id_list = app_id_list[:self.max_app_len]
            self.applicant_ids.append(torch.tensor(app_id_list, dtype=torch.long))
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        return (
            self.num_claims[idx],
            self.backward_cites[idx],
            torch.tensor(self.embeddings[idx], dtype=torch.float32),
            self.cpc_ids[idx],
            self.applicant_ids[idx],
            self.labels[idx]
        )


# ============================================================
# Training Function
# ============================================================

def train_model(model, train_loader, val_loader, device, config, logger):
    """Train binary classification model."""
    optimizer = Adam(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay']
    )
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-7
    )
    
    criterion = nn.BCELoss()
    
    best_loss = float('inf')
    patience = 0
    training_history = []
    
    for epoch in range(config['epochs']):
        # Training
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            optimizer.zero_grad()
            
            num_claims, backward_cites, embeddings, cpc_ids, app_ids, labels = [
                x.to(device) for x in batch
            ]
            
            # Forward pass
            outputs = model(num_claims, backward_cites, embeddings, cpc_ids, app_ids)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                num_claims, backward_cites, embeddings, cpc_ids, app_ids, labels = [
                    x.to(device) for x in batch
                ]
                outputs = model(num_claims, backward_cites, embeddings, cpc_ids, app_ids)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        training_history.append({
            'epoch': epoch + 1,
            'train_loss': float(avg_loss),
            'val_loss': float(avg_val_loss),
            'learning_rate': float(optimizer.param_groups[0]['lr'])
        })
        
        logger.info(
            f"Epoch {epoch+1}/{config['epochs']} | "
            f"Train Loss: {avg_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f}"
        )
        
        scheduler.step(avg_val_loss)
        
        # Early stopping
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            patience = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss
            }, config['model_path'])
            logger.info(f"  New best model saved (val_loss={best_loss:.4f})")
        else:
            patience += 1
            if patience >= config['early_stop']:
                logger.info("Early stopping triggered")
                break
    
    return {
        'best_val_loss': float(best_loss),
        'training_history': training_history
    }


# ============================================================
# Main Execution
# ============================================================

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Train binary classification DNN'
    )
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--results_path', type=str, required=True)
    parser.add_argument('--train_start', type=int, required=True)
    parser.add_argument('--train_end', type=int, required=True)
    parser.add_argument('--val_start', type=int, required=True)
    parser.add_argument('--val_end', type=int, required=True)
    parser.add_argument(
        '--horizon_years', type=str, default='1,3,5',
        help='Comma-separated time horizons in years (e.g., "1,3,5")'
    )
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--early_stop', type=int, default=10)
    parser.add_argument('--log_dir', type=str, default='logs')
    
    args = parser.parse_args()
    
    # Parse horizons
    horizon_years = [int(x) for x in args.horizon_years.split(',')]
    
    # Setup logging
    logger = setup_logging(log_dir=args.log_dir)
    logger.info("=== Binary Classification DNN Training ===")
    logger.info(f"Time horizons: {horizon_years} years")
    
    config = vars(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    
    # GPU memory clear
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info(f"GPU: {torch.cuda.get_device_name()}")
    
    # Load data
    logger.info("\n=== Data Loading ===")
    train_df = load_preprocessed_data(
        config['data_dir'], config['train_start'], config['train_end'], logger
    )
    val_df = load_preprocessed_data(
        config['data_dir'], config['val_start'], config['val_end'], logger
    )
    
    # Train model for each horizon
    all_results = {}
    
    for horizon_year in horizon_years:
        horizon_days = horizon_year * 365
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Training for {horizon_year}-year horizon ({horizon_days} days)")
        logger.info(f"{'='*60}")
        
        # Create labels
        logger.info("Creating binary labels...")
        train_labels = create_binary_labels(train_df, horizon_days)
        val_labels = create_binary_labels(val_df, horizon_days)
        
        pos_rate_train = train_labels.mean()
        pos_rate_val = val_labels.mean()
        
        logger.info(f"Training positive rate: {pos_rate_train:.4f}")
        logger.info(f"Validation positive rate: {pos_rate_val:.4f}")
        
        # Prepare datasets
        train_ds = BinaryDataset(train_df, train_labels)
        val_ds = BinaryDataset(val_df, val_labels)
        
        train_loader = DataLoader(
            train_ds, batch_size=config['batch_size'], shuffle=True,
            drop_last=True, collate_fn=custom_collate_fn_6
        )
        val_loader = DataLoader(
            val_ds, batch_size=config['batch_size'], shuffle=False,
            collate_fn=custom_collate_fn_6
        )
        
        # Initialize model
        model = SimpleDNN(
            cpc_vocab_size=129,
            app_vocab_size=207499,
            cpc_pad_idx=0,
            app_pad_idx=0
        ).to(device)
        
        # Update model path for this horizon
        model_path_base = Path(config['model_path'])
        horizon_model_path = model_path_base.parent / f"{model_path_base.stem}_{horizon_year}y{model_path_base.suffix}"
        config['model_path'] = str(horizon_model_path)
        
        horizon_model_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Train
        logger.info("Training started...")
        start_time = datetime.now()
        
        result = train_model(model, train_loader, val_loader, device, config, logger)
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        result['model_type'] = 'binary_dnn'
        result['horizon_years'] = horizon_year
        result['horizon_days'] = horizon_days
        result['train_pos_rate'] = float(pos_rate_train)
        result['val_pos_rate'] = float(pos_rate_val)
        result['duration'] = str(duration)
        
        all_results[f'{horizon_year}y'] = result
        
        logger.info(f"\nCompleted {horizon_year}-year model")
        logger.info(f"Duration: {duration}")
        logger.info(f"Best validation loss: {result['best_val_loss']:.4f}")
    
    # Save combined results
    save_json(all_results, config['results_path'])
    
    logger.info("\n=== All Training Complete ===")
    logger.info(f"Results saved: {config['results_path']}")


if __name__ == '__main__':
    main()
