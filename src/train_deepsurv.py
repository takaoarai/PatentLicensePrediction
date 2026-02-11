#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepSurv Model Training

DeepSurv is a deep learning extension of Cox proportional hazards model.
Instead of a linear predictor, it uses a neural network to estimate risk scores.

Features:
- Deep neural network for risk score prediction
- Cox partial likelihood loss function
- Handles variable-length categorical features (CPC codes, applicants)

Reference:
    Katzman et al. (2018). DeepSurv: personalized treatment recommender system
    using a Cox proportional hazards deep neural network. BMC Medical Research Methodology.

Usage:
    python train_deepsurv.py --data_dir data/processed \\
        --model_path models/deepsurv.pt --results_path results/deepsurv.json \\
        --train_start 2000 --train_end 2015 --val_start 2016 --val_end 2017
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

from utils import setup_logging, load_preprocessed_data, save_json, custom_collate_fn


# ============================================================
# Model Definition
# ============================================================

class DeepSurv(nn.Module):
    """
    DeepSurv: Deep learning extension of Cox proportional hazards model.
    
    Architecture:
    - Embedding layers for categorical features (CPC codes, applicants)
    - Deep neural network for risk score prediction
    - Single output node (unconstrained risk score)
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
        
        # Risk score prediction network (nonlinear extension of Cox linear predictor)
        feature_dim = emb_dim + cpc_emb_dim + app_emb_dim + 2
        
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
            nn.Linear(128, 1)  # Risk score (unconstrained)
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
        
        # Risk score
        risk_score = self.network(x).squeeze(1)
        
        return risk_score


# ============================================================
# Dataset Definition
# ============================================================

class DeepSurvDataset(Dataset):
    """Dataset for DeepSurv (survival analysis)."""
    
    def __init__(self, df, max_cpc_len=10, max_app_len=5):
        self.df = df.reset_index(drop=True)
        self.max_cpc_len = max_cpc_len
        self.max_app_len = max_app_len
        self._prepare()
    
    def _prepare(self):
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
                # Remove duplicates while preserving order
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
        
        # Survival data
        self.t_normalized = torch.tensor(self.df['t_normalized'].values, dtype=torch.float32)
        self.s = torch.tensor(self.df['s'].values, dtype=torch.float32)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        return (
            self.num_claims[idx],
            self.backward_cites[idx],
            torch.tensor(self.embeddings[idx], dtype=torch.float32),
            self.cpc_ids[idx],
            self.applicant_ids[idx],
            self.t_normalized[idx],
            self.s[idx]
        )


# ============================================================
# Loss Function
# ============================================================

def cox_partial_likelihood_loss(risk_scores, t_normalized, s):
    """
    Cox partial likelihood loss function.
    
    Args:
        risk_scores: Model predicted risk scores [batch_size]
        t_normalized: Normalized observation time [batch_size]
        s: Event indicator [batch_size]
    
    Returns:
        Negative log partial likelihood
    """
    # Focus on event samples only
    event_mask = s >= 0.5  # Safe comparison for float values
    
    if event_mask.sum() == 0:
        # Return zero connected to computation graph (for gradient)
        return risk_scores.sum() * 0.0
    
    # Sort by event time
    sorted_indices = torch.argsort(t_normalized)
    risk_scores_sorted = risk_scores[sorted_indices]
    s_sorted = s[sorted_indices]
    
    # Exponential of risk scores
    exp_risk = torch.exp(risk_scores_sorted)
    
    # Cumulative sum of risk set (reverse order)
    risk_set_sum = torch.flip(torch.cumsum(torch.flip(exp_risk, [0]), 0), [0])
    
    # Partial likelihood
    log_likelihood = torch.sum(
        s_sorted * (risk_scores_sorted - torch.log(risk_set_sum + 1e-8))
    )
    
    # Number of events
    n_events = s_sorted.sum()
    
    # Negative log likelihood
    loss = -log_likelihood / (n_events + 1e-8)
    
    return loss


# ============================================================
# Training Function
# ============================================================

def train_model(model, train_loader, val_loader, device, config, logger):
    """Train DeepSurv model."""
    optimizer = Adam(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay']
    )
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-7
    )
    
    best_loss = float('inf')
    patience = 0
    training_history = []
    
    for epoch in range(config['epochs']):
        # Training
        model.train()
        total_loss = 0
        batch_count = 0
        
        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            
            num_claims, backward_cites, embeddings, cpc_ids, app_ids, t_normalized, s = [
                x.to(device) for x in batch
            ]
            
            # Forward pass
            risk_scores = model(num_claims, backward_cites, embeddings, cpc_ids, app_ids)
            
            # Loss calculation
            loss = cox_partial_likelihood_loss(risk_scores, t_normalized, s)
            
            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(f"NaN/Inf loss at epoch {epoch+1}, batch {batch_idx+1}")
                continue
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
        
        if batch_count == 0:
            continue
        
        avg_loss = total_loss / batch_count
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_loss = 0
            val_batch_count = 0
            for batch in val_loader:
                num_claims, backward_cites, embeddings, cpc_ids, app_ids, t_normalized, s = [
                    x.to(device) for x in batch
                ]
                risk_scores = model(num_claims, backward_cites, embeddings, cpc_ids, app_ids)
                loss = cox_partial_likelihood_loss(risk_scores, t_normalized, s)
                
                if not (torch.isnan(loss) or torch.isinf(loss)):
                    val_loss += loss.item()
                    val_batch_count += 1
            
            if val_batch_count == 0:
                val_loss = float('inf')
            else:
                val_loss /= val_batch_count
        
        training_history.append({
            'epoch': epoch + 1,
            'train_loss': float(avg_loss),
            'val_loss': float(val_loss),
            'learning_rate': float(optimizer.param_groups[0]['lr'])
        })
        
        logger.info(
            f"Epoch {epoch+1}/{config['epochs']} | "
            f"Train Loss: {avg_loss:.6f} | Val Loss: {val_loss:.6f}"
        )
        
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_loss:
            best_loss = val_loss
            patience = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss
            }, config['model_path'])
            logger.info(f"  New best model saved (val_loss={best_loss:.6f})")
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
    parser = argparse.ArgumentParser(description='Train DeepSurv model')
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--results_path', type=str, required=True)
    parser.add_argument('--train_start', type=int, required=True)
    parser.add_argument('--train_end', type=int, required=True)
    parser.add_argument('--val_start', type=int, required=True)
    parser.add_argument('--val_end', type=int, required=True)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--early_stop', type=int, default=10)
    parser.add_argument('--log_dir', type=str, default='logs')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(log_dir=args.log_dir)
    logger.info("=== DeepSurv Model Training ===")
    
    config = vars(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    
    # GPU memory clear
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        logger.info(f"GPU: {torch.cuda.get_device_name()}")
        logger.info(
            f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
        )
    
    # Create output directories
    Path(config['model_path']).parent.mkdir(parents=True, exist_ok=True)
    
    # Load data
    logger.info("\n=== Data Loading ===")
    train_df = load_preprocessed_data(
        config['data_dir'], config['train_start'], config['train_end'], logger
    )
    val_df = load_preprocessed_data(
        config['data_dir'], config['val_start'], config['val_end'], logger
    )
    
    # Prepare datasets
    logger.info("\n=== Dataset Preparation ===")
    train_ds = DeepSurvDataset(train_df)
    val_ds = DeepSurvDataset(val_df)
    
    train_loader = DataLoader(
        train_ds, batch_size=config['batch_size'], shuffle=True,
        drop_last=True, collate_fn=custom_collate_fn
    )
    val_loader = DataLoader(
        val_ds, batch_size=config['batch_size'], shuffle=False,
        collate_fn=custom_collate_fn
    )
    
    # Initialize model
    logger.info("\n=== Model Initialization ===")
    model = DeepSurv(
        cpc_vocab_size=129,
        app_vocab_size=207499,
        cpc_pad_idx=0,
        app_pad_idx=0
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable parameters: {total_params:,}")
    
    # Train
    logger.info("\n=== Training ===")
    start_time = datetime.now()
    
    try:
        result = train_model(model, train_loader, val_loader, device, config, logger)
        
        result['model_type'] = 'deepsurv'
        result['n_train_samples'] = len(train_df)
        result['n_val_samples'] = len(val_df)
        
        save_json(result, config['results_path'])
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        logger.info("\n=== Training Complete ===")
        logger.info(f"Duration: {duration}")
        logger.info(f"Best validation loss: {result['best_val_loss']:.6f}")
        logger.info(f"Model saved: {config['model_path']}")
        
    except Exception as e:
        logger.error(f"Training error: {e}")
        import traceback
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        raise


if __name__ == '__main__':
    main()
