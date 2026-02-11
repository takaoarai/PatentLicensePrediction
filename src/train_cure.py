#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cox Cure Rate Model Training

This script trains a Cox proportional hazards model with cure fraction (cure rate model).
The model assumes that a portion of the population will never experience the event (cured),
while the remaining population follows a Cox proportional hazards model.

Model Components:
1. Cure Fraction: P(cured) = 1 - p(X) where p(X) is the probability of being susceptible
2. Survival for Susceptible: S(t|X, not cured) follows Cox proportional hazards
3. Baseline Hazard: Modeled directly using a neural network

Key Features:
- Semi-parametric approach: parametric cure fraction + neural baseline hazard
- Numerical integration for cumulative hazard calculation
- Complex likelihood combining cured and susceptible populations
"""

import argparse
import torch
import pandas as pd
import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR
from datetime import datetime
from pathlib import Path

from utils import setup_logging, load_preprocessed_data, save_json, custom_collate_fn


# ============================================================
# Baseline Hazard Network
# ============================================================

class BaselineHazardNet(nn.Module):
    """
    Neural network to directly model baseline hazard function h₀(t).
    
    The baseline hazard is constrained to be non-negative using Softplus activation.
    """
    
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus()  # Non-negative constraint
        )
    
    def forward(self, t):
        """
        Args:
            t: Time tensor [batch_size] or [batch_size, 1]
        
        Returns:
            Baseline hazard h₀(t) [batch_size]
        """
        if t.dim() == 1:
            t = t.unsqueeze(1)
        hazard = self.net(t)
        return hazard.squeeze(1)


# ============================================================
# Numerical Integration Utility
# ============================================================

def numerical_integration_trapezoid(hazard_net, t_end, n_steps=100):
    """
    Compute cumulative hazard function using trapezoidal rule.
    
    Integrates h₀(t) from 0 to t_end to get cumulative hazard Λ₀(t).
    
    Args:
        hazard_net: Baseline hazard network
        t_end: Integration upper limit (tensor) [batch_size]
        n_steps: Number of integration steps
    
    Returns:
        Cumulative hazard Λ₀(t) [batch_size]
    """
    device = t_end.device
    batch_size = t_end.shape[0]
    
    # Determine step size based on maximum t_end
    t_max = torch.max(t_end)
    dt = t_max / n_steps
    
    # Create integration points [n_steps+1, batch_size]
    t_points = torch.arange(0, n_steps + 1, device=device, dtype=torch.float32) * dt
    t_points = t_points.unsqueeze(1).expand(-1, batch_size)
    
    # Clip at each sample's t_end
    t_end_expanded = t_end.unsqueeze(0).expand(n_steps + 1, -1)
    t_points = torch.min(t_points, t_end_expanded)
    
    # Evaluate hazard function
    t_flat = t_points.flatten()
    hazard_values = hazard_net(t_flat)
    hazard_values = hazard_values.view(n_steps + 1, batch_size)
    
    # Trapezoidal rule: ∫f(x)dx ≈ Σ[f(x_i) + f(x_{i+1})] * dx/2
    cumulative_hazard = torch.zeros(batch_size, device=device)
    
    for i in range(batch_size):
        t_sample = t_end[i]
        n_steps_sample = int((t_sample / dt).item())
        n_steps_sample = min(n_steps_sample, n_steps)
        
        if n_steps_sample > 0:
            dt_sample = t_sample / n_steps_sample
            h_values = hazard_values[:n_steps_sample+1, i]
            
            # Apply trapezoidal rule
            integral = torch.sum(h_values[:-1] + h_values[1:]) * dt_sample / 2
            cumulative_hazard[i] = integral
    
    return cumulative_hazard


# ============================================================
# Model Definition
# ============================================================

class CoxCureModel(nn.Module):
    """
    Semi-parametric Cox Cure Rate Model.
    
    Architecture:
    1. Shared feature encoder
    2. Cure fraction predictor: P(susceptible to licensing)
    3. Cox linear predictor: log hazard ratio
    4. Baseline hazard network: h₀(t)
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
        
        # Shared feature encoder
        feature_dim = emb_dim + cpc_emb_dim + app_emb_dim + 2  # 1042
        self.feature_encoder = nn.Sequential(
            nn.Linear(feature_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Cure fraction predictor (licensability predictor)
        self.licensability_predictor = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()  # P(susceptible to licensing) ∈ [0, 1]
        )
        
        # Cox linear predictor (log hazard ratio)
        self.cox_predictor = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # β'X (unconstrained)
        )
        
        # Baseline hazard network
        self.baseline_hazard_net = BaselineHazardNet(hidden_dim=64)
    
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
        
        # Shared encoding
        encoded = self.feature_encoder(x)
        
        # Two outputs
        p_licensable = self.licensability_predictor(encoded).squeeze(1)
        cox_linear = self.cox_predictor(encoded).squeeze(1)
        
        return p_licensable, cox_linear


# ============================================================
# Dataset Definition
# ============================================================

class CureDataset(Dataset):
    """Dataset for Cox Cure Rate Model."""
    
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

def cox_cure_rate_loss(p_licensable, cox_linear, t_normalized, s, baseline_hazard_net):
    """
    Cox Cure Rate Model loss function.
    
    The likelihood combines two populations:
    1. Cured (never licensed): probability (1 - p)
    2. Susceptible (may be licensed): probability p, follows Cox model
    
    For event (s=1):
        L = p × h(t|X) × S(t|X)
    
    For censoring (s=0):
        L = p × S(t|X) + (1-p)
    
    where:
        h(t|X) = h₀(t) × exp(β'X)  : hazard function
        S(t|X) = exp(-Λ₀(t) × exp(β'X))  : survival function
        Λ₀(t) = ∫₀ᵗ h₀(u)du  : cumulative baseline hazard
    
    Args:
        p_licensable: Probability of being susceptible [batch_size]
        cox_linear: Cox linear predictor β'X [batch_size]
        t_normalized: Normalized time [batch_size]
        s: Event indicator [batch_size]
        baseline_hazard_net: Baseline hazard network
    
    Returns:
        Negative log likelihood
    """
    # Numerical stability
    p_licensable = torch.clamp(p_licensable, min=1e-8, max=1-1e-8)
    cox_linear = torch.clamp(cox_linear, min=-10, max=10)
    t_normalized = torch.clamp(t_normalized, min=1e-6, max=10.0)
    
    # Baseline hazard function
    hazard_0 = baseline_hazard_net(t_normalized)  # h₀(t)
    
    # Cumulative baseline hazard via numerical integration
    cum_hazard_0 = numerical_integration_trapezoid(
        baseline_hazard_net, t_normalized, n_steps=50
    )  # Λ₀(t)
    
    # Clamp for numerical stability
    hazard_0 = torch.clamp(hazard_0, min=1e-8, max=100.0)
    cum_hazard_0 = torch.clamp(cum_hazard_0, min=1e-8, max=100.0)
    
    # Cox components
    exp_cox = torch.exp(cox_linear)
    h_t = hazard_0 * exp_cox                    # h(t|X) = h₀(t) × exp(β'X)
    S_t = torch.exp(-cum_hazard_0 * exp_cox)    # S(t|X) = exp(-Λ₀(t) × exp(β'X))
    
    # Cure Rate Model likelihood
    # Event: p × h(t|X) × S(t|X)
    event_likelihood = s * p_licensable * h_t * S_t
    
    # Censoring: p × S(t|X) + (1-p)
    censoring_likelihood = (1-s) * (p_licensable * S_t + (1-p_licensable))
    
    # Total likelihood
    total_likelihood = event_likelihood + censoring_likelihood
    
    # Negative log likelihood
    neg_log_likelihood = -torch.log(total_likelihood + 1e-8)
    loss = neg_log_likelihood.mean()
    
    return loss


# ============================================================
# Training Function
# ============================================================

def warmup_lambda(epoch, warmup_epochs=5):
    """Learning rate warmup schedule."""
    if epoch < warmup_epochs:
        return (epoch + 1) / warmup_epochs
    return 1.0


def train_model(model, train_loader, val_loader, device, config, logger):
    """Train Cox Cure Rate Model."""
    optimizer = Adam(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay']
    )
    
    # Warmup + ReduceLROnPlateau
    warmup_scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: warmup_lambda(epoch, warmup_epochs=5)
    )
    main_scheduler = ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-7
    )
    
    best_loss = float('inf')
    patience = 0
    training_history = []
    warmup_complete = False
    
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
            p_licensable, cox_linear = model(
                num_claims, backward_cites, embeddings, cpc_ids, app_ids
            )
            
            # Loss calculation
            loss = cox_cure_rate_loss(
                p_licensable, cox_linear, t_normalized, s,
                model.baseline_hazard_net
            )
            
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
                p_licensable, cox_linear = model(
                    num_claims, backward_cites, embeddings, cpc_ids, app_ids
                )
                loss = cox_cure_rate_loss(
                    p_licensable, cox_linear, t_normalized, s,
                    model.baseline_hazard_net
                )
                
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
        
        # Learning rate scheduling
        if epoch < 5:
            warmup_scheduler.step()
        else:
            if not warmup_complete:
                logger.info("Warmup complete, switching to ReduceLROnPlateau")
                warmup_complete = True
            main_scheduler.step(val_loss)
        
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
    parser = argparse.ArgumentParser(
        description='Train Cox Cure Rate Model'
    )
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
    logger.info("=== Cox Cure Rate Model Training ===")
    
    config = vars(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    
    # GPU memory clear
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        logger.info(f"GPU: {torch.cuda.get_device_name()}")
    
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
    train_ds = CureDataset(train_df)
    val_ds = CureDataset(val_df)
    
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
    model = CoxCureModel(
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
        
        result['model_type'] = 'cox_cure'
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
