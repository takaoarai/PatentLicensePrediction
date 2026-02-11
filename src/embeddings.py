#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Text Embedding Generation for Patent Data

This script generates BERT embeddings for patent abstracts using a pretrained
model specifically designed for patents (anferico/bert-for-patents).

Features:
- Processes patents_{year}_*.parquet files
- Filters for utility patents only
- Generates 768-dimensional embeddings from abstracts
- Skips files that already have embeddings
- Saves embeddings to intermediate directory

Usage:
    python embeddings.py --start_year 2015 --end_year 2020 \\
        --input_dir data/raw --output_dir data/intermediate --batch_size 32
"""
import argparse
import os
import sys
from pathlib import Path
import glob
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModel

from utils import setup_logging

# Embedding column names
EMBED_COLS = [f"text_emb_{i}" for i in range(768)]


def build_argparser():
    """Build argument parser."""
    parser = argparse.ArgumentParser(
        description="Generate BERT embeddings for patent abstracts"
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
        "--input_dir", type=Path, default=Path("data/raw"),
        help="Input directory containing patents_*.parquet files"
    )
    parser.add_argument(
        "--output_dir", type=Path, default=Path("data/intermediate"),
        help="Output directory for processed files"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32,
        help="Batch size for embedding generation"
    )
    parser.add_argument(
        "--log_dir", type=str, default="logs",
        help="Directory for log files"
    )
    return parser


@torch.inference_mode()
def encode_texts(texts, tokenizer, model, device, batch_size):
    """
    Encode texts into embeddings using BERT model.
    
    Args:
        texts: List of text strings
        tokenizer: Pretrained tokenizer
        model: Pretrained BERT model
        device: Device to run inference on
        batch_size: Batch size for processing
    
    Returns:
        numpy.ndarray: Array of shape (len(texts), 768) with embeddings
    """
    all_vecs = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        
        # Tokenize
        tokens = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(device)
        
        # Get CLS token embeddings (768-dim)
        output = model(**tokens).last_hidden_state[:, 0, :]
        all_vecs.append(output.cpu())
    
    return torch.cat(all_vecs, dim=0).numpy()


def process_file(input_path, output_path, tokenizer, model, device, batch_size, logger):
    """
    Process a single parquet file: filter utility patents and add embeddings.
    
    Args:
        input_path: Path to input parquet file
        output_path: Path to output parquet file
        tokenizer: Pretrained tokenizer
        model: Pretrained BERT model
        device: Device for inference
        batch_size: Batch size for encoding
        logger: Logger instance
    """
    if not input_path.exists():
        logger.warning(f"Input file not found: {input_path}")
        return
    
    # Load data
    df = pd.read_parquet(input_path)
    
    # Check for required columns
    if "abstract" not in df.columns:
        logger.warning(f"Skipping {input_path.name} - 'abstract' column not found")
        return
    
    if "patent_type" not in df.columns:
        logger.warning(f"Skipping {input_path.name} - 'patent_type' column not found")
        return
    
    # Check if embeddings already exist
    if set(EMBED_COLS).issubset(df.columns):
        logger.info(f"Skipping {input_path.name} - embeddings already exist")
        return
    
    # Filter for utility patents only
    original_rows = len(df)
    df_utility = df[df["patent_type"] == "utility"].copy()
    utility_rows = len(df_utility)
    
    if utility_rows == 0:
        logger.warning(
            f"Skipping {input_path.name} - no utility patents found "
            f"(total {original_rows} rows)"
        )
        return
    
    logger.info(
        f"Processing {input_path.name} - extracted {utility_rows} utility patents "
        f"from {original_rows} total rows"
    )
    
    # Extract abstracts (fill empty with empty string)
    texts = df_utility["abstract"].fillna("").tolist()
    
    logger.info(f"Generating embeddings for {utility_rows} patents...")
    
    # Generate embeddings
    embeddings = encode_texts(texts, tokenizer, model, device, batch_size)
    
    # Add embedding columns
    for i, col in enumerate(EMBED_COLS):
        df_utility[col] = embeddings[:, i]
    
    # Save to output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_utility.to_parquet(output_path, index=False)
    
    logger.info(
        f"Completed {input_path.name} -> {output_path.name} "
        f"({utility_rows} utility patents with 768-dim embeddings)"
    )


def main():
    """Main execution function."""
    args = build_argparser().parse_args()
    
    # Setup logging
    logger = setup_logging(log_dir=args.log_dir)
    logger.info("=== Patent Text Embedding Generation ===")
    logger.info(f"Input directory: {args.input_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Year range: {args.start_year}-{args.end_year}")
    logger.info(f"Batch size: {args.batch_size}")
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup device and model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    logger.info("Loading pretrained model: anferico/bert-for-patents")
    tokenizer = AutoTokenizer.from_pretrained("anferico/bert-for-patents")
    model = AutoModel.from_pretrained("anferico/bert-for-patents").to(device)
    model.eval()
    
    # Process files for each year
    years = range(args.start_year, args.end_year + 1)
    total_files_processed = 0
    
    for year in years:
        # Find files matching patents_{year}_*.parquet
        pattern = str(args.input_dir / f"patents_{year}_*.parquet")
        input_files = glob.glob(pattern)
        
        if not input_files:
            logger.warning(f"No files found for year {year}: {pattern}")
            continue
        
        logger.info(f"Year {year}: found {len(input_files)} files to process")
        
        for input_file_str in sorted(input_files):
            input_file = Path(input_file_str)
            # Output file has the same name as input file
            output_file = args.output_dir / input_file.name
            
            process_file(
                input_file, output_file, tokenizer, model, device,
                args.batch_size, logger
            )
            total_files_processed += 1
    
    logger.info("=== Embedding Generation Complete ===")
    logger.info(f"Total files processed: {total_files_processed}")


if __name__ == "__main__":
    main()
