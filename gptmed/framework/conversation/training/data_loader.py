"""
Data Loading Utilities for Conversation Model

Loads pre-tokenized data from merged_tokens.jsonl format.
"""

import json
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import List, Tuple, Optional
import logging


logger = logging.getLogger(__name__)


class TokenDataset(Dataset):
    """
    Dataset for pre-tokenized conversations
    
    Loads token sequences from JSONL file and creates training samples.
    Each sample is a constant-length sequence (max_seq_len).
    """
    
    def __init__(
        self,
        data_file: str,
        max_seq_len: int = 512,
        pad_token_id: int = 0,
    ):
        """
        Args:
            data_file: Path to merged_tokens.jsonl file
            max_seq_len: Maximum sequence length for training samples
            pad_token_id: Token ID used for padding
        """
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id
        self.tokens = []
        
        # Load tokens from JSONL file
        self._load_tokens(data_file)
        
        # Create training samples from continuous token stream
        self._create_samples()
    
    def _load_tokens(self, data_file: str):
        """Load all tokens from JSONL file"""
        data_file = Path(data_file)
        
        if not data_file.exists():
            raise FileNotFoundError(f"Data file not found: {data_file}")
        
        logger.info(f"Loading tokens from {data_file}...")
        logger.info(f"File size: {data_file.stat().st_size / 1e6:.2f} MB")
        
        num_invalid = 0
        with open(data_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    record = json.loads(line.strip())
                    
                    # Extract tokens from record
                    if 'tokens' in record and record['status'] == 'success':
                        tokens = record['tokens']
                        
                        # Ensure all tokens are within valid range [0, vocab_size)
                        # Clamp any out-of-range tokens to valid range
                        vocab_size = 10000
                        tokens = [min(max(t, 0), vocab_size - 1) for t in tokens]
                        
                        # Count invalid tokens
                        invalid_count = sum(1 for t in record['tokens'] if t < 0 or t >= vocab_size)
                        if invalid_count > 0:
                            num_invalid += invalid_count
                        
                        # Add to continuous stream
                        self.tokens.extend(tokens)
                        
                        # Add separator token between documents (using pad token)
                        self.tokens.append(self.pad_token_id)
                    
                except json.JSONDecodeError as e:
                    logger.warning(f"Error parsing line {line_num}: {str(e)}")
        
        if num_invalid > 0:
            logger.warning(f"Clamped {num_invalid} out-of-range tokens to vocab range [0, {vocab_size-1}]")
        
        logger.info(f"Loaded {len(self.tokens)} tokens from {line_num} records")
    
    def _create_samples(self):
        """Create training samples from token stream"""
        self.samples = []
        
        logger.info(f"Total tokens loaded: {len(self.tokens)}")
        logger.info(f"Max sequence length: {self.max_seq_len}")
        logger.info(f"Required tokens for 1 sample: {self.max_seq_len + 1}")
        
        if len(self.tokens) < self.max_seq_len + 1:
            logger.error(f"ERROR: Total tokens ({len(self.tokens)}) < required ({self.max_seq_len + 1})")
            logger.error("Cannot create any training samples!")
            return
        
        # Sliding window to create samples
        stride = self.max_seq_len // 2
        
        # Use range that ensures we get at least max_seq_len + 1 tokens
        i = 0
        while i + self.max_seq_len < len(self.tokens):
            end_idx = min(i + self.max_seq_len + 1, len(self.tokens))
            sample_tokens = self.tokens[i:end_idx]
            
            # Only keep samples that have exactly the right length
            if len(sample_tokens) == self.max_seq_len + 1:
                self.samples.append(sample_tokens)
            
            i += stride
        
        logger.info(f"Created {len(self.samples)} training samples")
    
    def __len__(self) -> int:
        """Number of samples"""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sample
        
        Returns:
            input_ids: Token IDs [max_seq_len]
            target_ids: Next token IDs [max_seq_len]
        """
        if idx >= len(self.samples):
            raise IndexError(f"Sample index {idx} out of range for {len(self.samples)} samples")
        
        sample = self.samples[idx]
        
        if len(sample) != self.max_seq_len + 1:
            raise ValueError(
                f"Sample {idx} has {len(sample)} tokens, expected {self.max_seq_len + 1}"
            )
        
        # Input: all but last token
        input_ids = torch.tensor(sample[:-1], dtype=torch.long)
        
        # Target: all but first token (shifted by 1)
        target_ids = torch.tensor(sample[1:], dtype=torch.long)
        
        # Verify shapes
        assert input_ids.shape[0] == self.max_seq_len, \
            f"Input shape mismatch: {input_ids.shape[0]} != {self.max_seq_len}"
        assert target_ids.shape[0] == self.max_seq_len, \
            f"Target shape mismatch: {target_ids.shape[0]} != {self.max_seq_len}"
        
        return input_ids, target_ids


def get_data_loaders(
    data_file: str,
    batch_size: int = 32,
    max_seq_len: int = 512,
    train_ratio: float = 0.8,
    num_workers: int = 0,
    pad_token_id: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation data loaders
    
    Args:
        data_file: Path to merged_tokens.jsonl
        batch_size: Batch size
        max_seq_len: Maximum sequence length
        train_ratio: Fraction of data for training
        num_workers: Number of data loading workers
        pad_token_id: Padding token ID
        
    Returns:
        train_loader, val_loader
    """
    # Load dataset
    dataset = TokenDataset(
        data_file=data_file,
        max_seq_len=max_seq_len,
        pad_token_id=pad_token_id,
    )
    
    # Split into train and validation
    train_size = int(len(dataset) * train_ratio)
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Use GPU memory pinning only if CUDA is available
    pin_memory = torch.cuda.is_available()
    logger.info(f"pin_memory enabled: {pin_memory}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Disable multiprocessing for debugging
        pin_memory=pin_memory,  # GPU memory pinning
        drop_last=True,  # Drop incomplete batches
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # Disable multiprocessing for debugging
        pin_memory=pin_memory,  # GPU memory pinning
        drop_last=False,
    )
    
    logger.info(f"Train: {len(train_dataset)}, Validation: {len(val_dataset)}")
    
    return train_loader, val_loader
