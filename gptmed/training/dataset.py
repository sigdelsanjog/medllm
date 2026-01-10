"""
PyTorch Dataset for Tokenized Data

PURPOSE:
Load tokenized sequences from numpy arrays and provide them as PyTorch tensors
for training. Handles batching and shuffling efficiently.

WHAT THIS FILE DOES:
1. Load tokenized data from .npy files
2. Create training examples for next-token prediction
3. Provide batches to DataLoader

TRAINING OBJECTIVE EXPLAINED:
For causal language modeling, each sequence is both input and target:
- Input:  [token_0, token_1, token_2, ..., token_n-1]
- Target: [token_1, token_2, token_3, ..., token_n]

The model learns to predict token_i given tokens [0, 1, ..., i-1].

PACKAGES USED:
- torch: PyTorch tensors and Dataset
- numpy: Load .npy files

FILES FROM THIS PROJECT:
- data/tokenized/train.npy (created by tokenize_data.py)
- data/tokenized/val.npy

TENSOR SHAPES:
- Loaded data: [num_sequences, seq_len]
- Each batch: [batch_size, seq_len]
- Input: [batch_size, seq_len]
- Target: [batch_size, seq_len]

COMMON FAILURE MODES:
- Wrong data path → FileNotFoundError
- Mismatched shapes → crashes during training
- Not shuffling train data → poor generalization
- Loading entire dataset into memory → OOM (for large datasets)
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path


class TokenizedDataset(Dataset):
    """
    Dataset for tokenized sequences.

    This is a simple in-memory dataset. For larger datasets, you'd use
    memory-mapped arrays or streaming.

    Each item returns (input_ids, target_ids) for next-token prediction.
    """

    def __init__(self, data_path: Path):
        """
        Args:
            data_path: Path to .npy file with tokenized sequences
        """
        self.data_path = Path(data_path)

        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")

        # Load tokenized sequences
        # Shape: [num_sequences, seq_len]
        self.data = np.load(self.data_path)

        print(f"Loaded dataset from {data_path}")
        print(f"  Shape: {self.data.shape}")
        print(f"  Dtype: {self.data.dtype}")
        print(f"  Num sequences: {len(self.data)}")

    def __len__(self) -> int:
        """Number of sequences in dataset."""
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple:
        """
        Get a single training example.

        Args:
            idx: Sequence index

        Returns:
            (input_ids, target_ids) tuple

        Training objective:
            input:  [t0, t1, t2, ..., tn-1]
            target: [t1, t2, t3, ..., tn]

        The model predicts token at position i using tokens [0, ..., i-1].
        """
        sequence = self.data[idx]

        # Convert to torch tensor
        sequence = torch.from_numpy(sequence).long()

        # For causal LM, input and target are the same sequence, shifted by 1
        # Input:  [0, 1, 2, 3, ..., n-1]
        # Target: [1, 2, 3, 4, ..., n]
        input_ids = sequence[:-1]  # All tokens except last
        target_ids = sequence[1:]  # All tokens except first

        return input_ids, target_ids


def create_dataloaders(
    train_path: Path, val_path: Path, batch_size: int, num_workers: int = 0
) -> tuple:
    """
    Create train and validation dataloaders.

    Args:
        train_path: Path to train .npy file
        val_path: Path to validation .npy file
        batch_size: Batch size
        num_workers: Number of dataloader workers (0 = main process)

    Returns:
        (train_loader, val_loader) tuple

    Design decisions:
    - Shuffle train data (prevents overfitting to sequence order)
    - Don't shuffle val data (consistent evaluation)
    - num_workers=0 on small datasets (overhead not worth it)
    - drop_last=True to avoid partial batches (can cause issues with batch norm)
    """
    # Create datasets
    train_dataset = TokenizedDataset(train_path)
    val_dataset = TokenizedDataset(val_path)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # Shuffle for training
        num_workers=num_workers,
        pin_memory=True,  # Faster GPU transfer
        drop_last=True,  # Drop incomplete last batch
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # Don't shuffle validation
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,  # Keep all validation data
    )

    print(f"\nDataLoaders created:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Batch size: {batch_size}")

    return train_loader, val_loader


def get_batch_info(batch: tuple) -> dict:
    """
    Get information about a batch (for debugging).

    Args:
        batch: (input_ids, target_ids) tuple

    Returns:
        Dictionary with batch statistics
    """
    input_ids, target_ids = batch

    return {
        "batch_size": input_ids.size(0),
        "seq_len": input_ids.size(1),
        "input_shape": tuple(input_ids.shape),
        "target_shape": tuple(target_ids.shape),
        "input_dtype": input_ids.dtype,
        "target_dtype": target_ids.dtype,
        "input_device": input_ids.device,
        "target_device": target_ids.device,
    }
