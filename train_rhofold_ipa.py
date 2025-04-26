#!/usr/bin/env python3
"""
RhoFold+ IPA Training Pipeline for RNA Structure Prediction

This script implements a training pipeline that integrates the RhoFold+ 
Invariant-Point-Attention (IPA) structure module into the existing Betabend
RNA Feature-Embedding model. The pipeline is designed to maximize TM-score
for RNA structure prediction and produce a Kaggle-ready checkpoint.

Usage:
    python train_rhofold_ipa.py \
        --train_csv data/train_sequences.csv \
        --label_csv data/train_labels.csv \
        --feature_root data/processed \
        --val_csv data/validation_sequences.csv \
        --epochs 30 --batch 4 --lr 3e-4 \
        --ckpt_out checkpoints/rhofold_ipa_final.pt
"""

import os
import sys
import time
import datetime
import argparse
import logging
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, Set

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Dataset, Subset

# Add necessary paths for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
betabend_dir = os.path.join(current_dir, "betabend-refactor")
rhofold_dir = os.path.join(current_dir, "RhoFold-refactor")

sys.path.append(current_dir)
sys.path.append(betabend_dir)
sys.path.append(os.path.join(betabend_dir, "src"))
sys.path.append(rhofold_dir)
sys.path.append(os.path.join(rhofold_dir, "src"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)

# Import RhoFold+ IPA module implementation
from rhofold_ipa_module import RhoFoldIPAModule
from utils.model_utils import count_parameters


class RNAFeatureDataset(Dataset):
    """
    Dataset for RNA structure prediction that loads and assembles
    features from multiple sources (dihedral, MI, thermo).
    
    This implementation handles temporal cutoff filtering and
    caching for faster loading during repeated epochs.
    """
    
    def __init__(
        self,
        sequences_csv: str,
        labels_csv: Optional[str] = None,
        feature_root: str = "data/processed",
        temporal_cutoff: Optional[str] = None,
        mode: str = "train",
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize the RNA feature dataset.
        
        Args:
            sequences_csv: Path to CSV containing RNA sequences
            labels_csv: Path to CSV containing 3D coordinates (optional for inference)
            feature_root: Root directory containing feature subdirectories
            temporal_cutoff: Date string for temporal filtering (YYYY-MM-DD)
            mode: "train" or "val" - affects temporal cutoff handling
            cache_dir: Directory to store feature cache
        """
        self.mode = mode
        self.feature_root = feature_root
        self.temporal_cutoff = temporal_cutoff
        
        # Load sequences DataFrame
        self.sequences_df = pd.read_csv(sequences_csv)
        self.target_ids_all = self.sequences_df["target_id"].tolist()
        
        # Apply temporal cutoff filtering for training set
        if temporal_cutoff is not None and mode == "train":
            logger.info(f"Applying temporal cutoff: {temporal_cutoff}")
            valid_rows = self.sequences_df["temporal_cutoff"] < temporal_cutoff
            self.sequences_df = self.sequences_df[valid_rows]
            self.target_ids = self.sequences_df["target_id"].tolist()
        else:
            self.target_ids = self.target_ids_all
            
        logger.info(f"Loaded {len(self.target_ids)} {mode} sequences")
        self.sequences = {
            row["target_id"]: row["sequence"] 
            for _, row in self.sequences_df.iterrows()
        }
        
        # Load labels if available
        self.labels_df = None
        self.coordinates = {}
        if labels_csv and os.path.exists(labels_csv):
            self.labels_df = pd.read_csv(labels_csv)
            
            # Extract unique target IDs and their conformers
            target_conformers = {}
            for _, row in self.labels_df.iterrows():
                target_id = row["ID"].split("_")[0]
                if target_id not in target_conformers:
                    target_conformers[target_id] = []
                target_conformers[target_id].append(row["ID"])
            
            # Extract coordinates for each target
            for target_id in self.target_ids:
                if target_id in target_conformers:
                    conformer_ids = target_conformers[target_id]
                    self.coordinates[target_id] = self._load_coordinates(conformer_ids)
        
        # Initialize feature cache
        self.cache_dir = cache_dir
        self.feature_cache = {}
        
        # Nucleotide to integer mapping
        self.nuc_to_int = {"A": 0, "C": 1, "G": 2, "U": 3, "T": 3, "N": 4}
        
        # Print feature directory info
        self._log_feature_directories()
    
    def _load_coordinates(self, conformer_ids: List[str]) -> np.ndarray:
        """
        Load C1' coordinates for all conformers of a target.
        
        Args:
            conformer_ids: List of conformer IDs
            
        Returns:
            Array of shape [num_conformers, seq_len, 3]
        """
        conformer_coords = []
        
        for conf_id in conformer_ids:
            # Filter rows for this conformer
            conf_rows = self.labels_df[self.labels_df["ID"] == conf_id]
            
            # Extract coordinates and sort by residue ID
            conf_rows = conf_rows.sort_values(by="resid")
            
            # Get C1' coordinates (x_1, y_1, z_1)
            coords = conf_rows[["x_1", "y_1", "z_1"]].values.astype(np.float32)
            conformer_coords.append(coords)
        
        # Stack all conformers
        return np.stack(conformer_coords)
    
    def _log_feature_directories(self):
        """Log info about feature directories."""
        for subdir in ["mi_features", "dihedral_features", "thermo_features"]:
            path = os.path.join(self.feature_root, subdir)
            if os.path.exists(path):
                files = len([f for f in os.listdir(path) if f.endswith(".npz")])
                logger.info(f"Found {files} files in {subdir}")
            else:
                logger.warning(f"Feature directory not found: {path}")
    
    def sequence_to_int(self, sequence: str) -> List[int]:
        """Convert nucleotide sequence to integer indices."""
        return [self.nuc_to_int.get(nuc.upper(), 4) for nuc in sequence]
    
    def _load_features(self, target_id: str) -> Dict[str, Any]:
        """
        Load all features for a specific target ID.
        
        Args:
            target_id: Target ID to load features for
            
        Returns:
            Dictionary of features
        """
        # Check cache first
        if target_id in self.feature_cache:
            return self.feature_cache[target_id]
        
        features = {}
        
        # Load sequence
        if target_id in self.sequences:
            sequence = self.sequences[target_id]
            features["sequence"] = sequence
            features["sequence_int"] = self.sequence_to_int(sequence)
        else:
            logger.warning(f"No sequence found for {target_id}")
            return {}
        
        # Load MI features
        mi_file = os.path.join(self.feature_root, "mi_features", f"{target_id}_mi_features.npz")
        if os.path.exists(mi_file):
            with np.load(mi_file) as data:
                features["mi"] = {
                    "coupling_matrix": data["coupling_matrix"].astype(np.float32),
                    # Add other MI features as needed
                }
        else:
            # If MI features don't exist, create zeros
            seq_len = len(features["sequence"])
            features["mi"] = {
                "coupling_matrix": np.zeros((seq_len, seq_len), dtype=np.float32),
            }
        
        # Load dihedral features
        dihedral_file = os.path.join(self.feature_root, "dihedral_features", f"{target_id}_dihedral_features.npz")
        if os.path.exists(dihedral_file):
            with np.load(dihedral_file) as data:
                if "features" in data:
                    # Handle NaN values
                    dihedral_features = data["features"].astype(np.float32)
                    dihedral_features = np.nan_to_num(dihedral_features, nan=0.0)
                    features["dihedral"] = {
                        "features": dihedral_features,
                    }
                    
                    if "eta" in data and "theta" in data:
                        features["dihedral"]["eta"] = data["eta"].astype(np.float32)
                        features["dihedral"]["theta"] = data["theta"].astype(np.float32)
                else:
                    # Create zeros if the file exists but has no features
                    seq_len = len(features["sequence"])
                    features["dihedral"] = {
                        "features": np.zeros((seq_len, 4), dtype=np.float32),
                    }
        else:
            # Create zeros if the file doesn't exist
            seq_len = len(features["sequence"])
            features["dihedral"] = {
                "features": np.zeros((seq_len, 4), dtype=np.float32),
            }
        
        # Load thermo features
        thermo_file = os.path.join(self.feature_root, "thermo_features", f"{target_id}_thermo_features.npz")
        if os.path.exists(thermo_file):
            with np.load(thermo_file) as data:
                # Extract key arrays
                thermo_features = {}
                
                # Get pairing probabilities matrix (critical)
                if "pairing_probs" in data:
                    thermo_features["pairing_probs"] = data["pairing_probs"].astype(np.float32)
                elif "base_pair_probs" in data:
                    thermo_features["pairing_probs"] = data["base_pair_probs"].astype(np.float32)
                else:
                    seq_len = len(features["sequence"])
                    thermo_features["pairing_probs"] = np.zeros((seq_len, seq_len), dtype=np.float32)
                
                # Get positional entropy
                if "positional_entropy" in data:
                    thermo_features["positional_entropy"] = data["positional_entropy"].astype(np.float32)
                else:
                    seq_len = len(features["sequence"])
                    thermo_features["positional_entropy"] = np.zeros(seq_len, dtype=np.float32)
                
                # Get accessibility
                if "accessibility" in data:
                    thermo_features["accessibility"] = data["accessibility"].astype(np.float32)
                else:
                    seq_len = len(features["sequence"])
                    thermo_features["accessibility"] = np.zeros(seq_len, dtype=np.float32)
                
                features["thermo"] = thermo_features
        else:
            # Create zeros if the file doesn't exist
            seq_len = len(features["sequence"])
            features["thermo"] = {
                "pairing_probs": np.zeros((seq_len, seq_len), dtype=np.float32),
                "positional_entropy": np.zeros(seq_len, dtype=np.float32),
                "accessibility": np.zeros(seq_len, dtype=np.float32),
            }
        
        # Store in cache
        self.feature_cache[target_id] = features
        
        return features
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.target_ids)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single RNA sample with all features.
        
        Args:
            idx: Index in the dataset
            
        Returns:
            Dictionary containing all features and labels
        """
        target_id = self.target_ids[idx]
        
        # Load all features
        features = self._load_features(target_id)
        if not features:
            # Return empty sample if features couldn't be loaded
            return {"error": True, "target_id": target_id}
        
        # Create sample dictionary with tensors
        sample = {
            "target_id": target_id,
            "sequence_int": torch.tensor(features["sequence_int"], dtype=torch.long),
            "length": len(features["sequence"]),
        }
        
        # Add dihedral features
        if "dihedral" in features:
            sample["dihedral_features"] = torch.tensor(
                features["dihedral"]["features"], dtype=torch.float32
            )
        
        # Add MI features
        if "mi" in features:
            sample["coupling_matrix"] = torch.tensor(
                features["mi"]["coupling_matrix"], dtype=torch.float32
            )
        
        # Add thermo features
        if "thermo" in features:
            sample["pairing_probs"] = torch.tensor(
                features["thermo"]["pairing_probs"], dtype=torch.float32
            )
            sample["positional_entropy"] = torch.tensor(
                features["thermo"]["positional_entropy"], dtype=torch.float32
            )
            sample["accessibility"] = torch.tensor(
                features["thermo"]["accessibility"], dtype=torch.float32
            )
        
        # Add coordinates if available for training/validation
        if target_id in self.coordinates:
            sample["coordinates"] = torch.tensor(
                self.coordinates[target_id][0], dtype=torch.float32
            )  # Use first conformer by default
            
            # If there are multiple conformers, store them as well
            if self.coordinates[target_id].shape[0] > 1:
                sample["all_conformers"] = torch.tensor(
                    self.coordinates[target_id], dtype=torch.float32
                )
        
        return sample


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate a list of samples into a batch with padding.
    
    Args:
        batch: List of sample dictionaries
        
    Returns:
        Dictionary of batched tensors
    """
    # Filter out error samples
    batch = [sample for sample in batch if "error" not in sample]
    
    if not batch:
        return {}  # Return empty batch if all samples had errors
    
    # Get batch size and maximum sequence length
    batch_size = len(batch)
    max_len = max(sample["length"] for sample in batch)
    
    # Extract target IDs
    target_ids = [sample["target_id"] for sample in batch]
    
    # Initialize output dictionary
    output = {
        "target_ids": target_ids,
        "lengths": torch.tensor([sample["length"] for sample in batch], dtype=torch.long),
    }
    
    # Process each tensor in the batch
    for key in batch[0].keys():
        if key in ["target_id", "length", "all_conformers"]:
            continue  # Skip non-tensor fields or special handling
        
        if isinstance(batch[0][key], torch.Tensor):
            # Handle different tensor shapes based on key
            sample_shape = batch[0][key].shape
            
            if key == "sequence_int":
                # 1D sequence tensor (L) -> (B, L)
                padded = torch.zeros((batch_size, max_len), dtype=torch.long)
                for i, sample in enumerate(batch):
                    length = sample["length"]
                    padded[i, :length] = sample[key][:length]
                output[key] = padded
                
            elif key == "dihedral_features":
                # 2D tensor (L, F) -> (B, L, F)
                feature_dim = sample_shape[-1]
                padded = torch.zeros((batch_size, max_len, feature_dim), dtype=torch.float32)
                for i, sample in enumerate(batch):
                    if key in sample:
                        length = sample["length"]
                        padded[i, :length, :] = sample[key][:length, :]
                output[key] = padded
                
            elif key in ["positional_entropy", "accessibility"]:
                # 1D feature tensor (L) -> (B, L)
                padded = torch.zeros((batch_size, max_len), dtype=torch.float32)
                for i, sample in enumerate(batch):
                    if key in sample:
                        length = sample["length"]
                        padded[i, :length] = sample[key][:length]
                output[key] = padded
                
            elif key in ["pairing_probs", "coupling_matrix"]:
                # 2D square matrix (L, L) -> (B, L, L)
                padded = torch.zeros((batch_size, max_len, max_len), dtype=torch.float32)
                for i, sample in enumerate(batch):
                    if key in sample:
                        length = sample["length"]
                        padded[i, :length, :length] = sample[key][:length, :length]
                output[key] = padded
                
            elif key == "coordinates":
                # Coordinate tensor (L, 3) -> (B, L, 3)
                padded = torch.zeros((batch_size, max_len, 3), dtype=torch.float32)
                for i, sample in enumerate(batch):
                    if key in sample:
                        length = sample["length"]
                        padded[i, :length, :] = sample[key][:length, :]
                output[key] = padded
            
            else:
                # Handle any other tensors generically
                output[key] = torch.stack([sample[key] for sample in batch if key in sample])
    
    # Create attention mask (True for valid positions, False for padding)
    mask = torch.zeros((batch_size, max_len), dtype=torch.bool)
    for i, sample in enumerate(batch):
        mask[i, : sample["length"]] = True
    output["mask"] = mask
    
    return output


class RhoFoldIPAModel(nn.Module):
    """
    Complete RNA structure prediction model integrating the RhoFold+ IPA module
    with the Betabend RNA Feature Embedding model.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the complete model.
        
        Args:
            config: Configuration dictionary with model parameters
        """
        super().__init__()
        
        # Extract parameters from config
        self.residue_dim = config.get("residue_embed_dim", 128)
        self.pair_dim = config.get("pair_embed_dim", 64)
        self.num_blocks = config.get("num_blocks", 4)
        
        # Import embedding and transformer modules from betabend
        from src.models.embeddings import EmbeddingModule
        from src.models.transformer_block import TransformerBlock
        
        # Initialize the embedding module
        self.embedding_module = EmbeddingModule(config)
        
        # Initialize transformer blocks
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(self.num_blocks)]
        )
        
        # Initialize RhoFold IPA module
        self.ipa_module = RhoFoldIPAModule(config)
    
    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the complete model.
        
        Args:
            batch: Dictionary of input tensors from the data loader
            
        Returns:
            Dictionary of output tensors with predicted coordinates,
            confidence scores, and angles
        """
        # Create initial representations using embedding module
        residue_repr, pair_repr, mask = self.embedding_module(batch)
        # residue_repr: (batch_size, seq_len, residue_dim)
        # pair_repr: (batch_size, seq_len, seq_len, pair_dim)
        
        # Process through transformer blocks
        for block in self.transformer_blocks:
            residue_repr, pair_repr = block(residue_repr, pair_repr, mask)
        
        # Process through RhoFold IPA module
        outputs = self.ipa_module(
            residue_repr=residue_repr,
            pair_repr=pair_repr,
            mask=mask,
            sequences_int=batch.get("sequence_int")
        )
        
        return outputs


def compute_tm_score(
    pred_coords: torch.Tensor,
    true_coords: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    d0_scale: float = 1.24,
    d0_offset: float = -1.8,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """
    Calculate the Template-Modeling score (TM-score) between predicted and true coordinates.
    
    TM-score is a measure of similarity between two structures with values in (0,1].
    Higher values indicate better alignment. The score is normalized to avoid length dependency.
    
    Args:
        pred_coords: Predicted coordinates, shape (batch_size, seq_len, 3) or (seq_len, 3)
        true_coords: Ground truth coordinates, shape (batch_size, seq_len, 3) or (seq_len, 3)
        mask: Boolean mask, shape (batch_size, seq_len) or (seq_len), True for valid positions
        d0_scale: Scale factor for normalization parameter d0
        d0_offset: Offset for normalization parameter d0
        epsilon: Small constant for numerical stability
        
    Returns:
        TM-score value(s), shape (batch_size) or scalar
    """
    # Import Kabsch alignment function
    from src.losses import stable_kabsch_align
    
    # Add batch dimension if not present
    if len(pred_coords.shape) == 2:
        pred_coords = pred_coords.unsqueeze(0)
        true_coords = true_coords.unsqueeze(0)
        if mask is not None and len(mask.shape) == 1:
            mask = mask.unsqueeze(0)
    
    device = pred_coords.device
    dtype = pred_coords.dtype
    batch_size, seq_len, _ = pred_coords.shape
    
    # Create default mask if not provided
    if mask is None:
        mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)
    
    # Initialize results tensor
    tm_scores = torch.zeros(batch_size, dtype=dtype, device=device)
    
    for b in range(batch_size):
        valid_mask = mask[b]
        valid_count = valid_mask.sum().item()
        
        if valid_count < 3:  # Need at least 3 points for meaningful TM-score
            tm_scores[b] = 0.0
            continue
            
        # Extract valid coordinates
        p_valid = pred_coords[b, valid_mask]
        t_valid = true_coords[b, valid_mask]
        
        # Calculate normalization parameter d0 based on sequence length
        L = valid_count
        d0 = d0_scale * (L ** (1.0/3.0)) + d0_offset
        d0 = max(d0, 0.5)  # Lower bound
        
        # Use the Kabsch algorithm to optimally align the structures
        p_aligned = stable_kabsch_align(p_valid, t_valid, epsilon=epsilon)
        
        # Handle direct checks for identical tensors
        if torch.allclose(p_valid, t_valid, atol=1e-6):
            tm_scores[b] = 1.0  # Perfect score for identical structures
            continue
            
        # Compute distance squared for each atom pair
        squared_diff = torch.sum((p_aligned - t_valid) ** 2, dim=-1)
        
        # Calculate the TM-score term for each atom
        tm_terms = 1.0 / (1.0 + squared_diff / (d0 ** 2))
        
        # Calculate TM-score as the mean over all atoms, normalized by length
        tm_score = torch.sum(tm_terms) / L
        tm_scores[b] = tm_score
    
    # Return scalar if batch size is 1
    if batch_size == 1:
        return tm_scores[0]
    else:
        return tm_scores


def compute_fape_loss(
    pred_coords: torch.Tensor,
    true_coords: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    clamp_distance: float = 10.0,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """
    Calculate the Frame-Aligned Point Error (FAPE) loss.
    
    Args:
        pred_coords: Predicted coordinates (batch_size, seq_len, 3)
        true_coords: Ground truth coordinates (batch_size, seq_len, 3)
        mask: Boolean mask (batch_size, seq_len)
        clamp_distance: Maximum distance for clamping
        epsilon: Small constant for numerical stability
        
    Returns:
        FAPE loss
    """
    # Import alignment function
    from src.losses import stable_kabsch_align
    
    # Add batch dimension if not present
    if len(pred_coords.shape) == 2:
        pred_coords = pred_coords.unsqueeze(0)
        true_coords = true_coords.unsqueeze(0)
        if mask is not None and len(mask.shape) == 1:
            mask = mask.unsqueeze(0)
    
    batch_size, seq_len, _ = pred_coords.shape
    device = pred_coords.device
    
    # Create default mask if not provided
    if mask is None:
        mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)
    
    # Initialize results tensor
    fape_loss = torch.zeros(batch_size, device=device)
    
    for b in range(batch_size):
        valid_mask = mask[b]
        valid_count = valid_mask.sum().item()
        
        if valid_count < 3:  # Need at least 3 points for meaningful alignment
            fape_loss[b] = 0.0
            continue
            
        # Extract valid coordinates
        p_valid = pred_coords[b, valid_mask]
        t_valid = true_coords[b, valid_mask]
        
        # Use the Kabsch algorithm to optimally align the structures
        p_aligned = stable_kabsch_align(p_valid, t_valid, epsilon=epsilon)
        
        # Compute distances between aligned points
        distances = torch.norm(p_aligned - t_valid, dim=-1)
        
        # Clamp distances for stability
        clamped_distances = torch.clamp(distances, max=clamp_distance)
        
        # Calculate mean distance as FAPE
        fape_loss[b] = clamped_distances.mean()
    
    # Return mean over batch
    return fape_loss.mean()


def compute_contact_bce_loss(
    pred_coords: torch.Tensor,
    pairing_probs: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    distance_threshold: float = 15.0,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """
    Calculate binary cross-entropy loss for RNA base-pair contacts.
    
    Args:
        pred_coords: Predicted coordinates (batch_size, seq_len, 3)
        pairing_probs: Pairing probabilities (batch_size, seq_len, seq_len)
        mask: Boolean mask (batch_size, seq_len)
        distance_threshold: Distance threshold for contacts
        epsilon: Small constant for numerical stability
        
    Returns:
        Contact BCE loss
    """
    batch_size, seq_len, _ = pred_coords.shape
    device = pred_coords.device
    
    # Create default mask if not provided
    if mask is None:
        mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)
    
    # Create 2D mask for pairs
    pair_mask = mask.unsqueeze(-1) & mask.unsqueeze(-2)  # (batch_size, seq_len, seq_len)
    
    # Calculate pairwise distances from predicted coordinates
    # Reshape for broadcasting
    x_i = pred_coords.unsqueeze(2)  # (batch_size, seq_len, 1, 3)
    x_j = pred_coords.unsqueeze(1)  # (batch_size, 1, seq_len, 3)
    
    # Calculate squared distances
    squared_distances = torch.sum((x_i - x_j) ** 2, dim=-1)  # (batch_size, seq_len, seq_len)
    
    # Convert distances to contact probabilities
    pred_contacts = torch.sigmoid(-squared_distances / (distance_threshold ** 2) + epsilon)
    
    # Apply mask
    pair_mask_float = pair_mask.float()
    
    # Calculate binary cross-entropy loss
    bce_loss = F.binary_cross_entropy(
        pred_contacts * pair_mask_float,
        pairing_probs * pair_mask_float,
        reduction="none"
    )
    
    # Sum over valid pairs and normalize
    valid_pairs = pair_mask.sum(dim=(1, 2))
    loss = torch.sum(bce_loss * pair_mask_float, dim=(1, 2)) / (valid_pairs + epsilon)
    
    return loss.mean()


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    loss_weights: Dict[str, float],
    scaler: Optional[GradScaler] = None,
    grad_accum_steps: int = 1,
) -> Dict[str, float]:
    """
    Train model for one epoch with gradient accumulation and mixed precision.
    
    Args:
        model: Model to train
        train_loader: DataLoader for training data
        optimizer: Optimizer
        device: Device to train on
        loss_weights: Dictionary of loss weights
        scaler: GradScaler for mixed precision training
        grad_accum_steps: Number of gradient accumulation steps
        
    Returns:
        Dictionary of training metrics
    """
    model.train()
    total_tm_loss = 0.0
    total_fape_loss = 0.0
    total_contact_bce_loss = 0.0
    total_samples = 0
    
    # Zero gradients at the beginning
    optimizer.zero_grad()
    
    for batch_idx, batch in enumerate(train_loader):
        # Skip empty batches
        if not batch:
            continue
        
        # Move batch to device
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device)
        
        # Forward pass with mixed precision if enabled
        if scaler:
            with autocast():
                outputs = model(batch)
                
                # Calculate TM-score loss (1 - TM-score)
                tm_score = compute_tm_score(
                    pred_coords=outputs["pred_coords"],
                    true_coords=batch["coordinates"],
                    mask=batch["mask"],
                )
                tm_loss = 1.0 - tm_score
                
                # Calculate FAPE loss
                fape_loss = compute_fape_loss(
                    pred_coords=outputs["pred_coords"],
                    true_coords=batch["coordinates"],
                    mask=batch["mask"],
                )
                
                # Calculate contact BCE loss
                contact_bce_loss = compute_contact_bce_loss(
                    pred_coords=outputs["pred_coords"],
                    pairing_probs=batch["pairing_probs"],
                    mask=batch["mask"],
                )
                
                # Combine losses with weights
                loss = (
                    loss_weights["tm"] * tm_loss +
                    loss_weights["fape"] * fape_loss +
                    loss_weights["contact_bce"] * contact_bce_loss
                )
                
                # Scale loss for gradient accumulation
                loss = loss / grad_accum_steps
            
            # Backward pass with scaler
            scaler.scale(loss).backward()
            
            # Step every grad_accum_steps batches
            if (batch_idx + 1) % grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            # Standard forward pass
            outputs = model(batch)
            
            # Calculate TM-score loss (1 - TM-score)
            tm_score = compute_tm_score(
                pred_coords=outputs["pred_coords"],
                true_coords=batch["coordinates"],
                mask=batch["mask"],
            )
            tm_loss = 1.0 - tm_score
            
            # Calculate FAPE loss
            fape_loss = compute_fape_loss(
                pred_coords=outputs["pred_coords"],
                true_coords=batch["coordinates"],
                mask=batch["mask"],
            )
            
            # Calculate contact BCE loss
            contact_bce_loss = compute_contact_bce_loss(
                pred_coords=outputs["pred_coords"],
                pairing_probs=batch["pairing_probs"],
                mask=batch["mask"],
            )
            
            # Combine losses with weights
            loss = (
                loss_weights["tm"] * tm_loss +
                loss_weights["fape"] * fape_loss +
                loss_weights["contact_bce"] * contact_bce_loss
            )
            
            # Scale loss for gradient accumulation
            loss = loss / grad_accum_steps
            
            # Backward pass
            loss.backward()
            
            # Step every grad_accum_steps batches
            if (batch_idx + 1) % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
        
        # Accumulate metrics
        batch_size = batch["mask"].size(0)
        total_tm_loss += tm_loss.item() * batch_size
        total_fape_loss += fape_loss.item() * batch_size
        total_contact_bce_loss += contact_bce_loss.item() * batch_size
        total_samples += batch_size
        
        # Log progress
        if (batch_idx + 1) % 10 == 0:
            logger.info(
                f"Train Batch {batch_idx+1}/{len(train_loader)}: "
                f"TM-Loss: {tm_loss.item():.4f}, "
                f"FAPE: {fape_loss.item():.4f}, "
                f"Contact-BCE: {contact_bce_loss.item():.4f}"
            )
    
    # Handle any remaining gradients
    if len(train_loader) % grad_accum_steps != 0:
        if scaler:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        optimizer.zero_grad()
    
    # Calculate average metrics
    avg_tm_loss = total_tm_loss / total_samples
    avg_fape_loss = total_fape_loss / total_samples
    avg_contact_bce_loss = total_contact_bce_loss / total_samples
    
    return {
        "tm_loss": avg_tm_loss,
        "tm_score": 1.0 - avg_tm_loss,
        "fape_loss": avg_fape_loss,
        "contact_bce_loss": avg_contact_bce_loss,
    }


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    loss_weights: Dict[str, float],
) -> Dict[str, float]:
    """
    Validate model on validation set.
    
    Args:
        model: Model to validate
        val_loader: DataLoader for validation data
        device: Device to validate on
        loss_weights: Dictionary of loss weights
        
    Returns:
        Dictionary of validation metrics
    """
    model.eval()
    total_tm_loss = 0.0
    total_fape_loss = 0.0
    total_contact_bce_loss = 0.0
    total_samples = 0
    all_tm_scores = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            # Skip empty batches
            if not batch:
                continue
            
            # Move batch to device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)
            
            # Forward pass
            outputs = model(batch)
            
            # Calculate TM-score
            tm_score = compute_tm_score(
                pred_coords=outputs["pred_coords"],
                true_coords=batch["coordinates"],
                mask=batch["mask"],
            )
            tm_loss = 1.0 - tm_score
            
            # Calculate FAPE loss
            fape_loss = compute_fape_loss(
                pred_coords=outputs["pred_coords"],
                true_coords=batch["coordinates"],
                mask=batch["mask"],
            )
            
            # Calculate contact BCE loss
            contact_bce_loss = compute_contact_bce_loss(
                pred_coords=outputs["pred_coords"],
                pairing_probs=batch["pairing_probs"],
                mask=batch["mask"],
            )
            
            # Accumulate metrics
            batch_size = batch["mask"].size(0)
            total_tm_loss += tm_loss.item() * batch_size
            total_fape_loss += fape_loss.item() * batch_size
            total_contact_bce_loss += contact_bce_loss.item() * batch_size
            total_samples += batch_size
            
            # Store individual TM-scores for detailed analysis
            for i in range(batch_size):
                # Calculate TM-score for each sample
                sample_tm = compute_tm_score(
                    pred_coords=outputs["pred_coords"][i].unsqueeze(0),
                    true_coords=batch["coordinates"][i].unsqueeze(0),
                    mask=batch["mask"][i].unsqueeze(0),
                )
                all_tm_scores.append((batch["target_ids"][i], sample_tm.item()))
    
    # Calculate average metrics
    avg_tm_loss = total_tm_loss / total_samples
    avg_fape_loss = total_fape_loss / total_samples
    avg_contact_bce_loss = total_contact_bce_loss / total_samples
    
    return {
        "tm_loss": avg_tm_loss,
        "tm_score": 1.0 - avg_tm_loss,
        "fape_loss": avg_fape_loss,
        "contact_bce_loss": avg_contact_bce_loss,
        "all_tm_scores": all_tm_scores,
    }


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_metrics: Dict[str, float],
    save_path: str,
    config: Dict[str, Any],
) -> None:
    """
    Save model checkpoint and metrics.
    
    Args:
        model: Model to save
        optimizer: Optimizer
        epoch: Current epoch
        val_metrics: Validation metrics
        save_path: Path to save checkpoint
        config: Model and training configuration
    """
    # Create directory if not exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save checkpoint
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_metrics": val_metrics,
        "config": config,
    }
    
    torch.save(checkpoint, save_path)
    logger.info(f"Checkpoint saved to {save_path}")
    
    # Also save config as JSON for easier inspection
    config_path = save_path.replace(".pt", "_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)


def save_val_report(
    all_tm_scores: List[Tuple[str, float]],
    save_path: str,
) -> None:
    """
    Save validation report as CSV.
    
    Args:
        all_tm_scores: List of (target_id, tm_score) tuples
        save_path: Path to save CSV
    """
    # Create directory if not exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Create DataFrame
    df = pd.DataFrame(all_tm_scores, columns=["target_id", "tm_score"])
    
    # Save CSV
    df.to_csv(save_path, index=False)
    logger.info(f"Validation report saved to {save_path}")


def check_feature_availability(feature_root: str) -> Dict[str, int]:
    """
    Check feature availability in the feature directories.
    
    Args:
        feature_root: Root directory containing feature subdirectories
        
    Returns:
        Dictionary of feature counts
    """
    feature_counts = {}
    
    for subdir in ["mi_features", "dihedral_features", "thermo_features"]:
        path = os.path.join(feature_root, subdir)
        if os.path.exists(path):
            files = len([f for f in os.listdir(path) if f.endswith(".npz")])
            feature_counts[subdir] = files
        else:
            feature_counts[subdir] = 0
    
    return feature_counts


def get_target_ids_from_csv(csv_path: str) -> List[str]:
    """
    Extract target IDs from a CSV file.
    
    Args:
        csv_path: Path to CSV file
        
    Returns:
        List of target IDs
    """
    df = pd.read_csv(csv_path)
    return df["target_id"].tolist()


def loader_feature_test(feature_root: str, target_ids: List[str]) -> Dict[str, float]:
    """
    Test feature availability for a list of target IDs.
    
    Args:
        feature_root: Root directory containing feature subdirectories
        target_ids: List of target IDs
        
    Returns:
        Dictionary of feature availability ratios
    """
    # Initialize counters
    total = len(target_ids)
    found = {
        "mi_features": 0,
        "dihedral_features": 0,
        "thermo_features": 0,
        "all_features": 0,
    }
    
    # Check each target
    for target_id in target_ids:
        # Check each feature type
        has_mi = os.path.exists(os.path.join(feature_root, "mi_features", f"{target_id}_mi_features.npz"))
        has_dihedral = os.path.exists(os.path.join(feature_root, "dihedral_features", f"{target_id}_dihedral_features.npz"))
        has_thermo = os.path.exists(os.path.join(feature_root, "thermo_features", f"{target_id}_thermo_features.npz"))
        
        # Update counters
        if has_mi:
            found["mi_features"] += 1
        if has_dihedral:
            found["dihedral_features"] += 1
        if has_thermo:
            found["thermo_features"] += 1
        if has_mi and has_dihedral and has_thermo:
            found["all_features"] += 1
    
    # Calculate ratios
    ratios = {k: v / total for k, v in found.items()}
    
    return ratios


def tm_score_test() -> None:
    """
    Test TM-score calculation with toy coordinates.
    """
    # Create toy coordinates
    pred_coords = torch.tensor([
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],  # Perfect match
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]],  # Scaled x2
        [[1.0, 1.0, 1.0], [2.0, 1.0, 1.0], [1.0, 2.0, 1.0]],  # Translated
        [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],  # Different topology
    ])
    
    true_coords = torch.tensor([
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    ])
    
    # Calculate TM-scores
    tm_scores = compute_tm_score(pred_coords, true_coords)
    
    # Expected outcomes
    # Case 1: Perfect match -> TM-score = 1.0
    # Case 2: Scaled -> TM-score < 1.0 but still high
    # Case 3: Translated -> TM-score = 1.0 (after alignment)
    # Case 4: Different topology -> TM-score < 1.0
    
    logger.info("TM-score test results:")
    for i, tm in enumerate(tm_scores):
        logger.info(f"  Case {i+1}: {tm.item():.4f}")
    
    # Check if results match expectations
    assert tm_scores[0] > 0.99, "Perfect match should have TM-score close to 1"
    assert tm_scores[2] > 0.99, "Translated structure should have TM-score close to 1 after alignment"
    assert tm_scores[1] < 0.99, "Scaled structure should have TM-score < 1"
    assert tm_scores[3] < 0.8, "Different topology should have low TM-score"
    
    logger.info("TM-score test passed!")


def main():
    """Main function for the training pipeline."""
    parser = argparse.ArgumentParser(description="Train RhoFold IPA model")
    
    # Data parameters
    parser.add_argument("--train_csv", type=str, required=True, help="Path to training sequences CSV")
    parser.add_argument("--label_csv", type=str, required=True, help="Path to training labels CSV")
    parser.add_argument("--feature_root", type=str, required=True, help="Path to processed features directory")
    parser.add_argument("--val_csv", type=str, help="Path to validation sequences CSV")
    parser.add_argument("--temporal_cutoff", type=str, default="2022-05-27", help="Temporal cutoff date")
    
    # Model parameters
    parser.add_argument("--residue_embed_dim", type=int, default=128, help="Residue embedding dimension")
    parser.add_argument("--pair_embed_dim", type=int, default=64, help="Pair embedding dimension")
    parser.add_argument("--num_blocks", type=int, default=4, help="Number of transformer blocks")
    parser.add_argument("--num_ipa_blocks", type=int, default=4, help="Number of IPA iterations")
    parser.add_argument("--no_heads", type=int, default=4, help="Number of attention heads in IPA")
    parser.add_argument("--no_qk_points", type=int, default=4, help="Number of query/key points in IPA")
    parser.add_argument("--no_v_points", type=int, default=8, help="Number of value points in IPA")
    
    # Training parameters
    parser.add_argument("--batch", type=int, default=4, help="Batch size")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("--grad_accum_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--mixed_precision", action="store_true", help="Use mixed precision training")
    parser.add_argument("--grad_checkpoint", action="store_true", help="Use gradient checkpointing")
    
    # Loss weights
    parser.add_argument("--tm_weight", type=float, default=1.0, help="Weight for TM-score loss")
    parser.add_argument("--fape_weight", type=float, default=0.5, help="Weight for FAPE loss")
    parser.add_argument("--contact_bce_weight", type=float, default=0.1, help="Weight for contact BCE loss")
    
    # Output parameters
    parser.add_argument("--ckpt_out", type=str, required=True, help="Path to save checkpoint")
    parser.add_argument("--run_tests", action="store_true", help="Run tests before training")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(os.path.dirname(args.ckpt_out), exist_ok=True)
    
    # Configure file logging
    file_handler = logging.FileHandler(
        os.path.join(os.path.dirname(args.ckpt_out), "training.log")
    )
    file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(file_handler)
    
    # Log arguments
    logger.info("Training arguments:")
    for arg in vars(args):
        logger.info(f"  {arg}: {getattr(args, arg)}")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Run tests if requested
    if args.run_tests:
        logger.info("Running tests...")
        
        # Check feature availability
        feature_counts = check_feature_availability(args.feature_root)
        logger.info("Feature availability:")
        for subdir, count in feature_counts.items():
            logger.info(f"  {subdir}: {count} files")
        
        # Get target IDs from CSVs
        train_target_ids = get_target_ids_from_csv(args.train_csv)
        logger.info(f"Found {len(train_target_ids)} target IDs in training CSV")
        
        # Run loader feature test
        feature_ratios = loader_feature_test(args.feature_root, train_target_ids)
        logger.info("Feature availability ratios:")
        for feature_type, ratio in feature_ratios.items():
            logger.info(f"  {feature_type}: {ratio:.2%}")
            
        # Check if feature availability is sufficient
        if feature_ratios["all_features"] < 0.95:
            logger.warning(
                f"Only {feature_ratios['all_features']:.2%} of targets have all features. "
                f"This may cause issues during training."
            )
        
        # Run TM-score test
        tm_score_test()
    
    # Create model configuration
    model_config = {
        "residue_embed_dim": args.residue_embed_dim,
        "pair_embed_dim": args.pair_embed_dim,
        "num_blocks": args.num_blocks,
        "num_ipa_blocks": args.num_ipa_blocks,
        "no_heads": args.no_heads,
        "no_qk_points": args.no_qk_points,
        "no_v_points": args.no_v_points,
    }
    
    # Create model
    model = RhoFoldIPAModel(model_config)
    model = model.to(device)
    
    # Log model parameters
    num_params = count_parameters(model)
    logger.info(f"Model has {num_params:,} parameters")
    
    # Create optimizer
    optimizer = optim.Adam(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=args.weight_decay
    )
    
    # Create loss weights
    loss_weights = {
        "tm": args.tm_weight,
        "fape": args.fape_weight,
        "contact_bce": args.contact_bce_weight,
    }
    
    # Create scaler for mixed precision training
    scaler = GradScaler() if args.mixed_precision else None
    
    # Create datasets
    train_dataset = RNAFeatureDataset(
        sequences_csv=args.train_csv,
        labels_csv=args.label_csv,
        feature_root=args.feature_root,
        temporal_cutoff=args.temporal_cutoff,
        mode="train",
    )
    
    # Create validation dataset
    if args.val_csv:
        val_dataset = RNAFeatureDataset(
            sequences_csv=args.val_csv,
            labels_csv=args.label_csv,
            feature_root=args.feature_root,
            mode="val",
        )
    else:
        # If no validation CSV is provided, use a subset of the training set
        val_size = int(0.1 * len(train_dataset))
        val_indices = list(range(len(train_dataset) - val_size, len(train_dataset)))
        train_indices = list(range(len(train_dataset) - val_size))
        
        val_dataset = Subset(train_dataset, val_indices)
        train_dataset = Subset(train_dataset, train_indices)
    
    logger.info(f"Training on {len(train_dataset)} samples, validating on {len(val_dataset)} samples")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch, 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
    )
    
    # Create extended validation loader for CASP15 targets
    if args.val_csv:
        ext_val_dataset = RNAFeatureDataset(
            sequences_csv=args.train_csv,
            labels_csv=args.label_csv,
            feature_root=args.feature_root,
            temporal_cutoff=None,  # Include all dates
            mode="val",
        )
        
        # Filter for targets with temporal_cutoff >= 2022-05-27
        ext_val_targets = []
        for idx, target_id in enumerate(ext_val_dataset.target_ids):
            row = ext_val_dataset.sequences_df[ext_val_dataset.sequences_df["target_id"] == target_id]
            if not row.empty and pd.to_datetime(row["temporal_cutoff"].iloc[0]) >= pd.to_datetime("2022-05-27"):
                ext_val_targets.append(idx)
        
        ext_val_dataset = Subset(ext_val_dataset, ext_val_targets)
        
        ext_val_loader = DataLoader(
            ext_val_dataset, 
            batch_size=args.batch, 
            shuffle=False, 
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=True,
        )
        
        logger.info(f"Extended validation set has {len(ext_val_dataset)} samples")
    else:
        ext_val_loader = None
    
    # Training loop
    best_tm_score = 0.0
    best_epoch = 0
    no_improve_count = 0
    moving_avg_vals = []  # For moving average validation TM-score
    
    # Perform initial validation
    logger.info("Performing initial validation...")
    val_metrics = validate(model, val_loader, device, loss_weights)
    logger.info(
        f"Initial validation: "
        f"TM-score: {val_metrics['tm_score']:.4f}, "
        f"FAPE: {val_metrics['fape_loss']:.4f}, "
        f"Contact-BCE: {val_metrics['contact_bce_loss']:.4f}"
    )
    
    # Train for specified number of epochs
    for epoch in range(args.epochs):
        logger.info(f"Epoch {epoch+1}/{args.epochs}")
        
        # Train for one epoch
        start_time = time.time()
        train_metrics = train_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            device=device,
            loss_weights=loss_weights,
            scaler=scaler,
            grad_accum_steps=args.grad_accum_steps,
        )
        train_time = time.time() - start_time
        
        # Log training metrics
        logger.info(
            f"Training completed in {train_time:.2f}s: "
            f"TM-score: {train_metrics['tm_score']:.4f}, "
            f"FAPE: {train_metrics['fape_loss']:.4f}, "
            f"Contact-BCE: {train_metrics['contact_bce_loss']:.4f}"
        )
        
        # Validate
        val_metrics = validate(model, val_loader, device, loss_weights)
        
        # Log validation metrics
        logger.info(
            f"Validation: "
            f"TM-score: {val_metrics['tm_score']:.4f}, "
            f"FAPE: {val_metrics['fape_loss']:.4f}, "
            f"Contact-BCE: {val_metrics['contact_bce_loss']:.4f}"
        )
        
        # Evaluate on extended validation set if available
        if ext_val_loader is not None:
            ext_val_metrics = validate(model, ext_val_loader, device, loss_weights)
            logger.info(
                f"Extended validation: "
                f"TM-score: {ext_val_metrics['tm_score']:.4f}, "
                f"FAPE: {ext_val_metrics['fape_loss']:.4f}, "
                f"Contact-BCE: {ext_val_metrics['contact_bce_loss']:.4f}"
            )
            
            # Combine with core validation for moving average
            combined_tm = (val_metrics["tm_score"] + ext_val_metrics["tm_score"]) / 2
        else:
            combined_tm = val_metrics["tm_score"]
        
        # Update moving average
        moving_avg_vals.append(combined_tm)
        if len(moving_avg_vals) > 5:
            moving_avg_vals.pop(0)
        moving_avg_tm = sum(moving_avg_vals) / len(moving_avg_vals)
        
        # Check if this is the best model
        is_best = False
        if moving_avg_tm > best_tm_score:
            best_tm_score = moving_avg_tm
            best_epoch = epoch
            no_improve_count = 0
            is_best = True
            
            # Save best model
            save_path = args.ckpt_out.replace(".pt", f"_best.pt")
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                val_metrics=val_metrics,
                save_path=save_path,
                config=model_config,
            )
            
            # Save validation report
            report_path = os.path.join(os.path.dirname(args.ckpt_out), "val_report.csv")
            save_val_report(val_metrics["all_tm_scores"], report_path)
        else:
            no_improve_count += 1
        
        # Save checkpoint
        if (epoch + 1) % 5 == 0:
            save_path = args.ckpt_out.replace(".pt", f"_epoch_{epoch+1}.pt")
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                val_metrics=val_metrics,
                save_path=save_path,
                config=model_config,
            )
        
        # Check early stopping
        if no_improve_count >= 5:
            logger.info(f"No improvement for 5 epochs, stopping training.")
            break
    
    # Save final model
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=args.epochs - 1,
        val_metrics=val_metrics,
        save_path=args.ckpt_out,
        config=model_config,
    )
    
    logger.info(f"Training completed. Best model at epoch {best_epoch+1} with TM-score {best_tm_score:.4f}")
    
    # Run final validation with the best model
    try:
        # Load best model
        save_path = args.ckpt_out.replace(".pt", f"_best.pt")
        checkpoint = torch.load(save_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        
        # Validate on both validation sets
        final_val_metrics = validate(model, val_loader, device, loss_weights)
        logger.info(
            f"Final core validation with best model: "
            f"TM-score: {final_val_metrics['tm_score']:.4f}"
        )
        
        if ext_val_loader is not None:
            final_ext_metrics = validate(model, ext_val_loader, device, loss_weights)
            logger.info(
                f"Final extended validation with best model: "
                f"TM-score: {final_ext_metrics['tm_score']:.4f}"
            )
        
        # Save final validation report
        final_report_path = os.path.join(os.path.dirname(args.ckpt_out), "final_val_report.csv")
        save_val_report(final_val_metrics["all_tm_scores"], final_report_path)
    except Exception as e:
        logger.error(f"Error during final validation: {e}")


if __name__ == "__main__":
    main()
