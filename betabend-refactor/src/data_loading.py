import os
import time
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .utils.padding import pad_1d, pad_2d, pad_tensor


def load_coordinates(
    labels_df: pd.DataFrame, target_id: str
) -> Tuple[np.ndarray, List[str]]:
    """
    Extract C1' coordinates for a given target ID from labels DataFrame.

    Args:
        labels_df: DataFrame containing coordinates data
        target_id: Target ID to extract coordinates for

    Returns:
        Tuple of (coordinates array of shape (N, 3), list of residue names)
    """
    # Filter rows for the target_id
    target_rows = labels_df[labels_df["ID"].str.startswith(f"{target_id}_")]

    if len(target_rows) == 0:
        raise ValueError(f"No coordinates found for target {target_id}")

    # Extract coordinates and sort by residue ID
    target_rows = target_rows.sort_values(by="resid")

    # Get coordinates (x_1, y_1, z_1)
    coords = target_rows[["x_1", "y_1", "z_1"]].values.astype(np.float32)

    # Get residue names
    resnames = target_rows["resname"].tolist()

    return coords, resnames


def check_features_availability(target_id: str, features_dir: str) -> Dict[str, bool]:
    """Check which features are available for a given target.

    Args:
        target_id: The ID of the target RNA sequence
        features_dir: Directory containing feature subdirectories

    Returns:
        Dictionary mapping feature types to availability (True/False)
    """
    availability = {"dihedral": False, "thermo": False, "mi": False}

    # Check each feature type
    dihedral_path = os.path.join(
        features_dir, "dihedral_features", f"{target_id}_dihedral_features.npz"
    )
    thermo_path = os.path.join(
        features_dir, "thermo_features", f"{target_id}_thermo_features.npz"
    )
    
    # Try both possible locations for MI features
    mi_path = os.path.join(features_dir, "evolutionary_features", f"{target_id}_mi_features.npz")
    mi_path_alt = os.path.join(features_dir, "mi_features", f"{target_id}_mi_features.npz") 

    availability["dihedral"] = os.path.exists(dihedral_path)
    availability["thermo"] = os.path.exists(thermo_path)
    availability["mi"] = os.path.exists(mi_path) or os.path.exists(mi_path_alt)

    return availability


def get_dihedral_tensors(
    target_id: str, features_dir: str
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Get both input and target dihedral tensors.

    This function always returns two tensors: one for input features and one for
    target angles, even if the feature file only contains one. This design ensures
    V2 readiness when target angles might be used differently.

    Args:
        target_id: The ID of the target RNA sequence
        features_dir: Directory containing feature subdirectories

    Returns:
        Tuple of (input_tensor, target_tensor), each of shape (seq_len, 4) or (0, 4)
    """
    # Check if dihedral features exist
    dihedral_path = os.path.join(
        features_dir, "dihedral_features", f"{target_id}_dihedral_features.npz"
    )

    if not os.path.exists(dihedral_path):
        # Return empty tensors if features don't exist
        return torch.zeros((0, 4), dtype=torch.float32), torch.zeros(
            (0, 4), dtype=torch.float32
        )

    # Load dihedral features
    with np.load(dihedral_path) as data:
        if "features" in data:
            features = data["features"].astype(np.float32)
            # Handle NaN values
            if np.isnan(features).any():
                features = np.nan_to_num(features, nan=0.0)

            # For now, use the same tensor for both input and target
            # In V2, we might have separate input and target tensors
            return torch.tensor(features), torch.tensor(features)
        else:
            return torch.zeros((0, 4), dtype=torch.float32), torch.zeros(
                (0, 4), dtype=torch.float32
            )


def is_uniform_top_pairs(top_pairs: np.ndarray, epsilon: float = 1e-6) -> bool:
    """
    Check if top pairs from MI features have uniform scores, indicating a single sequence MSA.

    Args:
        top_pairs: Array of shape (P, 3) with format [pos_i, pos_j, score]
        epsilon: Threshold for standard deviation to consider uniform

    Returns:
        True if all scores are effectively identical
    """
    if len(top_pairs) == 0:
        return False

    # Extract scores (third column)
    scores = top_pairs[:, 2]

    # Check if standard deviation is near zero
    return np.std(scores) < epsilon


def is_uniform_mi_matrix(matrix: np.ndarray, epsilon: float = 1e-6) -> bool:
    """
    Check if an MI matrix contains uniform values, indicating a single sequence MSA.

    Args:
        matrix: Mutual information matrix
        epsilon: Threshold for standard deviation to consider uniform

    Returns:
        True if matrix appears to have uniform off-diagonal values
    """
    # Get values excluding diagonal
    off_diag = matrix[~np.eye(matrix.shape[0], dtype=bool)]

    # Empty matrix case
    if len(off_diag) == 0:
        return False

    # Check if standard deviation is near zero
    return np.std(off_diag) < epsilon


def load_precomputed_features(
    target_id: str, features_dir: str, temporal_cutoff: Optional[str] = None
) -> Dict[str, Union[Dict[str, np.ndarray], None]]:
    """Load all precomputed features for a target from .npz files.

    Args:
        target_id: RNA sequence identifier
        features_dir: Directory containing feature files
        temporal_cutoff: Optional temporal cutoff date for filtering

    Returns:
        Dictionary of feature dictionaries with structure:
        {
            'dihedral': {'features': array(...)},
            'thermo': {'pairing_probs': array(...), 'mfe': value, ...},
            'evolutionary': {'coupling_matrix': array(...), 'has_valid_mi': bool}
        }
    """
    features = {}

    # 1. Load dihedral features
    dihedral_path = os.path.join(
        features_dir, "dihedral_features", f"{target_id}_dihedral_features.npz"
    )
    if os.path.exists(dihedral_path):
        with np.load(dihedral_path) as data:
            # Check feature generation date if available for temporal cutoff
            if temporal_cutoff is not None and "metadata" in data:
                # Try to extract generation date from metadata
                try:
                    metadata_str = str(data["metadata"])
                    if "extraction_timestamp" in metadata_str:
                        # Extract date portion (assuming format like '2025-04-13 00:10:44')
                        timestamp_part = metadata_str.split("extraction_timestamp")[
                            1
                        ].split("'")[1]
                        generation_date = timestamp_part.split()[
                            0
                        ]  # Take just the date part

                        if pd.to_datetime(generation_date) > pd.to_datetime(
                            temporal_cutoff
                        ):
                            warnings.warn(
                                f"Dihedral features for {target_id} were generated after the temporal cutoff. Using zeros."
                            )
                            features["dihedral"] = None
                            return features
                except (KeyError, IndexError, ValueError):
                    # If any parsing error occurs, continue without temporal check
                    pass

            features["dihedral"] = {"features": data["features"].astype(np.float32)}
            # Handle NaN values if present
            if np.isnan(features["dihedral"]["features"]).any():
                features["dihedral"]["features"] = np.nan_to_num(
                    features["dihedral"]["features"], nan=0.0
                )
    else:
        # For test data or if file is missing, default will be created later based on sequence length
        features["dihedral"] = None
        warnings.warn(f"Dihedral features not found for {target_id}. Using zeros.")

    # 2. Load thermodynamic features (required)
    thermo_path = os.path.join(
        features_dir, "thermo_features", f"{target_id}_thermo_features.npz"
    )
    if not os.path.exists(thermo_path):
        raise ValueError(
            f"Thermodynamic features not found for {target_id}. Required for prediction."
        )

    with np.load(thermo_path) as data:
        # Check feature generation date if available for temporal cutoff
        if temporal_cutoff is not None and "generation_date" in data:
            generation_date = str(data["generation_date"])
            if pd.to_datetime(generation_date) > pd.to_datetime(temporal_cutoff):
                warnings.warn(
                    f"Thermo features for {target_id} were generated after the temporal cutoff. Using zeros."
                )
                features["thermo"] = None
                return features

        # Extract key arrays and scalar values
        thermo_features = {}

        # Get pairing probabilities matrix (critical)
        if "pairing_probs" in data:
            thermo_features["pairing_probs"] = data["pairing_probs"].astype(np.float32)
        elif "base_pair_probs" in data:
            thermo_features["pairing_probs"] = data["base_pair_probs"].astype(
                np.float32
            )
        else:
            raise ValueError(f"No pairing probability matrix found for {target_id}")

        # Get sequence length for defaults
        seq_len = thermo_features["pairing_probs"].shape[0]

        # Get positional entropy
        if "positional_entropy" in data:
            thermo_features["positional_entropy"] = data["positional_entropy"].astype(
                np.float32
            )
        elif "position_entropy" in data:
            thermo_features["positional_entropy"] = data["position_entropy"].astype(
                np.float32
            )
        else:
            thermo_features["positional_entropy"] = np.zeros(seq_len, dtype=np.float32)
            warnings.warn(f"No positional entropy found for {target_id}. Using zeros.")

        # Get accessibility
        if "accessibility" in data:
            thermo_features["accessibility"] = data["accessibility"].astype(np.float32)
        else:
            thermo_features["accessibility"] = np.zeros(seq_len, dtype=np.float32)
            warnings.warn(f"No accessibility found for {target_id}. Using zeros.")

        # Get sequence if available
        if "sequence" in data:
            thermo_features["sequence"] = str(data["sequence"])

        # Get scalar features
        scalar_features = [
            "mfe",
            "ensemble_energy",
            "mfe_probability",
            "gc_content",
            "paired_fraction",
        ]
        for key in scalar_features:
            if key in data:
                thermo_features[key] = float(data[key])

        features["thermo"] = thermo_features

    # 3. Load evolutionary coupling features (optional)
    # Try both possible directory names with fallback
    mi_path = os.path.join(features_dir, "evolutionary_features", f"{target_id}_mi_features.npz")
    
    # Fallback to mi_features directory if evolutionary_features doesn't exist
    if not os.path.exists(mi_path):
        mi_path = os.path.join(features_dir, "mi_features", f"{target_id}_mi_features.npz")
        
    if os.path.exists(mi_path):
        with np.load(mi_path) as data:
            # Check feature generation date if available for temporal cutoff
            if temporal_cutoff is not None and "generation_date" in data:
                generation_date = str(data["generation_date"])
                if pd.to_datetime(generation_date) > pd.to_datetime(temporal_cutoff):
                    warnings.warn(
                        f"MI features for {target_id} were generated after the temporal cutoff. Using zeros."
                    )
                    seq_len = features["thermo"]["pairing_probs"].shape[0]
                    features["evolutionary"] = {
                        "coupling_matrix": np.zeros(
                            (seq_len, seq_len), dtype=np.float32
                        ),
                        "has_valid_mi": False,
                    }
                    return features

            evolutionary_features = {}

            # OPTIMIZATION: First check if top_pairs indicates uniform MI
            if "top_pairs" in data:
                top_pairs = data["top_pairs"]
                if is_uniform_top_pairs(top_pairs):
                    # Uniform MI detected from top_pairs - no need to load full matrix
                    seq_len = features["thermo"]["pairing_probs"].shape[0]
                    evolutionary_features["coupling_matrix"] = np.zeros(
                        (seq_len, seq_len), dtype=np.float32
                    )
                    evolutionary_features["has_valid_mi"] = False

                    # Add other evolutionary features as needed
                    for key in data.files:
                        if key != "coupling_matrix":  # Skip loading the full matrix
                            evolutionary_features[key] = data[key]

                    features["evolutionary"] = evolutionary_features
                    return features

            # If top_pairs doesn't exist or indicates non-uniform MI, load full coupling_matrix
            for key in data.files:
                evolutionary_features[key] = data[key]

            # Check if MI matrix is uniform as a fallback
            if "coupling_matrix" in evolutionary_features:
                has_valid_mi = not is_uniform_mi_matrix(
                    evolutionary_features["coupling_matrix"]
                )
                evolutionary_features["has_valid_mi"] = has_valid_mi

                # If MI is not valid, zero out the coupling matrix to save memory
                if not has_valid_mi:
                    evolutionary_features["coupling_matrix"] = np.zeros_like(
                        evolutionary_features["coupling_matrix"]
                    )
            else:
                evolutionary_features["has_valid_mi"] = False

            features["evolutionary"] = evolutionary_features
    else:
        # Create empty evolutionary features based on sequence length
        if "thermo" in features and "pairing_probs" in features["thermo"]:
            seq_len = features["thermo"]["pairing_probs"].shape[0]
            features["evolutionary"] = {
                "coupling_matrix": np.zeros((seq_len, seq_len), dtype=np.float32),
                "has_valid_mi": False,
            }
        else:
            features["evolutionary"] = None
        warnings.warn(f"Evolutionary features not found for {target_id}. Using zeros.")

    return features


class RNADataset(Dataset):
    """Dataset class for RNA 3D structure prediction with partial data handling."""

    def __init__(
        self,
        sequences_csv_path: str,
        labels_csv_path: Optional[str] = None,
        features_dir: str = "",
        split_fn: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
        temporal_cutoff: Optional[str] = None,
        use_validation_set: bool = False,
        require_features: bool = True,
    ):
        """Initialize RNA dataset with pluggable split logic and feature filtering.

        Args:
            sequences_csv_path: Path to CSV file containing RNA sequences
            labels_csv_path: Path to CSV file containing 3D coordinates (optional for inference)
            features_dir: Directory containing feature subdirectories
            split_fn: Optional function to apply custom splitting logic
            temporal_cutoff: Optional date string for temporal validation split
            use_validation_set: If True, use validation set, otherwise training set
            require_features: If True, only use sequences with available features
        """
        # Store paths (NO hardcoded paths)
        self.features_dir = features_dir
        self.require_features = require_features

        # Store temporal cutoff parameters for potential reconfiguration
        self.temporal_cutoff = temporal_cutoff
        self.use_validation_set = use_validation_set

        # Load sequences
        self.sequences_df = pd.read_csv(sequences_csv_path)

        # Apply custom split function if provided
        if split_fn is not None:
            self.sequences_df = split_fn(self.sequences_df)
        # Otherwise, apply temporal cutoff if specified
        elif temporal_cutoff is not None and not use_validation_set:
            self.sequences_df = self.sequences_df[
                pd.to_datetime(self.sequences_df["temporal_cutoff"])
                <= pd.to_datetime(temporal_cutoff)
            ]

        # Extract all target IDs and sequences
        self.target_ids = self.sequences_df["target_id"].tolist()
        self.sequences = self.sequences_df["sequence"].tolist()

        # Initialize feature availability cache
        self._availability_cache = {}

        # Populate availability cache and filter sequences based on feature availability
        self._scan_available_features()
        self._update_filtered_sequences()

        # Load labels if available
        self.labels_df = None
        self.coordinates = {}
        if labels_csv_path is not None and os.path.exists(labels_csv_path):
            self.labels_df = pd.read_csv(labels_csv_path)

            # Pre-load coordinates for filtered sequences if not too memory intensive
            # For large datasets, you might want to load coordinates on-demand in __getitem__
            for target_id in self.filtered_sequences:
                try:
                    coords, _ = load_coordinates(self.labels_df, target_id)
                    self.coordinates[target_id] = coords
                except Exception as e:
                    warnings.warn(f"Could not load coordinates for {target_id}: {e}")

        # Nucleotide to integer mapping
        self.nuc_to_int = {"A": 0, "C": 1, "G": 2, "U": 3, "T": 3, "N": 4}

    def _scan_available_features(self):
        """Scan features directory and populate availability cache."""
        for target_id in self.target_ids:
            self._availability_cache[target_id] = check_features_availability(
                target_id, self.features_dir
            )

    def _get_temporal_filtered_ids(self) -> List[str]:
        """Get target IDs filtered by temporal cutoff."""
        if not hasattr(self, "_temporal_filtered_cache"):
            # Cache the results to avoid recomputing
            if self.temporal_cutoff is not None and not self.use_validation_set:
                cutoff_date = pd.to_datetime(self.temporal_cutoff)
                self._temporal_filtered_cache = set(
                    self.sequences_df[
                        pd.to_datetime(self.sequences_df["temporal_cutoff"])
                        <= cutoff_date
                    ]["target_id"].tolist()
                )
            else:
                self._temporal_filtered_cache = set(
                    self.sequences_df["target_id"].tolist()
                )

        return self._temporal_filtered_cache

    def _update_filtered_sequences(self):
        """Update the list of filtered sequences based on feature requirements."""
        if not self.require_features:
            # Include all sequences if features not required
            self.filtered_sequences = self.target_ids.copy()
        else:
            # Filter based on feature availability
            self.filtered_sequences = []
            for target_id in self.target_ids:
                if target_id not in self._availability_cache:
                    continue

                # Check if all required features are available
                avail = self._availability_cache[target_id]
                if all(avail.values()):
                    self.filtered_sequences.append(target_id)

        # Apply temporal filtering if needed
        if self.temporal_cutoff is not None and not self.use_validation_set:
            temporal_ids = self._get_temporal_filtered_ids()
            self.filtered_sequences = [
                target_id
                for target_id in self.filtered_sequences
                if target_id in temporal_ids
            ]

    def update_available_features(self) -> int:
        """Update the list of available features.

        Rescans the features directory to identify newly available feature files
        and updates the filtered sequence list accordingly, maintaining temporal boundaries.

        Returns:
            Number of sequences with complete feature sets
        """
        # Reset availability cache
        self._availability_cache = {}

        # Scan for available features
        self._scan_available_features()

        # Update filtered sequences with proper temporal boundary enforcement
        self._update_filtered_sequences()

        # Return number of valid sequences
        return len(self.filtered_sequences)

    def set_temporal_cutoff(self, new_cutoff: Optional[str] = None) -> None:
        """
        Update the temporal cutoff and refilter sequences.

        Args:
            new_cutoff: New temporal cutoff date or None to remove cutoff
        """
        self.temporal_cutoff = new_cutoff

        # Clear cached filtered IDs
        if hasattr(self, "_temporal_filtered_cache"):
            delattr(self, "_temporal_filtered_cache")

        # Reapply filtering with new cutoff
        self._update_filtered_sequences()

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.filtered_sequences)

    def sequence_to_int(self, sequence: str) -> List[int]:
        """Convert nucleotide sequence to integer indices."""
        return [self.nuc_to_int.get(nuc, 4) for nuc in sequence]

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get features and labels for a single RNA sequence.

        Args:
            idx: Index of the sequence

        Returns:
            Dictionary containing features and labels for the sequence
        """
        # Get target_id from filtered sequences
        target_id = self.filtered_sequences[idx]

        # Find the original index to get the sequence
        original_idx = self.target_ids.index(target_id)
        sequence = self.sequences[original_idx]
        sequence_length = len(sequence)

        # Load precomputed features with temporal cutoff
        try:
            features = load_precomputed_features(
                target_id, self.features_dir, self.temporal_cutoff
            )
        except Exception as e:
            raise RuntimeError(f"Error loading features for {target_id}: {e}")

        # Convert sequence to integers
        sequence_int = self.sequence_to_int(sequence)

        # Create sample dictionary
        sample = {
            "target_id": target_id,
            "sequence_int": torch.tensor(sequence_int, dtype=torch.long),
            "length": sequence_length,
        }

        # Generate metadata flags based on feature availability and MI validity
        meta = {}
        if target_id in self._availability_cache:
            avail = self._availability_cache[target_id]
            meta["has_dihedrals"] = torch.tensor(avail["dihedral"], dtype=torch.bool)
            meta["has_thermo"] = torch.tensor(avail["thermo"], dtype=torch.bool)

            # For MI, check both availability and validity
            has_mi = avail["mi"]
            has_valid_mi = (
                has_mi
                and features["evolutionary"] is not None
                and features["evolutionary"].get("has_valid_mi", False)
            )
            meta["has_msa"] = torch.tensor(has_valid_mi, dtype=torch.bool)
        else:
            # Default to False if not in cache
            meta["has_dihedrals"] = torch.tensor(False, dtype=torch.bool)
            meta["has_thermo"] = torch.tensor(False, dtype=torch.bool)
            meta["has_msa"] = torch.tensor(False, dtype=torch.bool)

        # Add temporal flag
        meta["before_cutoff"] = torch.tensor(
            True, dtype=torch.bool
        )  # Always true due to filtering

        # Add training/validation flag
        if hasattr(self.sequences_df, "temporal_cutoff"):
            is_train = True
            if "validation" in self.sequences_df.columns:
                original_row = self.sequences_df.iloc[original_idx]
                is_train = not original_row["validation"]
            meta["is_train"] = torch.tensor(is_train, dtype=torch.bool)

        # Add metadata to sample
        sample["meta"] = meta

        # Get dihedral tensors (input and target)
        dihedral_input, dihedral_target = get_dihedral_tensors(
            target_id, self.features_dir
        )

        # Add dihedral features
        if dihedral_input.shape[0] > 0:
            sample["dihedral_features"] = dihedral_input
        else:
            # Create default zero tensor
            sample["dihedral_features"] = torch.zeros(
                (sequence_length, 4), dtype=torch.float32
            )

        # Add target dihedral angles (for future use)
        if dihedral_target.shape[0] > 0:
            sample["dihedral_targets"] = dihedral_target
        else:
            sample["dihedral_targets"] = torch.zeros(
                (sequence_length, 4), dtype=torch.float32
            )

        # Add thermodynamic features
        if features["thermo"] is not None:
            sample["pairing_probs"] = torch.tensor(
                features["thermo"]["pairing_probs"], dtype=torch.float32
            )

            if "positional_entropy" in features["thermo"]:
                sample["positional_entropy"] = torch.tensor(
                    features["thermo"]["positional_entropy"], dtype=torch.float32
                )
            else:
                sample["positional_entropy"] = torch.zeros(
                    sequence_length, dtype=torch.float32
                )

            if "accessibility" in features["thermo"]:
                sample["accessibility"] = torch.tensor(
                    features["thermo"]["accessibility"], dtype=torch.float32
                )
            else:
                sample["accessibility"] = torch.zeros(
                    sequence_length, dtype=torch.float32
                )

            # Add scalar thermodynamic features (as individual scalars)
            scalar_features = [
                "mfe",
                "ensemble_energy",
                "mfe_probability",
                "gc_content",
            ]
            for key in scalar_features:
                if key in features["thermo"]:
                    # Store as scalar
                    sample[key] = features["thermo"][key]
        else:
            # Create default zero tensors if thermo features are filtered by temporal cutoff
            sample["pairing_probs"] = torch.zeros(
                (sequence_length, sequence_length), dtype=torch.float32
            )
            sample["positional_entropy"] = torch.zeros(
                sequence_length, dtype=torch.float32
            )
            sample["accessibility"] = torch.zeros(sequence_length, dtype=torch.float32)

        # Add evolutionary features if available
        if features["evolutionary"] is not None:
            sample["coupling_matrix"] = torch.tensor(
                features["evolutionary"]["coupling_matrix"], dtype=torch.float32
            )

            if "conservation" in features["evolutionary"]:
                sample["conservation"] = torch.tensor(
                    features["evolutionary"]["conservation"], dtype=torch.float32
                )
            else:
                sample["conservation"] = torch.zeros(
                    sequence_length, dtype=torch.float32
                )
        else:
            # Create default zero tensors
            sample["coupling_matrix"] = torch.zeros(
                (sequence_length, sequence_length), dtype=torch.float32
            )
            sample["conservation"] = torch.zeros(sequence_length, dtype=torch.float32)

        # Add coordinates if available (for training)
        if target_id in self.coordinates:
            sample["coordinates"] = torch.tensor(
                self.coordinates[target_id], dtype=torch.float32
            )

        # Verify shapes
        expected_length = len(sequence)
        for key, tensor in sample.items():
            if isinstance(tensor, torch.Tensor) and key != "meta":
                if key in [
                    "dihedral_features",
                    "dihedral_targets",
                    "positional_entropy",
                    "accessibility",
                    "conservation",
                ]:
                    assert (
                        tensor.shape[0] == expected_length
                    ), f"Feature {key} length mismatch: {tensor.shape[0]} vs {expected_length}"
                elif key in ["pairing_probs", "coupling_matrix"]:
                    assert (
                        tensor.shape[0] == expected_length
                        and tensor.shape[1] == expected_length
                    ), f"Feature {key} shape mismatch: {tensor.shape} vs ({expected_length}, {expected_length})"

        return sample


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate a list of samples into a batch with robust shape handling.

    Args:
        batch: List of dictionaries, each containing features for a single sequence

    Returns:
        Dictionary containing batched tensors
    """
    import logging
    logger = logging.getLogger(__name__)
    
    # Get batch size and maximum sequence length
    batch_size = len(batch)
    max_len = max(sample["length"] for sample in batch)

    # Extract target IDs
    target_ids = [sample["target_id"] for sample in batch]

    # Initialize output dictionary
    output = {
        "target_ids": target_ids,
        "lengths": torch.tensor(
            [sample["length"] for sample in batch], dtype=torch.long
        ),
    }

    # Process meta dictionary separately
    if "meta" in batch[0]:
        meta = {}
        for key in batch[0]["meta"].keys():
            # Collect values for this key from all samples
            meta[key] = torch.stack([sample["meta"][key] for sample in batch])
        output["meta"] = meta

    # Process each tensor in the batch
    for key in batch[0].keys():
        if key in ["target_id", "length", "meta"]:
            continue  # Already processed

        if isinstance(batch[0][key], torch.Tensor):
            # Process tensor based on its shape
            try:
                if len(batch) == 0:
                    logger.warning(f"Empty batch encountered for key: {key}")
                    continue
                    
                # Get sample shapes for debugging
                sample_shapes = [sample[key].shape for sample in batch if key in sample]
                
                # Check if shapes are compatible for stacking
                if len(sample_shapes) > 1 and any(shape != sample_shapes[0] for shape in sample_shapes[1:]):
                    logger.warning(f"Incompatible shapes for key {key}: {sample_shapes}")
                
                # Handle different tensor shapes
                sample_shape = batch[0][key].shape
                
                if len(sample_shape) == 1:
                    # 1D tensor (sequence, per-residue features)
                    output[key] = torch.stack(
                        [pad_1d(sample[key], max_len) for sample in batch if key in sample]
                    )
                elif len(sample_shape) == 2:
                    if sample_shape[0] != sample_shape[1] or key in ["dihedral_features", "dihedral_targets"]:
                        # 2D tensor with feature dimension (L, D)
                        output[key] = torch.stack(
                            [pad_2d(sample[key], max_len) for sample in batch if key in sample]
                        )
                    else:
                        # 2D square matrix (L, L) - e.g., pairing_probs, coupling_matrix
                        output[key] = torch.stack(
                            [pad_2d(sample[key], max_len) for sample in batch if key in sample]
                        )
                elif len(sample_shape) > 2:
                    # Handle higher dimensional tensors
                    # Create target shape with batch dimension
                    target_shape = (max_len,) + sample_shape[1:]
                    output[key] = torch.stack(
                        [pad_tensor(sample[key], target_shape) for sample in batch if key in sample]
                    )
            except Exception as e:
                logger.error(f"Error processing key {key}: {e}")
                logger.error(f"Shapes: {[sample[key].shape if key in sample and isinstance(sample[key], torch.Tensor) else None for sample in batch]}")
                # Don't include this tensor if there's an error
                continue

        elif isinstance(batch[0][key], (int, float)):
            # Handle scalar values
            values = [sample[key] for sample in batch if key in sample]
            if values:  # Only create tensor if we have values
                output[key] = torch.tensor(values)

    # Create attention mask (True for valid positions, False for padding)
    mask = torch.zeros((batch_size, max_len), dtype=torch.bool)
    for i, sample in enumerate(batch):
        mask[i, : sample["length"]] = True
    output["mask"] = mask

    # Check for missing required model inputs
    required_keys = ["sequence_int", "pairing_probs", "positional_entropy", "accessibility", "coupling_matrix"]
    for key in required_keys:
        if key not in output:
            logger.warning(f"Missing required key in batch: {key}")
            # Try to create default tensor if possible
            if key == "sequence_int":
                output[key] = torch.zeros((batch_size, max_len), dtype=torch.long)
            elif key in ["positional_entropy", "accessibility"]:
                output[key] = torch.zeros((batch_size, max_len), dtype=torch.float32)
            elif key in ["pairing_probs", "coupling_matrix"]:
                output[key] = torch.zeros((batch_size, max_len, max_len), dtype=torch.float32)

    # Rename 'coordinates' to 'coords' for validate function
    if "coordinates" in output:
        output["coords"] = output["coordinates"]

    return output


def create_data_loader(
    sequences_csv_path: str,
    labels_csv_path: Optional[str] = None,
    features_dir: str = "",
    batch_size: int = 32,
    split_fn: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
    temporal_cutoff: Optional[str] = None,
    use_validation_set: bool = False,
    require_features: bool = True,
    shuffle: bool = True,
    num_workers: int = 4,
    distributed: bool = False,
) -> torch.utils.data.DataLoader:
    """Create data loader for RNA structure prediction with flexible splitting and feature filtering.

    Args:
        sequences_csv_path: Path to CSV file containing RNA sequences
        labels_csv_path: Path to CSV file containing 3D coordinates (optional for inference)
        features_dir: Directory containing feature subdirectories
        batch_size: Number of sequences per batch
        split_fn: Optional function to apply custom splitting logic
        temporal_cutoff: Optional date string for temporal validation split
        use_validation_set: If True, use validation set, otherwise training set
        require_features: If True, only use sequences with available features
        shuffle: Whether to shuffle the dataset
        num_workers: Number of worker threads for data loading
        distributed: Whether to use DistributedSampler

    Returns:
        DataLoader object that yields batches with features and labels
    """
    # Verify paths
    if not os.path.exists(sequences_csv_path):
        raise FileNotFoundError(f"Sequences file not found: {sequences_csv_path}")

    if labels_csv_path is not None and not os.path.exists(labels_csv_path):
        warnings.warn(
            f"Labels file not found: {labels_csv_path}. Running in inference mode."
        )
        labels_csv_path = None

    if features_dir and not os.path.exists(features_dir):
        warnings.warn(f"Features directory not found: {features_dir}")

    # Log temporal cutoff settings if provided
    if temporal_cutoff is not None:
        if use_validation_set:
            print(
                f"Creating validation loader (ignoring temporal cutoff: {temporal_cutoff})"
            )
        else:
            print(f"Creating data loader with temporal cutoff: {temporal_cutoff}")

    # Create dataset
    dataset = RNADataset(
        sequences_csv_path=sequences_csv_path,
        labels_csv_path=labels_csv_path,
        features_dir=features_dir,
        split_fn=split_fn,
        temporal_cutoff=temporal_cutoff,
        use_validation_set=use_validation_set,
        require_features=require_features,
    )

    # Check dataset size
    if len(dataset) == 0:
        warnings.warn(
            f"Dataset is empty. Check feature filtering settings or data availability."
        )
        if require_features:
            warnings.warn(
                f"Try setting require_features=False to include sequences without features."
            )
        if temporal_cutoff is not None and not use_validation_set:
            warnings.warn(
                f"Try adjusting temporal_cutoff value (current: {temporal_cutoff}) or set use_validation_set=True"
            )

    # Create sampler for distributed training
    sampler = None
    if distributed and torch.distributed.is_initialized():
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, shuffle=shuffle
        )
        shuffle = False  # Sampling is handled by the DistributedSampler

    # Create data loader
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=False,
    )

    # Add reconfiguration method to data loader
    data_loader.set_temporal_cutoff = dataset.set_temporal_cutoff
    data_loader.update_available_features = dataset.update_available_features

    return data_loader