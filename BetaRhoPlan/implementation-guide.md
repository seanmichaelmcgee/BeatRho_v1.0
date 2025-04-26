# Detailed Implementation Guide: Integrating RhoFold+ IPA with RNA Feature Embedding Model

This guide provides step-by-step instructions for integrating the Invariant Point Attention (IPA) module from RhoFold+ with your existing RNA feature embedding model. Each section includes code examples, potential issues, and debugging strategies.

## 1. Environment Setup

First, ensure your development environment has all necessary dependencies for both models.

```bash
# Clone RhoFold+ repository
git clone https://github.com/ml4bio/RhoFold.git

# Install RhoFold+ dependencies
cd RhoFold
pip install -r requirements.txt

# Return to your project
cd ..

# Create symbolic links to required RhoFold+ files (optional)
mkdir -p external/rhofold
ln -s $(pwd)/RhoFold/rhofold/model/structure_module.py external/rhofold/structure_module.py
ln -s $(pwd)/RhoFold/rhofold/utils/rigid_utils.py external/rhofold/rigid_utils.py
```

## 2. Extract Required Components

Create a new module in your project to house the extracted components from RhoFold+.

```python
# src/model/ipa_module.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any

# Import RhoFold+ components
from external.rhofold.structure_module import InvariantPointAttention
from external.rhofold.rigid_utils import Rigid

# Create a simplified version for your needs
class IPAModule(nn.Module):
    """Invariant Point Attention module adapted from RhoFold+"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        # Extract dimensions from config
        self.residue_dim = config.get("residue_embed_dim", 128)
        self.pair_dim = config.get("pair_embed_dim", 64)
        
        # RhoFold+ IPA parameters
        self.c_s = config.get("ipa_single_dim", 384)  # RhoFold+ single representation dimension
        self.c_z = config.get("ipa_pair_dim", 128)    # RhoFold+ pair representation dimension
        self.c_ipa = config.get("ipa_hidden_dim", 16) # IPA hidden dimension
        self.no_heads_ipa = config.get("ipa_heads", 12)  # Number of IPA attention heads
        self.no_qk_points = config.get("ipa_qk_points", 4)  # Number of query/key points
        self.no_v_points = config.get("ipa_v_points", 8)    # Number of value points
        
        # Dimension adaptation layers
        self.single_adapter = nn.Linear(self.residue_dim, self.c_s)
        self.pair_adapter = nn.Linear(self.pair_dim, self.c_z)
        
        # IPA module from RhoFold+
        self.ipa = InvariantPointAttention(
            c_s=self.c_s,
            c_z=self.c_z,
            c_ipa=self.c_ipa,
            no_heads=self.no_heads_ipa,
            no_qk_points=self.no_qk_points,
            no_v_points=self.no_v_points,
        )
        
        # Output projection layers
        self.output_projection = nn.Linear(self.c_s, 6)  # 6 for 3 angles + 3 coordinates
```

## 3. Implement the Rigid Integration

The Rigid class is crucial for IPA to work properly. Add a method to create and manage these rigid transformations:

```python
# Add to IPAModule class

def initialize_rigid(self, batch_size: int, seq_len: int, 
                   device: torch.device) -> Rigid:
    """Initialize rigid transformations for the IPA module"""
    
    # Create identity quaternions and zero translations
    # This serves as the starting point for the rigid transforms
    rigid = Rigid.identity(
        shape=(batch_size, seq_len),
        dtype=torch.float32,
        device=device,
        fmt="quat"  # quaternion format for rotations
    )
    
    return rigid

def update_rigid_from_angles(self, rigid: Rigid, angles: torch.Tensor) -> Rigid:
    """Update rigid transformations based on predicted angles"""
    
    # The angles tensor is expected to be [batch_size, seq_len, 3]
    # We convert these angles to quaternion updates
    q_update_vec = angles
    
    # Apply the update to the rigid object
    updated_rigid = rigid.compose_q_update_vec(q_update_vec)
    
    return updated_rigid
```

## 4. Create the Full Integration Module

Now, implement the full forward pass connecting your model to the IPA module:

```python
# Complete the IPAModule forward method

def forward(
    self,
    residue_repr: torch.Tensor,  # [batch_size, seq_len, residue_dim]
    pair_repr: torch.Tensor,     # [batch_size, seq_len, seq_len, pair_dim]
    mask: Optional[torch.Tensor] = None,  # [batch_size, seq_len]
    initial_rigid: Optional[Rigid] = None
) -> Dict[str, torch.Tensor]:
    """
    Forward pass through the IPA module
    
    Args:
        residue_repr: Residue representations [batch_size, seq_len, residue_dim]
        pair_repr: Pair representations [batch_size, seq_len, seq_len, pair_dim]
        mask: Boolean mask where True indicates valid positions [batch_size, seq_len]
        initial_rigid: Optional initial rigid transformations
        
    Returns:
        Dictionary containing:
            - angles: Predicted torsion angles [batch_size, seq_len, 3]
            - coordinates: Predicted 3D coordinates [batch_size, seq_len, 3]
            - single_repr: Updated single representation [batch_size, seq_len, c_s]
    """
    batch_size, seq_len, _ = residue_repr.shape
    device = residue_repr.device
    
    # 1. Adapt dimensions to RhoFold+ expectations
    s = self.single_adapter(residue_repr)  # [batch_size, seq_len, c_s]
    z = self.pair_adapter(pair_repr)       # [batch_size, seq_len, seq_len, c_z]
    
    # 2. Initialize or use provided rigid transformations
    if initial_rigid is None:
        rigid = self.initialize_rigid(batch_size, seq_len, device)
    else:
        rigid = initial_rigid
    
    # 3. Apply IPA
    s_updated = self.ipa(
        s=s,
        z=z,
        r=rigid,
        mask=mask
    )
    
    # 4. Project to output values (angles and coordinates)
    output = self.output_projection(s_updated)
    
    # Split output into angles and coordinates
    angles = output[..., :3]       # [batch_size, seq_len, 3]
    coordinates = output[..., 3:]  # [batch_size, seq_len, 3]
    
    # 5. Update rigid transformations based on predicted angles
    updated_rigid = self.update_rigid_from_angles(rigid, angles)
    
    # 6. Apply mask if provided
    if mask is not None:
        mask_3d = mask.unsqueeze(-1).expand(-1, -1, 3).to(angles.dtype)
        angles = angles * mask_3d
        coordinates = coordinates * mask_3d
    
    return {
        "angles": angles,
        "coordinates": coordinates,
        "single_repr": s_updated,
        "rigid": updated_rigid
    }
```

## 5. Integrate with Existing Model

Modify your main RNA model class to incorporate the new IPA module:

```python
# src/model/rna_folding_model.py

from .ipa_module import IPAModule

class RNAFoldingModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        # Your existing embedding module
        self.embedding_module = EmbeddingModule(config)
        
        # Your existing transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.num_blocks)
        ])
        
        # Add IPA module
        self.ipa_module = IPAModule(config)
        
        # Optionally add refinement network
        if config.get("use_refinement", False):
            from external.rhofold.structure_module import RefineNet
            self.refinement = RefineNet(**config.get("refinement_config", {}))
        else:
            self.refinement = None
    
    def forward(self, batch):
        # Process through embedding module
        residue_repr, pair_repr, mask = self.embedding_module(batch)
        
        # Process through transformer blocks
        for block in self.transformer_blocks:
            residue_repr, pair_repr = block(residue_repr, pair_repr, mask)
        
        # Generate structure using IPA
        ipa_outputs = self.ipa_module(residue_repr, pair_repr, mask)
        
        # Optional refinement
        if self.refinement is not None:
            # Convert batch sequence to tokens for refinement
            tokens = batch.get("sequence_tokens", None)
            if tokens is not None:
                refined_coords = self.refinement(
                    tokens=tokens,
                    cords=ipa_outputs["coordinates"].reshape(
                        tokens.shape[0], -1, 3
                    )
                )
                ipa_outputs["refined_coordinates"] = refined_coords
        
        return ipa_outputs
```

## 6. RNA-Specific Coordinate Refinement

For better RNA structure prediction, you may want to add RNA-specific constraints:

```python
# Add to IPAModule class

def apply_rna_constraints(
    self,
    coordinates: torch.Tensor,
    mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Apply RNA-specific geometric constraints to coordinates"""
    
    batch_size, seq_len, _ = coordinates.shape
    device = coordinates.device
    
    # Skip constraint application for very short sequences
    if seq_len <= 3:
        return coordinates
    
    # Typical distance between adjacent C1' atoms in RNA (~6Å)
    c1_c1_distance = 6.0
    
    # Softly enforce this distance for adjacent residues
    refined_coords = coordinates.clone()
    
    # For each adjacent pair, adjust distances
    for i in range(seq_len - 1):
        # Get coordinates of adjacent residues
        p1 = coordinates[:, i, :]      # [batch_size, 3]
        p2 = coordinates[:, i + 1, :]  # [batch_size, 3]
        
        # Calculate current distance
        diff = p2 - p1  # [batch_size, 3]
        dist = torch.norm(diff, dim=-1, keepdim=True)  # [batch_size, 1]
        
        # Unit vector from p1 to p2
        unit_vec = diff / (dist + 1e-8)  # [batch_size, 3]
        
        # Calculate distance adjustment factor (softly)
        # Using a spring-like model: force ~ (dist - ideal_dist)
        adjustment = (dist - c1_c1_distance) * 0.1
        
        # Apply adjustment in opposite directions
        if mask is None or (mask[:, i].bool().all() and mask[:, i+1].bool().all()):
            refined_coords[:, i, :] += adjustment * unit_vec
            refined_coords[:, i+1, :] -= adjustment * unit_vec
    
    return refined_coords
```

## 7. Memory Optimization

IPA can be memory-intensive. Add memory optimization techniques:

```python
# Add to IPAModule forward method, before IPA application

# Apply gradient checkpointing for memory efficiency during training
if self.training and getattr(self, "use_gradient_checkpointing", False):
    # Wrap the IPA call in a checkpointed function
    def create_custom_forward(module):
        def custom_forward(*inputs):
            return module(*inputs)
        return custom_forward
    
    # Apply checkpointing
    s_updated = torch.utils.checkpoint.checkpoint(
        create_custom_forward(self.ipa),
        s, z, rigid, mask
    )
else:
    # Normal forward pass
    s_updated = self.ipa(s=s, z=z, r=rigid, mask=mask)
```

## 8. Handling Different Sequence Lengths

Make sure the model can handle variable sequence lengths efficiently:

```python
# Add to your data preparation code

def collate_fn(batch):
    """Custom collate function to handle variable sequence lengths"""
    
    # Find the maximum sequence length in this batch
    max_len = max(item["seq_len"] for item in batch)
    
    # Initialize batch tensors
    batch_size = len(batch)
    sequences = torch.zeros(batch_size, max_len, dtype=torch.long)
    masks = torch.zeros(batch_size, max_len, dtype=torch.bool)
    residue_features = torch.zeros(batch_size, max_len, batch[0]["residue_features"].shape[-1])
    pair_features = torch.zeros(batch_size, max_len, max_len, batch[0]["pair_features"].shape[-1])
    
    # Fill tensors with data
    for i, item in enumerate(batch):
        seq_len = item["seq_len"]
        sequences[i, :seq_len] = item["sequence"]
        masks[i, :seq_len] = True
        residue_features[i, :seq_len] = item["residue_features"]
        pair_features[i, :seq_len, :seq_len] = item["pair_features"]
    
    return {
        "sequences": sequences,
        "masks": masks,
        "residue_features": residue_features,
        "pair_features": pair_features
    }
```

## 9. Validation Strategy

Implement a validation strategy to ensure correctness:

```python
# src/validation.py

def validate_ipa_integration(model, validation_data, device):
    """Validate the IPA integration by checking outputs and structure quality"""
    
    model.eval()
    all_metrics = []
    
    with torch.no_grad():
        for batch in validation_data:
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            outputs = model(batch)
            
            # Check basic output shapes
            batch_size, seq_len = batch["masks"].shape
            assert outputs["angles"].shape == (batch_size, seq_len, 3), \
                f"Expected angles shape {(batch_size, seq_len, 3)}, got {outputs['angles'].shape}"
            assert outputs["coordinates"].shape == (batch_size, seq_len, 3), \
                f"Expected coordinates shape {(batch_size, seq_len, 3)}, got {outputs['coordinates'].shape}"
            
            # Calculate RNA-specific metrics
            metrics = calculate_rna_metrics(
                pred_coords=outputs["coordinates"],
                true_coords=batch.get("true_coordinates", None),
                sequences=batch["sequences"],
                masks=batch["masks"]
            )
            
            all_metrics.append(metrics)
    
    # Aggregate metrics
    avg_metrics = {k: sum(m[k] for m in all_metrics) / len(all_metrics) 
                  for k in all_metrics[0].keys()}
    
    return avg_metrics

def calculate_rna_metrics(pred_coords, true_coords, sequences, masks):
    """Calculate RNA-specific structural quality metrics"""
    
    metrics = {}
    
    # Calculate basic geometric statistics
    bond_lengths = calculate_bond_lengths(pred_coords, masks)
    metrics["mean_bond_length"] = bond_lengths.mean().item()
    metrics["std_bond_length"] = bond_lengths.std().item()
    
    # Calculate bond angles
    bond_angles = calculate_bond_angles(pred_coords, masks)
    metrics["mean_bond_angle"] = bond_angles.mean().item()
    metrics["std_bond_angle"] = bond_angles.std().item()
    
    # If ground truth is available, calculate RMSD
    if true_coords is not None:
        rmsd = calculate_rmsd(pred_coords, true_coords, masks)
        metrics["rmsd"] = rmsd.mean().item()
    
    return metrics
```

## 10. Debugging Strategies

Add logging for easier debugging during development:

```python
# Add to IPAModule forward method

import logging
logger = logging.getLogger(__name__)

# Debug logging for tensor shapes and values
if self.debug:
    logger.debug(f"residue_repr shape: {residue_repr.shape}")
    logger.debug(f"pair_repr shape: {pair_repr.shape}")
    logger.debug(f"adapted single shape: {s.shape}")
    logger.debug(f"adapted pair shape: {z.shape}")
    
    # Check for NaNs
    if torch.isnan(s).any():
        logger.warning("NaNs detected in adapted single representation")
    if torch.isnan(z).any():
        logger.warning("NaNs detected in adapted pair representation")
    
    # Log value ranges
    logger.debug(f"s range: [{s.min().item()}, {s.max().item()}]")
    logger.debug(f"z range: [{z.min().item()}, {z.max().item()}]")
```

## 11. Hyperparameter Tuning

Suggestions for hyperparameter tuning:

```python
# Example hyperparameter sweep configuration
sweep_config = {
    # IPA Module parameters
    "ipa_single_dim": [256, 384, 512],
    "ipa_pair_dim": [64, 128, 256],
    "ipa_hidden_dim": [16, 32],
    "ipa_heads": [8, 12, 16],
    "ipa_qk_points": [4, 8],
    "ipa_v_points": [8, 16],
    
    # Learning parameters
    "learning_rate": [1e-4, 5e-4, 1e-3],
    "weight_decay": [0.0, 1e-5, 1e-4],
    
    # Architecture parameters
    "use_refinement": [True, False],
}
```

## 12. Performance Profiling

Add profiling for performance optimization:

```python
# src/utils/profiling.py

import time
import torch

class ProfilingContext:
    """Context manager for profiling model components"""
    
    def __init__(self, name, enabled=True):
        self.name = name
        self.enabled = enabled
        
    def __enter__(self):
        if self.enabled:
            self.start_time = time.time()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                self.start_event = torch.cuda.Event(enable_timing=True)
                self.end_event = torch.cuda.Event(enable_timing=True)
                self.start_event.record()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.enabled:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                self.end_event.record()
                torch.cuda.synchronize()
                gpu_time = self.start_event.elapsed_time(self.end_event) / 1000
                print(f"{self.name} - GPU time: {gpu_time:.4f}s")
            
            cpu_time = time.time() - self.start_time
            print(f"{self.name} - CPU time: {cpu_time:.4f}s")

# Usage in model forward pass
def forward(self, batch):
    with ProfilingContext("Embedding"):
        residue_repr, pair_repr, mask = self.embedding_module(batch)
    
    with ProfilingContext("Transformer"):
        for block in self.transformer_blocks:
            residue_repr, pair_repr = block(residue_repr, pair_repr, mask)
    
    with ProfilingContext("IPA"):
        ipa_outputs = self.ipa_module(residue_repr, pair_repr, mask)
    
    return ipa_outputs
```

## 13. Final Integration Testing

Ensure the full integration works end-to-end:

```python
# src/test_integration.py

import torch
import numpy as np
from model.rna_folding_model import RNAFoldingModel
from utils.visualization import visualize_rna_structure

def test_integration(config_path, checkpoint_path=None):
    """Test the full integration with a simple example"""
    
    # Load configuration
    import json
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Create model
    model = RNAFoldingModel(config)
    
    # Load checkpoint if provided
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
    
    # Move to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Create a simple test example
    seq = "GGGAAACCC"  # Simple RNA hairpin
    seq_len = len(seq)
    
    # Create dummy features
    batch = {
        "sequences": torch.tensor([[0, 0, 0, 1, 1, 1, 2, 2, 2]]),  # G=0, A=1, C=2
        "masks": torch.ones(1, seq_len, dtype=torch.bool),
        "residue_features": torch.randn(1, seq_len, config["residue_embed_dim"]),
        "pair_features": torch.randn(1, seq_len, seq_len, config["pair_embed_dim"])
    }
    
    # Move to device
    batch = {k: v.to(device) for k, v in batch.items()}
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        outputs = model(batch)
    
    # Visualize results
    coords = outputs["coordinates"][0].cpu().numpy()
    visualize_rna_structure(coords, seq)
    
    print("Integration test passed!")

# Run the test
if __name__ == "__main__":
    test_integration("configs/integration_test.json")
```

## Common Issues and Solutions

### Issue 1: Dimension Mismatch in IPA Module

**Symptoms**: 
- RuntimeError mentioning tensor shape mismatch in the IPA module
- Typically involves unsqueezing or matrix multiplication operations

**Solutions**:
- Double-check all tensor dimensions, especially after adapter layers
- Add intermediate shape checks and assertions to narrow down the issue
- Ensure input and output dimensions match RhoFold+ expectations

### Issue 2: NaN Values in IPA Output

**Symptoms**:
- Structure coordinates contain NaN values
- Loss suddenly becomes NaN during training

**Solutions**:
- Add normalization layers before IPA to stabilize values
- Check for extreme values in the input features
- Add gradient clipping during training
- Use torch.nan_to_num to handle NaNs safely

### Issue 3: Memory Issues with Long Sequences

**Symptoms**:
- CUDA out of memory errors for longer sequences
- Slow training or inference

**Solutions**:
- Enable gradient checkpointing in the IPA module
- Use lower precision (e.g., mixed precision training)
- Process sequences in chunks with a sliding window approach
- Reduce batch size or model dimensions

### Issue 4: Unrealistic RNA Structures

**Symptoms**:
- Predicted structures have unrealistic bond lengths or angles
- Structures appear distorted or collapsed

**Solutions**:
- Add more stringent RNA-specific geometric constraints
- Incorporate a physics-based loss term during training
- Increase the weight of the refinement module
- Check the scale of coordinate predictions (may need normalization)
