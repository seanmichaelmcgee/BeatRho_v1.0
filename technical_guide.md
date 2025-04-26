# Technical Guide: RhoFold+ IPA Integration

## Overview and Architecture

This guide outlines the technical approach for integrating the Invariant Point Attention (IPA) module from RhoFold+ with the existing RNA Feature Embedding Model to enhance RNA structure prediction capabilities.

### Current Systems

1. **RNA Feature Embedding Model**:
   - Processes RNA sequences and related features through transformer blocks
   - Produces two representations:
     - `residue_repr`: Shape `[batch_size, seq_len, 128]` - per-residue features
     - `pair_repr`: Shape `[batch_size, seq_len, seq_len, 64]` - pairwise interactions
   - Currently uses a simplified IPA module that directly projects residue representations to 3D coordinates

2. **RhoFold+ IPA Module**:
   - Sophisticated structure module with Invariant Point Attention
   - Operates on rigid body transformations (rotations and translations)
   - Uses quaternions and rotation matrices for stable coordinate updates
   - Maintains geometric invariance during structure prediction

## Key Integration Challenges

1. **Representation Adaptation**:
   - RhoFold+ expects specific tensor shapes and types
   - The RNA embedding model outputs need to be adapted for the IPA input format

2. **Rigid Frame Initialization**:
   - IPA requires proper initialization of rigid frames for each residue
   - Need a strategy to convert embeddings to initial rigid frames

3. **Tensor Dimension Management**:
   - Managing consistent tensor dimensions through the pipeline
   - Ensuring proper shape transformations at integration points

4. **Software Dependencies**:
   - RhoFold+ utilities for rigid transformations need to be imported correctly
   - Careful management of the utility functions for quaternions and rotations

## Detailed Integration Architecture

### 1. Module Structure
The integration will create a new `RhoFoldIPAModule` class that will:

1. Take the outputs from the RNA Feature Embedding model
2. Convert them to the format expected by RhoFold+'s IPA
3. Initialize rigid frames
4. Apply the IPA module
5. Convert the outputs back to the format expected by the RNA model

```
RNA Embedding → RhoFoldIPAModule → Predicted Coordinates
   (inputs)          (adapter)           (outputs)
```

### 2. Data Flow and Transformations

```
residue_repr [B, L, 128] ┐
                         ├→ RhoFoldIPAModule → pred_coords [B, L, 3]
pair_repr [B, L, L, 64]  ┘                   → pred_angles [B, L, 7, 2]
```

The full transformation pipeline is:

1. Initial feature embedding through the EmbeddingModule
2. Processing through transformer blocks
3. Input adaptation for IPA
4. Rigid frame initialization
5. IPA application with multiple iterations
6. Coordinate extraction and postprocessing

### 3. Key Components to Implement

#### 3.1. IPAAdapter
- Adapts the dimensions of `residue_repr` and `pair_repr` to IPA's expected formats
- Creates position-specific single and pair representations as expected by IPA

#### 3.2. RigidInitialization
- Initializes rigid frames for each residue position
- Converts embeddings to reasonable initial coordinates and orientations

#### 3.3. RhoFoldIPAModule
- Core module for the integration
- Encapsulates the IPA from RhoFold+
- Manages the iteration loops for structure refinement
- Provides the interface between the two systems

#### 3.4. AnglePredictor
- Predicts dihedral angles for RNA backbones
- Leverages the angle prediction components from RhoFold+

## Implementation Details

### 1. Required Imports

```python
# From RhoFold+
from rhofold.model.primitives import Linear, LayerNorm
from rhofold.utils.rigid_utils import Rigid
from rhofold.utils.tensor_utils import dict_multimap, permute_final_dims, flatten_final_dims
from rhofold.model.rhofold_components.structure_module import InvariantPointAttention, AngleResnet

# From RNA Feature Embedding Model
from models.ipa_module import IPAModule  # For compatibility/fallback
```

### 2. Rigid Frame Initialization

The key challenge is initializing plausible rigid frames for each RNA residue. Options include:

1. **Learning-based initialization**:
   - Use an MLP to predict initial frames from residue embeddings
   - Train this component specifically for RNA frames

2. **Reference-based initialization**:
   - Use idealized RNA geometry as starting frames
   - Apply small learned perturbations based on residue types

3. **Identity initialization**:
   - Start with identity frames and rely on the IPA iterations for refinement
   - Simpler but may require more iterations to converge

The recommended approach is a hybrid of options 1 and 2:
```python
def initialize_rigid_frames(residue_repr, mask=None):
    """Initialize rigid frames from residue representations."""
    batch_size, seq_len, dim = residue_repr.shape
    device = residue_repr.device
    
    # 1. Create initial coordinates from residue representations
    coord_initializer = nn.Sequential(
        nn.Linear(dim, dim//2),
        nn.ReLU(),
        nn.Linear(dim//2, 3)
    )
    initial_coords = coord_initializer(residue_repr)
    
    # 2. Create frames from coordinates (simplified approach)
    # In practice, would use RNA-specific priors here
    frames = Rigid.identity(
        (batch_size, seq_len),
        dtype=initial_coords.dtype,
        device=initial_coords.device,
        fmt="quat"
    )
    
    return frames, initial_coords
```

### 3. Dimension Adaptation

The input/output dimension adapter needs to be implemented:

```python
def adapt_dimensions(residue_repr, pair_repr):
    """Adapt dimensions from RNA embedding model to RhoFold IPA format."""
    batch_size, seq_len, residue_dim = residue_repr.shape
    
    # Project residue representations if needed
    if residue_dim != 384:  # RhoFold expects 384 dimensions
        residue_projection = nn.Linear(residue_dim, 384)
        residue_repr = residue_projection(residue_repr)
    
    # Process pair representations
    pair_dim = pair_repr.shape[-1]
    if pair_dim != 128:  # RhoFold expects 128 dimensions
        pair_projection = nn.Linear(pair_dim, 128)
        # Reshape for efficient projection
        pair_repr_flat = pair_repr.view(-1, pair_dim)
        pair_repr_flat = pair_projection(pair_repr_flat)
        pair_repr = pair_repr_flat.view(batch_size, seq_len, seq_len, 128)
    
    return residue_repr, pair_repr
```

### 4. Core RhoFoldIPAModule Implementation

Here's the skeleton of the core module:

```python
class RhoFoldIPAModule(nn.Module):
    """
    Integration module for RhoFold+'s IPA with RNA Feature Embedding Model.
    """

    def __init__(self, config):
        super().__init__()
        
        # Extract parameters
        self.residue_dim = config.get("residue_embed_dim", 128)
        self.pair_dim = config.get("pair_embed_dim", 64)
        self.num_ipa_blocks = config.get("num_ipa_blocks", 4)
        
        # Initialize dimension adaptation layers
        self._init_adaptation_layers()
        
        # Initialize IPA components from RhoFold+
        self._init_ipa_components()
        
        # Initialize angle prediction
        self._init_angle_prediction()
        
    def _init_adaptation_layers(self):
        # Initialize layers for adapting dimensions
        # (implementation details as above)
        
    def _init_ipa_components(self):
        # Initialize IPA components from RhoFold+
        self.ipa = InvariantPointAttention(
            c_s=384,  # RhoFold's single representation dim
            c_z=128,  # RhoFold's pair representation dim
            c_hidden=16,
            no_heads=4,
            no_qk_points=4,
            no_v_points=8,
        )
        
        # Initialize backbone update
        self.backbone_update = BackboneUpdate(c_s=384)
        
        # Initialize angle resnet
        self.angle_resnet = AngleResnet(
            c_in=384,
            c_hidden=128,
            no_blocks=2,
            no_angles=7,  # RNA typically has 7 backbone torsion angles
            epsilon=1e-8,
        )
        
    def _init_angle_prediction(self):
        # Initialize angle prediction components
        # (implementation details)
        
    def forward(self, residue_repr, pair_repr, mask=None):
        """Forward pass through the RhoFold IPA module."""
        batch_size, seq_len, _ = residue_repr.shape
        
        # 1. Adapt dimensions
        adapted_residue, adapted_pair = self.adapt_dimensions(residue_repr, pair_repr)
        
        # 2. Initialize rigid frames
        rigid_frames, initial_coords = self.initialize_rigid_frames(adapted_residue, mask)
        
        # 3. Apply IPA iterations
        s_initial = adapted_residue  # Store initial representation
        s = adapted_residue
        
        outputs = []
        for i in range(self.num_ipa_blocks):
            # Update single representation with IPA
            s = s + self.ipa(s, adapted_pair, rigid_frames, mask)
            
            # Update rigid frames with backbone update
            rigid_frames = rigid_frames.compose_q_update_vec(self.backbone_update(s))
            
            # Predict angles
            unnormalized_angles, angles = self.angle_resnet(s, s_initial)
            
            # Store outputs for this iteration
            outputs.append({
                "frames": rigid_frames.to_tensor_7(),
                "unnormalized_angles": unnormalized_angles,
                "angles": angles,
                "single": s
            })
            
            # Stop gradient flow in rotations for stability
            if i != self.num_ipa_blocks - 1:
                rigid_frames = rigid_frames.stop_rot_gradient()
        
        # 4. Stack outputs and convert to final coordinates
        stacked_outputs = dict_multimap(torch.stack, outputs)
        
        # 5. Extract final coordinates from angles and frames
        final_coords = self.converter.build_cords(
            adapted_residue,  # Sequence info embedded in residue repr
            stacked_outputs['frames'][-1], 
            stacked_outputs['angles'][-1]
        )
        
        # 6. Return coordinates and other outputs as needed
        return {
            "pred_coords": final_coords,
            "pred_angles": stacked_outputs['angles'][-1],
            "confidence": self.predict_confidence(s)
        }
```

### 5. Error Handling and Fallbacks

Robust error handling is essential for integration stability:

```python
def forward(self, residue_repr, pair_repr, mask=None):
    """Forward pass with error handling and fallbacks."""
    try:
        # Primary implementation (as above)
        return self._forward_impl(residue_repr, pair_repr, mask)
    except Exception as e:
        print(f"RhoFold IPA error: {e}. Using fallback method.")
        
        # Fallback to original simple IPA
        fallback_ipa = IPAModule({
            "residue_embed_dim": self.residue_dim,
            "pair_embed_dim": self.pair_dim
        })
        
        coords = fallback_ipa(residue_repr, pair_repr, mask)
        return {
            "pred_coords": coords,
            "pred_angles": self._generate_fallback_angles(residue_repr),  # Simple angle prediction
            "confidence": torch.ones_like(coords[..., 0]) * 0.5  # Medium confidence
        }
```

## Performance Considerations

1. **Memory Efficiency**:
   - IPA operations can be memory-intensive
   - For long RNA sequences, consider chunking strategies
   - Implement gradient checkpointing for training

2. **Computation Optimization**:
   - Pre-compute invariant features where possible
   - Use fused operations for attention calculations
   - Consider lower precision for some operations

3. **Training Approach**:
   - Start with frozen RNA embedding components
   - Gradually unfreeze layers as integration stabilizes
   - Use a curriculum learning approach for increasing RNA sequence lengths

## Testing and Validation

1. **Unit Tests**:
   - Test rigid frame initialization
   - Test dimension adaptation
   - Test coordinate generation

2. **Integration Tests**:
   - Test end-to-end RNA structure prediction
   - Compare against original RhoFold+ results
   - Validate on standard RNA benchmark datasets

3. **Metrics**:
   - RMSD (Root Mean Square Deviation)
   - TM-score for global structure quality
   - RNA-specific metrics like base pair accuracy

## References

1. RhoFold+: https://github.com/ml4bio/RhoFold
2. InvariantPointAttention implementation in RhoFold+: `structure_module.py`
3. Rigid transformation utilities: `rigid_utils.py`
