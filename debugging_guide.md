# Debugging and Implementation Examples Guide

This guide provides recommended examples, debugging strategies, and workflow suggestions for implementing the integration of RhoFold+'s Invariant Point Attention (IPA) module with the RNA Feature Embedding Model.

## Recommended Examples

### 1. Simple Test Cases

#### 1.1 Single Short RNA Sequence
Start with a very simple example to validate basic functionality:

```python
def test_simple_sequence():
    # Create a small test example
    batch = {
        "sequence_int": torch.tensor([[0, 1, 2, 3, 0, 1]], dtype=torch.long),  # Simple ACGUA sequence
        "mask": torch.tensor([[1, 1, 1, 1, 1, 1]], dtype=torch.bool),
        "pairing_probs": torch.zeros((1, 6, 6)),
        "coupling_matrix": torch.zeros((1, 6, 6)),
        "positional_entropy": torch.zeros((1, 6)),
        "accessibility": torch.ones((1, 6)),
        "dihedral_features": torch.zeros((1, 6, 4)),
    }
    
    # Create embedding module
    config = {"residue_embed_dim": 128, "pair_embed_dim": 64}
    embedding_module = EmbeddingModule(config)
    
    # Get embeddings
    residue_repr, pair_repr, mask = embedding_module(batch)
    
    # Use these with the IPA module
    ipa_module = RhoFoldIPAModule(config)
    outputs = ipa_module(residue_repr, pair_repr, mask)
    
    # Validate outputs
    assert outputs["pred_coords"].shape == (1, 6, 3)
    assert outputs["pred_angles"].shape == (1, 6, 7, 2)  # 7 RNA angles, each as sin/cos pair
    
    # Plot structure for visual inspection
    plot_rna_structure(outputs["pred_coords"][0])
```

#### 1.2 Hairpin Structure Example
Create a test case with a known RNA hairpin:

```python
def test_hairpin_structure():
    # Create a hairpin test example with known pairing
    # GGGAAACCC - classic hairpin with GC pairs
    batch = {
        "sequence_int": torch.tensor([[1, 1, 1, 0, 0, 0, 2, 2, 2]], dtype=torch.long),
        "mask": torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1]], dtype=torch.bool),
    }
    
    # Set pairing probabilities for the hairpin
    pairing_probs = torch.zeros((1, 9, 9))
    pairing_probs[0, 0, 8] = pairing_probs[0, 8, 0] = 0.9  # G-C pair
    pairing_probs[0, 1, 7] = pairing_probs[0, 7, 1] = 0.9  # G-C pair
    pairing_probs[0, 2, 6] = pairing_probs[0, 6, 2] = 0.9  # G-C pair
    batch["pairing_probs"] = pairing_probs
    
    # Other features
    batch["coupling_matrix"] = torch.zeros((1, 9, 9))
    batch["positional_entropy"] = torch.zeros((1, 9))
    batch["accessibility"] = torch.ones((1, 9))
    batch["dihedral_features"] = torch.zeros((1, 9, 4))
    
    # Run through model and validate
    # ... (rest of test as above)
    
    # Additionally, verify hairpin shape - should have paired residues close to each other
    coords = outputs["pred_coords"][0]
    
    # Calculate distances between paired nucleotides
    dist_0_8 = torch.norm(coords[0] - coords[8])
    dist_1_7 = torch.norm(coords[1] - coords[7])
    dist_2_6 = torch.norm(coords[2] - coords[6])
    
    # Paired nucleotides should be closer than unpaired ones
    assert dist_0_8 < 15.0, f"Distance between first pair too large: {dist_0_8}"
    assert dist_1_7 < 15.0, f"Distance between second pair too large: {dist_1_7}"
    assert dist_2_6 < 15.0, f"Distance between third pair too large: {dist_2_6}"
```

### 2. Complex Examples

#### 2.1 Real RNA Structures
Use examples from databases like PDB or RNA3DHub:

```python
def test_real_structure(pdb_id="1EHZ"):
    """Test using real RNA structure from PDB."""
    # Load reference structure
    reference_coords = load_pdb_coordinates(pdb_id)
    sequence = get_pdb_sequence(pdb_id)
    
    # Prepare inputs
    batch = prepare_batch_from_sequence(sequence)
    
    # Run prediction
    residue_repr, pair_repr, mask = embedding_module(batch)
    outputs = ipa_module(residue_repr, pair_repr, mask)
    
    # Compare with reference using RMSD
    rmsd = calculate_rmsd(reference_coords, outputs["pred_coords"][0])
    print(f"RMSD to reference: {rmsd:.2f} Å")
    
    # Visualize both structures
    visualize_comparison(reference_coords, outputs["pred_coords"][0])
```

#### 2.2 Stress Testing with Long Sequences
Test performance and memory usage with long RNAs:

```python
def stress_test_long_sequence():
    """Test how the model handles long sequences."""
    # Create a long sequence (e.g., 500 nucleotides)
    seq_len = 500
    sequence = torch.randint(0, 4, (1, seq_len), dtype=torch.long)
    mask = torch.ones((1, seq_len), dtype=torch.bool)
    
    # Create other features
    batch = create_dummy_features(sequence, mask)
    
    # Track memory usage
    initial_memory = torch.cuda.memory_allocated()
    
    # Process through modules with memory tracking
    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()
    
    residue_repr, pair_repr, mask = embedding_module(batch)
    embedding_memory = torch.cuda.max_memory_allocated() - initial_memory
    embedding_time = time.time() - start_time
    
    # Reset for IPA module
    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()
    
    outputs = ipa_module(residue_repr, pair_repr, mask)
    
    ipa_memory = torch.cuda.max_memory_allocated() - initial_memory
    ipa_time = time.time() - start_time
    
    print(f"Sequence length: {seq_len}")
    print(f"Embedding module: {embedding_memory/1e6:.1f} MB, {embedding_time:.2f} seconds")
    print(f"IPA module: {ipa_memory/1e6:.1f} MB, {ipa_time:.2f} seconds")
```

### 3. Integration Examples

#### 3.1 Step-by-Step Integration Example
Show the incremental integration process with TorchScript inspection:

```python
def integration_example():
    """Demonstrate the integration process step by step."""
    # 1. Create inputs
    batch = create_test_batch()
    
    # 2. Get embeddings
    residue_repr, pair_repr, mask = embedding_module(batch)
    print(f"Residue repr: {residue_repr.shape}, Pair repr: {pair_repr.shape}")
    
    # 3. Adapt dimensions
    adapted_residue, adapted_pair = adapt_dimensions(residue_repr, pair_repr)
    print(f"Adapted residue: {adapted_residue.shape}, Adapted pair: {adapted_pair.shape}")
    
    # 4. Initialize rigid frames
    frames, initial_coords = initialize_rigid_frames(adapted_residue, mask)
    print(f"Frames shape: {frames.get_rots().get_rot_mats().shape}, Initial coords: {initial_coords.shape}")
    
    # 5. Apply single IPA iteration (simplified)
    s = adapted_residue
    s_update = ipa(s, adapted_pair, frames, mask)
    print(f"IPA update shape: {s_update.shape}")
    
    # 6. Update frames
    frames_update = backbone_update(s)
    frames = frames.compose_q_update_vec(frames_update)
    print(f"Updated frames quaternions shape: {frames.get_quats().shape}")
    
    # 7. Predict angles
    unnormalized_angles, angles = angle_resnet(s, adapted_residue)
    print(f"Angles shape: {angles.shape}")
    
    # 8. Build coordinates
    coords = converter.build_cords(sequence, frames.to_tensor_7(), angles)
    print(f"Final coordinates shape: {coords.shape}")
```

## Debugging Strategies

### 1. Tensor Shape and Value Inspection

#### 1.1 Shape Verification Tool
Create a tool to verify tensor shapes throughout the pipeline:

```python
def debug_shape(name, tensor):
    """Print tensor shape and basic statistics."""
    if isinstance(tensor, torch.Tensor):
        shape_str = str(tensor.shape)
        if tensor.numel() > 0:
            # Basic statistics
            stats = {
                "min": tensor.min().item(),
                "max": tensor.max().item(),
                "mean": tensor.mean().item(),
                "std": tensor.std().item(),
                "nan": torch.isnan(tensor).any().item(),
                "inf": torch.isinf(tensor).any().item(),
            }
            print(f"{name}: {shape_str} (min={stats['min']:.4f}, max={stats['max']:.4f}, "
                  f"mean={stats['mean']:.4f}, std={stats['std']:.4f}, "
                  f"has_nan={stats['nan']}, has_inf={stats['inf']})")
        else:
            print(f"{name}: {shape_str} (empty tensor)")
    elif isinstance(tensor, Rigid):
        # For Rigid objects
        rot_mats = tensor.get_rots().get_rot_mats()
        trans = tensor.get_trans()
        print(f"{name} (Rigid): rot_mats={rot_mats.shape}, trans={trans.shape}")
    else:
        print(f"{name}: {type(tensor)} (not a tensor)")
```

Use this utility throughout the implementation to track tensor shapes:

```python
class DebugRhoFoldIPAModule(RhoFoldIPAModule):
    """Debug version with shape tracking."""
    
    def forward(self, residue_repr, pair_repr, mask=None):
        debug_shape("input_residue_repr", residue_repr)
        debug_shape("input_pair_repr", pair_repr)
        debug_shape("mask", mask)
        
        # 1. Adapt dimensions
        adapted_residue, adapted_pair = self.adapt_dimensions(residue_repr, pair_repr)
        debug_shape("adapted_residue", adapted_residue)
        debug_shape("adapted_pair", adapted_pair)
        
        # Continue with additional debugging...
        # ...
```

#### 1.2 Visualizing Attention Maps
Visualize the attention patterns in the IPA:

```python
def visualize_attention(attn_weights, sequence):
    """Visualize attention weights for IPA."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Average attention over heads
    avg_attn = attn_weights.mean(dim=1)  # Shape: [batch, seq_len, seq_len]
    
    # Plot heatmap for the first batch item
    plt.figure(figsize=(10, 8))
    sns.heatmap(avg_attn[0].cpu().detach().numpy(), 
                cmap="viridis", 
                vmin=0, 
                vmax=avg_attn.max().item())
    
    # Add sequence labels if not too long
    if len(sequence) <= 50:
        plt.xticks(np.arange(len(sequence)) + 0.5, sequence)
        plt.yticks(np.arange(len(sequence)) + 0.5, sequence)
    
    plt.title("IPA Attention Weights")
    plt.tight_layout()
    plt.savefig("ipa_attention.png")
    plt.close()
```

#### 1.3 Tracking Numerical Stability
Monitor numerical stability during coordinate updates:

```python
def check_numerical_stability(name, tensor):
    """Check tensor for numerical instability."""
    if not isinstance(tensor, torch.Tensor):
        return
        
    has_nan = torch.isnan(tensor).any().item()
    has_inf = torch.isinf(tensor).any().item()
    
    if has_nan or has_inf:
        print(f"WARNING: {name} has {'NaN' if has_nan else ''} {'Inf' if has_inf else ''}")
        
        # Print more detailed information for debugging
        if has_nan:
            nan_indices = torch.nonzero(torch.isnan(tensor))
            print(f"NaN at indices: {nan_indices[:10]}")  # Show first 10
            
        if has_inf:
            inf_indices = torch.nonzero(torch.isinf(tensor))
            print(f"Inf at indices: {inf_indices[:10]}")  # Show first 10
            
        # Get context - values around the problematic areas
        if tensor.dim() >= 2:
            for idx in (nan_indices if has_nan else inf_indices)[:5]:
                idx_tuple = tuple(idx.tolist())
                try:
                    slice_indices = [slice(max(0, i-1), i+2) for i in idx_tuple]
                    context = tensor[slice_indices]
                    print(f"Context around {idx_tuple}: {context}")
                except Exception as e:
                    print(f"Could not get context: {e}")
```

### 2. Structure Validation Tools

#### 2.1 Basic Structure Validation
Create a tool to validate physical plausibility of predicted structures:

```python
def validate_rna_structure(coords, sequence):
    """Basic validation of RNA structure physical plausibility."""
    # 1. Check bond lengths
    bond_distances = torch.norm(coords[1:] - coords[:-1], dim=-1)
    avg_bond = bond_distances.mean().item()
    min_bond = bond_distances.min().item()
    max_bond = bond_distances.max().item()
    
    print(f"Bond distances - avg: {avg_bond:.2f}, min: {min_bond:.2f}, max: {max_bond:.2f}")
    if min_bond < 2.0 or max_bond > 6.0:
        print("WARNING: Bond distances outside expected range (2-6 Å)")
    
    # 2. Check for collisions (atoms too close)
    pairwise_dist = torch.cdist(coords, coords)
    
    # Exclude adjacent residues from collision check
    mask = torch.ones_like(pairwise_dist, dtype=torch.bool)
    for i in range(len(coords)):
        for j in range(max(0, i-1), min(len(coords), i+2)):
            mask[i, j] = False
    
    # Check min distance between non-adjacent residues
    masked_dist = pairwise_dist.masked_select(mask)
    min_non_adjacent = masked_dist.min().item()
    
    print(f"Minimum non-adjacent distance: {min_non_adjacent:.2f}")
    if min_non_adjacent < 3.0:
        print("WARNING: Potential steric clashes (non-adjacent residues < 3 Å apart)")
    
    # 3. Check structure compactness
    radius_of_gyration = torch.norm(coords - coords.mean(dim=0, keepdim=True), dim=1).mean().item()
    print(f"Radius of gyration: {radius_of_gyration:.2f}")
    
    # Rough estimate based on sequence length
    expected_rog = 3.0 * len(sequence)**0.33  # Simplified scaling law
    
    if radius_of_gyration > expected_rog * 1.5:
        print(f"WARNING: Structure may be too extended (RoG: {radius_of_gyration:.2f}, expected: ~{expected_rog:.2f})")
    elif radius_of_gyration < expected_rog * 0.5:
        print(f"WARNING: Structure may be too compact (RoG: {radius_of_gyration:.2f}, expected: ~{expected_rog:.2f})")
```

#### 2.2 Visualizing Intermediate Structures
Visualize the structures at each IPA iteration:

```python
def visualize_structure_evolution(all_coords, output_prefix="structure_iter"):
    """Visualize how the structure evolves through IPA iterations."""
    import py3Dmol
    
    for i, coords in enumerate(all_coords):
        # Convert to Angstroms and numpy
        coords_np = coords.detach().cpu().numpy() * 10.0  # nm to Angstrom
        
        # Generate PDB-like string
        pdb_str = "MODEL     1\n"
        for j, pos in enumerate(coords_np):
            atom = f"ATOM  {j+1:5d}  C   C A{j+1:4d}    {pos[0]:8.3f}{pos[1]:8.3f}{pos[2]:8.3f}  1.00  0.00           C  \n"
            pdb_str += atom
        pdb_str += "ENDMDL\n"
        
        # Visualize
        view = py3Dmol.view(width=400, height=400)
        view.addModel(pdb_str, "pdb")
        view.setStyle({"model": -1}, {"sphere": {"colorscheme": "Jmol"}})
        view.zoomTo()
        view.render()
        
        # Save to image
        view.png(f"{output_prefix}_{i}.png")
        
        # Also save coordinates to file
        with open(f"{output_prefix}_{i}.pdb", "w") as f:
            f.write(pdb_str)
```

### 3. Debugging Common Issues

#### 3.1 Gradient Instability Issues
Monitor and fix gradient issues:

```python
def debug_gradients(model):
    """Add hooks to track gradient statistics."""
    gradient_stats = {}
    
    def hook_fn(name):
        def fn(grad):
            if grad is not None:
                if torch.isnan(grad).any():
                    print(f"WARNING: NaN gradient in {name}")
                if torch.isinf(grad).any():
                    print(f"WARNING: Inf gradient in {name}")
                
                gradient_stats[name] = {
                    "min": grad.min().item(),
                    "max": grad.max().item(),
                    "mean": grad.mean().item(),
                    "norm": grad.norm().item()
                }
        return fn
    
    # Register hooks for all parameters
    for name, param in model.named_parameters():
        if param.requires_grad:
            param.register_hook(hook_fn(name))
    
    return gradient_stats

# Use during training
gradient_stats = debug_gradients(model)
# After backward pass
print("Gradient statistics:")
for name, stats in gradient_stats.items():
    print(f"{name}: min={stats['min']:.4e}, max={stats['max']:.4e}, "
          f"mean={stats['mean']:.4e}, norm={stats['norm']:.4e}")
```

#### 3.2 Memory Profiling
Profile memory usage:

```python
def memory_profile(func, *args, **kwargs):
    """Profile memory usage of a function."""
    import gc
    
    # Force garbage collection
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # Record initial memory
    mem_init = torch.cuda.memory_allocated()
    
    # Run function
    result = func(*args, **kwargs)
    
    # Get memory stats
    mem_peak = torch.cuda.max_memory_allocated()
    mem_current = torch.cuda.memory_allocated()
    
    print(f"Memory profiling for {func.__name__}:")
    print(f"  Peak memory: {(mem_peak-mem_init)/1e6:.2f} MB")
    print(f"  Current memory: {(mem_current-mem_init)/1e6:.2f} MB")
    print(f"  Memory delta: {(mem_current-mem_init)/1e6:.2f} MB")
    
    return result

# Example usage:
outputs = memory_profile(ipa_module, residue_repr, pair_repr, mask)
```

#### 3.3 Debugging Rigid Transformations
Validate rigid transformations:

```python
def validate_rigid_frames(frames):
    """Validate rigid frames for orthogonality and determinant."""
    rot_mats = frames.get_rots().get_rot_mats()
    batch_size, seq_len = rot_mats.shape[0], rot_mats.shape[1]
    
    # Check orthogonality: R^T R should be identity
    identities = torch.bmm(
        rot_mats.reshape(-1, 3, 3).transpose(-1, -2),
        rot_mats.reshape(-1, 3, 3)
    )
    
    # Calculate deviation from identity
    eye = torch.eye(3, device=rot_mats.device).unsqueeze(0).expand(batch_size * seq_len, -1, -1)
    deviation = torch.norm(identities - eye, dim=(-2, -1))
    
    print(f"Orthogonality deviation - mean: {deviation.mean().item()}, max: {deviation.max().item()}")
    
    # Check determinants (should be 1 for proper rotations)
    dets = torch.linalg.det(rot_mats.reshape(-1, 3, 3))
    print(f"Determinants - mean: {dets.mean().item()}, min: {dets.min().item()}, max: {dets.max().item()}")
    
    # Alert if severe issues
    if deviation.max().item() > 1e-5:
        print("WARNING: Rotation matrices not orthogonal")
    
    if abs(dets.mean().item() - 1.0) > 1e-5 or dets.min().item() < 0:
        print("WARNING: Rotation matrices have incorrect determinants")
```

## Workflow Recommendations

### 1. Development Workflow

#### 1.1 Incremental Development Strategy

Implement the integration in smaller, testable steps:

1. **Step 1: Framework**
   - Create the skeleton RhoFoldIPAModule with interfaces but simple implementation
   - Test input/output shapes and basic data flow

2. **Step 2: Dimension Adaptation**
   - Implement dimension adaptation layers
   - Test with dummy rigid frames and IPA
   - Verify shape transformations

3. **Step 3: Rigid Frame Initialization**
   - Implement proper frame initialization
   - Validate frame properties (orthogonality, etc.)
   - Test coordinate generation from frames

4. **Step 4: Basic IPA Integration**
   - Integrate InvariantPointAttention class
   - Implement single-iteration pipeline
   - Test attention calculation and updates

5. **Step 5: Multi-Iteration Refinement**
   - Add iterative refinement loop
   - Monitor convergence behavior
   - Implement gradient stopping for stability

6. **Step 6: Angle Prediction**
   - Integrate angle prediction components
   - Add RNA-specific constraints
   - Test coordinate generation from angles

7. **Step 7: Full Integration**
   - Connect with the RNA folding model
   - Implement end-to-end pipeline
   - Test with real RNA examples

#### 1.2 Test-Driven Development Approach

For each component, follow this pattern:

1. **Write test first**: Define what success looks like
2. **Implement minimally**: Make the test pass with the simplest code
3. **Refactor**: Clean up the implementation
4. **Add edge cases**: Test boundary conditions and failure modes
5. **Document**: Document the component interface and behavior

Example test framework:

```python
def test_rigid_frame_initialization():
    """Test rigid frame initialization from residue representation."""
    # Setup
    batch_size, seq_len, dim = 2, 10, 128
    residue_repr = torch.randn(batch_size, seq_len, dim)
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    mask[:, -2:] = False  # Mask out last two positions
    
    # Call function
    frames, coords = initialize_rigid_frames(residue_repr, mask)
    
    # Assertions
    assert isinstance(frames, Rigid), "Should return a Rigid object"
    assert frames.shape == (batch_size, seq_len), f"Expected shape {(batch_size, seq_len)}, got {frames.shape}"
    assert coords.shape == (batch_size, seq_len, 3), f"Expected shape {(batch_size, seq_len, 3)}, got {coords.shape}"
    
    # Check masking
    assert torch.all(coords[:, -2:] == 0), "Masked positions should have zero coordinates"
    
    # Validate frame properties
    rot_mats = frames.get_rots().get_rot_mats()
    
    # Check orthogonality
    products = torch.matmul(
        rot_mats.view(-1, 3, 3), 
        rot_mats.view(-1, 3, 3).transpose(-1, -2)
    )
    identity = torch.eye(3, device=rot_mats.device).unsqueeze(0).repeat(batch_size * seq_len, 1, 1)
    assert torch.allclose(products, identity, atol=1e-6), "Rotation matrices should be orthogonal"
    
    # Check determinant (should be +1 for proper rotation)
    dets = torch.linalg.det(rot_mats.view(-1, 3, 3))
    assert torch.allclose(dets, torch.ones_like(dets), atol=1e-6), "Determinants should be 1.0"
```

### 2. Integration Workflow

#### 2.1 Staged Integration Approach

Integrate the components in stages:

1. **Stage 1: Side-by-Side Testing**
   - Keep both the original IPAModule and new RhoFoldIPAModule
   - Compare outputs and performance
   - Identify discrepancies and resolve

2. **Stage 2: Optional Switching**
   - Make the choice of IPA module configurable
   - Enable easy switching for testing
   - Validate on diverse datasets

3. **Stage 3: Complete Integration**
   - Switch to RhoFoldIPAModule as the default
   - Keep fallback mechanisms for robustness
   - Finalize end-to-end tests

Example configuration system:

```python
class ConfigurableRNAFoldingModel(nn.Module):
    """RNA folding model with configurable IPA module."""
    
    def __init__(self, config):
        super().__init__()
        
        # Extract parameters
        self.ipa_type = config.get("ipa_type", "enhanced")  # "basic", "rhofold", or "enhanced"
        
        # Initialize embedding and transformer components
        self.embedding_module = EmbeddingModule(config)
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.get("num_blocks", 4))]
        )
        
        # Initialize appropriate IPA module
        if self.ipa_type == "basic":
            self.ipa_module = IPAModule(config)
        elif self.ipa_type == "rhofold":
            self.ipa_module = RhoFoldIPAModule(config)
        elif self.ipa_type == "enhanced":
            self.ipa_module = EnhancedIPAModule(config)
        else:
            raise ValueError(f"Unknown IPA type: {self.ipa_type}")
```

#### 2.2 Logging and Monitoring

Implement comprehensive logging for the integration process:

```python
import logging
import time

def setup_module_logging(name, level=logging.INFO):
    """Setup module-specific logger."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Create handler and formatter
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    return logger

# Create logger for the module
logger = setup_module_logging("rhofold_ipa")

class LoggingRhoFoldIPAModule(RhoFoldIPAModule):
    """RhoFoldIPAModule with detailed logging."""
    
    def forward(self, residue_repr, pair_repr, mask=None):
        """Forward pass with logging."""
        start_time = time.time()
        logger.info(f"Starting RhoFoldIPA forward pass: "
                   f"residue_repr={residue_repr.shape}, "
                   f"pair_repr={pair_repr.shape}")
        
        try:
            # Track timing for each step
            step_times = {}
            
            # 1. Adapt dimensions
            step_start = time.time()
            adapted_residue, adapted_pair = self.adapt_dimensions(residue_repr, pair_repr)
            step_times["adapt_dimensions"] = time.time() - step_start
            
            # 2. Initialize rigid frames
            step_start = time.time()
            frames, initial_coords = self.initialize_rigid_frames(adapted_residue, mask)
            step_times["initialize_frames"] = time.time() - step_start
            
            # Additional timing for other steps...
            # ...
            
            # Log timing information
            total_time = time.time() - start_time
            logger.info(f"RhoFoldIPA forward pass completed in {total_time:.2f}s")
            for step, step_time in step_times.items():
                logger.info(f"  - {step}: {step_time:.2f}s ({step_time/total_time*100:.1f}%)")
            
            return outputs
            
        except Exception as e:
            logger.error(f"Error in RhoFoldIPA forward pass: {e}", exc_info=True)
            # Use fallback mechanism
            return self._fallback_forward(residue_repr, pair_repr, mask)
```

### 3. Optimization Workflow

#### 3.1 Performance Profiling

Profile and optimize the implementation:

```python
def profile_ipa_module(module, inputs, iterations=10):
    """Profile the IPA module performance."""
    import time
    
    # Extract inputs
    residue_repr, pair_repr, mask = inputs
    
    # Warm-up
    for _ in range(3):
        with torch.no_grad():
            _ = module(residue_repr, pair_repr, mask)
    
    # Synchronize before timing
    torch.cuda.synchronize()
    
    # Profile forward pass
    forward_times = []
    for _ in range(iterations):
        start = time.time()
        with torch.no_grad():
            outputs = module(residue_repr, pair_repr, mask)
        torch.cuda.synchronize()
        forward_times.append(time.time() - start)
    
    # Profile backward pass if in training mode
    if module.training:
        backward_times = []
        for _ in range(iterations):
            with torch.enable_grad():
                outputs = module(residue_repr, pair_repr, mask)
                # Create dummy loss
                loss = outputs["pred_coords"].sum()
                
                start = time.time()
                loss.backward()
                torch.cuda.synchronize()
                backward_times.append(time.time() - start)
    else:
        backward_times = [0] * iterations
    
    # Report statistics
    print(f"Performance profiling ({iterations} iterations):")
    print(f"  Forward:  {sum(forward_times)/iterations*1000:.2f} ms (±{torch.tensor(forward_times).std()*1000:.2f} ms)")
    if module.training:
        print(f"  Backward: {sum(backward_times)/iterations*1000:.2f} ms (±{torch.tensor(backward_times).std()*1000:.2f} ms)")
    print(f"  Total:    {sum(forward_times+backward_times)/iterations*1000:.2f} ms")
    
    return forward_times, backward_times
```

#### 3.2 Memory Optimization

Implement memory efficiency techniques:

```python
class MemoryEfficientRhoFoldIPAModule(RhoFoldIPAModule):
    """Memory-optimized version of RhoFoldIPAModule."""
    
    def forward(self, residue_repr, pair_repr, mask=None):
        """Forward pass with memory optimizations."""
        # Use checkpointing for IPA blocks
        from torch.utils.checkpoint import checkpoint
        
        # 1. Adapt dimensions
        adapted_residue, adapted_pair = self.adapt_dimensions(residue_repr, pair_repr)
        
        # 2. Initialize rigid frames
        frames, initial_coords = self.initialize_rigid_frames(adapted_residue, mask)
        
        # 3. Apply IPA with checkpointing
        s = adapted_residue
        for i in range(self.num_ipa_blocks):
            def ipa_block(s, pair, frames, mask):
                # IPA update
                s_update = self.ipa(s, pair, frames, mask)
                s = s + s_update
                
                # Backbone update
                frames_update = self.backbone_update(s)
                frames = frames.compose_q_update_vec(frames_update)
                
                return s, frames
            
            # Use checkpointing to save memory
            if self.training:
                s, frames = checkpoint(
                    ipa_block, s, adapted_pair, frames, mask,
                    preserve_rng_state=False
                )
            else:
                s, frames = ipa_block(s, adapted_pair, frames, mask)
        
        # Rest of implementation...
```

#### 3.3 Hyperparameter Tuning

Systematically optimize hyperparameters:

```python
def hyperparameter_search(train_dataset, val_dataset):
    """Simple grid search for hyperparameters."""
    import itertools
    
    # Define hyperparameter grid
    param_grid = {
        "num_ipa_blocks": [2, 4, 6],
        "learning_rate": [1e-4, 5e-4, 1e-3],
        "no_qk_points": [4, 8],
        "no_v_points": [8, 16],
    }
    
    # Generate all combinations
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = list(itertools.product(*values))
    
    best_val_loss = float('inf')
    best_params = None
    
    for comb in combinations:
        # Create config from combination
        config = {keys[i]: comb[i] for i in range(len(keys))}
        print(f"Testing configuration: {config}")
        
        # Create model with this config
        model = RNAFoldingModel(**config)
        
        # Train and validate
        train_loss = train_model(model, train_dataset, epochs=5)
        val_loss = validate_model(model, val_dataset)
        
        print(f"Train loss: {train_loss}, Val loss: {val_loss}")
        
        # Update best params
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_params = config
    
    print(f"Best hyperparameters: {best_params} (val_loss: {best_val_loss})")
    return best_params
```

## Summary of Key Recommendations

1. **Develop Incrementally**: Start with simple test cases and add complexity gradually
2. **Monitor Carefully**: Add extensive logging and validation throughout the process
3. **Test Thoroughly**: Create comprehensive tests for each component and integration point
4. **Optimize Systematically**: Profile performance and optimize critical paths
5. **Handle Failures Gracefully**: Implement robust fallback mechanisms
6. **Visualize Results**: Create visualizations to validate structure predictions
7. **Document Extensively**: Document the integration process and the final solution

By following these recommendations, the integration process will be more manageable, testable, and ultimately more successful.
