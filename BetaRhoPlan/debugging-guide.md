# Likely Issues and Debugging Guide: RhoFold+ IPA Integration

This guide outlines common issues you're likely to encounter when integrating the RhoFold+ IPA module with your RNA feature embedding model, along with detailed debugging strategies and solutions.

## 1. Tensor Shape and Dimension Issues

### Common Manifestations
- `RuntimeError: The size of tensor a (128) must match the size of tensor b (384) at non-singleton dimension 2`
- `RuntimeError: mat1 and mat2 shapes cannot be multiplied (128x64 and 384x128)`
- `IndexError: Dimension out of range (expected to be in range of [-4, 3], but got 4)`

### Root Causes
1. **Dimension Adaptation Mismatch**: Incorrect input/output dimensions in adapter layers
2. **Missing Dimension Expansion**: Forgotten to add a dimension with `unsqueeze()`
3. **Batch Dimension Handling**: Inconsistent handling of batch dimensions
4. **Rigid Transformation Shape**: Incorrect shape in rigid transformation initialization

### Debugging Strategies
1. **Add Dimension Logging**:
   ```python
   def debug_shapes(module_name, **tensors):
       print(f"\n--- {module_name} Shapes ---")
       for name, tensor in tensors.items():
           if isinstance(tensor, torch.Tensor):
               print(f"{name}: {tensor.shape}")
           elif hasattr(tensor, 'shape'):
               print(f"{name}: {tensor.shape} (type: {type(tensor)})")
           else:
               print(f"{name}: (type: {type(tensor)})")
   
   # Use in forward method
   debug_shapes("IPA Input", 
               residue_repr=residue_repr, 
               pair_repr=pair_repr,
               adapted_s=s,
               adapted_z=z)
   ```

2. **Shape Assertion Checks**:
   ```python
   # Add to critical points in your code
   assert s.shape[-1] == self.c_s, f"Expected s dimension {self.c_s}, got {s.shape[-1]}"
   assert z.shape[-1] == self.c_z, f"Expected z dimension {self.c_z}, got {z.shape[-1]}"
   assert s.shape[:-1] == residue_repr.shape[:-1], f"Shape mismatch: {s.shape} vs {residue_repr.shape}"
   ```

3. **Visual Dimension Debugging**:
   Create a flowchart of tensor shapes throughout the pipeline using debug logs.

### Solutions
1. **Fix Adapter Dimensions**:
   ```python
   # Ensure adapter dimensions match exactly
   self.single_adapter = nn.Linear(self.residue_dim, self.c_s)
   self.pair_adapter = nn.Linear(self.pair_dim, self.c_z)
   ```

2. **Handle Dynamic Dimensions**:
   ```python
   # Dynamically set dimensions in __init__
   def __init__(self, config):
       self.residue_dim = config.get("residue_embed_dim")
       self.c_s = config.get("ipa_single_dim", 384)
       # Dynamically create adapter based on actual dimensions
       self.single_adapter = nn.Linear(self.residue_dim, self.c_s)
   ```

3. **Fix Batch Handling**:
   ```python
   # Initialize rigid with correct batch dimensions
   batch_size, seq_len = residue_repr.shape[:2]
   rigid = Rigid.identity(
       shape=(batch_size, seq_len),  # Ensure correct shape here
       dtype=torch.float32,
       device=device,
       fmt="quat"
   )
   ```

## 2. Runtime and Memory Efficiency Issues

### Common Manifestations
- `CUDA out of memory` errors 
- Extremely slow training or inference
- Memory usage grows with sequence length

### Root Causes
1. **Large Tensor Allocations**: IPA creates large intermediate tensors
2. **Redundant Computations**: Unnecessary recomputation of values
3. **Precision Issues**: Using full precision unnecessarily
4. **Inefficient Attention Implementation**: Default attention mechanism is memory-intensive

### Debugging Strategies
1. **Memory Profiling**:
   ```python
   def profile_memory(name):
       if torch.cuda.is_available():
           torch.cuda.synchronize()
           memory_allocated = torch.cuda.memory_allocated() / (1024 ** 2)
           memory_reserved = torch.cuda.memory_reserved() / (1024 ** 2)
           print(f"{name} - Allocated: {memory_allocated:.2f}MB, Reserved: {memory_reserved:.2f}MB")
   
   # Use before and after critical operations
   profile_memory("Before IPA")
   output = self.ipa_module(residue_repr, pair_repr, mask)
   profile_memory("After IPA")
   ```

2. **Time Profiling**:
   ```python
   import time
   
   def time_module(module, *args, **kwargs):
       start = time.time()
       result = module(*args, **kwargs)
       end = time.time()
       print(f"Module execution time: {end - start:.4f}s")
       return result
   
   # Use to measure execution time
   output = time_module(self.ipa_module, residue_repr, pair_repr, mask)
   ```

3. **Inspect Forward Pass with Hooks**:
   ```python
   # Register hooks to track tensor sizes
   activation_sizes = {}
   
   def hook_fn(name):
       def fn(module, input, output):
           activation_sizes[name] = {
               'input_size': [tuple(x.shape) for x in input if isinstance(x, torch.Tensor)],
               'output_size': tuple(output.shape) if isinstance(output, torch.Tensor) else 
                             [tuple(x.shape) for x in output if isinstance(x, torch.Tensor)]
           }
       return fn
   
   # Register hook
   module.register_forward_hook(hook_fn("module_name"))
   ```

### Solutions
1. **Enable Gradient Checkpointing**:
   ```python
   # Add to model configuration
   config["use_gradient_checkpointing"] = True
   
   # Implement in forward pass
   if self.training and getattr(self, "use_gradient_checkpointing", False):
       s_updated = torch.utils.checkpoint.checkpoint(
           lambda s, z, r, m: self.ipa(s, z, r, m),
           s, z, rigid, mask
       )
   else:
       s_updated = self.ipa(s, z, r=rigid, mask=mask)
   ```

2. **Use Mixed Precision Training**:
   ```python
   # Enable AMP for training
   from torch.cuda.amp import autocast, GradScaler
   
   # In training loop
   scaler = GradScaler()
   
   with autocast():
       outputs = model(batch)
       loss = loss_fn(outputs, targets)
   
   scaler.scale(loss).backward()
   scaler.step(optimizer)
   scaler.update()
   ```

3. **Implement Chunked Processing**:
   ```python
   def process_in_chunks(self, residue_repr, pair_repr, mask, chunk_size=128):
       """Process long sequences in chunks to save memory"""
       batch_size, seq_len, _ = residue_repr.shape
       
       # If sequence is short enough, process normally
       if seq_len <= chunk_size:
           return self.ipa_module(residue_repr, pair_repr, mask)
       
       # Otherwise, process in overlapping chunks with a sliding window
       results = []
       for start_idx in range(0, seq_len, chunk_size // 2):
           # Define chunk boundaries with overlap
           end_idx = min(start_idx + chunk_size, seq_len)
           start_idx = max(0, end_idx - chunk_size)
           
           # Extract chunk
           chunk_residue = residue_repr[:, start_idx:end_idx]
           chunk_pair = pair_repr[:, start_idx:end_idx, start_idx:end_idx]
           chunk_mask = mask[:, start_idx:end_idx] if mask is not None else None
           
           # Process chunk
           chunk_result = self.ipa_module(chunk_residue, chunk_pair, chunk_mask)
           
           # Store with position information for later merging
           results.append((start_idx, end_idx, chunk_result))
       
       # Merge results with position-based weighting for overlapping regions
       return self._merge_chunks(results, seq_len)
   ```

4. **Optimize Memory Layout**:
   ```python
   # Reuse tensors where possible
   def forward(self, residue_repr, pair_repr, mask=None):
       # Adapt dimensions in-place if possible
       s = residue_repr  # Keep reference to reuse memory
       s = self.single_adapter(s)  # Overwrite variable
       
       # Clear unused tensors to free memory
       del residue_repr  # Free memory if not needed anymore
       torch.cuda.empty_cache()  # Explicitly clear cache at critical points
   ```

## 3. Numerical Stability Issues

### Common Manifestations
- `NaN` or `Inf` values in outputs
- Exploding or vanishing gradients
- Unstable training dynamics (loss fluctuates wildly)

### Root Causes
1. **Division by Zero**: Often in normalization operations
2. **Exponentiation of Large Values**: In attention softmax
3. **Accumulation of Small Errors**: Throughout IPA iterations
4. **Underflow in Quaternion Normalization**: When computing rigid frames

### Debugging Strategies
1. **NaN Detection**:
   ```python
   def check_nan(name, tensor):
       if torch.isnan(tensor).any():
           print(f"NaN detected in {name}")
           # Print statistics about where NaNs occur
           nan_indices = torch.nonzero(torch.isnan(tensor))
           print(f"NaN indices: {nan_indices[:10]}...")  # Show first 10
           valid_values = tensor[~torch.isnan(tensor)]
           if valid_values.numel() > 0:
               print(f"Valid value range: [{valid_values.min()}, {valid_values.max()}]")
           return True
       return False
   
   # Use at critical points
   if check_nan("ipa_output", s_updated):
       # Take remedial action
       s_updated = torch.nan_to_num(s_updated)
   ```

2. **Gradient Monitoring**:
   ```python
   # In training loop, monitor gradients
   for name, param in model.named_parameters():
       if param.grad is not None:
           grad_norm = param.grad.norm()
           if torch.isnan(grad_norm) or torch.isinf(grad_norm):
               print(f"Problematic gradient in {name}: {grad_norm}")
   ```

3. **Value Range Tracking**:
   ```python
   def track_range(name, tensor):
       if tensor.numel() > 0:
           min_val = tensor.min().item()
           max_val = tensor.max().item()
           mean_val = tensor.mean().item()
           print(f"{name} range: [{min_val:.4f}, {max_val:.4f}], mean: {mean_val:.4f}")
           
           # Check for potential instability
           if abs(max_val) > 1e5 or abs(min_val) > 1e5:
               print(f"WARNING: Extreme values in {name}")
   
   # Use before and after operations
   track_range("Before IPA", s)
   s_updated = self.ipa(s, z, r=rigid, mask=mask)
   track_range("After IPA", s_updated)
   ```

### Solutions
1. **Add Numerical Safeguards**:
   ```python
   # Add small epsilon to denominators
   def safe_normalize(tensor, dim=-1, eps=1e-8):
       return tensor / (torch.norm(tensor, dim=dim, keepdim=True) + eps)
   
   # Use nan_to_num for worst-case recovery
   coordinates = torch.nan_to_num(
       coordinates, 
       nan=0.0, 
       posinf=1e6, 
       neginf=-1e6
   )
   ```

2. **Implement Gradient Clipping**:
   ```python
   # In training loop
   torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
   ```

3. **Stable Quaternion Operations**:
   ```python
   # More stable quaternion normalization
   def normalize_quaternion(quat, eps=1e-6):
       # Add small epsilon to avoid division by zero
       norm = torch.sqrt(torch.sum(quat * quat, dim=-1, keepdim=True) + eps)
       return quat / norm
   
   # Use in update_rigid_from_angles
   normalized_quat = normalize_quaternion(quat)
   ```

4. **Normalize Inputs and Outputs**:
   ```python
   # Add normalization to stabilize values
   class StableIPAModule(nn.Module):
       def __init__(self, config):
           super().__init__()
           # Add layer normalization
           self.input_norm_residue = nn.LayerNorm(config.residue_embed_dim)
           self.input_norm_pair = nn.LayerNorm(config.pair_embed_dim)
       
       def forward(self, residue_repr, pair_repr, mask=None):
           # Normalize inputs
           residue_repr = self.input_norm_residue(residue_repr)
           pair_repr = self.input_norm_pair(pair_repr)
           
           # Normal processing
           # ...
   ```

## 4. Integration with RhoFold+ Components

### Common Manifestations
- Import errors or missing attributes
- AttributeError when accessing RhoFold+ methods
- Version mismatch issues
- Dependency conflicts

### Root Causes
1. **Partial Extraction**: Not all required components were extracted
2. **Class Dependencies**: RhoFold+ classes rely on other components
3. **API Changes**: Differences in expected parameters or return values
4. **Inconsistent Tensor Types**: Mixed float types between models

### Debugging Strategies
1. **Trace Import Dependencies**:
   ```python
   import sys
   import inspect
   
   def trace_dependencies(module_name):
       """Trace dependencies of a module"""
       module = sys.modules[module_name]
       print(f"Dependencies for {module_name}:")
       
       for name, obj in inspect.getmembers(module):
           if inspect.ismodule(obj):
               print(f"  Module: {obj.__name__}")
           elif inspect.isclass(obj):
               print(f"  Class: {name}")
               # Print base classes
               if obj.__bases__:
                   print(f"    Bases: {[base.__name__ for base in obj.__bases__]}")
   
   # Use to trace dependencies
   trace_dependencies('external.rhofold.structure_module')
   ```

2. **Create Minimal Test Cases**:
   ```python
   def test_ipa_standalone():
       """Test IPA as a standalone component"""
       from external.rhofold.structure_module import InvariantPointAttention
       from external.rhofold.rigid_utils import Rigid
       
       # Create minimal inputs
       batch_size, seq_len = 2, 10
       s = torch.randn(batch_size, seq_len, 384)
       z = torch.randn(batch_size, seq_len, seq_len, 128)
       r = Rigid.identity((batch_size, seq_len), dtype=torch.float32)
       mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
       
       # Create IPA module
       ipa = InvariantPointAttention(
           c_s=384,
           c_z=128,
           c_ipa=16,
           no_heads=12,
           no_qk_points=4,
           no_v_points=8,
       )
       
       # Forward pass
       try:
           output = ipa(s, z, r, mask)
           print("IPA test passed!")
           print(f"Output shape: {output.shape}")
           return True
       except Exception as e:
           print(f"IPA test failed: {e}")
           import traceback
           traceback.print_exc()
           return False
   ```

3. **Check Version Compatibility**:
   ```python
   # Check PyTorch version compatibility
   import torch
   required_torch = "1.10.0"
   current_torch = torch.__version__
   
   from packaging import version
   if version.parse(current_torch) < version.parse(required_torch):
       print(f"Warning: RhoFold+ requires PyTorch {required_torch}, but found {current_torch}")
   ```

### Solutions
1. **Copy Full Module with Dependencies**:
   ```bash
   # Instead of copying individual files, copy entire modules
   mkdir -p external/rhofold/model external/rhofold/utils
   cp -r RhoFold/rhofold/model/structure_module.py external/rhofold/model/
   cp -r RhoFold/rhofold/utils/rigid_utils.py external/rhofold/utils/
   cp -r RhoFold/rhofold/utils/tensor_utils.py external/rhofold/utils/
   ```

2. **Use Module Wrapper**:
   ```python
   class IPAWrapper(nn.Module):
       """Wrapper for RhoFold+ IPA to handle API differences"""
       
       def __init__(self, config):
           super().__init__()
           # Import within init to contain exceptions
           try:
               from external.rhofold.structure_module import InvariantPointAttention
               self.ipa = InvariantPointAttention(
                   c_s=config.get("ipa_single_dim", 384),
                   c_z=config.get("ipa_pair_dim", 128),
                   c_ipa=config.get("ipa_hidden_dim", 16),
                   no_heads=config.get("ipa_heads", 12),
                   no_qk_points=config.get("ipa_qk_points", 4),
                   no_v_points=config.get("ipa_v_points", 8),
               )
               self.use_fallback = False
           except ImportError:
               print("WARNING: Could not import RhoFold+ IPA, using fallback implementation")
               self.use_fallback = True
               # Implement simple fallback
               self.fallback = nn.Sequential(
                   nn.Linear(config.residue_embed_dim, config.residue_embed_dim),
                   nn.ReLU(),
                   nn.Linear(config.residue_embed_dim, 6)  # 3 angles + 3 coords
               )
       
       def forward(self, s, z, r, mask):
           if self.use_fallback:
               # Simple fallback if IPA is not available
               return self.fallback(s)
           else:
               return self.ipa(s, z, r, mask)
   ```

3. **Ensure Type Consistency**:
   ```python
   def ensure_float32(tensor):
       """Ensure tensor is float32 for RhoFold+ compatibility"""
       if tensor.dtype != torch.float32:
           return tensor.to(torch.float32)
       return tensor
   
   # Use in forward pass
   s = ensure_float32(self.single_adapter(residue_repr))
   z = ensure_float32(self.pair_adapter(pair_repr))
   ```

4. **Use Docker for Isolation**:
   ```dockerfile
   # Dockerfile for environment isolation
   FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-runtime
   
   WORKDIR /app
   
   # Copy RhoFold+ code
   COPY RhoFold /app/RhoFold
   
   # Copy your code
   COPY src /app/src
   
   # Install dependencies
   RUN pip install -r RhoFold/requirements.txt
   RUN pip install -r src/requirements.txt
   
   # Set up pythonpath
   ENV PYTHONPATH="/app:${PYTHONPATH}"
   
   # Entry point
   CMD ["python", "src/train.py"]
   ```

## 5. Structure Quality Issues

### Common Manifestations
- Unrealistic RNA structures (distorted or collapsed)
- Invalid bond lengths or angles
- Missing structural motifs (hairpins, helices)
- Not matching expected RNA geometric constraints

### Root Causes
1. **Inadequate RNA Constraints**: Missing RNA-specific geometric knowledge
2. **Scale Mismatch**: Coordinate scale different from training data
3. **Insufficient Refinement**: IPA outputs need further refinement
4. **Untrained Parameters**: Initial random values causing poor structures

### Debugging Strategies
1. **Visualize Structures**:
   ```python
   def visualize_structure(coords, sequence, title="RNA Structure"):
       """Visualize RNA structure using matplotlib"""
       import matplotlib.pyplot as plt
       from mpl_toolkits.mplot3d import Axes3D
       
       fig = plt.figure(figsize=(10, 8))
       ax = fig.add_subplot(111, projection='3d')
       
       # Plot backbone
       ax.plot(coords[:, 0], coords[:, 1], coords[:, 2], 'b-', linewidth=2)
       
       # Color nucleotides
       colors = {'A': 'green', 'U': 'red', 'G': 'blue', 'C': 'orange'}
       for i, (nt, pos) in enumerate(zip(sequence, coords)):
           ax.scatter(pos[0], pos[1], pos[2], color=colors.get(nt, 'gray'), s=100)
           ax.text(pos[0], pos[1], pos[2], f"{i+1}:{nt}", fontsize=8)
       
       ax.set_title(title)
       plt.tight_layout()
       plt.show()
   
   # Use after prediction
   visualize_structure(
       outputs["coordinates"][0].detach().cpu().numpy(), 
       "".join([["G", "A", "U", "C"][idx] for idx in batch["sequences"][0].cpu().numpy() if idx < 4])
   )
   ```

2. **Calculate Structural Metrics**:
   ```python
   def calculate_rna_metrics(coords, sequence=None):
       """Calculate RNA structure quality metrics"""
       metrics = {}
       
       # Bond lengths (between adjacent residues)
       bond_vectors = coords[1:] - coords[:-1]
       bond_lengths = torch.norm(bond_vectors, dim=1)
       metrics["mean_bond_length"] = bond_lengths.mean().item()
       metrics["std_bond_length"] = bond_lengths.std().item()
       
       # Bond angles (between consecutive bonds)
       if len(coords) >= 3:
           v1 = bond_vectors[:-1]
           v2 = bond_vectors[1:]
           cosines = torch.sum(v1 * v2, dim=1) / (torch.norm(v1, dim=1) * torch.norm(v2, dim=1))
           # Clamp to avoid numerical errors
           cosines = torch.clamp(cosines, -1 + 1e-6, 1 - 1e-6)
           angles = torch.acos(cosines) * 180 / math.pi
           metrics["mean_angle"] = angles.mean().item()
           metrics["std_angle"] = angles.std().item()
       
       return metrics
   
   # Use after prediction
   metrics = calculate_rna_metrics(outputs["coordinates"][0])
   print(f"Structure metrics: {metrics}")
   
   # Define expected values for comparison
   expected_metrics = {
       "mean_bond_length": 6.0,  # Expected ~6Å between adjacent C1' atoms
       "std_bond_length": 0.5,   # Low variance expected
       "mean_angle": 120.0,      # Expected angle between consecutive bonds
       "std_angle": 15.0,        # Some variance expected
   }
   
   # Compare to expected values
   for key in metrics:
       if key in expected_metrics:
           diff = abs(metrics[key] - expected_metrics[key])
           status = "✓" if diff < expected_metrics[key] * 0.2 else "✗"
           print(f"{key}: {metrics[key]:.2f} vs expected {expected_metrics[key]:.2f} {status}")
   ```

3. **Track Training Progress**:
   ```python
   # In training loop
   if batch_idx % log_interval == 0:
       # Track structure metrics during training
       with torch.no_grad():
           metrics = calculate_rna_metrics(outputs["coordinates"][0])
           for key, value in metrics.items():
               writer.add_scalar(f"structure/{key}", value, global_step)
           
           # Visualize structure periodically
           if batch_idx % (log_interval * 10) == 0:
               fig = visualize_structure_to_figure(
                   outputs["coordinates"][0].cpu().numpy(),
                   batch["sequence_string"][0]
               )
               writer.add_figure("structure/prediction", fig, global_step)
   ```

### Solutions
1. **Add RNA-Specific Constraints**:
   ```python
   def apply_rna_constraints(coords, sequence=None, strength=1.0):
       """Apply RNA-specific geometric constraints to coordinates"""
       batch_size, seq_len, _ = coords.shape
       device = coords.device
       
       # Expected RNA parameters
       c1_c1_distance = 6.0  # Å between adjacent C1' atoms
       typical_angles = {
           # Different angle expectations based on sequence
           "AA": 120.0, "AU": 125.0, "AG": 122.0, "AC": 123.0,
           "UA": 125.0, "UU": 120.0, "UG": 122.0, "UC": 123.0,
           "GA": 122.0, "GU": 122.0, "GG": 120.0, "GC": 123.0,
           "CA": 123.0, "CU": 123.0, "CG": 123.0, "CC": 120.0,
       }
       
       # Create a copy for refinement
       refined_coords = coords.clone()
       
       # Apply distance constraints between adjacent residues
       for i in range(seq_len - 1):
           # Get coordinates of adjacent residues
           p1 = coords[:, i]       # [batch_size, 3]
           p2 = coords[:, i + 1]   # [batch_size, 3]
           
           # Calculate current distance
           diff = p2 - p1  # [batch_size, 3]
           dist = torch.norm(diff, dim=-1, keepdim=True)  # [batch_size, 1]
           
           # Unit vector from p1 to p2
           unit_vec = diff / (dist + 1e-8)  # [batch_size, 3]
           
           # Calculate distance adjustment (use a spring-like model)
           adjustment = (dist - c1_c1_distance) * 0.2 * strength
           
           # Apply adjustment in opposite directions
           refined_coords[:, i] += adjustment * unit_vec
           refined_coords[:, i + 1] -= adjustment * unit_vec
       
       # Apply angle constraints for three consecutive residues
       if seq_len >= 3 and sequence is not None:
           for i in range(seq_len - 2):
               # Get coordinates of three consecutive residues
               p1 = refined_coords[:, i]      # [batch_size, 3]
               p2 = refined_coords[:, i + 1]  # [batch_size, 3]
               p3 = refined_coords[:, i + 2]  # [batch_size, 3]
               
               # Calculate vectors and angle
               v1 = p2 - p1  # [batch_size, 3]
               v2 = p3 - p2  # [batch_size, 3]
               
               # Calculate current angle
               cos_angle = torch.sum(v1 * v2, dim=-1) / (
                   torch.norm(v1, dim=-1) * torch.norm(v2, dim=-1) + 1e-8
               )
               cos_angle = torch.clamp(cos_angle, -1 + 1e-6, 1 - 1e-6)
               angle = torch.acos(cos_angle) * 180 / math.pi  # [batch_size]
               
               # Get expected angle based on sequence context
               if sequence is not None:
                   seq_pair = sequence[i:i+2]
                   target_angle = typical_angles.get(seq_pair, 120.0)
                   
                   # Calculate angle adjustment (small adjustment)
                   adjustment = (angle - target_angle) * 0.01 * strength
                   
                   # TODO: Apply small rotations to improve angles
                   # This requires quaternion rotations which are complex
                   # For now, focus on distance constraints which are simpler
       
       return refined_coords
   ```

2. **Use Multi-stage Refinement**:
   ```python
   class MultiStageRefinement(nn.Module):
       """Multi-stage refinement of RNA structure"""
       
       def __init__(self, config):
           super().__init__()
           
           # Initial IPA module
           self.ipa_module = IPAModule(config)
           
           # Optional RhoFold+ refinement
           if config.get("use_rhofold_refine", False):
               from external.rhofold.structure_module import RefineNet
               self.refine_net = RefineNet(**config.get("refinement_config", {}))
           else:
               self.refine_net = None
           
           # Coordinate refinement MLP
           self.coord_refine = nn.Sequential(
               nn.Linear(3, 32),
               nn.ReLU(),
               nn.Linear(32, 32),
               nn.ReLU(),
               nn.Linear(32, 3)
           )
           
       def forward(self, residue_repr, pair_repr, mask=None, sequence=None):
           # Initial structure from IPA
           ipa_outputs = self.ipa_module(residue_repr, pair_repr, mask)
           
           # Stage 1: Apply RNA constraints
           stage1_coords = apply_rna_constraints(
               ipa_outputs["coordinates"], 
               sequence,
               strength=1.0
           )
           
           # Stage 2: Refine with RhoFold+ if available
           if self.refine_net is not None and "sequence_tokens" in batch:
               stage2_coords = self.refine_net(
                   tokens=batch["sequence_tokens"],
                   cords=stage1_coords.reshape(batch_size, -1, 3)
               )
           else:
               stage2_coords = stage1_coords
           
           # Stage 3: Fine MLP refinement
           residual = self.coord_refine(stage2_coords)
           final_coords = stage2_coords + residual * 0.1  # Small refinement
           
           # Apply mask if provided
           if mask is not None:
               mask_3d = mask.unsqueeze(-1).expand(-1, -1, 3).to(final_coords.dtype)
               final_coords = final_coords * mask_3d
           
           return {
               "coordinates": final_coords,
               "stage1_coordinates": stage1_coords,
               "stage2_coordinates": stage2_coords,
               "angles": ipa_outputs["angles"]
           }
   ```

3. **Add Physics-Based Loss Terms**:
   ```python
   def rna_structure_loss(pred_coords, sequence, weight=0.1):
       """Physics-based RNA structure loss term"""
       batch_size, seq_len, _ = pred_coords.shape
       
       # Bond length loss (adjacent C1' atoms should be ~6Å apart)
       bond_vectors = pred_coords[:, 1:] - pred_coords[:, :-1]
       bond_lengths = torch.norm(bond_vectors, dim=-1)
       ideal_length = 6.0
       bond_loss = F.mse_loss(bond_lengths, 
                              torch.ones_like(bond_lengths) * ideal_length)
       
       # Bond angle loss (consistency of angles)
       if seq_len >= 3:
           v1 = bond_vectors[:, :-1]  # [batch_size, seq_len-2, 3]
           v2 = bond_vectors[:, 1:]   # [batch_size, seq_len-2, 3]
           
           # Calculate cosine of angles
           cos_angles = torch.sum(v1 * v2, dim=-1) / (
               torch.norm(v1, dim=-1) * torch.norm(v2, dim=-1) + 1e-8
           )
           
           # Ideal cosine for RNA backbone (~120 degrees)
           ideal_cos = torch.cos(torch.tensor(120.0 * math.pi / 180))
           angle_loss = F.mse_loss(cos_angles, 
                                   torch.ones_like(cos_angles) * ideal_cos)
       else:
           angle_loss = torch.tensor(0.0, device=pred_coords.device)
       
       # Combine losses
       struct_loss = bond_loss + angle_loss
       
       return weight * struct_loss
   
   # Use in training loss
   loss = mse_loss(pred_coords, target_coords) + rna_structure_loss(pred_coords, sequence)
   ```

4. **Implement Iterative Refinement**:
   ```python
   def iterative_refinement(initial_coords, sequence, num_iterations=5):
       """Iteratively refine RNA structure with constraints"""
       coords = initial_coords.clone()
       
       for i in range(num_iterations):
           # Gradually decrease strength to avoid oscillations
           strength = 1.0 * (1.0 - i / num_iterations)
           
           # Apply constraints
           coords = apply_rna_constraints(coords, sequence, strength=strength)
           
           # Calculate metrics to monitor improvement
           metrics = calculate_rna_metrics(coords[0])
           print(f"Iteration {i+1} metrics: {metrics}")
       
       return coords
   
   # Use after initial prediction
   refined_coords = iterative_refinement(
       outputs["coordinates"], 
       batch["sequence_string"][0]
   )
   ```
