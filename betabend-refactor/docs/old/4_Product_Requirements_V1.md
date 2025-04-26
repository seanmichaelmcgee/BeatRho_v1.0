# Product Requirements Document: RNA Folding Model - V2.0 Architecture Implementation

**Version:** 2.0  
**Date:** 2025-04-20  
**Author:** RNA 3D Folding Team

## 1. Introduction

This document outlines the requirements for the initial implementation phase (V1) of the RNA 3D Folding project's core machine learning components, based on the architecture defined in `3_Architecture_Specification.md`. This phase focuses on establishing a robust PyTorch data loading pipeline, implementing the foundational Transformer-based fusion architecture (including auxiliary prediction heads and simplified update mechanisms), and creating a structured validation framework. The goal is to create a runnable, testable system capable of processing precomputed features, generating initial 3D coordinate predictions, and quantitatively evaluating prediction quality, serving as the base for subsequent refinement and optimization towards the Kaggle competition goal.

## 2. Goals

* Implement a PyTorch `Dataset` and `DataLoader` capable of loading, preprocessing, and batching precomputed RNA features and corresponding labels
* Implement a PyTorch `nn.Module` representing the V1 fusion model architecture, including embeddings, Transformer backbone (with standard attention and simplified pair updates), placeholder structure module, and auxiliary heads
* Define and implement V1 loss functions (simplified FAPE proxy, confidence loss, auxiliary angle prediction loss) and evaluation metrics (RMSD, confidence correlation)
* Create a structured, tiered validation framework with appropriate metrics for assessing model performance
* Implement experiment tracking and versioning infrastructure to enable systematic model improvement
* Ensure the implementation adheres to project design principles (Docker, reproducibility, **modularity**, **strict path parameterization**, configurability), critical for **Kaggle Notebook compatibility**
* Provide a foundation for integrating a full training loop and more complex components (e.g., functional IPA module, refined losses) in V2

## 3. Scope

### In Scope (V1 Implementation)

* **Data Loading:** (Requirements DL-01 to DL-09)
  * `RNADataset` implementation loading features specified in feature specification (.npz files: dihedral, thermo, evolutionary MI)
  * Loading sequence data and ground truth C1' coordinates
  * Temporal cutoff logic for train/validation splits
  * `collate_fn` handling variable sequence lengths via padding and masking
  * Support for `DistributedSampler` integration
  * Validation subset selection based on sequence diversity

* **Model Architecture (V1):** (Requirements MA-01 to MA-11)
  * Input embedding layers (sequence, positional, relative positional)
  * Linear projection layers for combined input features (residue & pair)
  * Implementation of `TransformerBlock` with standard MHA and simplified pair update mechanism
  * Stacking multiple `TransformerBlock` instances (initially scaled-down number/dims)
  * **Placeholder** implementation for the Structure Module (`IPAModule` stub predicting coordinates linearly)
  * Output head for predicted 3D coordinates (C1' atoms)
  * Output head for predicted per-residue confidence scores
  * **Auxiliary Output Head** for predicting pseudo-dihedral angles (eta, theta) via multi-task learning
  * Model configuration driven by `config/default_config.yaml`

* **Loss Functions & Evaluation (V1):** (Requirements LF-01 to LF-07)
  * Implementation of a simplified coordinate loss (`compute_fape_loss` proxy: clamped L2)
  * Implementation of a confidence prediction loss (`compute_confidence_loss`: MSE/BCE vs. derived lDDT proxy)
  * Implementation of an auxiliary pseudo-dihedral angle prediction loss (`compute_angle_loss`)
  * Evaluation metric implementations (RMSD, confidence correlation)
  * Integration with external TM-score calculation

* **Validation Framework:** (Requirements VF-01 to VF-09)
  * Tiered validation approach (technical, scientific, comprehensive)
  * Validation subset selection based on RNA diversity
  * Structured validation notebooks following standardized template
  * Experiment tracking and versioning infrastructure
  * Performance visualization components

* Adherence to Non-Functional Requirements (NF-01 to NF-08)

### Out of Scope (for this V1 phase)

* Full, functional implementation of the Invariant Point Attention (IPA) module
* Full implementation of pair-bias injection into attention mechanism
* Full implementation of Triangle Attention/Multiplication pair updates
* Implementation of the complete training and prediction loops (`train.py`, `predict.py`)
* Advanced/refined loss implementations (full FAPE, accurate lDDT targets)
* Teacher-Student distillation pipeline
* Hyperparameter optimization beyond initial scaling
* Automated submission generation for Kaggle
* Detailed performance profiling and memory optimization (beyond basic checks for VRAM)
* Multi-GPU training setup and testing

## 4. Target Users

* Machine Learning Engineer/Developer implementing and training the V1 model
* Data Scientists evaluating model performance and planning improvements
* Future team members extending/maintaining the codebase

## 5. Requirements

### 5.1. Data Loading (`src/data_loading.py`)

| ID    | Requirement                                                                                                | Priority | Verification Method              | Notes                                             |
| :---- | :--------------------------------------------------------------------------------------------------------- | :------- | :------------------------------- | :------------------------------------------------ |
| DL-01 | Implement `RNADataset` class inheriting from `torch.utils.data.Dataset`                                    | Must     | Code review, Unit tests          |                                                   |
| DL-02 | Constructor accepts `sequences_csv_path`, `labels_csv_path`, `features_dir`, `temporal_cutoff` arguments   | Must     | Unit tests (instantiation)       | Paths must be arguments                           |
| DL-03 | Implement logic to filter training sequences based on `temporal_cutoff` in `train_sequences.csv`           | Must     | Unit tests (cutoff logic)        | Use `pd.to_datetime`                              |
| DL-04 | Use `validation_sequences.csv` and `validation_labels.csv` entirely when `use_validation_set` is True      | Must     | Unit tests (validation mode)     | Ignore `temporal_cutoff`                          |
| DL-05 | Implement `__len__` method returning the number of eligible samples                                        | Must     | Unit tests                       |                                                   |
| DL-06 | Implement `__getitem__` method:                                                                            | Must     |                                  |                                                   |
| DL-06a| \- Loads sequence string for the given index                                                              | Must     | Unit tests                       |                                                   |
| DL-06b| \- Loads precomputed features (`dihedral`, `thermo`, `evolutionary`) from `.npz` files in `features_dir`   | Must     | Unit tests (shapes, types)       | Use helper `load_precomputed_features`            |
| DL-06c| \- Handles missing feature files gracefully (e.g., warning + zero tensor or error during debug)           | Should   | Test case (missing files)        | Return default zero tensors of correct shape      |
| DL-06d| \- Loads ground truth C1' coordinates (`x_1, y_1, z_1`) from `labels_csv` using helper `load_coordinates` | Must     | Unit tests (coords loading)      |                                                   |
| DL-06e| \- Performs basic consistency checks (e.g., sequence length vs. coordinate length vs. feature lengths)    | Must     | Test case (inconsistent data)    | Raise error or log warning                        |
| DL-06f| \- Converts all loaded data into appropriately typed PyTorch tensors                                      | Must     | Unit tests (output dict types)   | `float32` for features/coords, `long` for seq_int |
| DL-07 | Implement `collate_fn` function for batching:                                                              | Must     |                                  |                                                   |
| DL-07a| \- Identifies the maximum sequence length (`max_len`) within a batch                                      | Must     | Unit tests (variable lengths)    |                                                   |
| DL-07b| \- Pads all sequence-length-dependent tensors (1D, 2D-N, 2D-NxN) to `max_len` using `F.pad`              | Must     | Unit tests (batch shapes, padding) | Pad value = 0                                     |
| DL-07c| \- Generates a boolean attention mask tensor (`mask`, shape `(B, L)`) indicating valid positions          | Must     | Unit tests (mask correctness)    | `True` for valid, `False` for padded              |
| DL-07d| \- Stacks all tensors correctly into batch dimension using `torch.stack`                                  | Must     | Unit tests (final batch structure) | Handle non-tensor items (e.g., `target_id`)       |
| DL-08 | DataLoader setup design must be compatible with `DistributedSampler` for future DDP integration           | Must     | Code review                      | No hardcoded shuffling logic conflicting w/ sampler |
| DL-09 | Implement `create_validation_subsets` function for selecting diverse validation sequences                  | Must     | Unit tests (subset selection)    | Select by length categories and diversity         |

### 5.2. Model Architecture (`src/models/`)

| ID      | Requirement                                                                                                                              | Priority | Verification Method                | Notes                                                            |
| :------ | :--------------------------------------------------------------------------------------------------------------------------------------- | :------- | :--------------------------------- | :--------------------------------------------------------------- |
| MA-01   | Implement `RNAFoldingModel` class inheriting from `torch.nn.Module`                                                                     | Must     | Code review, Instantiation test    |                                                                  |
| MA-02   | Model hyperparameters (dimensions, layers, heads, dropout) loaded from a configuration object/dict (`config`)                           | Must     | Unit test (instantiation w/ config)  | Allows easy scaling                                              |
| MA-03   | Implement input embedding layers: `SequenceEmbedding`, `PositionalEncoding`, `RelativePositionalEncoding`                               | Must     | Unit tests (embedding shapes)      | In `src/models/embeddings.py`                                    |
| MA-04   | Implement input linear projection layers for residue features (seq, dihedral, pair status, etc.) & pair features (pair probs, MI, rel pos) | Must     | Code review, Shape checks in test  | Calculate `in_features` dimension based on concatenated inputs |
| MA-05   | Implement `TransformerBlock` module containing: LayerNorm, **standard Multi-Head Attention** (`batch_first=True`), FFN, and **simplified pair update MLP** | Must     | Unit test (block I/O shapes)       | In `src/models/transformer_block.py`                             |
| MA-06   | The main model backbone (`RNAFoldingModel.__init__`) stacks multiple `TransformerBlock` instances using `nn.ModuleList`                   | Must     | Code review (`__init__`)           | Number of blocks from config                                     |
| MA-07   | Implement a **placeholder** `IPAModule` that accepts residue features and outputs 3D coordinates linearly (shape `(B, L, 3)`)             | Must     | Code review, Shape checks in test  | In `src/models/ipa_module.py`. Document clearly as placeholder.  |
| MA-08   | Implement a `confidence_head` (`nn.Sequential`) projecting final residue features to a scalar confidence score per residue (shape `(B, L)`) | Must     | Code review, Shape checks in test  | Output logits for BCE/MSE loss                                   |
| MA-09   | Implement an auxiliary `angle_prediction_head` (`nn.Sequential`) projecting final residue features to predicted sin/cos eta/theta (shape `(B, L, 4)`) | Must     | Code review, Shape checks in test  | For multi-task learning                                          |
| MA-10   | `RNAFoldingModel.forward` method correctly processes a batch, passing data through all components in sequence                           | Must     | Integration test (data->model->loss) | Embeddings -> Projections -> Backbone -> IPA-Stub -> Heads       |
| MA-11   | Model `forward` output dictionary includes keys: `"pred_coords"`, `"pred_confidence"`, and `"pred_angles"`                              | Must     | Unit/Integration tests             | Check tensor shapes match requirements                           |

### 5.3. Loss Functions and Evaluation (`src/losses.py`)

| ID      | Requirement                                                                                                          | Priority | Verification Method              | Notes                                                        |
| :------ | :------------------------------------------------------------------------------------------------------------------- | :------- | :------------------------------- | :----------------------------------------------------------- |
| LF-01   | Implement `compute_fape_loss` function (**simplified proxy**: clamped L2 distance between predicted and true coords)   | Must     | Unit tests (scalar output, mask) | Clamp distance error (e.g., `torch.clamp(dist, max=10.0)`) |
| LF-02   | Implement `compute_confidence_loss` function (**proxy**: MSE or BCEWithLogitsLoss vs. derived lDDT proxy target)      | Must     | Unit tests (scalar output, mask) | Calculate target inside `torch.no_grad()`                  |
| LF-03   | Implement `compute_angle_loss` function (auxiliary loss comparing predicted vs. true sin/cos angle features)        | Must     | Unit tests (scalar output, mask) | Use `1 - F.cosine_similarity` or MSE on sin/cos values. Handle NaNs. |
| LF-04   | All loss functions must correctly ignore contributions from padded sequence positions using the input `mask`        | Must     | Unit tests (masked inputs)       | Apply mask before reduction (sum/mean)                   |
| LF-05   | Implement `calculate_rmsd` function for evaluation (Root Mean Square Deviation after optimal superposition)         | Must     | Unit tests (expected values)     | Implement Kabsch algorithm for alignment                   |
| LF-06   | Implement `confidence_correlation` function for evaluating relationship between predicted confidence and accuracy   | Must     | Unit tests (correlation range)   | Output in range [-1, 1], higher values are better          |
| LF-07   | Prepare interface to external TM-score calculation (primary Kaggle evaluation metric)                              | Should   | Integration test with sample data | May use subprocess call to external tool                    |

### 5.4. Validation Framework (`validation/`)

| ID      | Requirement                                                                                           | Priority | Verification Method              | Notes                                                   |
| :------ | :---------------------------------------------------------------------------------------------------- | :------- | :------------------------------- | :------------------------------------------------------ |
| VF-01   | Implement tiered validation directory structure with technical, scientific, and comprehensive tiers   | Must     | Directory structure verification | See structure in tactical plan                         |
| VF-02   | Implement Tier 1 validation notebook for technical validation                                         | Must     | Execution test                   | Should run in <5 minutes on target hardware            |
| VF-03   | Implement Tier 2 validation notebook for scientific validation                                        | Must     | Execution test                   | Should run in 15-30 minutes on target hardware         |
| VF-04   | Implement four-section structure in all validation notebooks (Setup, Model, Inference, Evaluation)    | Must     | Code review                      | Follow structure defined in validation strategy         |
| VF-05   | Implement visualization components for model performance                                              | Should   | Visual inspection                | Include TM-score distribution, error heatmaps, etc.     |
| VF-06   | Implement experiment tracking system with version numbering                                           | Must     | Metadata file verification       | Follow 0.1.x.y versioning scheme for V1                |
| VF-07   | Create experiment metadata recording utility that saves configuration and results                     | Must     | Function test                    | Save as JSON in experiments directory                   |
| VF-08   | Document validation procedures with instructions for interpreting results                             | Must     | Documentation review             | Create README.md in validation directory                |
| VF-09   | Include resource management utilities for memory optimization during validation                       | Should   | Function tests                   | GPU cache clearing, batch size optimization             |

### 5.5. Non-Functional Requirements

| ID      | Requirement                                                                                                      | Priority | Verification Method                | Notes                                                 |
| :------ | :--------------------------------------------------------------------------------------------------------------- | :------- | :--------------------------------- | :---------------------------------------------------- |
| NF-01   | Code organized according to the defined project structure (`src/`, `scripts/`, `tests/`, etc.)                  | Must     | Code review                        | Ref: `1_Context_and_Setup.md`                        |
| NF-02   | Implementation uses PyTorch as the primary deep learning framework                                              | Must     | Code review                        |                                                       |
| NF-03   | Code includes reasonable type hints (`typing` module) and docstrings for major classes and functions            | Should   | Code review                        | Improves maintainability                              |
| NF-04   | Design allows easy containerization via Docker (no dependencies outside `environment.yml`, uses relative imports) | Must     | Code review, Successful Docker build |                                                       |
| NF-05   | Initial V1 implementation runs on the prototype workstation GPU (RTX 4070 Ti, 16GB VRAM) with a small batch size (e.g., 1-4) without Out-Of-Memory (OOM) errors | Must     | Integration test run, `nvidia-smi` | Start with scaled-down config params                  |
| NF-06   | Adherence to Python code style guidelines (e.g., PEP 8)                                                         | Should   | Linting tool (flake8/black)        | Improves readability                                  |
| NF-07   | All code and referenced algorithms respect the knowledge cutoff date (September 18, 2024)                         | Must     | Code review, Algorithm justification | Check against `3_Architecture_Specification.md`       |
| NF-08   | **Strict Path Parameterization:** Core logic modules in `src/` must accept necessary file paths as arguments and contain **no hardcoded paths** | **Must** | **Code review, Tests w/ varied paths** | **Critical for Kaggle/Portability**                 |

## 6. Success Metrics

### 6.1. Technical Success Metrics

* `RNADataset` correctly loads, filters, and tensorizes all specified features from `.npz` and `.csv` files
* `DataLoader` produces batches with correctly padded tensors (1D, 2D-N, 2D-NxN) and boolean masks
* `RNAFoldingModel` (V1, scaled-down) instantiates from config and performs a forward pass on a batch without crashing on the target GPU
* Output tensors (`pred_coords`, `pred_confidence`, `pred_angles`) have the expected shapes (`B,L,3`, `B,L`, `B,L,4` respectively)
* Loss functions (`fape_proxy`, `confidence_proxy`, `angle_aux`) compute scalar, non-negative values and respect padding masks
* Evaluation metrics correctly calculate RMSD and confidence correlation
* Unit tests for implemented `src/` components pass
* Basic integration test (data -> model -> loss) runs successfully
* Tier 1 validation notebook executes without errors
* Code adheres to structural and path parameterization requirements

### 6.2. Scientific Success Metrics

* **Mean TM-score:** >0.4 on scientific validation set (target for V1-V2 transition)
* **RMSD Values:** Lower values indicate better structural prediction
* **Confidence Correlation:** >0.5 correlation between predicted confidence and actual accuracy
* **Loss Values:** Consistent decrease in combined loss during training iterations

## 7. V1-V2 Transition Criteria

The transition from V1 to V2 will be based on meeting one or more of the following criteria:

### 7.1. Performance Thresholds
* Mean TM-score >0.4 on scientific validation set, OR
* Per-residue confidence correlation >0.5, OR
* Performance plateau with no improvement for 3+ iterations

### 7.2. Technical Requirements
* All V1 components implemented and tested
* Full validation framework operational
* Core functionality verified on varying sequence lengths
* Memory efficiency demonstrated on target hardware

### 7.3. Documentation Requirements
* Performance analysis documented with identified limitations
* Clear hypotheses for V2 improvements formulated
* Implementation plan for V2 components approved

## 8. Future Considerations (Post V1)

* Replace placeholder IPA with a functional implementation
* Develop the full training loop (`scripts/train.py`) including optimization, scheduling, logging, and validation steps
* Refine loss functions (implement full FAPE, calculate accurate lDDT targets)
* Integrate official TM-score calculation (e.g., using US-align via subprocess) for validation
* Implement the 5-prediction generation strategy for Kaggle submission (`scripts/predict.py`)
* Scale up model parameters (layers, dimensions) and perform hyperparameter tuning
* Consider architectural enhancements (pair bias, triangle updates)
* Evaluate Teacher-Student distillation as an alternative to multi-task angle prediction
* Implement multi-GPU training (DDP) if performance requires it
* Implement performance optimizations (mixed precision, gradient checkpointing)

## 9. Validation Framework Details

### 9.1. Tiered Validation Approach

The validation strategy follows a three-tier approach:

| Validation Tier | Purpose | Data Size | Metrics | Frequency | Runtime Target |
|-----------------|---------|-----------|---------|-----------|----------------|
| **Tier 1: Technical** | Verify code functionality and basic correctness | 3-5 sequences | Shape checks, loss values, simple RMSD | After component changes | <5 minutes |
| **Tier 2: Scientific** | Assess model learning and prediction quality | 10-15 sequences | TM-score, RMSD, confidence correlation | Weekly | 15-30 minutes |
| **Tier 3: Comprehensive** | Full evaluation of model performance | 3-5 CASP15 targets | Complete metric suite | Before version changes | 1-2 hours |

### 9.2. Validation Data Selection

* **Technical Validation Subset:**
  * 5-10 diverse RNA sequences respecting temporal cutoff (pre-2022-05-27)
  * Select varied sequence lengths (short: <50nt, medium: 50-150nt, long: >150nt)
  * Include varied secondary structure patterns (hairpins, pseudoknots, multi-loops)

* **Limited Training Subset:**
  * 20-30 sequences for initial learning validation
  * Stratified selection across RNA classes (tRNA, rRNA fragments, ribozymes, etc.)
  * Balance between "easy" and "challenging" structures based on prediction difficulty

* **Scientific Validation Subset:**
  * Use a portion (3-5) of the CASP15 validation sequences
  * Reserve remaining CASP15 sequences for final validation
  * Create synthetic test cases with controlled properties for specific feature testing

### 9.3. Experiment Tracking

* **Version Numbering Scheme:**
  * **0.1.x.y:** Initial V1 validation iterations
    * x: Major changes to data processing
    * y: Model parameter adjustments
  * **0.2.x.y:** V1 submission candidates
    * x: Architecture refinements
    * y: Hyperparameter tuning
  * **0.5.x.y:** V2 implementations
    * x: Major V2 component additions
    * y: V2 component refinements

* **Experiment Metadata Recording:**
  * Model configuration
  * Validation subset used
  * Performance metrics (TM-score, RMSD, confidence correlation)
  * Notes and observations
  * Stored in JSON format in experiments directory

## 10. Implementation Dependencies

### 10.1. Core Dependencies
* PyTorch 2.1+
* NumPy
* Pandas
* PyYAML

### 10.2. File Dependencies
* `config/default_config.yaml`: Configuration file defining model parameters
* `data/raw/train_sequences.csv`: RNA sequence data
* `data/raw/train_labels.csv`: Ground truth 3D coordinates
* `data/processed/`: Directory for precomputed features (.npz files)

### 10.3. Containerization
* Docker environment as defined in `Dockerfile`
* Environment variables and dependencies in `environment.yml`

## 11. Glossary of Terms

* **FAPE**: Frame-Aligned Point Error, a coordinate loss function that is invariant to global rotations/translations
* **TM-score**: Template Modeling score, a metric measuring structural similarity
* **RMSD**: Root Mean Square Deviation, measures average distance between aligned atoms
* **IPA**: Invariant Point Attention, a mechanism for coordinate prediction in 3D space
* **lDDT**: local Distance Difference Test, a measure of local structural accuracy
* **MSA**: Multiple Sequence Alignment, used for evolutionary coupling detection
