# Tactical Implementation Guide: RNA Folding - V1 Architecture (v2.0)

**Objective:** Implement the foundational V1 PyTorch data loading pipeline and Transformer-based fusion model architecture, as defined in `Revised-architecture-plan-Apr9.md` and `4_Product_Requirements_V1.md`. Includes multi-task angle prediction, structured validation strategy, and starts with scaled-down parameters.

**Phase:** `Dev` Environment (Local Workstation, RTX 4070 Ti / 16GB VRAM focus)

**IMPORTANT NOTE:** Throughout implementation within `src/` modules, strictly adhere to `7_AI_Agent_Rules.md`:
*   **Rule 2.4 (Modularity):** Keep core logic in `src/`, use `scripts/` or notebooks for orchestration only.
*   **Rule 7.2 (Path Parameterization):** Pass all file/directory paths needed by `src/` functions/classes as arguments. **NO hardcoded paths** inside `src/`.

---

## I. Environment & Setup Verification (PRD NF-01, NF-02, NF-04)

*   `[ ]` **Verify Environment:** Activate `rna-3d-folding`. Check packages (`pytorch`, `pandas`, `numpy`, `pyyaml`) per `environment.yml`.
*   `[ ]` **Project Structure:** Ensure directories (`src`, `tests`, `data/processed`, `config`, `validation`) exist as per `1_Context_and_Setup.md`.
*   `[ ]` **Config File:** Create/update `config/default_config.yaml`. Define **scaled-down V1 parameters** first (e.g., `residue_embed_dim: 64`, `pair_embed_dim: 32`, `seq_embed_dim: 32`, `num_transformer_blocks: 4`, `num_attention_heads: 4`, `ffn_dim: 256`, `dropout: 0.1`). Include placeholder data paths (to be filled by orchestrator/args). Define `loss_weights` section (e.g., `fape: 1.0`, `confidence: 0.1`, `angle: 0.5`).
*   `[ ]` **Version Control Setup:** Create version tracking file at `experiments/version_metadata.json` to track implementation versions as defined in validation strategy.

## II. Data Loading Implementation (`src/data_loading.py`) (PRD DL-01 to DL-08)

*   `[ ]` **Implement `load_coordinates` Helper:**
    *   Input: `labels_df: pd.DataFrame`, `target_id: str`.
    *   Output: `coords: np.ndarray` (`(N, 3)`, `float32`), `resnames: List[str]`. Error check.
*   `[ ]` **Implement `load_precomputed_features` Helper:**
    *   Input: `target_id: str`, `features_dir: str` (**Argument**).
    *   Action: Load `.npz` files (`dihedral`, `thermo`, `evolutionary`/`mi`). Handle errors/missing files (return default zero arrays or `None`).
    *   Output: `Dict[str, Dict[str, np.ndarray]]` or `Dict[str, Optional[Dict]]`.
*   `[ ]` **Implement `RNADataset.__init__`:**
    *   Args: `sequences_csv_path`, `labels_csv_path`, `features_dir` (**Arguments**), `temporal_cutoff`, `use_validation_set`.
    *   Action: Load sequences CSV, filter by `temporal_cutoff`. Store `target_ids`, `sequences`. Load full `labels_df`. Define `nuc_to_int`. Store `features_dir`.
*   `[ ]` **Implement `RNADataset.__len__`:**
    *   Action: Return `len(self.target_ids)`.
*   `[ ]` **Implement `RNADataset.__getitem__`:**
    *   Action: Get `target_id`, `sequence_str`. Call `load_precomputed_features(target_id, self.features_dir)`. Call `load_coordinates(self.labels_df, target_id)`.
    *   Action: Perform length consistency checks.
    *   Action: Assemble feature dictionary (`sequence_int`, `dihedral_features` (handle NaNs, length mismatch), `pairing_status`, `pairing_probs`, `mi_matrix`, `delta_g`, `coordinates`, `length`). Convert to tensors. Use defaults for missing features.
    *   Action: Wrap loading in `try...except`.
    *   Output: Dictionary of tensors + metadata.
*   `[ ]` **Implement `collate_fn`:**
    *   Input: `batch: List[Dict]`.
    *   Action: Find `max_len`. Pad 1D, 2D-N, 2D-NxN tensors using `F.pad`. Stack tensors. Generate boolean `mask`.
    *   Output: Batch dictionary.
*   `[ ]` **Implement `create_validation_subsets`:**
    *   Input: `sequences_df`, `labels_df`, `temporal_cutoff`.
    *   Action: Filter by temporal cutoff, select diverse sequences by length (short/medium/long).
    *   Output: Dictionary with technical, training, and scientific validation subsets.
*   `[ ]` **Write Unit Tests (`tests/test_data_loading.py`):**
    *   Test instantiation (train/val modes, cutoff).
    *   Test `__getitem__` output dict keys, shapes, types; mock helpers. Test missing feature handling.
    *   Test `collate_fn` with variable lengths; check padding, mask, shapes.
    *   Test `create_validation_subsets` function.
*   `[ ]` **Run Tests & Debug:** `pytest tests/test_data_loading.py`. Fix failures.
*   `[ ]` **Commit Prompt:** *AI prompts user:* "`RNADataset` and `collate_fn` implemented and tests pass. Ready to commit? Suggest: `feat(data): Implement RNADataset and collate_fn for V1 features`"

## III. Model Architecture Implementation (`src/models/`) (PRD MA-01 to MA-11)

1.  `[ ]` **Implement Embedding Layers (`src/models/embeddings.py`):** (PRD MA-03)
    *   Implement `SequenceEmbedding`.
    *   Implement `PositionalEncoding`.
    *   Implement `RelativePositionalEncoding`.
    *   Write unit tests (`tests/test_embeddings.py`) checking output shapes.
    *   **Commit Prompt:** *AI prompts user:* "Embedding layers implemented and tests pass. Ready to commit? Suggest: `feat(model): Implement sequence, positional, relative positional embeddings`"
2.  `[ ]` **Implement Transformer Block (`src/models/transformer_block.py`):** (PRD MA-05)
    *   Implement `TransformerBlock(config)`:
        *   Use **`nn.MultiheadAttention`** (`batch_first=True`).
        *   Implement residue update path.
        *   Implement **simplified pair update path** (Outer Product Prep -> MLP). Calculate MLP input dim: `2 * config['residue_embed_dim'] + config['pair_embed_dim']`.
    *   Write unit tests (`tests/test_transformer_block.py`) checking I/O shapes.
    *   **Commit Prompt:** *AI prompts user:* "`TransformerBlock` implemented and tests pass. Ready to commit? Suggest: `feat(model): Implement TransformerBlock with std MHA and simplified pair update`"
3.  `[ ]` **Implement IPA Placeholder (`src/models/ipa_module.py`):** (PRD MA-07)
    *   Implement placeholder `IPAModule(config)` with `nn.Linear` outputting `(B, L, 3)`. Document clearly.
    *   **(No dedicated commit unless complex placeholder warrants it)**
4.  `[ ]` **Implement Main Model (`src/models/rna_folding_model.py`):** (PRD MA-01, MA-02, MA-04, MA-06, MA-08, MA-09, MA-10, MA-11)
    *   Implement `RNAFoldingModel(config)`:
        *   `__init__`: Instantiate embeddings, input projections (calculate `IN_RES_DIM`, `IN_PAIR_DIM` based on `config` and features used), `nn.ModuleList` of `TransformerBlock`s, IPA placeholder, `confidence_head`, `angle_prediction_head`.
        *   `forward(batch)`: Implement data flow through components. Return dict: `"pred_coords"`, `"pred_confidence"`, `"pred_angles"`. Apply masks correctly after backbone and potentially before heads.
    *   Write unit tests (`tests/test_model.py`): Test instantiation, forward pass with dummy batch, verify output shapes.
    *   **Commit Prompt:** *AI prompts user:* "`RNAFoldingModel` V1 implemented (with placeholders/aux heads) and tests pass. Ready to commit? Suggest: `feat(model): Implement V1 RNAFoldingModel structure`"

## IV. Loss Function Implementation (`src/losses.py`) (PRD LF-01, LF-02, LF-03, LF-04)

1.  `[ ]` **Implement `compute_fape_loss` (Simplified Proxy):** (PRD LF-01)
    *   Action: Clamped L2 distance, averaged over valid residues/batch.
2.  `[ ]` **Implement `compute_confidence_loss` (Proxy):** (PRD LF-02)
    *   Action: Calculate proxy lDDT target (distance-based) inside `torch.no_grad()`. Compute MSE loss (`sigmoid` output) or `BCEWithLogitsLoss` (logit output). Average over valid residues/batch.
3.  `[ ]` **Implement `compute_angle_loss` (Auxiliary):** (PRD LF-03)
    *   Action: Compare predicted vs. true sin/cos angle features (e.g., `1 - F.cosine_similarity`). Average over angles and valid residues/batch. Handle NaNs in true angles (mask out).
4.  `[ ]` **Implement Evaluation Metrics:**
    *   Implement `calculate_rmsd` function as defined in validation strategy.
    *   Implement `confidence_correlation` function for evaluating confidence predictions.
    *   Prepare code placeholder for `calculate_tm_score` (using US-align, may be external call).
5.  `[ ]` **Write Unit Tests (`tests/test_losses.py`):**
    *   Test each loss function: scalar output, non-negative, mask handling.
    *   Test each evaluation metric function.
6.  `[ ]` **Run Tests & Debug:** `pytest tests/test_losses.py`.
7.  `[ ]` **Commit Prompt:** *AI prompts user:* "V1 loss functions (FAPE proxy, conf proxy, angle aux) and evaluation metrics implemented and tests pass. Ready to commit? Suggest: `feat(loss): Implement V1 loss functions and evaluation metrics`"

## V. Basic Integration Test (`scripts/test_pipeline.py` or Notebook) (PRD NF-05, NF-08)

1.  `[ ]` **Create Test Script/Notebook (Orchestrator):**
    *   Import from `src/`. Load config. Set `device`.
    *   **Define/Load Paths:** Explicitly define `data_dir`, `features_dir`, `sequences_csv_path`, `labels_csv_path` for this test.
    *   Instantiate `RNADataset` (passing paths).
    *   Instantiate `DataLoader` (small `batch_size`).
    *   Instantiate `RNAFoldingModel(config).to(device)`.
    *   Fetch batch, move tensors to `device`.
    *   Run `outputs = model(batch)`.
    *   Extract true targets from batch, move to `device`.
    *   Calculate individual losses using functions from `src/losses.py`.
    *   Calculate `total_loss` using `config['loss_weights']`.
    *   Print key shapes and loss values.
2.  `[ ]` **Execute & Verify:** Run script. Check for runtime errors (shapes, device, OOM). Ensure losses are computed. Check VRAM (`nvidia-smi`). If OOM, reduce config parameters (layers/dims).
3.  `[ ]` **Commit (Optional):** *AI prompts user:* "Basic integration test (data->model->loss) passes. Ready to commit script? Suggest: `test(integration): Add basic script testing V1 data->model->loss flow`"

## VI. Validation Framework Implementation (New Section)

1.  `[ ]` **Implement Tiered Validation Structure:**
    *   Create `validation/` directory with subdirectories:
        *   `validation/tier1_technical/`
        *   `validation/tier2_scientific/`
        *   `validation/tier3_comprehensive/`
    *   Add README.md in each directory explaining purpose and usage.

2.  `[ ]` **Implement Tier 1 (Technical) Validation Notebook:**
    *   Create `validation/tier1_technical/validation_technical.ipynb`.
    *   Implement the four-section structure from validation strategy:
        *   **Section 1:** Setup & Data Loading (use minimal subset of 3-5 sequences)
        *   **Section 2:** Model Loading
        *   **Section 3:** Inference & Prediction
        *   **Section 4:** Basic Evaluation (shape checks, loss values, simple RMSD)
    *   Ensure notebook completes in <5 minutes on target hardware.
    *   Add markdown documentation explaining purpose and interpretation of results.

3.  `[ ]` **Implement Tier 2 (Scientific) Validation Notebook:**
    *   Create `validation/tier2_scientific/validation_scientific.ipynb`.
    *   Use the same four-section structure with more comprehensive evaluation:
        *   **Section 1:** Setup & Data Loading (use 10-15 balanced sequences)
        *   **Section 2:** Model Loading
        *   **Section 3:** Inference & Prediction
        *   **Section 4:** Scientific Evaluation (TM-score, RMSD, confidence correlation)
    *   Implement the visualization components from validation strategy.
    *   Target 15-30 minute runtime.

4.  `[ ]` **Implement Version Tracking:**
    *   Create `experiments/track_experiment.py` utility script for recording experiment metadata:
        *   Input: version string, model config, validation results
        *   Output: JSON metadata file with timestamp and results
    *   Follow versioning scheme: 0.1.x.y (V1 validation iterations)
    *   Create `experiments/version_registry.md` to document version changes.

5.  `[ ]` **Write Documentation:**
    *   Create `validation/README.md` explaining validation framework
    *   Document when to use each validation tier
    *   Explain how to interpret results
    *   Include V1-to-V2 transition criteria (TM-score >0.4, confidence correlation >0.5)

6.  `[ ]` **Commit Prompt:** *AI prompts user:* "Validation framework implemented with tiered notebooks and metrics. Ready to commit? Suggest: `feat(validation): Implement validation framework with tiered approach`"

## VII. Resource Management and Documentation

1.  `[ ]` **Implement Resource Management:**
    *   Create utility functions for memory management in `src/utils/resource_management.py`:
        *   `clear_gpu_cache()`: For clearing GPU memory between runs
        *   `estimate_memory_usage(seq_len, batch_size, config)`: For predicting memory requirements
        *   `optimize_batch_size(seq_lens, available_memory, config)`: For dynamically selecting optimal batch size
    *   Add these utilities to validation notebooks.

2.  `[ ]` **Document V1-V2 Transition Plan:**
    *   Create `docs/V1_V2_Transition.md` containing:
        *   Performance thresholds from validation strategy (TM-score >0.4)
        *   Technical requirements for transition
        *   Documentation requirements
        *   Planned V2 components and improvements

3.  `[ ]` **Create Kaggle Transition Guide:**
    *   Create `docs/Kaggle_Transition_Guide.md` containing:
        *   Step-by-step instructions for converting local code to Kaggle
        *   Path adjustment examples
        *   Common pitfalls and solutions
        *   Resource optimization strategies

4.  `[ ]` **Implement Baseline Comparison:**
    *   If time permits, create `validation/baselines/` directory
    *   Implement simplified baseline as described in validation strategy
    *   Add comparative analysis notebook

5.  `[ ]` **Final Documentation Review:**
    *   Ensure all components are documented with proper docstrings
    *   Check that path parameterization is strictly followed
    *   Verify all tests are passing
    *   Review memory usage and optimization opportunities

## VIII. Integration and Validation (Completion Review)

1.  `[ ]` **Run Complete Integration Pipeline:**
    *   Execute Tier 1 validation notebook with current implementation
    *   Document results and any issues
    *   Fix critical issues before considering V1 complete

2.  `[ ]` **Complete V1 Assessment:**
    *   Verify all core components implemented and tested
    *   Run full test suite: `pytest tests/`
    *   Check code against path parameterization and modularity principles
    *   Document current performance metrics
    *   Assess memory usage on target hardware

3.  `[ ]` **Prepare V2 Planning:**
    *   Document learnings from V1 implementation
    *   Identify performance bottlenecks
    *   Prioritize V2 components based on validation results
    *   Create preliminary V2 tactical plan

4.  `[ ]` **Final Documentation:**
    *   Update version tracking with final V1 status
    *   Document environment and hardware requirements
    *   Prepare handoff documentation if needed

---

**Completion Check (V1 Foundation):** Successful execution of Tier 1 validation notebook with basic metrics signifies the completion of this V1.0 phase. The system includes a working data pipeline, a foundational V1 model (scaled-down, aux heads, placeholders), V1 losses, and a structured validation framework, adhering to modularity and strict path parameterization. Ready for Tier 2 validation and iterative refinement toward V2.

### Version Tracking Framework

| Version | Description | Requirements |
|---------|-------------|--------------|
| 0.1.x.y | Initial V1 iterations | Basic functionality, passing integration tests |
| 0.2.x.y | V1 refinement | Improved performance, passing Tier 1 validation |
| 0.3.x.y | V1 optimization | Memory optimization, efficient validation |
| 0.5.x.y | V1 completion candidate | Passing Tier 2 validation, documented performance |
| 1.0.0 | V1 final version | Meets transition criteria or reached performance plateau |

### V1-V2 Transition Criteria

1. **Performance Thresholds:**
   * Mean TM-score >0.4 on scientific validation set
   * Per-residue confidence correlation >0.5
   * Or: Performance plateaus with no improvement for 3+ iterations

2. **Technical Requirements:**
   * All V1 components implemented and tested
   * Full validation framework operational
   * Verified memory efficiency on target hardware

3. **Documentation Requirements:**
   * Performance analysis documented
   * V2 improvement hypotheses formulated
   * Implementation plan for V2 components prepared
