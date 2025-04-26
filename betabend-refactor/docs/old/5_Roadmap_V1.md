# Roadmap: RNA Folding - V2.0 Foundational Build with Validation

**Version:** 2.0  
**Date:** 2025-04-20  
**Purpose:** This document outlines the high-level development plan and milestones for implementing Version 1 (V1) of the PyTorch-based RNA 3D structure prediction pipeline. V1 focuses on establishing the core data loading infrastructure, a functional (albeit simplified) version of the target architecture, and a structured validation framework. This provides a testable foundation compatible with our Dockerized workflow and prepares for subsequent V2+ iterations aiming for full Kaggle competition readiness.

## 1. V1 Goal

The primary goal of V1 is to create a **runnable end-to-end pipeline with validation capabilities** based on the foundational architecture. This includes:
*   A working PyTorch `Dataset` and `DataLoader` for precomputed features (`2_Feature_Specification.md`).
*   A V1 `RNAFoldingModel` implementing the core Transformer structure with scaled-down parameters, simplified pair updates, a placeholder structure module (IPA stub), and auxiliary heads (confidence, angle prediction) as specified in `4_Product_Requirements_V1.md`.
*   Basic proxy loss functions (FAPE, confidence, angle) and evaluation metrics (RMSD, TM-score, confidence correlation).
*   A structured, tiered validation framework for assessing model performance.
*   Experiment tracking and versioning infrastructure.
*   Verification through unit tests and validation notebooks demonstrating data flow and model performance.
*   Strict adherence to modularity and path parameterization principles for Kaggle compatibility.

## 2. High-Level Pipeline Overview (V1 Implementation)

1.  **Data Ingestion & Feature Loading (`src/data_loading.py`):** Implement `Dataset` and `DataLoader` to load sequences, labels, and precomputed `.npz` features, handling padding/masking via `collate_fn`. Apply temporal cutoff logic. Implement validation subset selection.
2.  **Feature Embedding (`src/models/embeddings.py`):** Implement modules for sequence, positional, and relative positional embeddings.
3.  **Fusion Backbone (V1) (`src/models/transformer_block.py`, `src/models/rna_folding_model.py`):** Implement scaled-down Transformer backbone with standard Multi-Head Attention and simplified pair updates.
4.  **Structure Module (Placeholder) (`src/models/ipa_module.py`):** Implement a simple linear layer predicting coordinates as a placeholder for the full IPA module.
5.  **Output Heads & Loss Calculation (`src/models/rna_folding_model.py`, `src/losses.py`):** Implement heads for coordinate, confidence, and auxiliary angle prediction. Implement V1 proxy loss functions and evaluation metrics.
6.  **Validation Framework (`validation/`):** Implement a tiered validation approach with three levels (technical, scientific, comprehensive) following structured notebook templates.
7.  **Experiment Tracking (`experiments/`):** Implement version tracking system and experiment metadata recording.
8.  **Orchestration (Basic Test & Validation):** Create both a test script (`scripts/test_pipeline.py`) and validation notebooks to verify the flow from data loading through loss calculation and performance evaluation.

*(Full training and prediction scripts (`train.py`, `predict.py`) are part of V2+ development).*

## 3. Estimated Timeline & Milestones (V1 Foundational Build: ~3-4 Weeks)

This timeline assumes focused effort (a few hours per day) following the detailed steps in the updated Tactical Plan.

| **Week**       | **Focus**                                                                                  | **Key Deliverables**                                                                                                   | **PRD Sections**    |
| :------------- | :----------------------------------------------------------------------------------------- | :--------------------------------------------------------------------------------------------------------------------- | :------------------ |
| **Week 1**     | Setup, Data Loading Pipeline, Embeddings                                                   | - Docker env stable & verified <br/> - `RNADataset`, `collate_fn` implemented & unit-tested <br/> - Embedding modules implemented & unit-tested <br/> - Validation subset selection implemented | DL-01 to DL-08, MA-03 |
| **Week 2**     | Model Backbone (V1), Placeholders, Auxiliary Heads                                         | - `TransformerBlock` (V1) implemented & unit-tested <br/> - `IPAModule` placeholder functional <br/> - `RNAFoldingModel` (V1) structure implemented <br/> - Confidence & Angle heads implemented & unit-tested | MA-01, 02, 04-11    |
| **Week 3**     | Loss Functions, Evaluation Metrics, Integration Testing                                     | - V1 Loss functions (FAPE proxy, Conf proxy, Angle aux) implemented & unit-tested <br/> - Evaluation metrics (RMSD, confidence correlation) implemented <br/> - Basic Integration test runs successfully on target hardware (GPU) | LF-01 to LF-04, NF-* |
| **Week 4**     | Validation Framework, Experiment Tracking, Documentation                                     | - Tiered validation notebooks implemented <br/> - Experiment tracking and versioning system <br/> - V1-V2 transition documentation <br/> - Tier 1 & 2 validation executed | All validation requirements |

## 4. Validation Framework

The validation framework consists of three tiers with increasing comprehensiveness:

### 4.1 Tier 1: Technical Validation
* **Purpose:** Verify code functionality and basic correctness
* **Data:** 3-5 sequences (covering short, medium, long lengths)
* **Metrics:** Basic shape checks, loss values, simple RMSD
* **Frequency:** After every significant code change
* **Runtime target:** <5 minutes

### 4.2 Tier 2: Scientific Validation
* **Purpose:** Assess model learning and prediction quality
* **Data:** 10-15 sequences (balanced distribution)
* **Metrics:** TM-score, RMSD, confidence correlation
* **Frequency:** Weekly during active development
* **Runtime target:** 15-30 minutes

### 4.3 Tier 3: Comprehensive Validation
* **Purpose:** Full evaluation of model performance
* **Data:** CASP15 validation sequences
* **Metrics:** Full suite of primary and secondary metrics
* **Frequency:** Before major version changes
* **Runtime target:** 1-2 hours

### 4.4 Experiment Tracking
* **Version numbering scheme:**
  * **0.1.x.y:** Initial V1 validation iterations
  * **0.2.x.y:** V1 submission candidates
  * **0.5.x.y:** V2 implementations
* **Metadata tracking:** Record performance metrics, configuration, and notes for each version

## 5. Transition to V2 and Beyond

*   **V1 Completion:** V1 is considered complete when one of the following conditions is met:
    *   Successful execution of Tier 2 validation with mean TM-score >0.4
    *   Per-residue confidence correlation >0.5
    *   Performance plateaus with no improvement for 3+ iterations

*   **V2 Planning:** Following V1 completion, a review will inform the specific requirements and tactical plan for V2. Key V2+ goals include:
    *   Implementing the full training loop (`scripts/train.py`) with optimization, logging, and validation.
    *   Replacing the `IPAModule` placeholder with a functional implementation.
    *   Refining loss functions (implementing full FAPE, accurate lDDT targets).
    *   Integrating TM-score calculation for validation.
    *   Implementing the 5-prediction strategy and finalizing `scripts/predict.py` for Kaggle submission format.
    *   Scaling up model parameters (layers, dimensions) based on performance and hardware capabilities.
    *   Potentially integrating more advanced architectural features (pair bias, triangle updates).

*   **Technical Requirements for Transition:**
    *   All V1 components implemented and tested
    *   Full validation framework operational
    *   Core functionality verified on varying sequence lengths
    *   Memory efficiency demonstrated on target hardware

*   **Documentation Requirements for Transition:**
    *   Performance analysis documented with identified limitations
    *   Clear hypotheses for V2 improvements formulated
    *   Implementation plan for V2 components approved

*   **Multi-GPU:** Implementation of multi-GPU training (DDP) is deferred. It will only be considered in later phases (V3+) **if** single-GPU training performance proves to be a significant bottleneck preventing necessary experimentation within the competition timeline. The V1/V2 design ensures readiness but avoids the upfront complexity.

## 6. Alignment with Kaggle

*   The focus on Docker, pinned dependencies (`environment.yml`), modularity (`src/` vs. `scripts/`), and strict path parameterization in V1 is explicitly designed to minimize friction when creating the final Kaggle Notebook submission.
*   The validation framework includes considerations for transitioning to Kaggle, with guidance on path adjustments and resource optimization.
*   While V1 does not produce the final `submission.csv`, it builds the core, reusable components (`src/`) that will be orchestrated within the Kaggle environment.

## 7. Resource Management

*   **Memory Efficiency:** Implementation includes utilities for memory management and optimization:
    *   GPU cache clearing between validation runs
    *   Memory usage estimation based on sequence length and batch size
    *   Dynamic batch size optimization

*   **Computation Efficiency:**
    *   Tiered validation approach to balance thoroughness with speed
    *   Resource-appropriate validation frequencies
    *   Optimized tensor operations where possible

## Conclusion

This updated V1 roadmap focuses on building a working, testable foundation for the RNA folding pipeline with a structured validation framework. By implementing a tiered approach to validation and systematic experiment tracking, we can assess model performance quantitatively and make informed decisions about the transition to V2. Starting with simplified components and scaled-down parameters, we establish a solid base that adheres to architectural plans and Kaggle compatibility requirements while enabling systematic evaluation and refinement.
