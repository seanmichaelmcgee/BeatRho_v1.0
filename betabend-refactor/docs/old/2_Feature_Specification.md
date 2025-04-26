# Feature Specification for ML Pipeline (V1.2)

**Version:** 1.2
**Date:** 2024-04-09
**Source Document:** `6.technical_report.md` (v0.2.1)
**Purpose:** This document specifies the exact format, content, and location of the precomputed input features available for the V1 machine learning model, based on the outputs of the feature extraction pipeline documented in the source technical report. This serves as the primary reference for implementing the data loading components (`src/data_loading.py`).

## 1. Overview

The machine learning model will consume features precomputed from raw RNA sequence and structure data. These features capture geometric, thermodynamic, and evolutionary properties. They are stored in separate `.npz` files for each RNA target, organized by feature type within the `data/processed/` directory.

The `DataLoader` is responsible for locating the correct files for a given `target_id`, loading the specified arrays, performing necessary checks, and converting them into PyTorch tensors for model consumption.

## 2. Data Storage Convention

*   **Base Directory:** `data/processed/`
*   **Subdirectories:** Features are organized into subdirectories based on type:
    *   `dihedral_features/`
    *   `thermo_features/` (*Note: Assumes standardized naming, potentially simplified from `features_all_TIMESTAMP` mentioned in source*)
    *   `mi_features/` (*Note: Contains Mutual Information based evolutionary features*)
*   **File Naming:** Each file corresponds to a specific `target_id` and feature type:
    *   Dihedral: `{target_id}_dihedral_features.npz`
    *   Thermo: `{target_id}_thermo_features.npz`
    *   Evolutionary MI: `{target_id}_features.npz` (within `mi_features/`)

## 3. Detailed Feature Breakdown

### 3.1 Pseudo-dihedral Angle Features

*   **File Location:** `data/processed/dihedral_features/{target_id}_dihedral_features.npz`
*   **Source:** Calculated from C1' coordinates of known 3D structures (available for training/validation data only).
*   **Primary ML Input Key:** `features`
    *   **Data Type:** `numpy.float32`
    *   **Shape:** `(N, 4)`, where N = number of residues in the sequence.
    *   **Content:** Sine and cosine transformations of pseudo-dihedral angles eta (η) and theta (θ).
        *   Column 0: `eta_sin`
        *   Column 1: `eta_cos`
        *   Column 2: `theta_sin`
        *   Column 3: `theta_cos`
    *   **Range:** Each value is between [-1, 1].
    *   **Notes:**
        *   The source report indicates raw angles (`eta`, `theta`) might also be present in the file but are not typically used as direct ML input due to periodicity. The 4D sin/cos representation is preferred.
        *   The source report mentions NaN handling for boundary residues during calculation. The final `features` array loaded by the `DataLoader` should be checked for NaNs, and a strategy for handling them (e.g., masking, imputation to 0) must be implemented in the `DataLoader` or model.

### 3.2 Thermodynamic Features

*   **File Location:** `data/processed/thermo_features/{target_id}_thermo_features.npz` (*Standardized path assumed*)
*   **Source:** Calculated using ViennaRNA based on sequence data.
*   **Content Keys:** The `.npz` file contains multiple arrays representing different thermodynamic properties.

    **Scalar Features (Shape `()`):**
    *   `mfe`: `float32`, Minimum Free Energy (kcal/mol).
    *   `ensemble_energy`: `float32`, Free energy of the ensemble.
    *   `energy_gap`: `float32`, Difference: `ensemble_energy` - `mfe`.
    *   `mfe_probability`: `float32`, Boltzmann probability of MFE structure [0, 1].
    *   `gc_content`: `float32`, Fraction of G-C pairs [0, 1].
    *   `paired_fraction`: `float32`, Fraction of paired nucleotides in MFE structure [0, 1].
    *   `avg_pair_distance_mean`: `float32`, Mean base pairing distance.
    *   `avg_pair_distance_std`: `float32`, Std dev of base pairing distances.
    *   `free_energy_per_nucleotide`: `float32`, Normalized MFE.
    *   `accessibility_mean`: `float32`, Mean accessibility over sequence [0, 1].
    *   `accessibility_variance`: `float32`, Variance of accessibility [0, 1].
    *   **Notes:** These global features might be broadcasted or appended to per-residue features within the model's embedding layer.

    **Vector Features (Per-Residue, Shape `(N,)`):**
    *   `positional_entropy` (or alias `position_entropy`): `float32`, Shannon entropy at each position [0, log2(4)].
    *   `accessibility`: `float32`, Per-nucleotide unpaired probability [0, 1].
    *   **Notes:** These provide per-residue structural context.

    **Matrix Features (Pairwise, Shape `(N, N)`):**
    *   `pairing_probs` (or alias `base_pair_probs`): `float32`, Probability of base pair formation between residue *i* and *j* [0, 1]. Symmetric matrix.
    *   **Notes:** This is a key input for representing secondary structure probabilities.

    **String Features (Metadata, Not Direct Tensor Input):**
    *   `structure` (or alias `mfe_structure`): `string`, Dot-bracket notation of MFE structure. Can be used to derive binary pairing status if needed.

### 3.3 Evolutionary Coupling Features (Mutual Information Based)

*   **File Location:** `data/processed/mi_features/{target_id}_features.npz`
*   **Source:** Calculated from Multiple Sequence Alignments (MSAs). Requires MSA availability.
*   **Primary ML Input Key:** `coupling_matrix`
    *   **Data Type:** `numpy.float32`
    *   **Shape:** `(N, N)`, where N = number of residues.
    *   **Content:** Matrix of evolutionary coupling scores (e.g., APC-corrected Mutual Information) between position pairs *i* and *j*.
    *   **Range:** Typically non-negative values (>= 0). Scale might vary.
    *   **Notes:** This represents co-evolutionary signals. It's expected to be symmetric. The `DataLoader` should handle cases where the file might be missing (if no MSA was available) by providing a default zero tensor.

*   **Other Potentially Useful Keys (Consider for V1/V2 input):**
    *   `conservation`: `float32`, shape `(N,)`. Per-position conservation score [0, 1]. Could be included as a per-residue feature.
    *   `contact_scores`: `float32`, shape `(N, N)`. Normalized version of `coupling_matrix`, potentially useful. Clarify if this or `coupling_matrix` is preferred. *Assumption for V1: Use `coupling_matrix` directly.*

*   **Metadata Keys (Not direct tensor input):**
    *   `top_contacts`: List of pairs.
    *   `score_distance_correlation`, `precision_L/X`: Validation metrics (only present if 3D structure was available during feature extraction).

## 4. Integration and Loading Notes

*   The `DataLoader` (`src/data_loading.py`) will be responsible for:
    *   Identifying the correct `.npz` file paths for a given `target_id` based on the conventions above.
    *   Loading the required arrays (keys) from each file.
    *   Handling cases where optional files (like `mi_features`) might be missing, providing appropriate default tensors (e.g., zero matrices/vectors of the correct expected shape).
    *   Handling potential NaNs in the loaded `dihedral_features['features']` array.
    *   Combining these features into a single dictionary per sample, ready for the `collate_fn`.
    *   Loading corresponding ground truth coordinates (`x_1, y_1, z_1`) from the relevant labels CSV (`train_labels.csv` or `validation_labels.csv`).
    *   Loading the raw sequence string.

## 5. Summary Table of Key V1 ML Inputs

| Feature Category      | File Suffix                       | Key ML Input(s)                      | Shape        | Data Type   | Notes                                             |
| :-------------------- | :-------------------------------- | :----------------------------------- | :----------- | :---------- | :------------------------------------------------ |
| **Per-Residue**       |                                   |                                      |              |             |                                                   |
| Dihedral (Sin/Cos)    | `_dihedral_features.npz`          | `features`                           | `(N, 4)`     | `float32`   | Range [-1, 1]. Check/Handle NaNs. Training only. |
| Thermo (Entropy)      | `_thermo_features.npz`            | `positional_entropy`                 | `(N,)`       | `float32`   | Range [0, log2(4)]                               |
| Thermo (Accessibility)| `_thermo_features.npz`            | `accessibility`                      | `(N,)`       | `float32`   | Range [0, 1]                                      |
| Evolutionary (Consrv) | `_features.npz` (in `mi_features/`) | `conservation`                       | `(N,)`       | `float32`   | Range [0, 1]. Optional V1 input. Check availability. |
| **Pairwise**          |                                   |                                      |              |             |                                                   |
| Thermo (Pair Probs)   | `_thermo_features.npz`            | `pairing_probs`                      | `(N, N)`     | `float32`   | Range [0, 1]. Symmetric.                         |
| Evolutionary (MI)     | `_features.npz` (in `mi_features/`) | `coupling_matrix`                    | `(N, N)`     | `float32`   | Range [0, inf). Symmetric. Check availability.  |
| **Global Scalars**    | `_thermo_features.npz`            | `mfe`, `ensemble_energy`, ...        | `()`         | `float32`   | See Sec 3.2. Broadcast/append in model.           |
| **Sequence**          | (From `sequences.csv`)            | Raw Sequence String                  | `(N,)` chars | `string`    | Convert to one-hot/embedding in model.            |
| **Labels**            | (From `labels.csv`)               | C1' Coordinates (`x_1,y_1,z_1`)      | `(N, 3)`     | `float32`   | Ground truth for training/validation.           |

This specification provides the necessary detail for implementing the `DataLoader` to correctly process the available precomputed features for the V1 model.
