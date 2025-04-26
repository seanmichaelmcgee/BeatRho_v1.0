# Architecture Specification: RNA 3D Structure Prediction (vApr9)

## Introduction

This proposal outlines a comprehensive neural architecture for end-to-end RNA 3D structure prediction, implemented in PyTorch. It integrates rich thermodynamic and evolutionary features with a Transformer-based backbone. The design meets the specified constraints by using proven pre-September 18, 2024 techniques (inspired by methods like AlphaFold and RNA secondary structure algorithms) while avoiding any post-cutoff innovations. We describe the model’s inputs, feature embeddings, backbone network, 3D structure generation module, output format, training losses, and deployment strategy. The goal is a Dockerized, scalable system that predicts multiple candidate RNA conformations (with confidences) from sequence alone, leveraging secondary structure thermodynamics, mutual information from sequence covariation, and potentially pseudo-dihedral angle supervision as auxiliary knowledge during training.

## Inputs and Feature Precomputation

### Overview
For each input RNA sequence, we first compute a variety of sequence-based features that serve as the model’s inputs. These include thermodynamic secondary structure features, evolutionary couplings (mutual information), basic sequence descriptors, and pseudo-dihedral angle annotations (for training only, if available). All features are either per-residue (1D) or per-residue-pair (2D) and encapsulate known RNA folding information using pre-cutoff techniques.

### Thermodynamic Secondary Structure Features
Leveraging RNA secondary structure prediction tools (e.g., ViennaRNA RNAfold) based on the Turner energy model, we compute:

*   **Minimum Free Energy (MFE):** Scalar energy (kcal/mol) of the most stable predicted 2D structure.
*   **Ensemble Free Energy:** Scalar energy of the thermodynamic ensemble (reflects overall stability).
*   **Energy Gap:** Scalar difference between MFE and ensemble energy (indicates fold definition).
*   **MFE Structure Probability:** Boltzmann probability of the MFE structure (confidence in the dominant pairing).
*   **MFE Pairing Status:** Binary indicator per nucleotide (1 if paired in MFE structure, 0 if unpaired). Derived from the MFE dot-bracket structure.
*   **Base Pairing Probability Matrix:** An $N \times N$ matrix $P_{ij}$ giving the probability that nucleotide *i* and *j* form a pair in the equilibrium ensemble (captures alternative foldings).
*   **Positional Entropy:** Shannon entropy of the base-pairing distribution for each nucleotide *i* (measures local structural uncertainty).
*   **Accessibility:** Probability that each nucleotide *i* is unpaired in the ensemble (potentially windowed, e.g., via RNAplfold).
*   **Global Descriptors:** Mean/variance of accessibility, sequence GC content, fraction of paired bases in MFE structure.
*   **(Optional) Derived Features:** Average pair distance, free energy per nucleotide.

These features provide a robust summary of 2D folding tendencies and are crucial for informing the 3D prediction.

### Evolutionary Coupling Feature (Mutual Information)
To capture coevolution signals while adhering to the knowledge cutoff:

*   **Input:** Multiple Sequence Alignment (MSA) of homologous RNAs.
*   **Method:** Calculate the **Mutual Information (MI)** matrix ($N \times N$). For each pair of positions (*i*, *j*), MI measures the reduction in uncertainty about nucleotide *j* given knowledge of nucleotide *i*, indicating potential coupling.
*   **Corrections:** Apply standard corrections (e.g., sequence weighting for phylogenetic bias, Average Product Correction - APC - for background noise) to highlight true covariation.
*   **Constraint:** Use only the corrected MI matrix. **Do not use** more complex Direct Coupling Analysis (DCA) methods or post-processed coupling scores popular after the mid-2010s.

### Single-site Sequence Profile Features
From the MSA:

*   **Frequency Profiles:** Per-position frequencies of A, C, G, U.
*   **Conservation:** Per-position Shannon entropy (low entropy indicates high conservation).

If no MSA is available, use a one-hot encoding of the sequence itself.

### Pseudo-dihedral Angle Annotations (Training only)
For training sequences with known 3D structures:

*   **Calculation:** Compute standard pseudo-torsion angles $\eta$ (eta) and $\theta$ (theta) defined by four consecutive backbone atoms (P and C4′).
*   **Purpose:** These angles describe backbone conformation (helix vs. loop).
*   **Usage:** Used as **auxiliary information during training only**. They are *not* provided as input during inference on unknown structures. Strategies include multi-task learning (predicting angles) or teacher-student distillation.

### Basic Sequence Features
*   **Encoding:** One-hot encoding or a trainable embedding vector (e.g., dimension 16) for each nucleotide (A, C, G, U).
*   **Positional Encoding:** Sine-cosine positional embeddings or normalized position index to inform the model of sequence order and distance. Relative positional encodings can also be used.

### Feature Precomputation Workflow
All features are computed in a preprocessing stage (CPU-based, containerized, cached). Results (e.g., `.npz` files per sequence) contain per-residue vectors and per-pair matrices ready for the neural network.

## Feature Embedding and Representation Initialization

Before entering the main model, features are embedded into initial residue (1D) and pair (2D) representations:

*   **Residue Embeddings ($h_i^{(0)}$):**
    *   Concatenate per-residue features: base embedding/one-hot, MSA frequency profile (if available), MFE pairing flag, positional entropy, accessibility, relevant global scalars (broadcasted).
    *   If using pseudo-dihedral angles during training (e.g., for multi-task loss), they are *not* included here but predicted later. If using a Teacher model that *does* see angles, embed them as $\sin/\cos$ for the Teacher.
    *   Project the concatenated vector to the residue embedding dimension ($d_{\text{res}}$, e.g., 128) using a linear layer (`nn.Linear`).
    *   Add absolute positional encoding.
*   **Pair Embeddings ($P_{ij}^{(0)}$):**
    *   Concatenate per-pair features: corrected Mutual Information score, base-pairing probability $P_{ij}$. Add a binary MFE pairing indicator.
    *   Encode sequence separation $|i - j|$ using a learned relative positional embedding.
    *   Project the concatenated vector to the pair embedding dimension ($d_{\text{pair}}$, e.g., 64) using a linear layer.
    *   Result is an $N \times N \times d_{\text{pair}}$ tensor $P^{(0)}$. Ensure symmetry ($P_{ij} = P_{ji}$).

## Backbone Model: Transformer-Based Multimodal Fusion

The core is a stack of $L$ identical fusion blocks ($L \approx 8-16$) that iteratively refine residue ($h$) and pair ($P$) representations.

### Each Fusion Block:

1.  **Residue Update with Pair-aware Attention:**
    *   Update $h_i$ using multi-head self-attention over all $h_j$.
    *   **Pair Bias (Concept):** The attention mechanism is biased by the current pair representation $P_{ij}$. This can be implemented by projecting $P_{ij}$ to a scalar or per-head bias $b_{ij}$ and adding it to the attention logits before the softmax:
        $$ \alpha_{ij} \propto \exp\left(\frac{(h_i W_Q)\cdot(h_j W_K)^T}{\sqrt{d_k}} + b_{ij}\right) $$
        *(Note: V1 implementation may start with standard MHA without explicit bias).*
    *   The output of attention is passed through a position-wise Feed-Forward Network (FFN).
    *   Layer normalization and residual connections are used throughout.

2.  **Pair Representation Update (Simplified):**
    *   Update $P_{ij}$ using information from the updated residue embeddings $h_i, h_j$ and potentially neighboring pair features (simplified triangle update). **Avoids** full triangle attention/multiplication from AlphaFold2.
    *   **Outer Product Style:** Combine $h_i$ and $h_j$ to influence $P_{ij}$. Example: Concatenate $h_i$, $h_j$, and the current $P_{ij}$ and pass through an MLP:
        $$ P_{ij}^{\text{update}} = \text{MLP}(\text{concat}(h_i, h_j, P_{ij})) $$
    *   **(Optional) Simplified Triangle Update:** Aggregate information from paths $i \to k \to j$. Example additive message passing:
        $$ m_{ij} = \sum_k \text{Activation}(\dots P_{ik} \dots P_{kj} \dots) $$
        Then $P_{ij} \leftarrow \text{LayerNorm}(P_{ij} + P_{ij}^{\text{update}} + (\text{optional } m_{ij}))$.
    *   Pass updated $P_{ij}$ through a small FFN. Enforce symmetry.

### Iteration:
Stacking $L$ blocks allows iterative refinement, capturing local secondary structure in early layers and long-range tertiary contacts in later layers. Gradient checkpointing can be used here to manage memory for large $L$.

## Structure Prediction Module (3D Coordinate Generation)

Converts final representations ($h^{(L)}, P^{(L)}$) into 3D coordinates ($x_i$ for C1' atoms). Aiming for direct coordinate generation:

*   **(Target Approach) Invariant Point Attention (IPA) based Coordinate Regression:**
    *   **Concept:** Iteratively refine residue coordinates and orientations using a spatially aware attention mechanism (IPA) that is invariant/equivariant to global rotations and translations.
    *   **Initialization:** Start with initial coordinates (e.g., random, linear chain, or zero) and frames (e.g., identity rotation at origin).
    *   **IPA Cycles:** Repeat for several cycles (e.g., 8 times):
        1.  **IPA Attention:** Attend between residues $i$ and $j$ using embeddings ($h_i, h_j$), pair features ($P_{ij}$), and current relative 3D positions/orientations in an invariant way. Produces updated embeddings.
        2.  **Equivariant Update:** Use updated embeddings to predict updates ($\Delta x_i$, $\Delta R_i$) to each residue's frame (position $x_i$, rotation $R_i$). Apply updates.
    *   **Output:** Final coordinates $x_i$ of C1' atoms after the last cycle.
    *   **V1 Implementation:** Start with a **placeholder** module (e.g., `nn.Linear` predicting $x_i$ directly from $h_i^{(L)}$) and integrate a full IPA implementation later.

*   **(Fallback/Alternative) Distance Matrix Regression + Distance Geometry:**
    *   Predict pairwise distances $\hat{d}_{ij}$ from $P_{ij}^{(L)}$ (using an MLP head, potentially predicting distance bins/distograms).
    *   Use a distance geometry algorithm (e.g., gradient descent optimization, MDS) to find coordinates $x_i$ that best satisfy the predicted distances $\hat{d}_{ij}$, potentially constrained by known chain connectivity distances (~5.9 Å for adjacent C1').
    *   Simpler to implement initially but may lead to inconsistencies if predicted distances aren't perfectly metric.

## Loss Functions

Train end-to-end using a combination of losses:

*   **Coordinate/Structure Loss (Primary):**
    *   **Frame Aligned Point Error (FAPE):** Preferred loss if using IPA. Compares predicted local atomic positions to true positions within aligned local reference frames. Invariant to global rotation/translation.
    *   **(V1 Proxy) Clamped L2/L1 Loss:** Simpler alternative. Calculate L1 or L2 distance between predicted ($x_i$) and true ($x_i^{\text{true}}$) C1' coordinates after optimal global alignment (e.g., Kabsch algorithm). Clamp large distance errors to improve robustness.
    *   **lDDT-based Loss:** Minimize `1 - lDDT` score (local Distance Difference Test), which measures preservation of local distances. Can be made differentiable.
*   **Auxiliary Losses (for Regularization & Intermediate Supervision):**
    *   **Pseudo-dihedral Angle Loss:** If using multi-task learning. Compare predicted angles (from `angle_prediction_head`) with true angles using a suitable periodic loss (e.g., `1 - cos(angle_diff)` on $\eta, \theta$).
    *   **Confidence Loss:** Train the `confidence_head` prediction (pLDDT proxy) to match the actual quality (e.g., true lDDT score) of the model's prediction on the training sample. Use MSE or BCE loss.
    *   **(Optional) Distogram Loss:** If predicting distance distributions, use cross-entropy against true distance bins.
    *   **(Optional) Contact Prediction Loss:** Add a head to predict binary contacts (< 8Å or < 12Å) from $P_{ij}^{(L)}$ and train with cross-entropy.

Total loss is a weighted sum: $\mathcal{L} = w_{\text{structure}} \mathcal{L}_{\text{structure}} + w_{\text{confidence}} \mathcal{L}_{\text{confidence}} + w_{\text{angle}} \mathcal{L}_{\text{angle}} + \dots$

## Output Format and Model Confidence

*   **Output:** For each sequence, predict **5 candidate 3D structures** (sets of C1' coordinates $x_1, ..., x_5$).
    *   **Generation Strategy:** Achieve diversity via multiple inference runs with dropout enabled, multiple random seeds for IPA initialization, or sampling from predicted distributions.
*   **Confidence Score:** Predict a per-residue confidence score (e.g., pLDDT proxy via `confidence_head`, trained using $\mathcal{L}_{\text{confidence}}$). Average per-residue scores for an overall model confidence, used potentially for ranking the 5 outputs.
*   **Format:** Generate `submission.csv` matching Kaggle format: `ID, resname, resid, x_1, y_1, z_1, ..., x_5, y_5, z_5`. Coordinates are for C1' atoms. Output structures might be centered at origin; global alignment for scoring is handled by Kaggle's backend (US-align).

## Training Strategy and Auxiliary Techniques

*   **Dataset:** Train on known RNA 3D structures (PDB, RNA Puzzles, Kaggle provided training data). Use temporal cutoffs rigorously. Hold out a validation set.
*   **Angle Handling (Multi-task):** Train the main model (without angle inputs) to simultaneously predict coordinates AND angles (via `angle_prediction_head` and $\mathcal{L}_{\text{angle}}$). This forces the model to internalize angle information.
*   **(Alternative) Angle Handling (Teacher-Student):** Train a "Teacher" model *with* angle inputs. Train a "Student" model (no angle inputs) to mimic the Teacher's outputs (e.g., predicted coordinates) using a distillation loss, in addition to supervising against ground truth. (More complex, potential V2+ approach).
*   **Data Augmentation:** Add noise to input features (e.g., thermodynamic predictions) or mask out features (e.g., MI matrix) during training to improve robustness.
*   **Long Sequences:** Use gradient checkpointing in Transformer blocks for memory efficiency. Consider sequence chunking or sparse attention mechanisms if needed for very long RNAs (potential V2+).
*   **Optimization:** Use AdamW optimizer with a learning rate scheduler (e.g., cosine annealing or reduce on plateau).

## Deployment and Engineering Considerations

*   **Dockerization:** Entire pipeline (preprocessing, training, inference) containerized using Docker and `environment.yml` for reproducibility (`1_Context_and_Setup.md`).
*   **Hardware:** PyTorch implementation targets GPU acceleration (`.to(device)`). Designed to run on local dev GPU (RTX 4070 Ti 16GB) for V1 (scaled-down), potentially scaling to larger cloud GPUs later.
*   **Scalability (Design):** Code structured for potential multi-GPU training using DDP (stateless, parameterized), but implementation deferred.
*   **Modularity:** Code organized into `src/data_loading`, `src/models`, `src/losses`, etc., orchestrated by `scripts/`.
*   **Knowledge Cutoff:** Strict adherence to pre-September 18, 2024 techniques. Core ideas (Transformers, MI, Thermodynamics, IPA concepts) satisfy this. No reliance on newer foundation models or DCA variants.
*   **Testing:** Unit tests for modules, integration tests for pipeline flow, validation against held-out set using TM-score (via external tool initially). Potential final light geometry refinement if needed.

This architecture provides a strong foundation for tackling the RNA 3D folding problem within the specified constraints, balancing proven techniques with modern deep learning capabilities.

