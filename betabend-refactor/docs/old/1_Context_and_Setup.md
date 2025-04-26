# Project Context and Setup: RNA 3D Folding - PyTorch Implementation

## 1. Project Goal and Development Philosophy

**Objective:** Develop a robust and high-performing PyTorch-based machine learning pipeline for RNA 3D structure prediction. The primary target is to create a model capable of achieving competitive results in the Stanford RNA 3D Folding Kaggle competition.

**Core Development Principles:**

*   **Reproducibility:** Ensure every step of the pipeline, from data loading to model prediction, is reproducible across different runs and environments. This is achieved through containerization and environment management.
*   **Containerization (Docker):**  The entire development and deployment workflow is centered around Docker containers. This guarantees consistent execution across local development, validation/testing, and the final Kaggle submission environment.
*   **Environment Management (Mamba/Conda):** Use Mamba (or Conda) and `environment.yml` to precisely define and manage software dependencies, including Python versions, PyTorch, CUDA, and other libraries. This ensures environment mirroring between local and cloud.
*   **Modularity:** Structure the codebase into modular components (`src/data_loading.py`, `src/models/`, `src/losses.py`, `scripts/`) for maintainability, testability, and easier extension.
*   **Test-Driven Development:** Implement unit tests for all core modules and functionalities to ensure correctness, prevent regressions, and facilitate future refactoring.
*   **Configuration-Driven:** Manage hyperparameters, file paths, and environment settings through a central configuration file (`config/default_config.yaml`) and command-line arguments, avoiding hardcoded values.
*   **Stateless Execution:** Design pipeline components to be stateless. Avoid saving runtime state within containers. Log all artifacts (checkpoints, logs, metrics) to mountable volumes.
*   **Strict Path Parameterization:** A key principle is strict path parameterization: core logic modules in `src/` **must** receive necessary file/directory paths as arguments from the orchestrating layer (scripts/notebooks) and contain **no hardcoded paths**. This ensures portability and enables seamless execution across different environments with varying mount points or fixed paths (e.g., local Docker vs. Kaggle).


## 2. Development Environment Setup

**2.1. Local Development Workstation (`Dev` Environment)**

*   **Hardware Specifications:** (Refer to `2.Technical-Specs-Prototype-Machine-and-Project-v1.md` for detailed specifications)
    *   **Processor:** Intel Core i9 (14th Gen) or equivalent.
    *   **GPU:** NVIDIA RTX 4070 Ti (16GB VRAM). This is the primary GPU target for local development and initial testing.
    *   **Memory:** 128 GB DDR5 RAM.
    *   **Storage:** 4 TB NVMe SSD (Gen 4).
*   **Software Environment (using Docker and Mamba):**
    1.  **Install Docker:** Ensure Docker is installed and configured on your development machine.
    2.  **Build Docker Image:** Navigate to the project root directory (containing `Dockerfile`) and build the Docker image:
        ```bash
        docker build -t rna-3d:dev .
        ```
        This Dockerfile (defined in project root) will:
        *   Use a suitable NVIDIA CUDA base image.
        *   Install Miniconda/Mamba within the container.
        *   Create a Conda environment named `rna-3d-env` inside the container based on `environment.yml`. This `environment.yml` pins specific versions of Python, PyTorch, CUDA toolkit, cuDNN, and other dependencies, ensuring consistent environments.
        *   Copy project code into the container.
        
    3.  **Run Development Container:** Launch a Docker container from the built image, mounting your local project directory and data directories for development:
        ```bash
        docker run --rm -it --gpus all \
          -v $(pwd):/app  # Mount current project directory
          -v /path/to/local/data:/app/data  # Mount your local data directory to /app/data inside container
          rna-3d:dev /bin/bash # Start a bash shell inside the container
        ```
        *   `-v $(pwd):/app`: Mounts your current project directory on your host machine to `/app` inside the container. Code changes in your local project will be immediately reflected inside the container.
        *   `-v /path/to/local/data:/app/data`: Mounts your local data directory to `/app/data` within the container. Replace `/path/to/local/data` with the actual path to your data on your machine.
        *   `--gpus all`: Makes all GPUs available inside the container.
        *   `/bin/bash`: Starts an interactive bash shell within the container, allowing you to execute commands and develop within the consistent environment.
    4.  **Activate Conda Environment Inside Container:** Once inside the Docker container's bash shell, activate the `rna-3d-env` Conda environment:
        ```bash
        conda activate rna-3d-env
        ```
        You are now working within a consistent, reproducible PyTorch development environment inside the Docker container. Use your preferred IDE (e.g., VS Code connected to the Docker container) to edit code within the mounted project directory (`/app`).

**2.2. Validation/Testing (`Test` Environment) & Production (`Prod` Environment - Kaggle)**

*   The Docker container prepared for the `Dev` environment is designed to be directly portable to `Test` (local validation/testing) and `Prod` (Kaggle submission) environments.
*   For `Test`, you would typically run the same Docker image, potentially mounting different datasets and output directories, to execute validation scripts and performance evaluations.
*   For `Prod` (Kaggle), your Kaggle Notebook submission will essentially replicate a similar containerized execution environment.
*   The key is that the code *within* the container, when executed with the same configuration and data, should produce consistent and reproducible results across all these environments.

## 3. Project Directory Structure

```
rna_3d_project/
├── Dockerfile                 # Dockerfile for building containerized environment
├── environment.yml            # Mamba/Conda environment definition
├── config/                    # Configuration files (e.g., default_config.yaml)
│   └── default_config.yaml
├── data/                      # Local data directory (typically .gitignored) - mount point for data within container
│   ├── raw/                   # Raw input data (CSVs)
│   │   ├── train_sequences.csv
│   │   ├── train_labels.csv
│   │   └── ...
│   ├── processed/             # Directory for precomputed features (output of feature extraction pipeline)
│   │   └── ...
│   └── ...
├── notebooks/                 # Jupyter Notebooks for exploration, prototyping, and debugging
│   ├── exploration.ipynb
│   └── ...
├── scripts/                   # Python scripts for command-line execution of pipeline stages
│   ├── train.py               # Script to train the RNA folding model
│   └── predict.py             # Script to run inference and generate submission file
├── src/                       # Source code directory - core modules of the project
│   ├── __init__.py
│   ├── data_loading.py        # PyTorch Dataset, DataLoader, data processing logic
│   ├── losses.py              # PyTorch loss functions
│   ├── models/                # PyTorch model components
│   │   ├── __init__.py
│   │   ├── embeddings.py      # Embedding layers
│   │   ├── transformer_block.py # Transformer block implementation
│   │   ├── ipa_module.py      # Invariant Point Attention module (or placeholder)
│   │   └── rna_folding_model.py # Main RNA folding model definition
│   ├── training_loop.py       # (Optional) Helper functions for training loop logic
│   └── utils/                 # Utility modules (logging, metrics, etc.)
├── tests/                     # Unit tests for source code modules
│   ├── test_data_loading.py
│   ├── test_model.py
│   └── ...
└── README.md                  # Project README file
```

*   **`config/`**: Stores configuration files (e.g., `default_config.yaml`) to manage hyperparameters and settings.
*   **`data/`**: Root directory for all data. Subdirectories: `raw/` (input CSVs), `processed/` (precomputed features). This directory is intended to be mounted from your local machine into the Docker container.
*   **`notebooks/`**: Contains Jupyter Notebooks for interactive exploration, prototyping, and debugging. Keep notebooks lean and primarily for calling functions from `src/`.
*   **`scripts/`**: Houses Python scripts for command-line execution of the main pipeline stages (`train.py`, `predict.py`). These are the entry points for running the pipeline within Docker and in the cloud.
*   **`src/`**: Contains the core Python source code, organized into modules:
    *   `data_loading.py`: PyTorch `Dataset`, `DataLoader`, and data processing logic.
    *   `losses.py`: PyTorch loss functions.
    *   `models/`: Subdirectory for all model-related code, further organized into:
        *   `embeddings.py`: Embedding layers.
        *   `transformer_block.py`: Implementation of the Transformer block.
        *   `ipa_module.py`: Invariant Point Attention module (or placeholder).
        *   `rna_folding_model.py`: Main model definition, integrating all components.
    *   `training_loop.py`: (Optional) Helper functions to structure the training loop in `scripts/train.py`.
    *   `utils/`: Utility functions (e.g., for logging, metrics calculation).
*   **`tests/`**: Contains Pytest unit tests, mirroring the structure of `src/`. Write tests for each module in `src/` to ensure code correctness.

## 4. Development Workflow Overview

1.  **Local Development (`Dev` Environment):**
    *   Perform the majority of coding, feature implementation, and unit testing within the Docker containerized `Dev` environment (as described in Section 2.1).
    *   Use Jupyter Notebooks in `notebooks/` for exploratory data analysis and prototyping, but ensure core logic resides in `src/` modules.
    *   Run unit tests frequently to validate code changes.
    *   Iterate on model architecture and training procedures based on the tactical implementation plan.

2.  **Validation and Testing (`Test` Environment):**
    *   Once core components are implemented and unit-tested, move to integration testing and validation.
    *   Run `scripts/train.py` and potentially `scripts/predict.py` within the Docker container (or a similar containerized environment) using validation datasets.
    *   Evaluate model performance using appropriate metrics (e.g., TM-score - to be integrated later).
    *   Debug and refine the pipeline based on validation results.

3.  **Kaggle Submission (`Prod` Environment):**
    *   For final submission to the Kaggle competition, prepare your `predict.py` script to run within the Kaggle Notebook environment, ensuring it adheres to Kaggle's code requirements and runtime limits.
    *   Test your submission notebook thoroughly before final submission.

## 5. Scalability Considerations (Future Multi-GPU Readiness)

*   While the initial development is focused on single-GPU training on the local workstation, the project is designed with future multi-GPU scalability in mind (for potentially improved model performance and handling larger datasets in later phases).
*   Key design choices to support future multi-GPU training (using PyTorch's DistributedDataParallel - DDP):
    *   **Stateless Pipeline:** Components are designed to be stateless, facilitating parallel execution across multiple GPUs.
    *   **No Hardcoded Device IDs:** Code should not contain hardcoded GPU device IDs (e.g., `device=0`). Use a configurable `device` parameter and `torch.device` objects.
    *   **Configurable Batch Sizes:** Design the model and data loading to easily adjust batch sizes to fit available GPU memory when scaling to multi-GPU setups.
    *   **Modular Code Structure:**  The modular structure facilitates wrapping model components with `DistributedDataParallel` if needed.
    *   **Parameterization:** All hardware-specific configurations (GPU flags, number of GPUs, etc.) should be managed through configuration files or command-line arguments, not inlined in code.
*   **Note:**  **Implementing multi-GPU training is NOT in the scope of the current implementation phase.** The focus is on building a functional single-GPU pipeline first.  However, the *design* should inherently support future scaling.

## 6. Essential External References

*   **Kaggle Competition Overview (`Kaggle_Overview.md`):** Understand the competition goals, evaluation metric (TM-score), timeline, and prizes.
*   **Kaggle Data Description (`Kaggle_Data.md`):**  Detailed information about the training, validation, and test datasets, file formats, and temporal cutoffs.

By adhering to these guidelines and utilizing the provided structure, you will establish a solid foundation for developing a competitive RNA 3D folding prediction pipeline in PyTorch.
