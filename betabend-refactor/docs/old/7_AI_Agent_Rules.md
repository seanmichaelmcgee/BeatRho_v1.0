# RULES.md: AI Agent Development Guidelines for RNA Folding Project v1.2

## Preamble

These rules guide the AI agent's development process for the Stanford RNA 3D Folding Kaggle project. The primary goal is to ensure focused, high-quality, test-driven, reproducible, and maintainable code development optimized for **both local iteration and seamless Kaggle Notebook execution**. Adherence to these rules is crucial.

## 1. Environment Awareness & Scope

1.1. **Environment Definitions:** Understand the project's operational environments:
    *   **`Dev` (Development):** Local workstation (`2.Technical-Specs...`), focus on implementation, testing, debugging. Clarity and correctness are paramount.
    *   **`Test` (Testing/Validation):** Integration testing, validation runs, performance checks (local or containerized).
    *   **`Prod` (Production/Submission):** Kaggle Notebook environment. Focus on reproducibility, adherence to format/runtime limits. Multi-GPU implementation is **out of scope** for V1/V2.
1.2. **Current Phase Focus:** Primarily **`Dev`**, moving towards **`Test`**. Prioritize code clarity, correctness, and testability, while **strictly adhering** to design choices enabling `Prod` (Kaggle) execution.
1.3. **Future-Proofing (Design Only):** Design for potential multi-GPU (DDP) scaling (statelessness, no hardcoded devices, config-driven) per `1.Proj-Cont-and-Setup...`, but **do not implement** DDP logic now.
1.4. **Hardware Constraints:** Be mindful of local 16GB GPU VRAM. Recommend initial configurations accordingly.

*   **Rationale:** Tailors behavior to the current stage while ensuring compatibility with the ultimate Kaggle deployment environment.

## 2. Task Focus, Modularity & PRD Alignment

2.1. **Single Task Focus:** Work on **one** specific, well-defined task at a time, linked to the `Product Requirements Document (PRD)`.
2.2. **Clarify Scope:** If a task is ambiguous or broad, request breakdown.
2.3. **PRD-Driven Development:**
    *   **Consult PRD First:** Review relevant requirement IDs before starting.
    *   **Continuous Reference:** Cross-reference implementation against PRD requirements.
    *   **Flag Deviations:** Stop and seek user approval if deviating from the PRD.
2.4. **Strict Modularity for Reusability:**
    *   **Core Logic in `src/`:** Implement **all** core data processing logic, model components, loss calculations, and utility functions exclusively within modules in the `src/` directory (`src/data_loading.py`, `src/models/`, `src/losses.py`, `src/utils/`).
    *   **Orchestration in `scripts/` & Notebooks:** Treat files in `scripts/` (`train.py`, `predict.py`) and any Jupyter Notebooks (`notebooks/`) primarily as **orchestrators**. They should **import** functions/classes from `src/` and handle workflow control (argument parsing, setup, calling `src/` functions in sequence, basic logging/output) but contain minimal complex algorithmic logic themselves.
    *   **Rationale:** This separation is **critical** for Kaggle. It allows the core, tested logic in `src/` to be easily imported and called from notebook cells, mirroring how it's called from `scripts/`, thus minimizing code duplication and ensuring consistency between local testing and Kaggle execution.

## 3. Test-Driven/Concurrent Development

3.1. **Test Concurrently:** Write unit tests in `tests/` for **every** new function/class/logic block in `src/` **concurrently** with implementation.
3.2. **Test Coverage:** Cover core functionality, expected outputs, edge cases, error handling.
3.3. **Test Location:** `tests/` directory mirrors `src/`.
3.4. **No Feature Without Tests:** Implementation is incomplete until tests pass.

*   **Rationale:** Ensures correctness of `src/` modules, enabling reliable orchestration.

## 4. Execution, Verification & Debugging

4.1. **Run Tests:** Execute relevant `pytest` commands after implementing features and tests.
4.2. **Ensure Passing:** Confirm tests pass in the `Dev` environment (Docker container).
4.3. **Debug Failures:** Systematically debug code/tests until passing.

*   **Rationale:** Verifies `src/` module correctness before integration/orchestration.

## 5. Commit Prompting

5.1. **Prompt on Success:** **Immediately after** verifying a feature/module in `src/` passes its tests, prompt the user to review and commit.
5.2. **Suggest Message:** Provide a Conventional Commits style message.

*   **Rationale:** Encourages atomic commits tied to tested `src/` components.

## 6. Code Hygiene & Temporary Code

6.1. **Identify Temporary Code:** Clearly mark temporary validation snippets.
6.2. **Temporary Script Validation:** Execute as needed.
6.3. **Prompt for Cleanup:** Prompt user for disposition (delete/integrate) after use, recommending deletion.

*   **Rationale:** Keeps codebase clean.

## 7. Configuration & Path Management (CRITICAL for Kaggle)

7.1. **Use Config File:** Manage hyperparameters, settings via `config/default_config.yaml`.
7.2. **Strict Path Parameterization:**
    *   **NO Hardcoded Paths in `src/`:** Absolutely **no** hardcoded file paths (e.g., `/app/data`, `../data`, `data/raw`) should exist within any module inside the `src/` directory.
    *   **Pass Paths via Args/Config:** All required paths (input data directories, feature directories, model checkpoints, output files/directories) must be passed into the relevant `src/` functions/classes as arguments.
    *   **Orchestrator Responsibility:** The orchestrating layer (`scripts/train.py`, `scripts/predict.py`, or notebook cells) is responsible for determining the correct paths (e.g., reading from the config file, parsing CLI arguments, using Kaggle's fixed paths like `/kaggle/input/...` and `/kaggle/working/...`) and passing them down to the `src/` components.
    *   **Example:** A function in `src/data_loading.py` should accept `features_dir` as an argument, not assume `data/processed`. The `scripts/train.py` script would read the path from config/args and pass it to the function.
    *   **Rationale:** This decoupling is essential for running the same `src/` code unchanged in different environments (local Docker with mounted volumes vs. Kaggle Notebook with fixed input/output paths).

## 8. Reproducibility & Output Formatting

8.1. **Environment Consistency:** Use pinned `environment.yml` and `Dockerfile`.
8.2. **Seed Usage:** Use and parameterize random seeds for stochastic operations.
8.3. **Strict Output Formatting:**
    *   **Adhere to Kaggle Spec:** The logic responsible for generating the final `submission.csv` (likely within `src/utils/` or called by `scripts/predict.py`) **must** precisely match the format specified in `Kaggle_Data.md` (columns: `ID,resname,resid,x_1,y_1,z_1,...,x_5,y_5,z_5`).
    *   **Implement & Test Early:** Implement and unit-test the output formatting logic relatively early in the process, even using dummy prediction data, to ensure correctness. Do not leave this solely for the final 'Prod' stage check.
    *   **Rationale:** Ensures the final output is directly submittable to Kaggle without formatting errors.

## 9. Library-Specific Rules & Best Practices (PyTorch, Pandas, NumPy)

9.1. **Consult Documentation & Guides:** Refer to official docs and guides like awesome-cursorrules.
9.2. **PyTorch Specifics:** Use `.to(device)`, prefer vectorization, use `torch.no_grad()`, configure `DataLoader` appropriately, build `nn.Module`s, be mindful of memory/views.
9.3. **Pandas Specifics:** Use vectorization, `.loc`/`.iloc`, chaining, watch for `SettingWithCopyWarning`.
9.4. **NumPy Specifics:** Use vectorization, broadcasting, be aware of views vs. copies.

*   **Rationale:** Promotes efficient, maintainable, standard code.

## 10. Communication & Clarification

10.1. **Ask When Unsure:** Ask the user for clarification on rules, tasks, PRD items before proceeding.
10.2. **Provide Context:** Frame questions with specific context.

*   **Rationale:** Ensures mutual understanding and prevents wasted effort.
