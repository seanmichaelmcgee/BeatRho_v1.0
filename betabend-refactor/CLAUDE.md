# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands
- Run validation: `python validation/validation_runner.py --model_path /path/to/checkpoint --output_dir /path/to/results`
- Test device handling: `python scripts/test_device_handling.py`
- Test checkpoint: `python scripts/test_checkpoint.py /path/to/checkpoint.pt`
- Generate validation comparison: `python scripts/generate_validation_comparison.py --run1 /path/to/run1 --run2 /path/to/run2`
- Monitor GPU usage: `python scripts/monitor_gpu.py`

## Code Style Guidelines
- **Indentation**: 4 spaces, no tabs
- **Imports**: Group as (1) standard library, (2) third-party, (3) local modules
- **Types**: Use comprehensive type annotations with `Optional`, `Union`, `List`, `Dict`, `Tuple`
- **Naming**: CamelCase for classes, snake_case for functions/variables
- **Private methods**: Use underscore prefix (e.g., `_init_weights`)
- **Error handling**: Use specific exception types with context-rich messages
- **Documentation**: Module docstrings + function docstrings with Args/Returns sections
- **Tensor handling**: Always validate tensor shapes and check device placement
- **Logging**: Use Python's logging module with appropriate severity levels