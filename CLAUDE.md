# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CVXPYlayers is a Python library for constructing differentiable convex optimization layers in PyTorch, JAX, and MLX using CVXPY. It solves parametrized convex optimization problems in the forward pass and computes gradients via implicit differentiation in the backward pass.

## Build and Development Commands

```bash
# Install in development mode with all dependencies
pip install -e ".[all]"

# Install for specific framework
pip install -e ".[torch]"   # PyTorch only
pip install -e ".[jax]"     # JAX only
pip install -e ".[mlx]"     # MLX only

# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_torch.py

# Run single test
pytest tests/test_torch.py::test_example -v

# Lint and format (via pre-commit)
pre-commit run --all-files

# Run ruff directly
ruff check src/ tests/
ruff format src/ tests/
```

## Architecture

### Source Layout (`src/cvxpylayers/`)

- **Framework-specific layers** (`torch/`, `jax/`, `mlx/`): Each contains a `cvxpylayer.py` implementing `CvxpyLayer` for that framework. The PyTorch layer extends `torch.nn.Module`; JAX and MLX layers are callable classes.

- **Solver interfaces** (`interfaces/`): Abstractions for different solver backends:
  - `diffcp_if.py` - Default CPU solver using diffcp
  - `cuclarabel_if.py` - GPU-accelerated solver using CuClarabel (requires Julia)
  - `mpax_if.py` - MPAX solver interface
  - `moreau_if.py` - Moreau envelope solver

- **Utilities** (`utils/`):
  - `parse_args.py` - Core canonicalization logic that converts CVXPY problems to parametrized cone programs. Defines `LayersContext` dataclass holding problem matrices and solver context.
  - `solver_utils.py` - Solver selection dispatch

### Key Concepts

1. **DPP Compliance**: All problems must follow CVXPY's Disciplined Parametrized Programming (DPP) rules. Check with `problem.is_dpp()`.

2. **Parametrized Cone Programs**: CVXPY problems are canonicalized into the form:
   - Quadratic objective: `P @ params`
   - Linear objective: `q @ params`
   - Constraints: `A @ params`
   The framework layers store these as sparse matrices.

3. **Batched Execution**: Parameters can be batched (first dimension is batch size). Non-batched parameters are broadcast automatically.

4. **GP Support**: Geometric programs use `gp=True` flag. Parameters are log-transformed before solving the DCP-equivalent problem.

### Data Flow

```
User Parameters (torch/jax/numpy tensors)
    ↓
validate_params() - check shapes, determine batch size
    ↓
_apply_gp_log_transform() - transform GP params to log-space if needed
    ↓
_flatten_and_batch_params() - flatten and concatenate into p_stack
    ↓
Matrix multiply: P @ p_stack, q @ p_stack, A @ p_stack
    ↓
Solver (via solver_ctx interface) - returns primal/dual solutions
    ↓
_recover_results() - extract variables, apply exp() for GP
    ↓
Output tensors with gradients attached
```

## Testing

Tests are in `tests/` using pytest. Framework-specific tests use `pytest.importorskip()` to gracefully skip if the framework isn't installed.

Key test patterns:
- `torch.autograd.gradcheck()` for verifying gradients
- Compare against closed-form solutions (e.g., least squares)
- Test batched vs unbatched consistency

## Dependencies

- Python >= 3.11
- CVXPY >= 1.7.4 (for native DGP→DCP reduction)
- diffcp >= 1.1.0 (default solver backend)
- Framework: PyTorch >= 2.0, JAX >= 0.4.0, or MLX
