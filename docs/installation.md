# Installation

## Quick Install

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Most Users
:class-card: sd-border-primary

```bash
pip install cvxpylayers[torch]
```

Includes PyTorch and all dependencies.
:::

:::{grid-item-card} Minimal Install
:class-card: sd-border-secondary

```bash
pip install cvxpylayers
```

Core only — add framework extras as needed.
:::

::::

---

## Choose Your Framework

::::{tab-set}

:::{tab-item} PyTorch
:sync: pytorch

```bash
pip install cvxpylayers[torch]
```

**Requirements:** PyTorch >= 2.0

The most popular choice. Full `torch.nn.Module` integration with autograd support.
:::

:::{tab-item} JAX
:sync: jax

```bash
pip install cvxpylayers[jax]
```

**Requirements:** JAX >= 0.4.0

Functional style with `jax.grad`, `jax.jit`, and `jax.vmap`.
:::

:::{tab-item} MLX
:sync: mlx

```bash
pip install cvxpylayers[mlx]
```

**Requirements:** MLX >= 0.27.1, Apple Silicon Mac

Optimized for M1/M2/M3 chips with unified memory.
:::

:::{tab-item} All Frameworks
:sync: all

```bash
pip install cvxpylayers[all]
```

Install everything — useful for development or testing.
:::

::::

---

## Verify Installation

::::{tab-set}

:::{tab-item} PyTorch
:sync: pytorch

```python
import cvxpy as cp
import torch
from cvxpylayers.torch import CvxpyLayer

x = cp.Variable(2)
A = cp.Parameter((2, 2))
b = cp.Parameter(2)
problem = cp.Problem(cp.Minimize(cp.sum_squares(A @ x - b)))

layer = CvxpyLayer(problem, parameters=[A, b], variables=[x])
A_t = torch.eye(2, requires_grad=True)
b_t = torch.ones(2, requires_grad=True)
(sol,) = layer(A_t, b_t)

print(f"Solution: {sol}")  # tensor([1., 1.])
print("Installation successful!")
```
:::

:::{tab-item} JAX
:sync: jax

```python
import cvxpy as cp
import jax.numpy as jnp
from cvxpylayers.jax import CvxpyLayer

x = cp.Variable(2)
A = cp.Parameter((2, 2))
b = cp.Parameter(2)
problem = cp.Problem(cp.Minimize(cp.sum_squares(A @ x - b)))

layer = CvxpyLayer(problem, parameters=[A, b], variables=[x])
A_jax = jnp.eye(2)
b_jax = jnp.ones(2)
(sol,) = layer(A_jax, b_jax)

print(f"Solution: {sol}")  # [1., 1.]
print("Installation successful!")
```
:::

:::{tab-item} MLX
:sync: mlx

```python
import cvxpy as cp
import mlx.core as mx
from cvxpylayers.mlx import CvxpyLayer

x = cp.Variable(2)
A = cp.Parameter((2, 2))
b = cp.Parameter(2)
problem = cp.Problem(cp.Minimize(cp.sum_squares(A @ x - b)))

layer = CvxpyLayer(problem, parameters=[A, b], variables=[x])
A_mx = mx.eye(2)
b_mx = mx.ones(2)
(sol,) = layer(A_mx, b_mx)

print(f"Solution: {sol}")  # [1., 1.]
print("Installation successful!")
```
:::

::::

---

## GPU Acceleration

:::{admonition} CuClarabel — NVIDIA GPU Support
:class: tip

For large-scale problems, CuClarabel keeps everything on GPU with no CPU transfers.
:::

::::{grid} 1 1 3 3
:gutter: 2

:::{grid-item-card} Step 1: Julia
Install from [julialang.org](https://julialang.org/)
:::

:::{grid-item-card} Step 2: CuClarabel
```julia
using Pkg
Pkg.add(url="https://github.com/oxfordcontrol/Clarabel.jl", rev="CuClarabel")
```
:::

:::{grid-item-card} Step 3: Python
```bash
pip install juliacall cupy diffqcp
```
:::

::::

**Usage:**

```python
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer

layer = CvxpyLayer(
    problem,
    parameters=[A, b],
    variables=[x],
    solver=cp.CUCLARABEL
).to("cuda")

# Parameters must be on GPU
A_gpu = A_t.cuda()
b_gpu = b_t.cuda()
(solution,) = layer(A_gpu, b_gpu)
```

---

## Dependencies

:::{dropdown} Core Dependencies
| Package | Version | Purpose |
|---------|---------|---------|
| Python | >= 3.11 | Runtime |
| NumPy | >= 1.22.4 | Array operations |
| CVXPY | >= 1.7.4 | Problem specification |
| diffcp | >= 1.1.0 | Differentiable cone programming |
:::

:::{dropdown} Framework Dependencies
| Framework | Version |
|-----------|---------|
| PyTorch | >= 2.0 |
| JAX | >= 0.4.0 |
| MLX | >= 0.27.1 |
:::

---

## Development Setup

For contributing:

```bash
git clone https://github.com/cvxpy/cvxpylayers.git
cd cvxpylayers
pip install -e ".[all,dev]"
```

Run tests:

```bash
pytest tests/
```
