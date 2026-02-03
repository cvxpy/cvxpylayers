# CVXPYlayers

```{raw} html
<p align="center" style="margin: 2rem 0;">
  <a href="https://pypi.org/project/cvxpylayers/"><img src="https://img.shields.io/pypi/v/cvxpylayers?style=flat-square&color=blue" alt="PyPI"></a>
  <a href="https://github.com/cvxpy/cvxpylayers"><img src="https://img.shields.io/github/stars/cvxpy/cvxpylayers?style=flat-square&color=yellow" alt="GitHub stars"></a>
  <a href="https://github.com/cvxpy/cvxpylayers/blob/master/LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-green?style=flat-square" alt="License"></a>
  <a href="https://arxiv.org/abs/1910.12430"><img src="https://img.shields.io/badge/NeurIPS-2019-red?style=flat-square" alt="NeurIPS 2019"></a>
</p>
```

```{raw} html
<p align="center" style="font-size: 1.4rem; color: var(--color-foreground-secondary); margin-bottom: 2rem;">
  <strong>Differentiable convex optimization layers for deep learning</strong>
</p>
```

Embed convex optimization problems directly into your neural networks. CVXPYlayers solves parametrized problems in the forward pass and computes gradients via implicit differentiation in the backward pass.

---

## Frameworks

::::{grid} 1 1 3 3
:gutter: 3

:::{grid-item-card} **PyTorch**
:link: api/torch
:link-type: doc
:class-card: sd-card-pytorch

Full `torch.nn.Module` integration with autograd support. The most popular choice for deep learning.
:::

:::{grid-item-card} **JAX**
:link: api/jax
:link-type: doc
:class-card: sd-card-jax

Works with `jax.grad`, `jax.vmap`, and `jax.jit` (Moreau solver).
:::

:::{grid-item-card} **MLX**
:link: api/mlx
:link-type: doc
:class-card: sd-card-mlx

Optimized for Apple Silicon. Unified memory architecture for M1/M2/M3 chips.
:::

::::

---

## Get Started in 30 Seconds

::::{grid} 1 1 2 2
:gutter: 4

:::{grid-item}
:columns: 12 12 7 7

```python
import cvxpy as cp
import torch
from cvxpylayers.torch import CvxpyLayer

# Define optimization problem
x = cp.Variable(2)
A = cp.Parameter((3, 2))
b = cp.Parameter(3)
problem = cp.Problem(
    cp.Minimize(cp.sum_squares(A @ x - b)),
    [x >= 0]
)

# Wrap as differentiable layer
layer = CvxpyLayer(problem,
    parameters=[A, b],
    variables=[x]
)

# Solve + backprop
A_t = torch.randn(3, 2, requires_grad=True)
b_t = torch.randn(3, requires_grad=True)
(solution,) = layer(A_t, b_t)
solution.sum().backward()  # Gradients flow!
```
:::

:::{grid-item}
:columns: 12 12 5 5

**Install**
```bash
pip install cvxpylayers[torch]
```

**What's happening?**

1. Define a convex problem with CVXPY
2. Wrap it as a `CvxpyLayer`
3. Use it like any PyTorch layer
4. Gradients computed automatically

```{button-ref} quickstart
:color: primary
:expand:

Quickstart Guide
```

:::

::::

---

## Why CVXPYlayers?

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Encode Domain Knowledge
:class-card: sd-card-feature

Inject constraints and structure into your models. Physics, fairness, safety — if you can write it as a convex program, you can differentiate through it.
:::

:::{grid-item-card} GPU Accelerated
:class-card: sd-card-feature

CuClarabel solver keeps everything on GPU. No CPU-GPU transfers for large-scale optimization.
:::

:::{grid-item-card} Batched Solving
:class-card: sd-card-feature

Solve thousands of problem instances in parallel. First dimension is batch — just like PyTorch.
:::

:::{grid-item-card} Multiple Solvers
:class-card: sd-card-feature

Clarabel, SCS, and CuClarabel. Pick the right solver for your problem structure.
:::

::::

---

## Used For

::::{grid} 2 2 4 4
:gutter: 2

:::{grid-item-card} Control
:text-align: center
:class-card: sd-card-usecase

MPC, LQR, path planning
:::

:::{grid-item-card} Finance
:text-align: center
:class-card: sd-card-usecase

Portfolio optimization
:::

:::{grid-item-card} ML
:text-align: center
:class-card: sd-card-usecase

Constrained learning
:::

:::{grid-item-card} Robotics
:text-align: center
:class-card: sd-card-usecase

Motion planning
:::

::::

```{button-ref} examples/index
:color: primary
:outline:
:expand:

Browse Examples
```

---

## Research

This library accompanies our **NeurIPS 2019 paper**:

> Agrawal, A., Amos, B., Barratt, S., Boyd, S., Diamond, S., & Kolter, Z. (2019).
> [Differentiable Convex Optimization Layers](https://web.stanford.edu/~boyd/papers/pdf/diff_cvxpy.pdf).
> *Advances in Neural Information Processing Systems*.

For an introduction, see our [blog post](https://locuslab.github.io/2019-10-28-cvxpylayers/).

:::{dropdown} BibTeX Citation
```bibtex
@inproceedings{cvxpylayers2019,
  author={Agrawal, A. and Amos, B. and Barratt, S. and Boyd, S. and Diamond, S. and Kolter, Z.},
  title={Differentiable Convex Optimization Layers},
  booktitle={Advances in Neural Information Processing Systems},
  year={2019},
}
```
:::

```{toctree}
:maxdepth: 2
:hidden:

installation
quickstart
guide/index
api/index
examples/index
```
