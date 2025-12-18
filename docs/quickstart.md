# Quickstart

Build your first differentiable optimization layer in 4 steps.

:::{admonition} Prerequisites
:class: tip

- Python 3.11+
- CVXPY (`pip install cvxpy`)
- One of: PyTorch, JAX, or MLX
:::

```{raw} html
<style>
.step-number {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 2rem;
    height: 2rem;
    background: var(--color-brand-primary);
    color: white;
    border-radius: 50%;
    font-weight: bold;
    margin-right: 0.75rem;
}
</style>
```

---

## <span class="step-number">1</span> Define the Problem

Create a parametrized convex optimization problem with CVXPY:

```python
import cvxpy as cp

# Problem dimensions
n, m = 2, 3

# Decision variable (what we solve for)
x = cp.Variable(n)

# Parameters (inputs that change at runtime)
A = cp.Parameter((m, n))
b = cp.Parameter(m)

# Build the problem
problem = cp.Problem(
    cp.Minimize(cp.sum_squares(A @ x - b)),  # Objective
    [x >= 0]                                   # Constraints
)
```

:::{tip}
**Parameters** are placeholders for values you'll provide later. **Variables** are what the solver finds.
:::

---

## <span class="step-number">2</span> Check DPP Compliance

Your problem must follow [Disciplined Parametrized Programming](https://www.cvxpy.org/tutorial/dpp/index.html) rules:

```python
assert problem.is_dpp(), "Problem must be DPP-compliant"
```

:::{dropdown} What is DPP?
DPP ensures the problem structure is fixed — only parameter *values* change, not the problem shape. This is required for implicit differentiation.

**Valid:** Parameters in linear/affine expressions
```python
A @ x - b  # Good: A and b appear linearly
```

**Invalid:** Parameters that change problem structure
```python
cp.quad_form(x, P)  # P as parameter may break DPP
```
:::

---

## <span class="step-number">3</span> Create the Layer

Wrap your problem as a differentiable layer:

::::{tab-set}

:::{tab-item} PyTorch
:sync: pytorch

```python
import torch
from cvxpylayers.torch import CvxpyLayer

layer = CvxpyLayer(
    problem,
    parameters=[A, b],   # CVXPY parameters (in order)
    variables=[x]        # Variables to return
)
```
:::

:::{tab-item} JAX
:sync: jax

```python
import jax
from cvxpylayers.jax import CvxpyLayer

layer = CvxpyLayer(
    problem,
    parameters=[A, b],
    variables=[x]
)
```
:::

:::{tab-item} MLX
:sync: mlx

```python
import mlx.core as mx
from cvxpylayers.mlx import CvxpyLayer

layer = CvxpyLayer(
    problem,
    parameters=[A, b],
    variables=[x]
)
```
:::

::::

---

## <span class="step-number">4</span> Solve & Differentiate

Pass tensor values and backpropagate:

::::{tab-set}

:::{tab-item} PyTorch
:sync: pytorch

```python
# Create tensors with gradients enabled
A_t = torch.randn(m, n, requires_grad=True)
b_t = torch.randn(m, requires_grad=True)

# Forward: solve the optimization
(solution,) = layer(A_t, b_t)

# Backward: compute gradients
loss = solution.sum()
loss.backward()

print(f"Solution: {solution}")
print(f"dL/dA: {A_t.grad}")
print(f"dL/db: {b_t.grad}")
```
:::

:::{tab-item} JAX
:sync: jax

```python
import jax.numpy as jnp

# Create arrays
key = jax.random.PRNGKey(0)
A_jax = jax.random.normal(key, (m, n))
b_jax = jax.random.normal(key, (m,))

# Forward
(solution,) = layer(A_jax, b_jax)

# Gradients via jax.grad
def loss_fn(A, b):
    (sol,) = layer(A, b)
    return sol.sum()

dA, db = jax.grad(loss_fn, argnums=[0, 1])(A_jax, b_jax)

print(f"Solution: {solution}")
print(f"dL/dA: {dA}")
print(f"dL/db: {db}")
```
:::

:::{tab-item} MLX
:sync: mlx

```python
# Create arrays
A_mx = mx.random.normal((m, n))
b_mx = mx.random.normal((m,))

# Forward
(solution,) = layer(A_mx, b_mx)

# Gradients via mx.grad
def loss_fn(A, b):
    (sol,) = layer(A, b)
    return mx.sum(sol)

grad_fn = mx.grad(loss_fn, argnums=[0, 1])
dA, db = grad_fn(A_mx, b_mx)

print(f"Solution: {solution}")
print(f"dL/dA: {dA}")
print(f"dL/db: {db}")
```
:::

::::

---

## Complete Example

Here's everything together — a training loop that learns matrix `A`:

```python
import cvxpy as cp
import torch
from cvxpylayers.torch import CvxpyLayer

# 1. Define problem
n, m = 5, 10
x = cp.Variable(n)
A = cp.Parameter((m, n))
b = cp.Parameter(m)

problem = cp.Problem(
    cp.Minimize(cp.sum_squares(A @ x - b)),
    [x >= 0]
)

# 2. Create layer
layer = CvxpyLayer(problem, parameters=[A, b], variables=[x])

# 3. Training setup
A_true = torch.randn(m, n)
x_true = torch.abs(torch.randn(n))
b_true = A_true @ x_true

A_learn = torch.randn(m, n, requires_grad=True)
optimizer = torch.optim.Adam([A_learn], lr=0.1)

# 4. Training loop
for i in range(100):
    optimizer.zero_grad()
    (x_pred,) = layer(A_learn, b_true)
    loss = torch.sum((x_pred - x_true) ** 2)
    loss.backward()
    optimizer.step()

    if i % 25 == 0:
        print(f"Step {i:3d} | Loss: {loss.item():.4f}")
```

---

## Next Steps

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Basic Usage
:link: guide/basic-usage
:link-type: doc

Constructor options, parameter handling, error handling.
:::

:::{grid-item-card} Batching
:link: guide/batching
:link-type: doc

Solve multiple problems in parallel for 10-100x speedup.
:::

:::{grid-item-card} Solvers
:link: guide/solvers
:link-type: doc

Choose the right solver: SCS, Clarabel, CuClarabel.
:::

:::{grid-item-card} Examples
:link: examples/index
:link-type: doc

Real-world applications: control, finance, ML, robotics.
:::

::::
