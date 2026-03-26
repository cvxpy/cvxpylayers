# Solvers

CVXPYlayers supports multiple solver backends for different use cases.

## Available Solvers

| Solver | Type | Best For |
|--------|------|----------|
| **diffcp w/ SCS** (default) | CPU | General use, most problem types |
| **diffcp w/ Clarabel** | CPU | Higher accuracy |
| **[Moreau](https://docs.moreau.so/)** | CPU/GPU | Best performance |
| **MPAX*** | CPU | LPs/QPs |
| **CuClarabel w/ diffqcp** | GPU | Open-source GPU alternative |

\* Gradient support is currently broken.

## Specifying a Solver

### At Construction

```python
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer

layer = CvxpyLayer(
    problem,
    parameters=[A, b],
    variables=[x],
    solver=cp.DIFFCP,
    solver_args={'solver': cp.CLARABEL}  # Use Clarabel
)
```

### At Call Time

```python
# Use default solver
(x,) = layer(A_tensor, b_tensor)

# Override with different solver
(x,) = layer(A_tensor, b_tensor, solver_args={"solver": cp.SCS})
```

## Solver Arguments

Pass solver-specific settings via `solver_args`:

```python
# At construction (defaults for all calls)
layer = CvxpyLayer(
    problem,
    parameters=[A, b],
    variables=[x],
    solver_args={"max_iters": 5000, "eps": 1e-8}
)

# At call time (override for this call)
(x,) = layer(A_tensor, b_tensor, solver_args={"max_iters": 10000})
```

### Common Arguments

| Argument | Solver | Description |
|----------|--------|-------------|
| `eps` | SCS, Clarabel | Convergence tolerance |
| `max_iters` | All | Maximum iterations |
| `verbose` | All | Print solver output |
| `acceleration_lookback` | SCS | Anderson acceleration window |

## SCS Tuning

SCS is robust but may need tuning for difficult problems:

```python
# Recommended settings for convergence issues
solver_args = {
    "eps": 1e-8,              # Tighter tolerance
    "max_iters": 10000,       # More iterations
    "acceleration_lookback": 0  # Disable acceleration (more stable)
}

(x,) = layer(A_tensor, b_tensor, solver_args=solver_args)
```

If SCS still struggles, try Clarabel:

```python
# Clarabel for better cone support
layer = CvxpyLayer(problem, parameters=[A, b], variables=[x], solver=cp.CLARABEL)
```

## Moreau

[Moreau](https://docs.moreau.so/) is the recommended solver for best performance on both CPU and GPU.
It supports PyTorch and JAX with native autograd integration, warm starts, and `jax.jit` compatibility.

### Setup

Moreau is available by request through a private package index.
See the [Moreau installation guide](https://docs.moreau.so/installation.html) for access and setup instructions.

### Usage (PyTorch)

```python
import cvxpy as cp
import torch
from cvxpylayers.torch import CvxpyLayer

layer = CvxpyLayer(
    problem,
    parameters=[A, b],
    variables=[x],
    solver="MOREAU"
)

A_tch = torch.randn(m, n, requires_grad=True)
b_tch = torch.randn(m, requires_grad=True)

(x_sol,) = layer(A_tch, b_tch)
x_sol.sum().backward()
```

### Usage (JAX)

```python
import cvxpy as cp
import jax
from cvxpylayers.jax import CvxpyLayer

layer = CvxpyLayer(
    problem,
    parameters=[A, b],
    variables=[x],
    solver="MOREAU"
)

A_jax = jax.random.normal(jax.random.PRNGKey(0), shape=(m, n))
b_jax = jax.random.normal(jax.random.PRNGKey(1), shape=(m,))

(x_sol,) = layer(A_jax, b_jax)
```

### Warm Starts

Moreau supports warm starting to speed up sequential solves:

```python
(x_sol,) = layer(A_tch, b_tch, warm_start=True)
```

### When to Use Moreau

Moreau is beneficial when:
- You want the best solve + differentiation performance
- You need `jax.jit` compatibility
- You're solving sequences of similar problems (warm starts)
- You want CPU or GPU support without Julia dependencies

---

## CuClarabel (Open-Source Alternative)

[CuClarabel](https://github.com/oxfordcontrol/Clarabel.jl/tree/CuClarabel/) is an open-source GPU solver alternative. It requires Julia and several additional dependencies. For NVIDIA GPUs, it keeps all data on the GPU:

See {doc}`../installation` for CuClarabel setup instructions.

## Troubleshooting

### Solver Failed

```
SolverError: Solver 'SCS' failed. Try another solver or adjust solver settings.
```

**Solutions:**
1. Try a different solver
2. Increase `max_iters`
3. Loosen tolerance (`eps`)
4. Check problem feasibility

### Numerical Issues

```
Warning: Solution may be inaccurate.
```

**Solutions:**
1. Scale your data (normalize matrices)
2. Use tighter tolerance
3. Try Clarabel (often more numerically stable)

### Slow Convergence

**Solutions:**
1. Warm-starting (if supported)
2. Problem reformulation
3. Use CuClarabel for large problems
