# Solvers

CVXPYlayers supports multiple solver backends for different use cases.

## Available Solvers

| Solver | Type | Best For |
|--------|------|----------|
| **diffcp w/ SCS** (default) | CPU | General use, most problem types |
| **diffcp w/ Clarabel** | CPU | Higher accuracy |
| **Moreau** | CPU/GPU | Best performance |
| **MPAX*** | CPU | LPs/QPs |
| **CuClarabel w/ diffqcp** | GPU | Large problems on NVIDIA GPUs |

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

## GPU Acceleration with CuClarabel

For NVIDIA GPUs, CuClarabel keeps all data on the GPU:

### Setup

See {doc}`../installation` for CuClarabel installation.

### Usage (PyTorch)

```python
import cvxpy as cp
import torch
from cvxpylayers.torch import CvxpyLayer

device = torch.device("cuda")

layer = CvxpyLayer(
    problem,
    parameters=[A, b],
    variables=[x],
    solver=cp.CUCLARABEL
).to(device)

# Parameters must be on GPU
A_gpu = torch.randn(m, n, device=device, requires_grad=True)
b_gpu = torch.randn(m, device=device, requires_grad=True)

(x_gpu,) = layer(A_gpu, b_gpu)
x_gpu.sum().backward()  # Gradients computed on GPU
```

### When to Use CuClarabel

CuClarabel is beneficial when:
- Problems are large (1000+ variables/constraints)
- You're already using GPU tensors
- Batch sizes are large
- You want to avoid CPU-GPU transfers

For small problems, CPU solvers may be faster due to GPU overhead.

## Solver Comparison

```python
import time
import cvxpy as cp
import torch
from cvxpylayers.torch import CvxpyLayer

# Create problem
n = 100
x = cp.Variable(n)
A = cp.Parameter((n, n))
b = cp.Parameter(n)
problem = cp.Problem(cp.Minimize(cp.sum_squares(A @ x - b)))

# Benchmark different solvers
A_t = torch.randn(n, n)
b_t = torch.randn(n)

for solver in [None, cp.CLARABEL, cp.SCS]:
    layer = CvxpyLayer(problem, parameters=[A, b], variables=[x], solver=solver)

    start = time.time()
    for _ in range(10):
        (x_sol,) = layer(A_t, b_t)
    elapsed = time.time() - start

    solver_name = solver if solver else "diffcp (default)"
    print(f"{solver_name}: {elapsed:.3f}s for 10 solves")
```

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
