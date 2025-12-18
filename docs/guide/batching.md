# Batched Execution

CVXPYlayers supports solving multiple problem instances in parallel through batching.

## How Batching Works

Add a batch dimension as the **first dimension** of your parameter tensors:

```python
import cvxpy as cp
import torch
from cvxpylayers.torch import CvxpyLayer

# Problem with parameters of shape (3, 2) and (3,)
x = cp.Variable(2)
A = cp.Parameter((3, 2))
b = cp.Parameter(3)
problem = cp.Problem(cp.Minimize(cp.sum_squares(A @ x - b)), [x >= 0])

layer = CvxpyLayer(problem, parameters=[A, b], variables=[x])

# Single instance: shapes (3, 2) and (3,)
A_single = torch.randn(3, 2)
b_single = torch.randn(3)
(x_single,) = layer(A_single, b_single)  # x_single shape: (2,)

# Batched: shapes (batch_size, 3, 2) and (batch_size, 3)
batch_size = 10
A_batch = torch.randn(batch_size, 3, 2)
b_batch = torch.randn(batch_size, 3)
(x_batch,) = layer(A_batch, b_batch)  # x_batch shape: (10, 2)
```

## Broadcasting

Parameters can be mixed batched and unbatched. Unbatched parameters are broadcast:

```python
batch_size = 10

# A is batched, b is shared across the batch
A_batch = torch.randn(batch_size, 3, 2)  # Different A for each instance
b_shared = torch.randn(3)                 # Same b for all instances

(x_batch,) = layer(A_batch, b_shared)    # x_batch shape: (10, 2)
```

This is useful when some parameters are fixed and others vary.

## Performance Considerations

### Batch Size Selection

Larger batches are more efficient due to:
- Parallelized matrix operations
- Reduced Python overhead
- Better GPU utilization (if using CuClarabel)

```python
# Efficient: solve 100 problems at once
A_batch = torch.randn(100, m, n)
(x_batch,) = layer(A_batch, b)

# Less efficient: solve 100 problems one at a time
solutions = []
for i in range(100):
    (x,) = layer(A_batch[i], b)
    solutions.append(x)
```

### Memory Trade-offs

Batching increases memory usage linearly with batch size. If you run out of memory:

1. Reduce batch size
2. Process in chunks:

```python
def chunked_solve(layer, A_all, b, chunk_size=32):
    results = []
    for i in range(0, len(A_all), chunk_size):
        chunk = A_all[i:i+chunk_size]
        (x_chunk,) = layer(chunk, b)
        results.append(x_chunk)
    return torch.cat(results, dim=0)
```

## Batched Gradients

Gradients work naturally with batching:

```python
batch_size = 10
A_batch = torch.randn(batch_size, 3, 2, requires_grad=True)
b_batch = torch.randn(batch_size, 3, requires_grad=True)

(x_batch,) = layer(A_batch, b_batch)

# Sum over batch for scalar loss
loss = x_batch.sum()
loss.backward()

# Gradients have same shape as inputs
print(A_batch.grad.shape)  # (10, 3, 2)
print(b_batch.grad.shape)  # (10, 3)
```

## JAX: vmap Integration

In JAX, you can also use `jax.vmap` for batching:

```python
import jax
import jax.numpy as jnp
from cvxpylayers.jax import CvxpyLayer

layer = CvxpyLayer(problem, parameters=[A, b], variables=[x])

# Option 1: Pass batched arrays directly (built-in support)
A_batch = jax.random.normal(key, (batch_size, 3, 2))
(x_batch,) = layer(A_batch, b_single)

# Option 2: Use vmap explicitly
def solve_single(A):
    (x,) = layer(A, b_single)
    return x

solve_batched = jax.vmap(solve_single)
x_batch = solve_batched(A_batch)
```

## Example: Batch Portfolio Optimization

```python
import cvxpy as cp
import torch
from cvxpylayers.torch import CvxpyLayer

n_assets = 10
n_scenarios = 50

# Portfolio optimization problem
weights = cp.Variable(n_assets)
expected_returns = cp.Parameter(n_assets)
risk_aversion = cp.Parameter(nonneg=True)

# Simplified: maximize return - risk_aversion * variance
problem = cp.Problem(
    cp.Maximize(expected_returns @ weights - risk_aversion * cp.sum_squares(weights)),
    [cp.sum(weights) == 1, weights >= 0]
)

layer = CvxpyLayer(
    problem,
    parameters=[expected_returns, risk_aversion],
    variables=[weights]
)

# Different return predictions for each scenario
returns_batch = torch.randn(n_scenarios, n_assets)
risk_aversion_val = torch.tensor(0.5)  # Shared across scenarios

# Solve all scenarios at once
(optimal_weights,) = layer(returns_batch, risk_aversion_val)
print(f"Optimal weights shape: {optimal_weights.shape}")  # (50, 10)
```
