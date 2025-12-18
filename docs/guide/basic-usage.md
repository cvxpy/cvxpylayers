# Basic Usage

This guide covers the `CvxpyLayer` constructor and common usage patterns.

## Constructor

```python
CvxpyLayer(
    problem,           # CVXPY Problem object
    parameters,        # List of cp.Parameter objects
    variables,         # List of cp.Variable objects to return
    solver=None,       # Solver to use (optional)
    gp=False,          # True for geometric programs
    verbose=False,     # Print solver output
    solver_args=None,  # Default solver arguments
)
```

### Parameters

| Argument | Type | Description |
|----------|------|-------------|
| `problem` | `cp.Problem` | A DPP-compliant CVXPY problem |
| `parameters` | `list[cp.Parameter]` | Parameters that will receive values at runtime |
| `variables` | `list[cp.Variable]` | Variables to return from the solution |
| `solver` | `str` or `None` | Solver backend (e.g., `cp.CLARABEL`, `cp.SCS`) |
| `gp` | `bool` | Set `True` for geometric/log-log convex programs |
| `verbose` | `bool` | Print solver output |
| `solver_args` | `dict` | Default arguments passed to the solver |

## DPP Requirements

Your problem must be DPP-compliant. This means:

1. **Parameters appear affinely** in the objective and constraints
2. **Problem structure is fixed** (same constraints for all parameter values)
3. **No parameter-dependent domains**

```python
import cvxpy as cp

x = cp.Variable(2)
A = cp.Parameter((3, 2))
b = cp.Parameter(3)

# Good: Parameter appears affinely
problem = cp.Problem(cp.Minimize(cp.sum_squares(A @ x - b)))
assert problem.is_dpp()  # True

# Bad: Parameter in non-affine position
c = cp.Parameter(nonneg=True)
bad_problem = cp.Problem(cp.Minimize(cp.quad_form(x, c * np.eye(2))))
# This may not be DPP-compliant
```

## Calling the Layer

The layer is called with parameter values in the same order as the `parameters` list:

```python
layer = CvxpyLayer(problem, parameters=[A, b, c], variables=[x, y])

# Call with matching parameter values
x_sol, y_sol = layer(A_tensor, b_tensor, c_tensor)
```

### Return Values

The layer returns a tuple of tensors corresponding to the `variables` list:

```python
# Single variable
layer = CvxpyLayer(problem, parameters=[A], variables=[x])
(x_sol,) = layer(A_tensor)  # Note the tuple unpacking

# Multiple variables
layer = CvxpyLayer(problem, parameters=[A], variables=[x, y, z])
x_sol, y_sol, z_sol = layer(A_tensor)
```

## Parameter Shapes

Parameters must match their CVXPY declaration shapes:

```python
A = cp.Parameter((m, n))  # Shape (m, n)
b = cp.Parameter(m)       # Shape (m,)
c = cp.Parameter()        # Scalar

layer = CvxpyLayer(problem, parameters=[A, b, c], variables=[x])

# Tensor shapes must match
A_tensor = torch.randn(m, n)  # Shape (m, n)
b_tensor = torch.randn(m)     # Shape (m,)
c_tensor = torch.tensor(1.0)  # Scalar

(solution,) = layer(A_tensor, b_tensor, c_tensor)
```

## Using with Neural Networks (PyTorch)

The PyTorch layer extends `torch.nn.Module`, so it integrates naturally:

```python
import torch.nn as nn

class OptimizationNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 6)  # Outputs A (2x3) flattened

        # Create CVXPY problem
        x = cp.Variable(3)
        A = cp.Parameter((2, 3))
        b = cp.Parameter(2)
        problem = cp.Problem(
            cp.Minimize(cp.sum_squares(A @ x - b)),
            [x >= 0]
        )

        self.cvx_layer = CvxpyLayer(problem, parameters=[A, b], variables=[x])

    def forward(self, input_features, b):
        # Neural network predicts A
        A_flat = self.linear(input_features)
        A = A_flat.view(-1, 2, 3)

        # Optimization layer finds optimal x
        (x,) = self.cvx_layer(A, b)
        return x
```

## Solver Arguments at Call Time

Override solver settings per-call:

```python
# Default solver args in constructor
layer = CvxpyLayer(problem, parameters=[A], variables=[x],
                   solver_args={"max_iters": 1000})

# Override at call time
(solution,) = layer(A_tensor, solver_args={"max_iters": 5000, "eps": 1e-8})
```

## Error Handling

Common errors and solutions:

### Problem Not DPP

```
AssertionError: Problem must be DPP
```

Check with `problem.is_dpp()`. Ensure parameters appear affinely.

### Shape Mismatch

```
ValueError: Parameter A has shape (3, 2) but tensor has shape (2, 3)
```

Ensure tensor shapes match parameter declarations.

### Solver Failure

```
SolverError: Solver 'SCS' failed.
```

Try:
1. Different solver: `solver=cp.CLARABEL`
2. More iterations: `solver_args={"max_iters": 10000}`
3. Looser tolerance: `solver_args={"eps": 1e-6}`
