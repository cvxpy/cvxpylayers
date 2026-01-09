# Dual Variables

CVXPYlayers can return constraint dual variables (Lagrange multipliers) alongside the primal solution. Dual variables represent the sensitivity of the optimal objective value to changes in the constraint right-hand side.

## Basic Usage

Include a constraint's dual variable in the `variables` list:

```python
import cvxpy as cp
import torch
from cvxpylayers.torch import CvxpyLayer

n = 2
x = cp.Variable(n)
c = cp.Parameter(n)
b = cp.Parameter()

# Define constraint and store reference
eq_con = cp.sum(x) == b
prob = cp.Problem(cp.Minimize(c @ x), [eq_con, x >= 0])

# Request primal variable AND the constraint's dual
layer = CvxpyLayer(
    prob,
    parameters=[c, b],
    variables=[x, eq_con.dual_variables[0]],
)

c_t = torch.tensor([1.0, 2.0], requires_grad=True)
b_t = torch.tensor(1.0, requires_grad=True)

# Forward pass returns both
x_opt, eq_dual = layer(c_t, b_t)

# Gradients flow through dual variables
loss = eq_dual.sum()
loss.backward()
```

## Multiple Dual Variables

You can request duals from multiple constraints:

```python
eq_con = cp.sum(x) == b
ineq_con = x >= 0
prob = cp.Problem(cp.Minimize(c @ x), [eq_con, ineq_con])

layer = CvxpyLayer(
    prob,
    parameters=[c, b],
    variables=[x, eq_con.dual_variables[0], ineq_con.dual_variables[0]],
)

x_opt, eq_dual, ineq_dual = layer(c_t, b_t)
```

## Dual-Only Output

You can request only dual variables without primal:

```python
layer = CvxpyLayer(
    prob,
    parameters=[c, b],
    variables=[eq_con.dual_variables[0]],
)

(eq_dual,) = layer(c_t, b_t)
```

## Batching

Dual variables work with batched parameters:

```python
# Batch of 32 problems
c_batch = torch.randn(32, n, requires_grad=True)
b_batch = torch.randn(32, requires_grad=True)

x_opt, eq_dual = layer(c_batch, b_batch)
# x_opt: (32, n)
# eq_dual: (32,) for scalar constraint
```

## Accessing Dual Variables

Every CVXPY constraint has a `dual_variables` attribute:

```python
con = A @ x == b
dual = con.dual_variables[0]  # First (usually only) dual variable
```

Most constraints have a single dual variable. Some conic constraints created explicitly (like `cp.SOC(t, x)` or `cp.PowCone3D(...)`) have multiple dual variables that you can access by index.
