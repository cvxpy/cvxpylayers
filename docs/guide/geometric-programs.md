# Geometric Programs

CVXPYlayers supports differentiating through geometric programs (GPs) and log-log convex programs (LLCPs).

## What are Geometric Programs?

Geometric programs are optimization problems where:
- Variables are positive
- Objectives and constraints are formed from monomials and posynomials
- The problem can be transformed to a convex problem via log transformation

CVXPY extends this to **log-log convex programs (LLCPs)**, which include GPs as a special case.

## Basic Usage

Set `gp=True` when creating the layer:

```python
import cvxpy as cp
import torch
from cvxpylayers.torch import CvxpyLayer

# Variables must be positive
x = cp.Variable(pos=True)
y = cp.Variable(pos=True)
z = cp.Variable(pos=True)

# Parameters
a = cp.Parameter(pos=True)
b = cp.Parameter(pos=True)
c = cp.Parameter()

# GP objective and constraints
objective = cp.Minimize(1 / (x * y * z))
constraints = [
    a * (x * y + x * z + y * z) <= b,
    x >= y ** c
]

problem = cp.Problem(objective, constraints)

# Verify it's a valid DGP (Disciplined Geometric Program)
assert problem.is_dgp(dpp=True)

# Create layer with gp=True
layer = CvxpyLayer(
    problem,
    parameters=[a, b, c],
    variables=[x, y, z],
    gp=True  # Important!
)
```

## Solving and Differentiating

```python
# Parameter values
a_tch = torch.tensor(2.0, requires_grad=True)
b_tch = torch.tensor(1.0, requires_grad=True)
c_tch = torch.tensor(0.5, requires_grad=True)

# Solve
x_star, y_star, z_star = layer(a_tch, b_tch, c_tch)

# Differentiate
loss = x_star + y_star + z_star
loss.backward()

print(f"x* = {x_star.item():.4f}")
print(f"y* = {y_star.item():.4f}")
print(f"z* = {z_star.item():.4f}")
print(f"d(loss)/da = {a_tch.grad.item():.4f}")
```

## How It Works

Internally, CVXPYlayers:

1. **Log-transforms** positive parameters before solving
2. Solves the equivalent **convex problem** in log-space
3. **Exponentiates** the solution to get back to original space
4. Computes gradients through this chain

This is handled automatically when `gp=True`.

## DGP Requirements

For a problem to work with `gp=True`:

1. **Check with `is_dgp(dpp=True)`**:
   ```python
   assert problem.is_dgp(dpp=True), "Problem must be DGP with DPP"
   ```

2. **Variables must be positive**:
   ```python
   x = cp.Variable(pos=True)  # Correct
   x = cp.Variable()          # Wrong for GP
   ```

3. **Use GP-compatible atoms**:
   - Products: `x * y`
   - Powers: `x ** c` (where c is a constant or parameter)
   - Sums of posynomials
   - Divisions
   - `cp.geo_mean`, `cp.pf_eigenvalue`

## Example: Optimal Sizing

A classic GP application is optimal component sizing:

```python
import cvxpy as cp
import torch
from cvxpylayers.torch import CvxpyLayer

# Design a box with minimum surface area for given volume
# Variables: dimensions
w = cp.Variable(pos=True)  # width
h = cp.Variable(pos=True)  # height
d = cp.Variable(pos=True)  # depth

# Parameters
min_volume = cp.Parameter(pos=True)
aspect_max = cp.Parameter(pos=True)  # max aspect ratio

# Objective: minimize surface area
surface_area = 2 * (w * h + h * d + d * w)
objective = cp.Minimize(surface_area)

# Constraints
constraints = [
    w * h * d >= min_volume,      # Volume constraint
    w <= aspect_max * h,          # Aspect ratio
    h <= aspect_max * d,
]

problem = cp.Problem(objective, constraints)
assert problem.is_dgp(dpp=True)

layer = CvxpyLayer(
    problem,
    parameters=[min_volume, aspect_max],
    variables=[w, h, d],
    gp=True
)

# Find optimal dimensions
vol = torch.tensor(100.0, requires_grad=True)
aspect = torch.tensor(2.0, requires_grad=True)

w_opt, h_opt, d_opt = layer(vol, aspect)
print(f"Optimal dimensions: {w_opt.item():.2f} x {h_opt.item():.2f} x {d_opt.item():.2f}")

# Gradient: how does surface area change with volume requirement?
surface = 2 * (w_opt * h_opt + h_opt * d_opt + d_opt * w_opt)
surface.backward()
print(f"d(surface)/d(volume) = {vol.grad.item():.4f}")
```

## Citing

If you use CVXPYlayers for log-log convex programs, please cite:

```bibtex
@article{agrawal2020differentiating,
  title={Differentiating through log-log convex programs},
  author={Agrawal, Akshay and Boyd, Stephen},
  journal={arXiv},
  archivePrefix={arXiv},
  eprint={2004.12553},
  primaryClass={math.OC},
  year={2020},
}
```
