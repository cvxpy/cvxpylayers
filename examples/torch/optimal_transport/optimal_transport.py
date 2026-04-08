"""
Entropy-regularized optimal transport as a differentiable layer.

Computes the optimal transport plan P between two discrete distributions,
then backpropagates through P to find how to change the support points
to increase a specific coupling.
"""

import argparse
import torch
import cvxpy as cp
import numpy as np
from cvxpylayers.torch import CvxpyLayer

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--solver", default="DIFFCP", help="DIFFCP, MOREAU, CUCLARABEL, MPAX")
args = parser.parse_args()

torch.set_default_dtype(torch.double)

n, m = 3, 3

# Define the OT problem
C = cp.Parameter((n, m))
a = cp.Parameter(n)
b = cp.Parameter(m)
eps = cp.Parameter(1, nonneg=True)
P = cp.Variable((n, m))

objective = cp.Minimize(cp.trace(P.T @ C) - eps * (cp.sum(cp.entr(P)) + cp.sum(P)))
constraints = [P @ np.ones(m) == a, P.T @ np.ones(n) == b, P >= 0]
layer = CvxpyLayer(cp.Problem(objective, constraints), [C, a, b, eps], [P], solver=args.solver)

# Support points and probability vectors
torch.manual_seed(6)
x = torch.randn(n, requires_grad=True)
y = torch.randn(m, requires_grad=True)
a_val = torch.full((n,), 1.0 / n, requires_grad=True)
b_val = torch.full((m,), 1.0 / m, requires_grad=True)
eps_val = torch.tensor([1.0], requires_grad=True)

# Compute transport plan
C_val = (x[:, None] - y[None, :]).pow(2)
P_val, = layer(C_val, a_val, b_val, eps_val)

print("Support points x:", x.detach().numpy().round(3))
print("Support points y:", y.detach().numpy().round(3))
print("Transport plan P:\n", P_val.detach().numpy().round(4))

# Sensitivity: how to increase coupling P[2,2]?
P_val[2, 2].backward()
print(f"\nGradients to increase P[{2},{2}] (coupling x[2]={x[2]:.3f} -> y[2]={y[2]:.3f}):")
print("  dP/dx:", x.grad.numpy().round(4))
print("  dP/dy:", y.grad.numpy().round(4))
print("  dP/da:", a_val.grad.numpy().round(4))
print("  dP/db:", b_val.grad.numpy().round(4))
print("  dP/deps:", eps_val.grad.numpy().round(4))
