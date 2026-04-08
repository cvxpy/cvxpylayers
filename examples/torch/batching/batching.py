"""
Batched least-squares as a convex optimization layer.
x* = argmin ||Ax-b||_2^2, solved for a batch of (A, b) pairs.
"""

import argparse
import torch
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--solver", default="DIFFCP", help="DIFFCP, MOREAU, CUCLARABEL, MPAX")
args = parser.parse_args()

torch.set_default_dtype(torch.double)

n, m = 2, 3
batch_size = 4

A = cp.Parameter((m, n))
b = cp.Parameter(m)
x = cp.Variable(n)
objective = cp.Minimize(cp.sum_squares(A @ x - b))
problem = cp.Problem(objective)
layer = CvxpyLayer(problem, parameters=[A, b], variables=[x], solver=args.solver)

torch.manual_seed(0)
A_batch = torch.randn(batch_size, m, n, requires_grad=True)
b_batch = torch.randn(batch_size, m, requires_grad=True)

x_batch, = layer(A_batch, b_batch)
print("x_batch shape:", x_batch.shape)
print("x_batch:\n", x_batch)

x_batch.sum().backward()
print("dL/dA shape:", A_batch.grad.shape)
print("dL/db shape:", b_batch.grad.shape)
