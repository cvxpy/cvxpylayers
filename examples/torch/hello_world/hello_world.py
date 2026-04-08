"""
Least-squares as a convex optimization layer.
x* = argmin ||Ax-b||_2^2
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

A = cp.Parameter((m, n))
b = cp.Parameter(m)
x = cp.Variable(n)
objective = cp.Minimize(cp.sum_squares(A @ x - b))
problem = cp.Problem(objective)
layer = CvxpyLayer(problem, parameters=[A, b], variables=[x], solver=args.solver)

torch.manual_seed(0)
A_val = torch.randn(m, n, requires_grad=True)
b_val = torch.randn(m, requires_grad=True)
x, = layer(A_val, b_val)
print("With cvxpylayers:")
print("x", x)
x.sum().backward()
print("dL/dA", A_val.grad)
print("dL/db", b_val.grad)

torch.manual_seed(0)
A_val = torch.randn(m, n, requires_grad=True)
b_val = torch.randn(m, requires_grad=True)
x_tch = torch.linalg.lstsq(A_val, b_val).solution
print("With torch.linalg.lstsq:")
print("x", x_tch)
x_tch.sum().backward()
print("dL/dA", A_val.grad)
print("dL/db", b_val.grad)

