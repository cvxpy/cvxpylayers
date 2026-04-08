import argparse

import cvxpy as cp
import torch

from cvxpylayers.torch import CvxpyLayer

parser = argparse.ArgumentParser()
parser.add_argument("--solver", type=str, default=None, help="DIFFCP, MOREAU, CUCLARABEL, MPAX")
args = parser.parse_args()

solver = getattr(cp, args.solver) if args.solver else None

n, m = 2, 3
x = cp.Variable(n)
A = cp.Parameter((m, n))
b = cp.Parameter(m)
constraints = [x >= 0]
objective = cp.Minimize(0.5 * cp.pnorm(A @ x - b, p=1))
problem = cp.Problem(objective, constraints)
assert problem.is_dpp()

cvxpylayer = CvxpyLayer(problem, parameters=[A, b], variables=[x], solver=solver)
A_tch = torch.randn(m, n, requires_grad=True)
b_tch = torch.randn(m, requires_grad=True)

# solve the problem
(solution,) = cvxpylayer(A_tch, b_tch)

# compute the gradient of the sum of the solution with respect to A, b
solution.sum().backward()
print(solution)
print(A_tch.grad)
print(b_tch.grad)
