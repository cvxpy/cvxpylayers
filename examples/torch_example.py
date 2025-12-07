import cvxpy as cp
import torch

from cvxpylayers.torch import CvxpyLayer

n, m = 2, 3
x = cp.Variable(n)
A = cp.Parameter((m, n))
b = cp.Parameter(m)
constraints = [x >= 0]
objective = cp.Minimize(0.5 * cp.pnorm(A @ x - b, p=1))
problem = cp.Problem(objective, constraints)
assert problem.is_dpp()

# On the CPU:
cvxpylayer = CvxpyLayer(problem, parameters=[A, b], variables=[x])
A_tch = torch.randn(m, n, requires_grad=True)
b_tch = torch.randn(m, requires_grad=True)

# solve the problem
(solution,) = cvxpylayer(A_tch, b_tch)

# compute the gradient of the sum of the solution with respect to A, b
solution.sum().backward()
print(solution)
print(A_tch.grad)
print(b_tch.grad)

# On the GPU:
device = torch.device("cuda")
cvxpylayer = CvxpyLayer(problem, parameters=[A, b], variables=[x], solver=cp.CUCLARABEL).to(device)
A_tch = torch.randn(m, n, requires_grad=True, device=device)
b_tch = torch.randn(m, requires_grad=True, device=device)

# solve the problem
(solution,) = cvxpylayer(A_tch, b_tch)

# compute the gradient of the sum of the solution with respect to A, b
solution.sum().backward()
print(solution)
print(A_tch.grad)
print(b_tch.grad)
