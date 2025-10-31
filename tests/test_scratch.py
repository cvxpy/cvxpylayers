from cvxpylayers.torch import CvxpyLayer
import cvxpy as cp
import torch 
from torch.autograd import grad
m, n = 100, 20
X = cp.Parameter((m, n))
y = cp.Parameter(m)
b = cp.Variable(n)
prob = cp.Problem(cp.Minimize(cp.sum_squares(X @ b - y) + cp.norm(b, 1)))



layer = CvxpyLayer(prob, [X, y], [b])
X_th = torch.randn(m, n).to(torch.float64).requires_grad_()
y_th= (X_th @ torch.ones(n, dtype=torch.float64) + 0.1 * torch.randn(m).to(torch.float64)).requires_grad_()
x = layer(X_th, y_th)[0]

grad_X_cvxpy, grad_y_cvxpy = grad(x.sum(), [X_th, y_th])
