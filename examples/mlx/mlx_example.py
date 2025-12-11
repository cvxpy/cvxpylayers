import cvxpy as cp
import mlx.core as mx

from cvxpylayers.mlx import CvxpyLayer

n, m = 2, 3
x = cp.Variable(n)
A = cp.Parameter((m, n))
b = cp.Parameter(m)
constraints = [x >= 0]
objective = cp.Minimize(0.5 * cp.pnorm(A @ x - b, p=1))
problem = cp.Problem(objective, constraints)
assert problem.is_dpp()

cvxpylayer = CvxpyLayer(problem, parameters=[A, b], variables=[x])

# Generate random parameters
mx.random.seed(0)
A_mlx = mx.random.normal(shape=(m, n))
b_mlx = mx.random.normal(shape=(m,))

# Solve the problem
(solution,) = cvxpylayer(A_mlx, b_mlx)

# Compute the gradient of the summed solution with respect to A, b
def loss_fn(A, b):
    (sol,) = cvxpylayer(A, b)
    return mx.sum(sol)

grad_fn = mx.grad(loss_fn, argnums=[0, 1])
gradA, gradb = grad_fn(A_mlx, b_mlx)

print("solution:", solution)
print("gradA:", gradA)
print("gradb:", gradb)
