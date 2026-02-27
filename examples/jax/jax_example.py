import cvxpy as cp
import jax

from cvxpylayers.jax import CvxpyLayer

n, m = 2, 3
x = cp.Variable(n)
A = cp.Parameter((m, n))
b = cp.Parameter(m)
constraints = [x >= 0]
objective = cp.Minimize(0.5 * cp.pnorm(A @ x - b, p=1))
problem = cp.Problem(objective, constraints)
assert problem.is_dpp()

cvxpylayer = CvxpyLayer(problem, parameters=[A, b], variables=[x])
key = jax.random.PRNGKey(0)
key, k1, k2 = jax.random.split(key, 3)
A_jax = jax.random.normal(k1, shape=(m, n))
b_jax = jax.random.normal(k2, shape=(m,))

(solution,) = cvxpylayer(A_jax, b_jax)

# compute the gradient of the summed solution with respect to A, b
dcvxpylayer = jax.grad(lambda A, b: sum(cvxpylayer(A, b)[0]), argnums=[0, 1])
gradA, gradb = dcvxpylayer(A_jax, b_jax)

print("solution:", solution)
print("gradA", gradA)
print("gradb", gradb)
