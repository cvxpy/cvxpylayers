"""
Least-squares as a convex optimization layer.
x* = argmin ||Ax-b||_2^2
"""

import argparse
import jax
import jax.numpy as jnp
import cvxpy as cp
import numpy as np
from cvxpylayers.jax import CvxpyLayer

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--solver", default="DIFFCP", help="DIFFCP, MOREAU, CUCLARABEL, MPAX")
args = parser.parse_args()

n, m = 2, 3

A = cp.Parameter((m, n))
b = cp.Parameter(m)
x = cp.Variable(n)
objective = cp.Minimize(cp.sum_squares(A @ x - b))
problem = cp.Problem(objective)
layer = CvxpyLayer(problem, parameters=[A, b], variables=[x], solver=args.solver)

key = jax.random.PRNGKey(0)
k1, k2 = jax.random.split(key)
A_val = jax.random.normal(k1, (m, n))
b_val = jax.random.normal(k2, (m,))


def loss_fn(A_val, b_val):
    (x,) = layer(A_val, b_val)
    return x.sum(), x


(_, x_sol), (dA, db) = jax.value_and_grad(loss_fn, argnums=(0, 1), has_aux=True)(A_val, b_val)
print("With cvxpylayers:")
print("x", x_sol)
print("dL/dA\n", dA)
print("dL/db", db)

# Compare with jax.numpy.linalg.lstsq
def lstsq_fn(A_val, b_val):
    x = jnp.linalg.lstsq(A_val, b_val)[0]
    return x.sum(), x

(_, x_lstsq), (dA_lstsq, db_lstsq) = jax.value_and_grad(lstsq_fn, argnums=(0, 1), has_aux=True)(A_val, b_val)
print("\nWith jnp.linalg.lstsq:")
print("x", x_lstsq)
print("dL/dA\n", dA_lstsq)
print("dL/db", db_lstsq)
