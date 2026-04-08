"""
Batched least-squares as a convex optimization layer.
x* = argmin ||Ax-b||_2^2, solved for a batch of (A, b) pairs.
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
batch_size = 4

A = cp.Parameter((m, n))
b = cp.Parameter(m)
x = cp.Variable(n)
objective = cp.Minimize(cp.sum_squares(A @ x - b))
problem = cp.Problem(objective)
layer = CvxpyLayer(problem, parameters=[A, b], variables=[x], solver=args.solver)

key = jax.random.PRNGKey(0)
k1, k2 = jax.random.split(key)
A_batch = jax.random.normal(k1, (batch_size, m, n))
b_batch = jax.random.normal(k2, (batch_size, m))

x_batch, = layer(A_batch, b_batch)
print("x_batch shape:", x_batch.shape)
print("x_batch:\n", x_batch)


def loss_fn(A_batch, b_batch):
    x_batch, = layer(A_batch, b_batch)
    return x_batch.sum()


dA, db = jax.grad(loss_fn, argnums=(0, 1))(A_batch, b_batch)
print("dL/dA shape:", dA.shape)
print("dL/db shape:", db.shape)
