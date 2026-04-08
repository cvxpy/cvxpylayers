"""
Entropy-regularized optimal transport as a differentiable layer.

Computes the optimal transport plan P between two discrete distributions,
then differentiates through P to find how to change the support points
to increase a specific coupling.
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
key = jax.random.PRNGKey(6)
k1, k2 = jax.random.split(key)
x = jax.random.normal(k1, (n,))
y = jax.random.normal(k2, (m,))
a_val = jnp.full((n,), 1.0 / n)
b_val = jnp.full((m,), 1.0 / m)
eps_val = jnp.array([1.0])

# Compute transport plan
C_val = (x[:, None] - y[None, :]) ** 2
(P_val,) = layer(C_val, a_val, b_val, eps_val)

print("Support points x:", np.array(x).round(3))
print("Support points y:", np.array(y).round(3))
print("Transport plan P:\n", np.array(P_val).round(4))


# Sensitivity: how to increase coupling P[2,2]?
def get_coupling(x, y, a_val, b_val, eps_val):
    C_val = (x[:, None] - y[None, :]) ** 2
    (P_val,) = layer(C_val, a_val, b_val, eps_val)
    return P_val[2, 2]


grads = jax.grad(get_coupling, argnums=(0, 1, 2, 3, 4))(x, y, a_val, b_val, eps_val)
dx, dy, da, db, deps = grads

print(f"\nGradients to increase P[{2},{2}] (coupling x[2]={x[2]:.3f} -> y[2]={y[2]:.3f}):")
print("  dP/dx:", np.array(dx).round(4))
print("  dP/dy:", np.array(dy).round(4))
print("  dP/da:", np.array(da).round(4))
print("  dP/db:", np.array(db).round(4))
print("  dP/deps:", np.array(deps).round(4))
