"""
Learning the LQR value function via a differentiable optimization layer.

We learn P_sqrt such that the policy u = argmin u'Ru + ||P_sqrt(Ax+Bu)||^2
matches the optimal LQR policy. P_sqrt is trained by differentiating through
batched trajectory rollouts to minimize cost.
"""

import argparse
import time
import jax
import jax.numpy as jnp
import cvxpy as cp
import numpy as np
from cvxpylayers.jax import CvxpyLayer
from scipy.linalg import sqrtm, solve_discrete_are

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--solver", default="DIFFCP", help="DIFFCP, MOREAU, CUCLARABEL, MPAX")
args = parser.parse_args()

np.random.seed(0)

# System dynamics
n, m = 4, 2
noise = 1.0
A = np.eye(n) + np.random.randn(n, n)
A /= np.max(np.abs(np.linalg.eig(A)[0]))  # stabilize
B = np.random.randn(n, m)
Q0 = np.eye(n)
R0 = np.eye(m)

# Optimal LQR solution (for comparison)
P_lqr = solve_discrete_are(A, B, Q0, R0)

# Define the policy as a CvxpyLayer
x_param = cp.Parameter((n, 1))
P_sqrt_param = cp.Parameter((n, n))
u_var = cp.Variable((m, 1))
x_next = cp.Variable((n, 1))
objective = cp.Minimize(cp.quad_form(u_var, R0) + cp.sum_squares(P_sqrt_param @ x_next))
constraints = [x_next == A @ x_param + B @ u_var]
policy = CvxpyLayer(cp.Problem(objective, constraints), [x_param, P_sqrt_param], [u_var], solver=args.solver)

At = jnp.array(A)
Bt = jnp.array(B)
Qt = jnp.array(Q0)
Rt = jnp.array(R0)

time_horizon = 50
batch_size = 32


def rollout_cost(P_sqrt, key):
    """Simulate trajectories and return average cost."""
    k1, k2 = jax.random.split(key)
    x = jax.random.normal(k1, (batch_size, n, 1))
    P_sqrt_batch = jnp.broadcast_to(P_sqrt, (batch_size, n, n))
    total = 0.0
    for t in range(time_horizon):
        (u,) = policy(x, P_sqrt_batch)
        state_cost = jnp.squeeze(x.transpose(0, 2, 1) @ Qt @ x)
        control_cost = jnp.squeeze(u.transpose(0, 2, 1) @ Rt @ u)
        total += (state_cost + control_cost).sum() / (time_horizon * batch_size)
        k2, subkey = jax.random.split(k2)
        x = At @ x + Bt @ u + noise * jax.random.normal(subkey, (batch_size, n, 1))
    return total


# Optimal cost (baseline)
P_sqrt_lqr = jnp.array(sqrtm(P_lqr))
cost_lqr = rollout_cost(P_sqrt_lqr, jax.random.PRNGKey(0)).item()

# Learn P_sqrt via gradient descent
P_sqrt = jnp.eye(n)
lr = 0.5

for k in range(20):
    t0 = time.time()

    test_cost = rollout_cost(P_sqrt, jax.random.PRNGKey(0)).item()

    grad = jax.grad(rollout_cost)(P_sqrt, jax.random.PRNGKey(k + 1))
    P_sqrt = P_sqrt - lr * grad
    dt = time.time() - t0

    n_probs = 2 * time_horizon * batch_size
    print(f"iter {k+1:3d} | cost gap: {test_cost - cost_lqr:8.3f} | {dt:.2f}s | {n_probs/dt:.0f} probs/s")

    if k == 10:
        lr = 0.1

print(f"\nOptimal LQR cost: {cost_lqr:.3f}")
print(f"Learned cost:     {test_cost:.3f}")
