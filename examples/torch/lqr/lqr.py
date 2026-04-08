"""
Learning the LQR value function via a differentiable optimization layer.

We learn P_sqrt such that the policy u = argmin u'Ru + ||P_sqrt(Ax+Bu)||^2
matches the optimal LQR policy. P_sqrt is trained by differentiating through
batched trajectory rollouts to minimize cost.
"""

import argparse
import time
import torch
import cvxpy as cp
import numpy as np
from cvxpylayers.torch import CvxpyLayer
from scipy.linalg import sqrtm, solve_discrete_are

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--solver", default="DIFFCP", help="DIFFCP, MOREAU, CUCLARABEL, MPAX")
args = parser.parse_args()

torch.set_default_dtype(torch.double)
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
K_lqr = np.linalg.solve(R0 + B.T @ P_lqr @ B, -B.T @ P_lqr @ A)

# Define the policy as a CvxpyLayer
x_param = cp.Parameter((n, 1))
P_sqrt_param = cp.Parameter((n, n))
u_var = cp.Variable((m, 1))
x_next = cp.Variable((n, 1))
objective = cp.Minimize(cp.quad_form(u_var, R0) + cp.sum_squares(P_sqrt_param @ x_next))
constraints = [x_next == A @ x_param + B @ u_var]
policy = CvxpyLayer(cp.Problem(objective, constraints), [x_param, P_sqrt_param], [u_var], solver=args.solver)

# Torch tensors for dynamics
At = torch.from_numpy(A)
Bt = torch.from_numpy(B)
Qt = torch.from_numpy(Q0)
Rt = torch.from_numpy(R0)


def rollout_cost(P_sqrt, time_horizon=50, batch_size=32, seed=None):
    """Simulate trajectories and return average cost."""
    if seed is not None:
        torch.manual_seed(seed)
    x = torch.randn(batch_size, n, 1)
    P_sqrt_batch = P_sqrt.expand(batch_size, -1, -1)
    total = 0.0
    for _ in range(time_horizon):
        u, = policy(x, P_sqrt_batch)
        state_cost = (x.transpose(1, 2) @ Qt @ x).squeeze()
        control_cost = (u.transpose(1, 2) @ Rt @ u).squeeze()
        total += (state_cost + control_cost).sum() / (time_horizon * batch_size)
        x = At @ x + Bt @ u + noise * torch.randn(batch_size, n, 1)
    return total


# Optimal cost (baseline)
P_sqrt_lqr = torch.from_numpy(sqrtm(P_lqr))
cost_lqr = rollout_cost(P_sqrt_lqr, seed=0).item()

# Learn P_sqrt via SGD
P_sqrt = torch.eye(n, dtype=torch.double, requires_grad=True)
optimizer = torch.optim.SGD([P_sqrt], lr=0.5)

for k in range(20):
    t0 = time.time()
    with torch.no_grad():
        test_cost = rollout_cost(P_sqrt.detach(), seed=0).item()

    optimizer.zero_grad()
    cost = rollout_cost(P_sqrt, seed=k + 1)
    cost.backward()
    optimizer.step()
    dt = time.time() - t0
    n_probs = 2 * 50 * 32  # eval + train rollouts
    print(f"iter {k+1:3d} | cost gap: {test_cost - cost_lqr:8.3f} | {dt:.2f}s | {n_probs/dt:.0f} probs/s")

    if k == 10:
        optimizer = torch.optim.SGD([P_sqrt], lr=0.1)

print(f"\nOptimal LQR cost: {cost_lqr:.3f}")
print(f"Learned cost:     {test_cost:.3f}")
