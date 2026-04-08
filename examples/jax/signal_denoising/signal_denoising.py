"""
Signal denoising with learned regularization.

Learn a weighting matrix theta and regularization strength lambda for
a total-variation denoising layer, trained end-to-end on noisy cosine signals.
"""

import argparse
import math
import time
import jax
import jax.numpy as jnp
import cvxpy as cp
import numpy as np
from cvxpylayers.jax import CvxpyLayer

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--solver", default="DIFFCP", help="DIFFCP, MOREAU, CUCLARABEL, MPAX")
args = parser.parse_args()

# Problem size
n = 50  # signal length
n_train = 192  # divisible by batch_size
n_val = 32
epochs = 20
batch_size = 32

# Generate synthetic data: noisy cosines with random frequency
np.random.seed(0)
Sigma_sqrt = 0.1 * np.random.randn(n, n)
Sigma = Sigma_sqrt.T @ Sigma_sqrt

eval_pts = np.linspace(0, 2 * math.pi, n)
inputs, targets = [], []
rng = np.random.RandomState(0)
for _ in range(n_train + n_val):
    freq = rng.uniform(1, 3)
    y = np.cos(freq * eval_pts)
    x = y + rng.multivariate_normal(np.zeros(n), Sigma)
    inputs.append(x)
    targets.append(y)

inputs = jnp.array(np.stack(inputs))
targets = jnp.array(np.stack(targets))
X_train, Y_train = inputs[:n_train], targets[:n_train]
X_val, Y_val = inputs[n_train:], targets[n_train:]

# Denoising layer:
#   min ||theta(x - y)||^2 + lambda * ||diff(y)||^2
y_var = cp.Variable(n)
x_minus_y = cp.Variable(n)
x_par = cp.Parameter(n)
theta_par = cp.Parameter((n, n))
lam_par = cp.Parameter(pos=True)
objective = cp.Minimize(
    cp.sum_squares(theta_par @ x_minus_y) + lam_par * cp.sum_squares(cp.diff(y_var))
)
constraints = [x_minus_y == x_par - y_var]
layer = CvxpyLayer(
    cp.Problem(objective, constraints), [x_par, theta_par, lam_par], [y_var],
    solver=args.solver,
)

# Learnable parameters
theta = jnp.eye(n)
lam = jnp.array(0.5)

# Adam state
import optax

optimizer = optax.adam(1e-2)
opt_state = optimizer.init((theta, lam))


def batch_loss(theta, lam, X_batch, Y_batch):
    preds, = layer(X_batch, theta, lam)
    return ((preds - Y_batch) ** 2).mean()


grad_fn = jax.grad(batch_loss, argnums=(0, 1))

for epoch in range(epochs):
    t0 = time.time()

    # Validation loss
    val_preds, = layer(X_val, theta, lam)
    val_mse = ((val_preds - Y_val) ** 2).mean().item()

    # Train (manual batching)
    key = jax.random.PRNGKey(epoch)
    perm = jax.random.permutation(key, n_train)
    train_loss = 0.0
    n_batches = 0
    for i in range(0, n_train, batch_size):
        idx = perm[i : i + batch_size]
        if len(idx) < batch_size:
            break
        X_batch = X_train[idx]
        Y_batch = Y_train[idx]
        g_theta, g_lam = grad_fn(theta, lam, X_batch, Y_batch)
        updates, opt_state = optimizer.update((g_theta, g_lam), opt_state, (theta, lam))
        theta = theta + updates[0]
        lam = lam + updates[1]
        train_loss += batch_loss(theta, lam, X_batch, Y_batch).item()
        n_batches += 1

    dt = time.time() - t0
    print(f"epoch {epoch+1:3d} | train: {train_loss/n_batches:.5f} | val: {val_mse:.5f} | {dt:.2f}s")

print(f"\nlambda: {lam.item():.4f}")
print(f"theta diag (first 5): {np.array(jnp.diag(theta)[:5]).round(3)}")
