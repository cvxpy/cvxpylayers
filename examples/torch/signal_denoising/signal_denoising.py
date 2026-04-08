"""
Signal denoising with learned regularization.

Learn a weighting matrix theta and regularization strength lambda for
a total-variation denoising layer, trained end-to-end on noisy cosine signals.
"""

import argparse
import math
import time
import torch
import cvxpy as cp
import numpy as np
from cvxpylayers.torch import CvxpyLayer
from torch.utils.data import TensorDataset, DataLoader

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--solver", default="DIFFCP", help="DIFFCP, MOREAU, CUCLARABEL, MPAX")
args = parser.parse_args()

torch.set_default_dtype(torch.double)

# Problem size
n = 50  # signal length
n_train = 192  # divisible by batch_size
n_val = 32
epochs = 20
batch_size = 32

# Generate synthetic data: noisy cosines with random frequency
torch.manual_seed(0)
np.random.seed(0)
Sigma_sqrt = 0.1 * np.random.randn(n, n)
Sigma = torch.from_numpy(Sigma_sqrt.T @ Sigma_sqrt)
noise_dist = torch.distributions.MultivariateNormal(torch.zeros(n), Sigma)
eval_pts = torch.linspace(0, 2 * math.pi, n)

inputs, targets = [], []
for _ in range(n_train + n_val):
    freq = torch.empty(1).uniform_(1, 3).item()
    y = torch.cos(freq * eval_pts)
    x = y + noise_dist.sample()
    inputs.append(x)
    targets.append(y)

inputs = torch.stack(inputs)
targets = torch.stack(targets)
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
theta = torch.eye(n, requires_grad=True)
lam = torch.tensor(0.5, requires_grad=True)

optimizer = torch.optim.Adam([theta, lam], lr=1e-2)
train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=batch_size, shuffle=True, drop_last=True)

for epoch in range(epochs):
    t0 = time.time()

    # Validation loss
    with torch.no_grad():
        val_preds = layer(X_val, theta, lam)[0]
        val_mse = (val_preds - Y_val).pow(2).mean().item()

    # Train
    train_loss = 0.0
    n_batches = 0
    for X_batch, Y_batch in train_loader:
        optimizer.zero_grad()
        preds = layer(X_batch, theta, lam)[0]
        loss = (preds - Y_batch).pow(2).mean()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        n_batches += 1

    dt = time.time() - t0
    print(f"epoch {epoch+1:3d} | train: {train_loss/n_batches:.5f} | val: {val_mse:.5f} | {dt:.2f}s")

print(f"\nlambda: {lam.item():.4f}")
print(f"theta diag (first 5): {torch.diag(theta)[:5].detach().numpy().round(3)}")
