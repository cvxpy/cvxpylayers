"""Tests for quad_form with parametric P matrix.

Requires CVXPY >= 1.9 with quad_form DPP support (PR #3121).
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import cvxpy as cp
from cvxpy.utilities import scopes

# Skip entire module if quad_form_dpp_scope not available
pytestmark = pytest.mark.skipif(
    not hasattr(scopes, "quad_form_dpp_scope"),
    reason="Requires CVXPY >= 1.9 with quad_form DPP support",
)

from cvxpylayers.torch import CvxpyLayer


class TestQuadFormParametricP:
    """Test quad_form(x, P) where P is a Parameter."""

    def test_forward_pass(self):
        """Test that forward pass produces correct solution."""
        n = 2
        x = cp.Variable(n)
        P = cp.Parameter((n, n), PSD=True)
        q = cp.Parameter(n)

        objective = cp.Minimize(cp.quad_form(x, P) + q @ x)
        constraints = [cp.sum(x) == 1, x >= 0]
        problem = cp.Problem(objective, constraints)

        layer = CvxpyLayer(problem, parameters=[P, q], variables=[x], solver="MOREAU")

        P_t = torch.tensor([[2.0, 0.5], [0.5, 1.0]], dtype=torch.float64)
        q_t = torch.tensor([0.1, 0.2], dtype=torch.float64)

        (x_out,) = layer(P_t, q_t)

        # Verify against CVXPY
        P.value = P_t.numpy()
        q.value = q_t.numpy()
        problem.solve()

        np.testing.assert_allclose(x_out.detach().numpy(), x.value, rtol=1e-5)

    def test_backward_pass_symmetric_fd(self):
        """Test gradients using symmetric finite differences.

        Standard torch.autograd.gradcheck fails for PSD parameters because it
        perturbs P[i,j] and P[j,i] independently, violating symmetry.

        For PSD parameters, we must perturb both off-diagonal entries together
        to maintain symmetry.
        """
        n = 2
        x = cp.Variable(n)
        P = cp.Parameter((n, n), PSD=True)
        q = cp.Parameter(n)

        objective = cp.Minimize(cp.quad_form(x, P) + q @ x)
        constraints = [cp.sum(x) == 1, x >= 0]
        problem = cp.Problem(objective, constraints)

        layer = CvxpyLayer(problem, parameters=[P, q], variables=[x], solver="MOREAU")

        P_t = torch.tensor([[2.0, 0.5], [0.5, 1.0]], dtype=torch.float64, requires_grad=True)
        q_t = torch.tensor([0.1, 0.2], dtype=torch.float64, requires_grad=True)

        (x_out,) = layer(P_t, q_t)
        loss = x_out[0]  # Gradient of x[0] w.r.t. parameters
        loss.backward()

        autograd_P = P_t.grad.clone()
        autograd_q = q_t.grad.clone()

        # Symmetric finite difference for P
        eps = 1e-5
        fd_grad_P = torch.zeros_like(P_t)

        for i in range(n):
            for j in range(i, n):  # Only upper triangle
                P_plus = P_t.detach().clone()
                P_minus = P_t.detach().clone()

                P_plus[i, j] += eps
                P_minus[i, j] -= eps
                if i != j:  # Off-diagonal: perturb both to maintain symmetry
                    P_plus[j, i] += eps
                    P_minus[j, i] -= eps

                (x_plus,) = layer(P_plus, q_t.detach())
                (x_minus,) = layer(P_minus, q_t.detach())

                grad = (x_plus[0] - x_minus[0]) / (2 * eps)
                fd_grad_P[i, j] = grad
                if i != j:
                    # For off-diagonal, we perturbed both entries by eps each,
                    # so the gradient per entry is half the total
                    fd_grad_P[i, j] = grad / 2
                    fd_grad_P[j, i] = grad / 2

        # Compare autograd to symmetric finite difference
        for i in range(n):
            for j in range(i, n):
                if i == j:
                    np.testing.assert_allclose(
                        autograd_P[i, j].item(), fd_grad_P[i, j].item(), rtol=1e-3
                    )
                else:
                    # For symmetric P, autograd splits gradient equally between
                    # P[i,j] and P[j,i]. Compare each entry.
                    np.testing.assert_allclose(
                        autograd_P[i, j].item(), fd_grad_P[i, j].item(), rtol=1e-3
                    )
                    np.testing.assert_allclose(
                        autograd_P[j, i].item(), fd_grad_P[j, i].item(), rtol=1e-3
                    )

        # Finite difference for q (no symmetry constraint)
        fd_grad_q = torch.zeros_like(q_t)
        for i in range(n):
            q_plus = q_t.detach().clone()
            q_minus = q_t.detach().clone()
            q_plus[i] += eps
            q_minus[i] -= eps

            (x_plus,) = layer(P_t.detach(), q_plus)
            (x_minus,) = layer(P_t.detach(), q_minus)

            fd_grad_q[i] = (x_plus[0] - x_minus[0]) / (2 * eps)

        np.testing.assert_allclose(autograd_q.numpy(), fd_grad_q.detach().numpy(), rtol=1e-3)

    def test_batched(self):
        """Test batched forward and backward pass."""
        n = 2
        batch_size = 3

        x = cp.Variable(n)
        P = cp.Parameter((n, n), PSD=True)
        q = cp.Parameter(n)

        objective = cp.Minimize(cp.quad_form(x, P) + q @ x)
        constraints = [cp.sum(x) == 1, x >= 0]
        problem = cp.Problem(objective, constraints)

        layer = CvxpyLayer(problem, parameters=[P, q], variables=[x], solver="MOREAU")

        # Batched inputs
        P_batch = torch.stack(
            [
                torch.tensor([[2.0, 0.5], [0.5, 1.0]], dtype=torch.float64),
                torch.tensor([[1.0, 0.2], [0.2, 2.0]], dtype=torch.float64),
                torch.tensor([[3.0, 0.1], [0.1, 1.5]], dtype=torch.float64),
            ]
        ).requires_grad_(True)

        q_batch = torch.tensor(
            [[0.1, 0.2], [-0.1, 0.3], [0.0, 0.0]], dtype=torch.float64
        ).requires_grad_(True)

        (x_batch,) = layer(P_batch, q_batch)

        assert x_batch.shape == (batch_size, n)

        # Verify each batch element against CVXPY
        for i in range(batch_size):
            P.value = P_batch[i].detach().numpy()
            q.value = q_batch[i].detach().numpy()
            problem.solve()
            np.testing.assert_allclose(
                x_batch[i].detach().numpy(), x.value, rtol=1e-5
            )

        # Test backward
        loss = x_batch.sum()
        loss.backward()

        assert P_batch.grad is not None
        assert P_batch.grad.shape == (batch_size, n, n)
        assert q_batch.grad is not None
        assert q_batch.grad.shape == (batch_size, n)
