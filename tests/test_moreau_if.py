"""Tests for moreau_if.py — constant P/A batching, detection, and edge cases."""

import cvxpy as cp
import numpy as np
import pytest
import torch

from cvxpylayers.torch import CvxpyLayer

moreau = pytest.importorskip("moreau")

torch.set_default_dtype(torch.double)


# ---------------------------------------------------------------------------
# Bug 2: Constant P/A unsqueeze — setup() must get 1D (nnz,), not (1, nnz)
# ---------------------------------------------------------------------------


class TestConstantPABatched:
    """When PA_is_constant=True, setup() receives 1D tensors that Moreau
    broadcasts to any batch size. Previously .unsqueeze(0) created (1,nnz)
    which Moreau treated as batch=1, breaking batch>1 solves.
    """

    @staticmethod
    def _make_simplex_layer(n=4):
        """min c'x  s.t.  x >= 0, sum(x) == 1.  Only c is a parameter."""
        x = cp.Variable(n)
        c = cp.Parameter(n)
        prob = cp.Problem(cp.Minimize(c @ x), [x >= 0, cp.sum(x) == 1])
        layer = CvxpyLayer(prob, parameters=[c], variables=[x], solver="MOREAU")
        assert layer.ctx.solver_ctx.PA_is_constant
        return layer, n

    # -- forward --

    def test_batched_forward(self):
        layer, n = self._make_simplex_layer()
        c = torch.tensor(
            [[3., 1., 4., 2.],
             [2., 3., 1., 4.],
             [4., 2., 3., 1.]],
            requires_grad=True,
        )
        (x,) = layer(c)
        assert x.shape == (3, n)
        for i in range(3):
            expected = torch.zeros(n)
            expected[c[i].detach().argmin()] = 1.0
            assert torch.allclose(x[i], expected, atol=1e-4), f"batch {i}: {x[i]}"

    def test_batch_size_2(self):
        """Smallest batch > 1 — the exact case that triggered the old bug."""
        layer, n = self._make_simplex_layer(3)
        c = torch.tensor([[1., 5., 5.], [5., 1., 5.]], requires_grad=True)
        (x,) = layer(c)
        assert x.shape == (2, 3)
        assert torch.allclose(x[0], torch.tensor([1., 0., 0.]), atol=1e-4)
        assert torch.allclose(x[1], torch.tensor([0., 1., 0.]), atol=1e-4)

    def test_large_batch(self):
        """Batch of 16 — stress the broadcasting."""
        layer, n = self._make_simplex_layer(3)
        rng = np.random.default_rng(42)
        c_np = rng.standard_normal((16, 3))
        c = torch.tensor(c_np, requires_grad=True)
        (x,) = layer(c)
        assert x.shape == (16, 3)
        for i in range(16):
            idx = c_np[i].argmin()
            assert x[i, idx].item() > 0.99, f"batch {i}: expected weight at {idx}, got {x[i]}"

    # -- backward --

    def test_batched_backward(self):
        layer, n = self._make_simplex_layer()
        c = torch.tensor(
            [[1., 5., 5., 5.],
             [5., 1., 5., 5.]],
            requires_grad=True,
        )
        (x,) = layer(c)
        loss = x.sum()
        loss.backward()
        assert c.grad is not None
        assert c.grad.shape == c.shape
        assert torch.isfinite(c.grad).all(), f"non-finite grad: {c.grad}"

    def test_batched_gradcheck(self):
        """torch.autograd.gradcheck for batched constant-PA problem."""
        layer, n = self._make_simplex_layer(3)

        # Use well-separated costs so the optimum is non-degenerate
        c = torch.tensor([[1., 10., 10.], [10., 1., 10.]], dtype=torch.float64, requires_grad=True)

        def func(c_in):
            (x,) = layer(c_in)
            return x

        assert torch.autograd.gradcheck(func, (c,), atol=1e-3, rtol=1e-3)

    def test_batched_objective_gradient(self):
        """Gradient of c'x* w.r.t. c should be x* for simplex LP."""
        layer, _ = self._make_simplex_layer(3)
        c = torch.tensor([[1., 5., 5.], [5., 5., 1.]], requires_grad=True)
        (x,) = layer(c)
        obj = (c * x).sum()
        obj.backward()
        # d(c'x*)/dc = x* (envelope theorem)
        assert torch.allclose(c.grad, x.detach(), atol=1e-3)

    # -- multiple forward passes with varying batch sizes --

    def test_varying_batch_sizes(self):
        """Setup is cached once; solves with different batch sizes must all work."""
        layer, n = self._make_simplex_layer(3)

        for batch_size in [1, 2, 4, 1, 8, 2]:
            rng = np.random.default_rng(batch_size)
            c = torch.tensor(rng.standard_normal((batch_size, 3)), requires_grad=True)
            (x,) = layer(c)
            assert x.shape == (batch_size, 3), f"batch_size={batch_size}: shape={x.shape}"
            loss = x.sum()
            loss.backward()
            assert c.grad is not None

    def test_unbatched_still_works(self):
        """After batched calls, unbatched should still work."""
        layer, n = self._make_simplex_layer(3)

        # Batched first
        c_b = torch.tensor([[1., 5., 5.], [5., 1., 5.]], requires_grad=True)
        (x_b,) = layer(c_b)
        assert x_b.shape == (2, 3)

        # Unbatched after
        c_u = torch.tensor([5., 5., 1.], requires_grad=True)
        (x_u,) = layer(c_u)
        assert x_u.shape == (3,)
        assert torch.allclose(x_u, torch.tensor([0., 0., 1.]), atol=1e-4)


# ---------------------------------------------------------------------------
# PA_is_constant detection correctness
# ---------------------------------------------------------------------------


class TestPAConstantDetection:
    """Verify PA_is_constant is set correctly for various problem structures."""

    def test_only_linear_cost_parametrized(self):
        x = cp.Variable(3)
        c = cp.Parameter(3)
        prob = cp.Problem(cp.Minimize(c @ x), [x >= 0, cp.sum(x) == 1])
        layer = CvxpyLayer(prob, parameters=[c], variables=[x], solver="MOREAU")
        assert layer.ctx.solver_ctx.PA_is_constant

    def test_rhs_parametrized(self):
        """b is a parameter — in conic form Ax+s=b, b is embedded in the
        reduced_A parametrization matrix, so PA_is_constant should be False."""
        x = cp.Variable(3)
        b = cp.Parameter()
        prob = cp.Problem(cp.Minimize(cp.sum_squares(x)), [cp.sum(x) == b])
        layer = CvxpyLayer(prob, parameters=[b], variables=[x], solver="MOREAU")
        assert not layer.ctx.solver_ctx.PA_is_constant

    def test_constraint_matrix_parametrized(self):
        x = cp.Variable(3)
        A = cp.Parameter((1, 3))
        b = cp.Parameter(1)
        prob = cp.Problem(cp.Minimize(cp.sum_squares(x)), [A @ x == b])
        layer = CvxpyLayer(prob, parameters=[A, b], variables=[x], solver="MOREAU")
        assert not layer.ctx.solver_ctx.PA_is_constant

    def test_quadratic_cost_parametrized(self):
        x = cp.Variable(2)
        P_param = cp.Parameter((2, 2), PSD=True)
        prob = cp.Problem(cp.Minimize(cp.quad_form(x, P_param)), [cp.sum(x) == 1])
        layer = CvxpyLayer(prob, parameters=[P_param], variables=[x], solver="MOREAU")
        assert not layer.ctx.solver_ctx.PA_is_constant


# ---------------------------------------------------------------------------
# Batched vs unbatched consistency
# ---------------------------------------------------------------------------


class TestBatchUnbatchConsistency:
    """Batched solutions should match unbatched solutions element-by-element."""

    def test_forward_consistency(self):
        n = 3
        x = cp.Variable(n)
        c = cp.Parameter(n)
        prob = cp.Problem(cp.Minimize(c @ x), [x >= 0, cp.sum(x) == 1])
        layer = CvxpyLayer(prob, parameters=[c], variables=[x], solver="MOREAU")

        c_vals = [
            torch.tensor([1., 5., 5.]),
            torch.tensor([5., 1., 5.]),
            torch.tensor([5., 5., 1.]),
        ]

        # Unbatched solutions
        unbatched = []
        for cv in c_vals:
            (sol,) = layer(cv)
            unbatched.append(sol.detach())

        # Batched solution
        c_batch = torch.stack(c_vals)
        (x_batch,) = layer(c_batch)

        for i in range(3):
            assert torch.allclose(x_batch[i], unbatched[i], atol=1e-6), (
                f"batch[{i}]={x_batch[i]} vs unbatched={unbatched[i]}"
            )

    def test_gradient_consistency(self):
        """Gradients from batched solve should match unbatched."""
        n = 3
        x = cp.Variable(n)
        c = cp.Parameter(n)
        prob = cp.Problem(cp.Minimize(c @ x), [x >= 0, cp.sum(x) == 1])
        layer = CvxpyLayer(prob, parameters=[c], variables=[x], solver="MOREAU")

        c_vals_np = [[1., 10., 10.], [10., 1., 10.]]

        # Unbatched gradients
        unbatched_grads = []
        for cv_np in c_vals_np:
            cv = torch.tensor(cv_np, requires_grad=True)
            (sol,) = layer(cv)
            sol.sum().backward()
            unbatched_grads.append(cv.grad.detach().clone())

        # Batched gradients
        c_batch = torch.tensor(c_vals_np, requires_grad=True)
        (x_batch,) = layer(c_batch)
        x_batch.sum().backward()

        for i in range(2):
            assert torch.allclose(c_batch.grad[i], unbatched_grads[i], atol=1e-5), (
                f"grad batch[{i}]={c_batch.grad[i]} vs unbatched={unbatched_grads[i]}"
            )


# ---------------------------------------------------------------------------
# QP with constant P/A (P is non-None)
# ---------------------------------------------------------------------------


class TestConstantPAWithQuadratic:
    """PA_is_constant with a quadratic objective (P != None).

    min (1/2)||x||^2 + c'x  s.t.  sum(x) == 1, x >= 0
    P = I (constant), A is constant, only c is parametrized.
    """

    @staticmethod
    def _make_layer(n=3):
        x = cp.Variable(n)
        c = cp.Parameter(n)
        prob = cp.Problem(
            cp.Minimize(0.5 * cp.sum_squares(x) + c @ x),
            [x >= 0, cp.sum(x) == 1],
        )
        layer = CvxpyLayer(prob, parameters=[c], variables=[x], solver="MOREAU")
        assert layer.ctx.solver_ctx.PA_is_constant
        return layer, n

    def test_batched_forward(self):
        layer, n = self._make_layer()
        c = torch.tensor([[0., 0., 10.], [10., 0., 0.]], requires_grad=True)
        (x,) = layer(c)
        assert x.shape == (2, n)
        # With large penalty on x[2], solution should put less weight there
        assert x[0, 2].item() < x[0, 0].item()
        assert x[1, 0].item() < x[1, 2].item()

    def test_batched_backward(self):
        layer, n = self._make_layer()
        c = torch.tensor([[0., 0., 10.], [10., 0., 0.]], requires_grad=True)
        (x,) = layer(c)
        x.sum().backward()
        assert c.grad is not None
        assert torch.isfinite(c.grad).all()

    def test_batched_varying_sizes(self):
        layer, n = self._make_layer()
        for bs in [1, 3, 5, 2]:
            c = torch.randn(bs, n, dtype=torch.float64, requires_grad=True)
            (x,) = layer(c)
            assert x.shape == (bs, n)
            x.sum().backward()
            assert c.grad is not None
