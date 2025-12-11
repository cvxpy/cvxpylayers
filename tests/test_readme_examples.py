"""Tests for README examples to ensure documentation stays accurate."""

import importlib.util

import cvxpy as cp
import pytest

# Check if modules are available
HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_JAX = importlib.util.find_spec("jax") is not None


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
class TestPyTorchExamples:
    """Test PyTorch examples from README."""

    def test_basic_pytorch(self):
        """Test basic PyTorch example."""
        import torch

        from cvxpylayers.torch import CvxpyLayer

        n, m = 2, 3
        x = cp.Variable(n)
        A = cp.Parameter((m, n))
        b = cp.Parameter(m)
        constraints = [x >= 0]
        objective = cp.Minimize(0.5 * cp.pnorm(A @ x - b, p=1))
        problem = cp.Problem(objective, constraints)
        assert problem.is_dpp()

        layer = CvxpyLayer(problem, parameters=[A, b], variables=[x])
        A_tch = torch.randn(m, n, requires_grad=True)
        b_tch = torch.randn(m, requires_grad=True)

        # solve the problem
        (solution,) = layer(A_tch, b_tch)

        # compute the gradient of the sum of the solution with respect to A, b
        solution.sum().backward()

        assert solution is not None
        assert A_tch.grad is not None
        assert b_tch.grad is not None

    def test_pytorch_gpu_cuclarabel(self):
        """Test PyTorch GPU example with CuClarabel."""
        import torch

        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        if "CUCLARABEL" not in cp.installed_solvers():
            pytest.skip("CUCLARABEL not installed")

        from cvxpylayers.torch import CvxpyLayer

        n, m = 2, 3
        x = cp.Variable(n)
        A = cp.Parameter((m, n))
        b = cp.Parameter(m)
        constraints = [x >= 0]
        objective = cp.Minimize(0.5 * cp.pnorm(A @ x - b, p=1))
        problem = cp.Problem(objective, constraints)
        assert problem.is_dpp()

        device = torch.device("cuda")
        layer = CvxpyLayer(
            problem, parameters=[A, b], variables=[x], solver=cp.CUCLARABEL
        ).to(device)
        A_tch = torch.randn(m, n, requires_grad=True, device=device)
        b_tch = torch.randn(m, requires_grad=True, device=device)

        # solve the problem
        (solution,) = layer(A_tch, b_tch)

        # compute the gradient of the sum of the solution with respect to A, b
        solution.sum().backward()

        assert solution is not None
        assert A_tch.grad is not None
        assert b_tch.grad is not None

    def test_log_log_convex_programs(self):
        """Test log-log convex programs (geometric programs) example."""
        import torch

        from cvxpylayers.torch import CvxpyLayer

        x = cp.Variable(pos=True)
        y = cp.Variable(pos=True)
        z = cp.Variable(pos=True)

        a = cp.Parameter(pos=True, value=2.0)
        b = cp.Parameter(pos=True, value=1.0)
        c = cp.Parameter(value=0.5)

        objective_fn = 1 / (x * y * z)
        objective = cp.Minimize(objective_fn)
        constraints = [a * (x * y + x * z + y * z) <= b, x >= y**c]
        problem = cp.Problem(objective, constraints)
        assert problem.is_dgp(dpp=True)

        layer = CvxpyLayer(problem, parameters=[a, b, c], variables=[x, y, z], gp=True)
        a_tch = torch.tensor(a.value, requires_grad=True)
        b_tch = torch.tensor(b.value, requires_grad=True)
        c_tch = torch.tensor(c.value, requires_grad=True)

        x_star, y_star, z_star = layer(a_tch, b_tch, c_tch)
        sum_of_solution = x_star + y_star + z_star
        sum_of_solution.backward()

        assert x_star is not None
        assert y_star is not None
        assert z_star is not None


@pytest.mark.skipif(not HAS_JAX, reason="jax not installed")
class TestJAXExamples:
    """Test JAX examples from README."""

    def test_basic_jax(self):
        """Test basic JAX example."""
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

        layer = CvxpyLayer(problem, parameters=[A, b], variables=[x])
        key = jax.random.PRNGKey(0)
        key, k1, k2 = jax.random.split(key, 3)
        A_jax = jax.random.normal(k1, shape=(m, n))
        b_jax = jax.random.normal(k2, shape=(m,))

        (solution,) = layer(A_jax, b_jax)

        # compute the gradient of the summed solution with respect to A, b
        dlayer = jax.grad(lambda A, b: sum(layer(A, b)[0]), argnums=[0, 1])
        gradA, gradb = dlayer(A_jax, b_jax)

        assert solution is not None
        assert gradA is not None
        assert gradb is not None
