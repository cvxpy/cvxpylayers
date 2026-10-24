"""Unit tests for dual variable support in cvxpylayers (MLX backend).

These tests compare MLX results against PyTorch as the reference implementation,
following the existing MLX test pattern.
"""

import cvxpy as cp
import numpy as np
import pytest

torch = pytest.importorskip("torch")
mx = pytest.importorskip("mlx.core")

from cvxpylayers.mlx import CvxpyLayer as MLXCvxpyLayer  # noqa: E402
from cvxpylayers.torch import CvxpyLayer as TorchCvxpyLayer  # noqa: E402

torch.set_default_dtype(torch.double)


def _compare(mlx_val, torch_val, atol=1e-4, rtol=1e-4):
    """Compare MLX and torch results."""
    mlx_np = np.array(mlx_val, dtype=np.float64)
    torch_np = (
        torch_val.detach().numpy() if isinstance(torch_val, torch.Tensor) else np.asarray(torch_val)
    )
    np.testing.assert_allclose(mlx_np, torch_np, atol=atol, rtol=rtol)


def test_equality_dual_vs_torch():
    """Test equality constraint dual comparing MLX with PyTorch."""
    n = 2
    x = cp.Variable(n)
    c = cp.Parameter(n)
    b = cp.Parameter()

    eq_con = cp.sum(x) == b
    prob = cp.Problem(cp.Minimize(c @ x), [eq_con, x >= 0])

    torch_layer = TorchCvxpyLayer(
        prob,
        parameters=[c, b],
        variables=[x, eq_con.dual_variables[0]],
    )
    mlx_layer = MLXCvxpyLayer(
        prob,
        parameters=[c, b],
        variables=[x, eq_con.dual_variables[0]],
    )

    c_np = np.array([1.0, 2.0])
    b_np = np.array(1.0)

    c_torch = torch.tensor(c_np)
    b_torch = torch.tensor(b_np)
    x_torch, eq_dual_torch = torch_layer(c_torch, b_torch)

    c_mlx = mx.array(c_np)
    b_mlx = mx.array(b_np)
    x_mlx, eq_dual_mlx = mlx_layer(c_mlx, b_mlx)

    _compare(x_mlx, x_torch)
    _compare(eq_dual_mlx, eq_dual_torch)


def test_inequality_dual_vs_torch():
    """Test inequality constraint dual comparing MLX with PyTorch."""
    n = 2
    x = cp.Variable(n)
    c = cp.Parameter(n)

    ineq_con = x >= 0
    prob = cp.Problem(cp.Minimize(c @ x + cp.sum_squares(x)), [ineq_con])

    torch_layer = TorchCvxpyLayer(
        prob,
        parameters=[c],
        variables=[x, ineq_con.dual_variables[0]],
    )
    mlx_layer = MLXCvxpyLayer(
        prob,
        parameters=[c],
        variables=[x, ineq_con.dual_variables[0]],
    )

    c_np = np.array([1.0, -1.0])

    c_torch = torch.tensor(c_np)
    x_torch, ineq_dual_torch = torch_layer(c_torch)

    c_mlx = mx.array(c_np)
    x_mlx, ineq_dual_mlx = mlx_layer(c_mlx)

    _compare(x_mlx, x_torch)
    _compare(ineq_dual_mlx, ineq_dual_torch)


def test_multiple_duals_vs_torch():
    """Test multiple dual variables comparing MLX with PyTorch."""
    n = 2
    x = cp.Variable(n)
    c = cp.Parameter(n)
    b = cp.Parameter()

    eq_con = cp.sum(x) == b
    ineq_con = x >= 0
    prob = cp.Problem(cp.Minimize(c @ x), [eq_con, ineq_con])

    torch_layer = TorchCvxpyLayer(
        prob,
        parameters=[c, b],
        variables=[x, eq_con.dual_variables[0], ineq_con.dual_variables[0]],
    )
    mlx_layer = MLXCvxpyLayer(
        prob,
        parameters=[c, b],
        variables=[x, eq_con.dual_variables[0], ineq_con.dual_variables[0]],
    )

    c_np = np.array([1.0, 2.0])
    b_np = np.array(1.0)

    c_torch = torch.tensor(c_np)
    b_torch = torch.tensor(b_np)
    x_torch, eq_dual_torch, ineq_dual_torch = torch_layer(c_torch, b_torch)

    c_mlx = mx.array(c_np)
    b_mlx = mx.array(b_np)
    x_mlx, eq_dual_mlx, ineq_dual_mlx = mlx_layer(c_mlx, b_mlx)

    _compare(x_mlx, x_torch)
    _compare(eq_dual_mlx, eq_dual_torch)
    _compare(ineq_dual_mlx, ineq_dual_torch)


def test_batched_dual_vs_torch():
    """Test batched dual variables comparing MLX with PyTorch."""
    n = 2
    batch_size = 3

    x = cp.Variable(n)
    c = cp.Parameter(n)
    b = cp.Parameter()

    eq_con = cp.sum(x) == b
    prob = cp.Problem(cp.Minimize(c @ x), [eq_con, x >= 0])

    torch_layer = TorchCvxpyLayer(
        prob,
        parameters=[c, b],
        variables=[x, eq_con.dual_variables[0]],
    )
    mlx_layer = MLXCvxpyLayer(
        prob,
        parameters=[c, b],
        variables=[x, eq_con.dual_variables[0]],
    )

    np.random.seed(42)
    c_np = np.random.randn(batch_size, n)
    b_np = np.ones(batch_size)

    c_torch = torch.tensor(c_np)
    b_torch = torch.tensor(b_np)
    x_torch, eq_dual_torch = torch_layer(c_torch, b_torch)

    c_mlx = mx.array(c_np)
    b_mlx = mx.array(b_np)
    x_mlx, eq_dual_mlx = mlx_layer(c_mlx, b_mlx)

    assert x_mlx.shape == (batch_size, n)
    assert eq_dual_mlx.shape == (batch_size,)

    _compare(x_mlx, x_torch)
    _compare(eq_dual_mlx, eq_dual_torch)


def test_soc_dual_vs_torch():
    """Test SOC dual comparing MLX with PyTorch."""
    n = 3
    x = cp.Variable(n)
    c = cp.Parameter(n)
    t = cp.Parameter(nonneg=True)

    soc_con = cp.norm(x) <= t
    prob = cp.Problem(cp.Minimize(c @ x), [soc_con])

    torch_layer = TorchCvxpyLayer(
        prob,
        parameters=[c, t],
        variables=[x, soc_con.dual_variables[0]],
    )
    mlx_layer = MLXCvxpyLayer(
        prob,
        parameters=[c, t],
        variables=[x, soc_con.dual_variables[0]],
    )

    c_np = np.array([1.0, 0.5, -0.5])
    t_np = np.array(2.0)

    c_torch = torch.tensor(c_np)
    t_torch = torch.tensor(t_np)
    x_torch, soc_dual_torch = torch_layer(c_torch, t_torch)

    c_mlx = mx.array(c_np)
    t_mlx = mx.array(t_np)
    x_mlx, soc_dual_mlx = mlx_layer(c_mlx, t_mlx)

    _compare(x_mlx, x_torch)
    _compare(soc_dual_mlx, soc_dual_torch)


def test_psd_dual_vs_torch():
    """Test PSD dual comparing MLX with PyTorch."""
    n = 2
    X = cp.Variable((n, n), symmetric=True)
    C = cp.Parameter((n, n), symmetric=True)

    psd_con = X >> 0
    trace_con = cp.trace(X) == 1
    prob = cp.Problem(cp.Minimize(cp.trace(C @ X)), [psd_con, trace_con])

    torch_layer = TorchCvxpyLayer(
        prob,
        parameters=[C],
        variables=[X, psd_con.dual_variables[0]],
    )
    mlx_layer = MLXCvxpyLayer(
        prob,
        parameters=[C],
        variables=[X, psd_con.dual_variables[0]],
    )

    C_np = np.array([[1.0, 0.5], [0.5, 2.0]])

    C_torch = torch.tensor(C_np)
    X_torch, psd_dual_torch = torch_layer(C_torch)

    C_mlx = mx.array(C_np)
    X_mlx, psd_dual_mlx = mlx_layer(C_mlx)

    _compare(X_mlx, X_torch)
    _compare(psd_dual_mlx, psd_dual_torch)


def test_gp_dual_vs_torch():
    """Test GP dual comparing MLX with PyTorch."""
    x = cp.Variable(pos=True)
    y = cp.Variable(pos=True)
    z = cp.Variable(pos=True)

    a = cp.Parameter(pos=True)
    b = cp.Parameter(pos=True)

    ineq_con = a * (x * y + x * z + y * z) <= b
    prob = cp.Problem(cp.Minimize(1 / (x * y * z)), [ineq_con])

    torch_layer = TorchCvxpyLayer(
        prob,
        parameters=[a, b],
        variables=[x, y, z, ineq_con.dual_variables[0]],
        gp=True,
    )
    mlx_layer = MLXCvxpyLayer(
        prob,
        parameters=[a, b],
        variables=[x, y, z, ineq_con.dual_variables[0]],
        gp=True,
    )

    a_np = np.array(2.0)
    b_np = np.array(1.0)

    a_torch = torch.tensor(a_np)
    b_torch = torch.tensor(b_np)
    x_torch, y_torch, z_torch, dual_torch = torch_layer(a_torch, b_torch)

    a_mlx = mx.array(a_np)
    b_mlx = mx.array(b_np)
    x_mlx, y_mlx, z_mlx, dual_mlx = mlx_layer(a_mlx, b_mlx)

    _compare(x_mlx, x_torch)
    _compare(y_mlx, y_torch)
    _compare(z_mlx, z_torch)
    _compare(dual_mlx, dual_torch)


def test_dual_gradient():
    """Test gradient computation through dual variables."""
    n = 2
    x = cp.Variable(n)
    c = cp.Parameter(n)
    b = cp.Parameter()

    eq_con = cp.sum(x) == b
    prob = cp.Problem(cp.Minimize(c @ x + 0.5 * cp.sum_squares(x)), [eq_con])

    layer = MLXCvxpyLayer(
        prob,
        parameters=[c, b],
        variables=[x, eq_con.dual_variables[0]],
    )

    c_mlx = mx.array([1.0, 2.0])
    b_mlx = mx.array(1.0)

    def loss_fn(c, b):
        _, eq_dual = layer(c, b)
        return mx.sum(eq_dual)

    # Compute gradients using MLX
    grad_fn = mx.grad(loss_fn, argnums=[0, 1])
    grads = grad_fn(c_mlx, b_mlx)

    # Check that gradients exist and are finite
    assert grads[0] is not None
    assert grads[1] is not None
    assert mx.all(mx.isfinite(grads[0]))
    assert mx.isfinite(grads[1])
