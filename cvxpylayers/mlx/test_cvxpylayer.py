
# Import MLX implementation
import os
import sys

sys.path.append(os.path.dirname(__file__))
from cvxpylayer import CvxpyLayer, to_numpy  # noqa: E402
from cvxpylayers.torch import CvxpyLayer as TorchCvxpyLayer  # noqa: E402
import cvxpy as cp  # noqa: E402
import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402


# =====================All the following tests assume
# the torch cvxpylayer as golden reference/ ground truth
# =============


def _compare(a, b, atol=1e-4, rtol=1e-4):
    """Compare apple mlx and torch results."""
    a_np = to_numpy(a)
    b_np = b.detach().numpy() if isinstance(b, torch.Tensor) else np.asarray(b)
    assert np.allclose(
        a_np, b_np, atol=atol, rtol=rtol
    ), f"Mismatch:\nmlx={a_np}\ntorch={b_np}"


@pytest.mark.parametrize("n", [101])
def test_relu(n):
    x_param = cp.Parameter(n)
    y_var = cp.Variable(n)
    prob = cp.Problem(cp.Minimize(cp.sum_squares(
        y_var - x_param)), [y_var >= 0])

    # Torch CVXPY layer
    torch_layer = TorchCvxpyLayer(prob, parameters=[x_param],
                                  variables=[y_var])
    mlx_layer = CvxpyLayer(prob, parameters=[x_param], variables=[y_var])

    # Input
    x_np = np.linspace(-5, 5, n).astype(np.float32)
    x_mx = mx.array(x_np)
    x_torch = torch.tensor(x_np, requires_grad=True)

    # Forward comparison
    y_mx = mlx_layer(x_mx)
    (y_torch,) = torch_layer(x_torch)
    _compare(y_mx, y_torch)

    # Gradient comparison
    y_torch.sum().backward()  # scalar gradient hence sum
    grad_torch = x_torch.grad
    grad_loss = mx.grad(lambda x: mx.sum(mlx_layer(x)))
    grad_mx = grad_loss(x_mx)
    _compare(grad_mx, grad_torch)


@pytest.mark.parametrize("n", [100])
def test_sigmoid(n):
    x_param = cp.Parameter(n)
    y_var = cp.Variable(n)
    obj = cp.Minimize(
        -x_param.T @ y_var - cp.sum(cp.entr(y_var) + cp.entr(1.0 - y_var))
    )
    prob = cp.Problem(obj)

    torch_layer = TorchCvxpyLayer(prob, parameters=[x_param],
                                  variables=[y_var])
    mlx_layer = CvxpyLayer(prob, parameters=[x_param], variables=[y_var])

    x_np = np.linspace(-5, 5, n).astype(np.float32)
    x_mx = mx.array(x_np)
    x_torch = torch.tensor(x_np, requires_grad=True)

    y_mx = mlx_layer(x_mx)
    (y_torch,) = torch_layer(x_torch)
    _compare(y_mx, y_torch)

    y_torch.sum().backward()
    grad_torch = x_torch.grad
    grad_loss = mx.grad(lambda x: mx.sum(mlx_layer(x)))
    grad_mx = grad_loss(x_mx)
    _compare(grad_mx, grad_torch)


@pytest.mark.parametrize("n", [4])
def test_sparsemax(n):
    x = cp.Parameter(n)
    y = cp.Variable(n)
    constraint = [cp.sum(y) == 1, 0 <= y, y <= 1]
    prob = cp.Problem(cp.Minimize(cp.sum_squares(x - y)), constraint)
    torch_layer = TorchCvxpyLayer(prob, parameters=[x], variables=[y])
    mlx_layer = CvxpyLayer(prob, parameters=[x], variables=[y])
    np.random.seed(0)
    x_np = np.random.randn(n).astype(np.float32)
    x_mx = mx.array(x_np)
    x_torch = torch.tensor(x_np, requires_grad=False)
    y_mx = mlx_layer(x_mx)
    (y_torch,) = torch_layer(x_torch)
    _compare(y_mx, y_torch)


@pytest.mark.parametrize("n,k", [(4, 2)])
def test_csoftmax(n, k):
    x = cp.Parameter(n)
    y = cp.Variable(n)
    u = np.full((n,), 1.0 / k)
    constraint = [cp.sum(y) == 1.0, y <= u]
    prob = cp.Problem(cp.Minimize(-x @ y - cp.sum(cp.entr(y))), constraint)
    torch_layer = TorchCvxpyLayer(prob, parameters=[x], variables=[y])
    mlx_layer = CvxpyLayer(prob, parameters=[x], variables=[y])
    np.random.seed(0)
    x_np = np.random.randn(n).astype(np.float32)
    x_mx = mx.array(x_np)
    x_torch = torch.tensor(x_np, requires_grad=False)
    y_mx = mlx_layer(x_mx)
    (y_torch,) = torch_layer(x_torch)
    _compare(y_mx, y_torch)


@pytest.mark.parametrize("n,k", [(4, 2)])
def test_csparsemax(n, k):
    x = cp.Parameter(n)
    y = cp.Variable(n)
    u = np.full((n,), 1.0 / k)
    obj = cp.sum_squares(x - y)
    constrainnt = [cp.sum(y) == 1.0, 0.0 <= y, y <= u]
    prob = cp.Problem(cp.Minimize(obj), constrainnt)
    torch_layer = TorchCvxpyLayer(prob, [x], [y])
    mlx_layer = CvxpyLayer(prob, [x], [y])
    x_np = np.random.randn(n).astype(np.float32)
    x_mx = mx.array(x_np)
    x_torch = torch.tensor(x_np, requires_grad=False)
    y_mx = mlx_layer(x_mx)
    (y_torch,) = torch_layer(x_torch)
    _compare(y_mx, y_torch)


@pytest.mark.parametrize("n,k", [(4, 2)])
def test_limited_multilayer_proj(n, k):

    x = cp.Parameter(n)
    y = cp.Variable(n)
    obj = -x * y - cp.sum(cp.entr(y)) - cp.sum(cp.entr(1.0 - y))
    cons = [cp.sum(y) == k]
    prob = cp.Problem(cp.Minimize(obj), cons)
    torch_layer = TorchCvxpyLayer(prob, [x], [y])
    mlx_layer = CvxpyLayer(prob, [x], [y])
    x_np = np.random.randn(n).astype(np.float32)
    x_mx = mx.array(x_np)
    x_torch = torch.tensor(x_np, requires_grad=False)
    y_mx = mlx_layer(x_mx)
    (y_torch,) = torch_layer(x_torch)
    _compare(y_mx, y_torch)


@pytest.mark.parametrize("n", [2])
def test_multiple_variables_vs_torch(n):
    """Test optimization with multiple variables"""
    x = cp.Variable(n)
    y = cp.Variable(n)
    c = cp.Parameter(n)
    problem = cp.Problem(
        cp.Minimize(cp.sum_squares(x) + cp.sum_squares(y)), [x + y == c]
    )
    mlx_layer = CvxpyLayer(problem, parameters=[c], variables=[x, y])
    torch_layer = TorchCvxpyLayer(problem, parameters=[c], variables=[x, y])
    c_val_np = np.array([2.0, 4.0], dtype=np.float32)
    c_val_mx = mx.array(c_val_np)
    c_val_torch = torch.tensor(c_val_np, requires_grad=True)
    # Forward pass
    x_mx, y_mx = mlx_layer(c_val_mx)
    x_torch, y_torch = torch_layer(c_val_torch)
    # Compare forward outputs
    _compare(x_mx, x_torch)
    _compare(y_mx, y_torch)
    # Gradient comparison
    (x_torch.sum() + y_torch.sum()).backward()
    grad_loss = mx.grad(lambda c_: mx.sum(mlx_layer(c_)[0] + mlx_layer(c_)[1]))
    grad_mx = grad_loss(c_val_mx)
    _compare(grad_mx, c_val_torch.grad)


@pytest.mark.parametrize("batch_size,n,m", [(5, 3, 4)])
def test_batched_solver(batch_size, n, m):
    """Test batched problem solving using PyTorch as reference"""
    x = cp.Variable(n)
    A = cp.Parameter((m, n))
    b = cp.Parameter(m)
    problem = cp.Problem(cp.Minimize(cp.sum_squares(A @ x - b)), [x >= 0])
    mlx_layer = CvxpyLayer(problem, parameters=[A, b], variables=[x])
    torch_layer = TorchCvxpyLayer(problem, parameters=[A, b], variables=[x])
    np.random.seed(42)
    A_batch_np = np.random.randn(batch_size, m, n).astype(np.float32)
    b_batch_np = np.random.randn(batch_size, m).astype(np.float32)
    A_batch_mx = mx.array(A_batch_np)
    b_batch_mx = mx.array(b_batch_np)
    A_batch_torch = torch.tensor(A_batch_np, requires_grad=True)
    b_batch_torch = torch.tensor(b_batch_np, requires_grad=True)
    # Forward pass
    y_mx = mlx_layer(A_batch_mx, b_batch_mx)
    (y_torch,) = torch_layer(A_batch_torch, b_batch_torch)
    _compare(y_mx, y_torch)
    # Gradient comparison
    y_torch.sum().backward()
    grad_loss_A = mx.grad(
        lambda A_: mx.sum(mlx_layer(A_, b_batch_mx)))(A_batch_mx)
    grad_loss_b = mx.grad(
        lambda b_: mx.sum(mlx_layer(A_batch_mx, b_)))(b_batch_mx)
    _compare(grad_loss_A, A_batch_torch.grad)
    _compare(grad_loss_b, b_batch_torch.grad)


@pytest.mark.parametrize("n", [4])
def test_ellipsoid_projection(n):
    """Test a QP with two variables and constraints ,
    which is an ellipsoid projection"""
    # Define problem
    _A = cp.Parameter((n, n))
    _z = cp.Parameter(n)
    _x = cp.Parameter(n)
    _y = cp.Variable(n)
    _t = cp.Variable(n)
    obj = cp.Minimize(0.5 * cp.sum_squares(_x - _y))
    cons = [0.5 * cp.sum_squares(_A * _t) <= 1, _t == (_y - _z)]
    prob = cp.Problem(obj, cons)
    # MLX and Torch layers
    mlx_layer = CvxpyLayer(prob, parameters=[_A, _z, _x], variables=[_y, _t])
    torch_layer = TorchCvxpyLayer(prob, parameters=[_A, _z, _x],
                                  variables=[_y, _t])
    # Random input
    torch.manual_seed(0)
    A_val = torch.randn(n, n, requires_grad=True)
    z_val = torch.randn(n, requires_grad=True)
    x_val = torch.randn(n, requires_grad=True)

    # Forward pass
    y_torch, t_torch = torch_layer(A_val, z_val, x_val)

    A_mx = mx.array(A_val.detach().numpy())
    z_mx = mx.array(z_val.detach().numpy())
    x_mx = mx.array(x_val.detach().numpy())

    y_mx, t_mx = mlx_layer(A_mx, z_mx, x_mx)

    # Compare outputs
    _compare(y_mx, y_torch)
    _compare(t_mx, t_torch)

    # Gradients
    (y_torch.sum() + t_torch.sum()).backward()

    # MLX gradients per parameter
    grad_y_A = mx.grad(
        lambda A_: mx.sum(
            mlx_layer(
                A_, z_mx, x_mx)[0] + mlx_layer(
                    A_, z_mx, x_mx)[1])
    )(A_mx)
    grad_y_z = mx.grad(
        lambda z_: mx.sum(
            mlx_layer(
                A_mx, z_, x_mx)[0] + mlx_layer(
                    A_mx, z_, x_mx)[1])
    )(z_mx)
    grad_y_x = mx.grad(
        lambda x_: mx.sum(
            mlx_layer(
                A_mx, z_mx, x_)[0] + mlx_layer(A_mx, z_mx, x_)[1])
    )(x_mx)
    # Compare gradients
    _compare(grad_y_A, A_val.grad)
    _compare(grad_y_z, z_val.grad)
    _compare(grad_y_x, x_val.grad)
