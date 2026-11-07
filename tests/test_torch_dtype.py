"""Check dtype preservation when composing Torch optimization layers."""

import cvxpy as cp
import pytest

torch = pytest.importorskip("torch")

from cvxpylayers.torch import CvxpyLayer


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("batch_size", [None, 1, 3])
@pytest.mark.parametrize("grad_enabled", [False, True])
def test_output_dtype_primal_dual(dtype, batch_size, grad_enabled):
    p = cp.Parameter(2)
    shift = cp.Parameter()
    x = cp.Variable(2)
    constraint = x >= 0
    problem = cp.Problem(cp.Minimize(cp.sum_squares(x - p - shift)), [constraint])
    assert problem.is_dpp()
    layer = CvxpyLayer(problem, [p, shift], [x, constraint.dual_variables[0]])

    value = torch.tensor([-1.0, 2.0], dtype=dtype)
    if batch_size is not None:
        value = value.repeat(batch_size, 1)
    value.requires_grad_()
    offset = torch.tensor(0.25, dtype=dtype, requires_grad=True)
    primal_linear = torch.nn.Linear(2, 1, bias=False, dtype=dtype)
    dual_linear = torch.nn.Linear(2, 1, bias=False, dtype=dtype)
    with torch.no_grad():
        primal_linear.weight.copy_(torch.tensor([[1.0, 3.0]], dtype=dtype))
        dual_linear.weight.copy_(torch.tensor([[2.0, 4.0]], dtype=dtype))

    with torch.set_grad_enabled(grad_enabled):
        primal, dual = layer(value, offset, solver_args={"eps": 1e-8})
        assert primal.dtype == dtype
        assert dual.dtype == dtype
        torch.testing.assert_close(primal, (value + offset).clamp(min=0), atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(dual, (-2 * (value + offset)).clamp(min=0), atol=1e-5, rtol=1e-5)
        loss = (primal_linear(primal) + dual_linear(dual)).sum()

    if grad_enabled:
        loss.backward()
        expected_grad = torch.tensor([-4.0, 3.0], dtype=dtype).expand_as(value)
        torch.testing.assert_close(value.grad, expected_grad, atol=1e-5, rtol=1e-5)
        expected_offset_grad = torch.tensor(-float(batch_size or 1), dtype=dtype)
        torch.testing.assert_close(offset.grad, expected_offset_grad, atol=1e-5, rtol=1e-5)
    else:
        assert not loss.requires_grad


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("batch_size", [None, 1, 3])
def test_output_dtype_gp(dtype, batch_size):
    p = cp.Parameter(2, pos=True, value=[1.0, 2.0])
    x = cp.Variable(2, pos=True)
    problem = cp.Problem(cp.Minimize(cp.sum(x)), [x >= p])
    assert problem.is_dgp(dpp=True)
    layer = CvxpyLayer(problem, [p], [x], gp=True)
    value = torch.tensor([1.0, 2.0], dtype=dtype)
    if batch_size is not None:
        value = value.repeat(batch_size, 1)
    value.requires_grad_()

    (solution,) = layer(value, solver_args={"eps": 1e-8})
    assert solution.dtype == dtype
    torch.testing.assert_close(solution, value, atol=1e-5, rtol=1e-5)
    solution.sum().backward()
    torch.testing.assert_close(value.grad, torch.ones_like(value), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_output_dtype_mixed_parameters(dtype):
    p = cp.Parameter()
    q = cp.Parameter()
    x = cp.Variable()
    problem = cp.Problem(cp.Minimize(cp.square(x - p - q)))
    layer = CvxpyLayer(problem, [p, q], [x])
    left = torch.tensor(1.0, dtype=dtype, requires_grad=True)
    right = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)

    (solution,) = layer(left, right, solver_args={"eps": 1e-8})
    assert solution.dtype == torch.float64
    torch.testing.assert_close(solution, left + right, atol=1e-5, rtol=1e-5)
    solution.backward()
    torch.testing.assert_close(left.grad, torch.ones_like(left), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(right.grad, torch.ones_like(right), atol=1e-5, rtol=1e-5)


def test_integer_parameter_retains_fractional_solution():
    p = cp.Parameter()
    x = cp.Variable()
    problem = cp.Problem(cp.Minimize(cp.square(2 * x - p)))
    layer = CvxpyLayer(problem, [p], [x])

    (solution,) = layer(torch.tensor(3), solver_args={"eps": 1e-8})
    assert solution.is_floating_point()
    torch.testing.assert_close(
        solution, torch.tensor(1.5, dtype=solution.dtype), atol=1e-5, rtol=1e-5
    )
