"""Unit tests for cvxpylayers.torch."""

import importlib.util

import cvxpy as cp
import diffcp
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from torch.autograd import grad  # noqa: E402

from cvxpylayers.torch import CvxpyLayer  # noqa: E402

torch.set_default_dtype(torch.double)


def set_seed(x: int) -> np.random.Generator:
    """Set the random seed for torch and return a numpy random generator.

    Parameters
    ----------
    x : int
        The seed value to use for random number generators.

    Returns
    -------
    np.random.Generator
        A numpy random number generator instance with the specified seed.

    """
    torch.manual_seed(x)
    return np.random.default_rng(x)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def test_example():
    n, m = 2, 3
    x = cp.Variable(n)
    A = cp.Parameter((m, n))
    b = cp.Parameter(m)
    constraints = [x >= 0]
    objective = cp.Minimize(0.5 * cp.pnorm(A @ x - b, p=1))
    problem = cp.Problem(objective, constraints)
    assert problem.is_dpp()

    cvxpylayer = CvxpyLayer(problem, parameters=[A, b], variables=[x])
    A_tch = torch.randn(m, n, requires_grad=True)
    b_tch = torch.randn(m, requires_grad=True)

    # solve the problem
    (solution,) = cvxpylayer(A_tch, b_tch)

    # compute the gradient of the sum of the solution with respect to A, b
    solution.sum().backward()


@pytest.mark.skip
def test_simple_batch_socp():
    _ = set_seed(243)
    n = 5
    m = 1
    batch_size = 4

    P_sqrt = cp.Parameter((n, n), name="P_sqrt")
    q = cp.Parameter((n, 1), name="q")
    A = cp.Parameter((m, n), name="A")
    b = cp.Parameter((m, 1), name="b")

    x = cp.Variable((n, 1), name="x")

    objective = 0.5 * cp.sum_squares(P_sqrt @ x) + q.T @ x
    constraints = [A @ x == b, cp.norm(x) <= 1]
    prob = cp.Problem(cp.Minimize(objective), constraints)

    prob_tch = CvxpyLayer(prob, [P_sqrt, q, A, b], [x])

    P_sqrt_tch = torch.randn(batch_size, n, n, requires_grad=True)
    q_tch = torch.randn(batch_size, n, 1, requires_grad=True)
    A_tch = torch.randn(batch_size, m, n, requires_grad=True)
    b_tch = torch.randn(batch_size, m, 1, requires_grad=True)

    torch.autograd.gradcheck(prob_tch, (P_sqrt_tch, q_tch, A_tch, b_tch))


def test_least_squares():
    _ = set_seed(243)
    m, n = 100, 20

    A = cp.Parameter((m, n))
    b = cp.Parameter(m)
    x = cp.Variable(n)
    obj = cp.sum_squares(A @ x - b) + cp.sum_squares(x)
    prob = cp.Problem(cp.Minimize(obj))
    prob_th = CvxpyLayer(prob, [A, b], [x])

    A_th = torch.randn(m, n).double().requires_grad_()
    b_th = torch.randn(m).double().requires_grad_()

    x = prob_th(A_th, b_th, solver_args={"eps": 1e-10})[0]

    def lstsq(A, b):
        return torch.linalg.solve(
            A.t() @ A + torch.eye(n, dtype=torch.float64),
            (A.t() @ b).unsqueeze(1),
        )

    x_lstsq = lstsq(A_th, b_th)

    grad_A_lstsq, grad_b_lstsq = grad(x_lstsq.sum(), [A_th, b_th])
    grad_A_cvxpy, grad_b_cvxpy = grad(x.sum(), [A_th, b_th])

    assert torch.allclose(grad_A_cvxpy, grad_A_lstsq, atol=1e-6)
    assert torch.allclose(grad_b_cvxpy, grad_b_lstsq.squeeze(), atol=1e-6)


@pytest.mark.skip
def test_least_squares_custom_method():
    _ = set_seed(243)
    m, n = 100, 20

    A = cp.Parameter((m, n))
    b = cp.Parameter(m)
    x = cp.Variable(n)
    obj = cp.sum_squares(A @ x - b) + cp.sum_squares(x)
    prob = cp.Problem(cp.Minimize(obj))
    prob_th = CvxpyLayer(
        prob,
        [A, b],
        [x],
        custom_method=(forward_numpy, backward_numpy),  # noqa: F821
    )

    A_th = torch.randn(m, n).double().requires_grad_()
    b_th = torch.randn(m).double().requires_grad_()

    x = prob_th(A_th, b_th, solver_args={"eps": 1e-10})[0]

    def lstsq(A, b):
        return torch.linalg.solve(
            A.t() @ A + torch.eye(n, dtype=torch.float64),
            (A.t() @ b).unsqueeze(1),
        )

    x_lstsq = lstsq(A_th, b_th)

    grad_A_cvxpy, grad_b_cvxpy = grad(x.sum(), [A_th, b_th])
    grad_A_lstsq, grad_b_lstsq = grad(x_lstsq.sum(), [A_th, b_th])

    assert torch.allclose(grad_A_cvxpy, grad_A_lstsq, atol=1e-6)
    assert torch.allclose(grad_b_cvxpy, grad_b_lstsq.squeeze(), atol=1e-6)


def test_logistic_regression():
    rng = set_seed(0)

    N, n = 5, 2

    X_np = rng.standard_normal((N, n))
    a_true = rng.standard_normal((n, 1))
    y_np = np.round(sigmoid(X_np.dot(a_true) + rng.standard_normal((N, 1)) * 0.5))

    X_th = torch.from_numpy(X_np).requires_grad_()
    lam_th = torch.tensor([0.1]).requires_grad_()

    a = cp.Variable((n, 1))
    X = cp.Parameter((N, n))
    lam = cp.Parameter(1, nonneg=True)
    y = y_np

    log_likelihood = cp.sum(
        cp.multiply(y, X @ a)
        - cp.log_sum_exp(
            cp.hstack([np.zeros((N, 1)), X @ a]).T,
            axis=0,
            keepdims=True,
        ).T,
    )
    prob = cp.Problem(cp.Minimize(-log_likelihood + lam * cp.sum_squares(a)))

    fit_logreg = CvxpyLayer(prob, [X, lam], [a])

    torch.autograd.gradcheck(fit_logreg, (X_th, lam_th), atol=1e-4)


@pytest.mark.skip
def test_entropy_maximization():
    rng = set_seed(243)
    n, m, p = 5, 3, 2

    tmp = rng.standard_normal(n)
    A_np = rng.standard_normal((m, n))
    b_np = A_np.dot(tmp)
    F_np = rng.standard_normal((p, n))
    g_np = F_np.dot(tmp) + rng.standard_normal(p)

    x = cp.Variable(n)
    A = cp.Parameter((m, n))
    b = cp.Parameter(m)
    F = cp.Parameter((p, n))
    g = cp.Parameter(p)
    obj = cp.Maximize(cp.sum(cp.entr(x)) - 0.01 * cp.sum_squares(x))
    constraints = [A @ x == b, F @ x <= g]
    prob = cp.Problem(obj, constraints)
    layer = CvxpyLayer(prob, [A, b, F, g], [x])

    A_th, b_th, F_th, g_th = map(
        lambda x: torch.from_numpy(x).requires_grad_(),
        [A_np, b_np, F_np, g_np],
    )

    torch.autograd.gradcheck(layer, (A_th, b_th, F_th, g_th))


def test_lml():
    _ = set_seed(243)
    k = 2
    x = cp.Parameter(4)
    y = cp.Variable(4)
    obj = -x @ y - cp.sum(cp.entr(y)) - cp.sum(cp.entr(1.0 - y))
    cons = [cp.sum(y) == k]
    prob = cp.Problem(cp.Minimize(obj), cons)
    lml = CvxpyLayer(prob, [x], [y])

    x_th = torch.tensor([1.0, -1.0, -1.0, -1.0]).requires_grad_()
    torch.autograd.gradcheck(lml, (x_th,), atol=1e-3)


def test_sdp():
    """Test SDP with symmetric parameters."""
    n = 3
    X = cp.Variable((n, n), symmetric=True)
    C = cp.Parameter((n, n), symmetric=True)

    psd_con = X >> 0
    trace_con = cp.trace(X) == 1
    prob = cp.Problem(cp.Minimize(cp.trace(C @ X)), [psd_con, trace_con])

    layer = CvxpyLayer(prob, parameters=[C], variables=[X])

    # Use a well-conditioned symmetric matrix
    C_t = torch.tensor([[2.0, 0.5, 0.1], [0.5, 3.0, 0.2], [0.1, 0.2, 1.5]], requires_grad=True)

    torch.autograd.gradcheck(layer, (C_t,), atol=1e-4, rtol=1e-3)


def test_not_enough_parameters():
    x = cp.Variable(1)
    lam = cp.Parameter(1, nonneg=True)
    lam2 = cp.Parameter(1, nonneg=True)
    objective = lam * cp.norm(x, 1) + lam2 * cp.sum_squares(x)
    prob = cp.Problem(cp.Minimize(objective))
    with pytest.raises(ValueError, match="must exactly match problem.parameters"):
        layer = CvxpyLayer(prob, [lam], [x])  # noqa: F841


def test_not_enough_parameters_at_call_time():
    x = cp.Variable(1)
    lam = cp.Parameter(1, nonneg=True)
    lam2 = cp.Parameter(1, nonneg=True)
    objective = lam * cp.norm(x, 1) + lam2 * cp.sum_squares(x)
    prob = cp.Problem(cp.Minimize(objective))
    layer = CvxpyLayer(prob, [lam, lam2], [x])
    lam_th = torch.ones(1)
    with pytest.raises(
        ValueError,
        match="A tensor must be provided for each CVXPY parameter.*",
    ):
        layer(lam_th)


def test_none_parameter_at_call_time():
    """Test that passing None as a parameter raises an appropriate error."""
    x = cp.Variable(1)
    lam = cp.Parameter(1, nonneg=True)
    lam2 = cp.Parameter(1, nonneg=True)
    objective = lam * cp.norm(x, 1) + lam2 * cp.sum_squares(x)
    prob = cp.Problem(cp.Minimize(objective))
    layer = CvxpyLayer(prob, [lam, lam2], [x])
    lam_th = torch.ones(1)
    with pytest.raises(AttributeError):
        layer(lam_th, None)


def test_too_many_variables():
    x = cp.Variable(1)
    y = cp.Variable(1)
    lam = cp.Parameter(1, nonneg=True)
    objective = lam * cp.norm(x, 1)
    prob = cp.Problem(cp.Minimize(objective))
    with pytest.raises(ValueError, match="must be a subset of problem.variables"):
        layer = CvxpyLayer(prob, [lam], [x, y])  # noqa: F841


def test_infeasible():
    x = cp.Variable(1)
    param = cp.Parameter(1)
    prob = cp.Problem(cp.Minimize(param), [x >= 1, x <= -1])
    layer = CvxpyLayer(prob, [param], [x])
    param_th = torch.ones(1)
    with pytest.raises(diffcp.SolverError):
        layer(param_th)


def test_unbounded():
    x = cp.Variable(1)
    param = cp.Parameter(1)
    prob = cp.Problem(cp.Minimize(x), [x <= param])
    layer = CvxpyLayer(prob, [param], [x])
    param_th = torch.ones(1)
    with pytest.raises(diffcp.SolverError):
        layer(param_th)


def test_incorrect_parameter_shape():
    _ = set_seed(243)
    m, n = 100, 20

    A = cp.Parameter((m, n))
    b = cp.Parameter(m)
    x = cp.Variable(n)
    obj = cp.sum_squares(A @ x - b) + cp.sum_squares(x)
    prob = cp.Problem(cp.Minimize(obj))
    prob_th = CvxpyLayer(prob, [A, b], [x])

    A_th = torch.randn(32, m, n).double()
    b_th = torch.randn(20, m).double()

    with pytest.raises(ValueError, match="Inconsistent batch sizes"):
        prob_th(A_th, b_th)

    A_th = torch.randn(32, m, n).double()
    b_th = torch.randn(32, 2 * m).double()

    with pytest.raises(ValueError, match="Invalid parameter shape"):
        prob_th(A_th, b_th)

    A_th = torch.randn(m, n).double()
    b_th = torch.randn(2 * m).double()

    with pytest.raises(ValueError, match="Invalid parameter shape"):
        prob_th(A_th, b_th)

    A_th = torch.randn(32, m, n).double()
    b_th = torch.randn(32, 32, m).double()

    with pytest.raises(ValueError, match="Invalid parameter dimensionality"):
        prob_th(A_th, b_th)


def test_broadcasting():
    _ = set_seed(243)
    m, n = 100, 20

    A = cp.Parameter((m, n))
    b = cp.Parameter(m)
    x = cp.Variable(n)
    obj = cp.sum_squares(A @ x - b) + cp.sum_squares(x)
    prob = cp.Problem(cp.Minimize(obj))
    prob_th = CvxpyLayer(prob, [A, b], [x])

    A_th = torch.randn(m, n).double().requires_grad_()
    b_th_0 = torch.randn(m).double().requires_grad_()
    b_th = torch.stack((b_th_0, b_th_0))

    x = prob_th(A_th, b_th, solver_args={"eps": 1e-10})[0]

    def lstsq(A, b):
        return torch.linalg.solve(
            A.t() @ A + torch.eye(n, dtype=torch.float64),
            A.t() @ b,
        )

    x_lstsq = lstsq(A_th, b_th_0)

    grad_A_cvxpy, grad_b_cvxpy = grad(x.sum(), [A_th, b_th])
    grad_A_lstsq, grad_b_lstsq = grad(x_lstsq.sum(), [A_th, b_th_0])

    assert torch.allclose(grad_A_cvxpy / 2.0, grad_A_lstsq, atol=1e-6)
    assert torch.allclose(grad_b_cvxpy[0], grad_b_lstsq, atol=1e-6)


def test_shared_parameter():
    rng = set_seed(243)
    m, n = 10, 5

    A = cp.Parameter((m, n))
    x = cp.Variable(n)
    b1 = rng.standard_normal(m)
    b2 = rng.standard_normal(m)
    prob1 = cp.Problem(cp.Minimize(cp.sum_squares(A @ x - b1)))
    layer1 = CvxpyLayer(prob1, parameters=[A], variables=[x])
    prob2 = cp.Problem(cp.Minimize(cp.sum_squares(A @ x - b2)))
    layer2 = CvxpyLayer(prob2, parameters=[A], variables=[x])

    A_th = torch.randn(m, n).double().requires_grad_()
    solver_args = {
        "eps": 1e-10,
        "acceleration_lookback": 0,
        "max_iters": 10000,
    }

    def f(A_th):
        (x1,) = layer1(A_th, solver_args=solver_args)
        (x2,) = layer2(A_th, solver_args=solver_args)
        return torch.cat((x1, x2))

    torch.autograd.gradcheck(f, A_th)


def test_equality():
    _ = set_seed(243)
    n = 10
    A = np.eye(n)
    x = cp.Variable(n)
    b = cp.Parameter(n)
    prob = cp.Problem(cp.Minimize(cp.sum_squares(x)), [A @ x == b])
    layer = CvxpyLayer(prob, parameters=[b], variables=[x])

    b_th = torch.randn(n).double().requires_grad_()

    torch.autograd.gradcheck(layer, b_th)


def test_basic_gp():
    _ = set_seed(0)
    x = cp.Variable(pos=True)
    y = cp.Variable(pos=True)
    z = cp.Variable(pos=True)

    a = cp.Parameter(pos=True, value=2.0)
    b = cp.Parameter(pos=True, value=1.0)
    c = cp.Parameter(value=0.5)

    objective_fn = 1 / (x * y * z)
    constraints = [a * (x * y + x * z + y * z) <= b, x >= y**c]
    problem = cp.Problem(cp.Minimize(objective_fn), constraints)
    problem.solve(cp.CLARABEL, gp=True)

    layer = CvxpyLayer(problem, parameters=[a, b, c], variables=[x, y, z], gp=True)
    a_th = torch.tensor([2.0]).requires_grad_()
    b_th = torch.tensor([1.0]).requires_grad_()
    c_th = torch.tensor([0.5]).requires_grad_()
    x_th, y_th, z_th = layer(a_th, b_th, c_th)

    assert torch.allclose(torch.tensor(x.value), x_th, atol=1e-5)
    assert torch.allclose(torch.tensor(y.value), y_th, atol=1e-5)
    assert torch.allclose(torch.tensor(z.value), z_th, atol=1e-5)

    def f(a, b, c):
        res = layer(a, b, c, solver_args={"acceleration_lookback": 0})
        return res[0].sum()

    torch.autograd.gradcheck(f, (a_th, b_th, c_th), atol=1e-4)


def test_batched_gp():
    """Test GP with batched parameters."""
    _ = set_seed(0)
    x = cp.Variable(pos=True)
    y = cp.Variable(pos=True)
    z = cp.Variable(pos=True)

    # Batched parameters (need initial values for GP)
    a = cp.Parameter(pos=True, value=2.0)
    b = cp.Parameter(pos=True, value=1.0)
    c = cp.Parameter(value=0.5)

    # Objective and constraints
    objective_fn = 1 / (x * y * z)
    constraints = [a * (x * y + x * z + y * z) <= b, x >= y**c]
    problem = cp.Problem(cp.Minimize(objective_fn), constraints)

    # Create layer
    layer = CvxpyLayer(problem, parameters=[a, b, c], variables=[x, y, z], gp=True)

    # Batched parameters - test with batch size 4 (double precision)
    # For scalar parameters, batching means 1D tensors
    batch_size = 4
    a_batch = torch.tensor([2.0, 1.5, 2.5, 1.8], dtype=torch.float64, requires_grad=True)
    b_batch = torch.tensor([1.0, 1.2, 0.8, 1.5], dtype=torch.float64, requires_grad=True)
    c_batch = torch.tensor([0.5, 0.6, 0.4, 0.5], dtype=torch.float64, requires_grad=True)

    # Forward pass
    x_batch, y_batch, z_batch = layer(a_batch, b_batch, c_batch)

    # Check shapes - batched results are (batch_size,) for scalar variables
    assert x_batch.shape == (batch_size,)
    assert y_batch.shape == (batch_size,)
    assert z_batch.shape == (batch_size,)

    # Verify each batch element by solving individually
    for i in range(batch_size):
        a.value = a_batch[i].item()
        b.value = b_batch[i].item()
        c.value = c_batch[i].item()
        problem.solve(cp.CLARABEL, gp=True)

        assert torch.allclose(torch.tensor(x.value), x_batch[i], atol=1e-4, rtol=1e-4), (
            f"Mismatch in batch {i} for x"
        )
        assert torch.allclose(torch.tensor(y.value), y_batch[i], atol=1e-4, rtol=1e-4), (
            f"Mismatch in batch {i} for y"
        )
        assert torch.allclose(torch.tensor(z.value), z_batch[i], atol=1e-4, rtol=1e-4), (
            f"Mismatch in batch {i} for z"
        )

    # Test gradients on batched problem
    def f_batch(a, b, c):
        res = layer(a, b, c, solver_args={"acceleration_lookback": 0})
        return res[0].sum()

    torch.autograd.gradcheck(f_batch, (a_batch, b_batch, c_batch), atol=1e-3, rtol=1e-3)


def test_gp_without_param_values():
    """Test that GP layers can be created without setting parameter values."""
    _ = set_seed(0)
    x = cp.Variable(pos=True)
    y = cp.Variable(pos=True)
    z = cp.Variable(pos=True)

    # Create parameters WITHOUT setting values (this is the key test!)
    a = cp.Parameter(pos=True, name="a")
    b = cp.Parameter(pos=True, name="b")
    c = cp.Parameter(name="c")

    # Build GP problem
    objective_fn = 1 / (x * y * z)
    constraints = [a * (x * y + x * z + y * z) <= b, x >= y**c]
    problem = cp.Problem(cp.Minimize(objective_fn), constraints)

    # This should work WITHOUT needing to set a.value, b.value, c.value
    layer = CvxpyLayer(problem, parameters=[a, b, c], variables=[x, y, z], gp=True)

    # Now use the layer with actual parameter values
    a_th = torch.tensor([2.0], dtype=torch.float64, requires_grad=True)
    b_th = torch.tensor([1.0], dtype=torch.float64, requires_grad=True)
    c_th = torch.tensor([0.5], dtype=torch.float64, requires_grad=True)

    # Forward pass
    x_th, y_th, z_th = layer(a_th, b_th, c_th)

    # Verify solution against CVXPY direct solve
    a.value = 2.0
    b.value = 1.0
    c.value = 0.5
    problem.solve(cp.CLARABEL, gp=True)

    assert torch.allclose(torch.tensor(x.value), x_th, atol=1e-5)
    assert torch.allclose(torch.tensor(y.value), y_th, atol=1e-5)
    assert torch.allclose(torch.tensor(z.value), z_th, atol=1e-5)

    # Test gradients
    def f(a, b, c):
        res = layer(a, b, c, solver_args={"acceleration_lookback": 0})
        return res[0].sum()

    torch.autograd.gradcheck(f, (a_th, b_th, c_th), atol=1e-4)


def test_gp_reversed_parameter_order():
    """Test that GP layers produce correct results regardless of parameter order.

    Regression test for a bug where the GP path in _build_user_order_mapping
    did not sort by column position, causing parameters to be concatenated
    in wrong order when user declaration order differed from CVXPY's internal
    column assignment order.
    """
    _ = set_seed(42)
    x = cp.Variable(pos=True)
    y = cp.Variable(pos=True)
    z = cp.Variable(pos=True)

    a = cp.Parameter(pos=True, name="a")
    b = cp.Parameter(pos=True, name="b")
    c = cp.Parameter(name="c")

    objective_fn = 1 / (x * y * z)
    constraints = [a * (x * y + x * z + y * z) <= b, x >= y**c]
    problem = cp.Problem(cp.Minimize(objective_fn), constraints)

    # Create layers with parameters in different orders
    layer_abc = CvxpyLayer(problem, parameters=[a, b, c], variables=[x, y, z], gp=True)
    layer_cba = CvxpyLayer(problem, parameters=[c, b, a], variables=[x, y, z], gp=True)

    a_th = torch.tensor([2.0], dtype=torch.float64, requires_grad=True)
    b_th = torch.tensor([1.0], dtype=torch.float64, requires_grad=True)
    c_th = torch.tensor([0.5], dtype=torch.float64, requires_grad=True)

    # Forward pass with both orderings
    x1, y1, z1 = layer_abc(a_th, b_th, c_th)
    x2, y2, z2 = layer_cba(c_th, b_th, a_th)

    # Both orderings should produce same results
    assert torch.allclose(x1, x2, atol=1e-5), f"x mismatch: {x1} vs {x2}"
    assert torch.allclose(y1, y2, atol=1e-5), f"y mismatch: {y1} vs {y2}"
    assert torch.allclose(z1, z2, atol=1e-5), f"z mismatch: {z1} vs {z2}"

    # Verify against CVXPY ground truth
    a.value = 2.0
    b.value = 1.0
    c.value = 0.5
    problem.solve(cp.CLARABEL, gp=True)

    assert torch.allclose(torch.tensor(x.value), x1, atol=1e-5)
    assert torch.allclose(torch.tensor(y.value), y1, atol=1e-5)
    assert torch.allclose(torch.tensor(z.value), z1, atol=1e-5)

    # Test gradients for reversed order
    def f_cba(c, b, a):
        res = layer_cba(c, b, a, solver_args={"acceleration_lookback": 0})
        return res[0].sum()

    c_th2 = torch.tensor([0.5], dtype=torch.float64, requires_grad=True)
    b_th2 = torch.tensor([1.0], dtype=torch.float64, requires_grad=True)
    a_th2 = torch.tensor([2.0], dtype=torch.float64, requires_grad=True)
    torch.autograd.gradcheck(f_cba, (c_th2, b_th2, a_th2), atol=1e-4)


def test_no_grad_context():
    n, m = 2, 3
    x = cp.Variable(n)
    A = cp.Parameter((m, n))
    b = cp.Parameter(m)
    constraints = [x >= 0]
    objective = cp.Minimize(0.5 * cp.pnorm(A @ x - b, p=1))
    problem = cp.Problem(objective, constraints)
    assert problem.is_dpp()

    cvxpylayer = CvxpyLayer(problem, parameters=[A, b], variables=[x])
    A_tch = torch.randn(m, n)
    b_tch = torch.randn(m)

    with torch.no_grad():
        (solution,) = cvxpylayer(A_tch, b_tch)
        # These tensors should not require grad when in no_grad context
        assert torch.is_tensor(solution)
        assert not solution.requires_grad


def test_requires_grad_false():
    n, m = 2, 3
    x = cp.Variable(n)
    A = cp.Parameter((m, n))
    b = cp.Parameter(m)
    constraints = [x >= 0]
    objective = cp.Minimize(0.5 * cp.pnorm(A @ x - b, p=1))
    problem = cp.Problem(objective, constraints)
    assert problem.is_dpp()

    cvxpylayer = CvxpyLayer(problem, parameters=[A, b], variables=[x])
    A_tch = torch.randn(m, n, requires_grad=False)
    b_tch = torch.randn(m, requires_grad=False)

    # solve the problem
    (solution,) = cvxpylayer(A_tch, b_tch)
    # These tensors should not require grad when inputs don't require grad
    assert torch.is_tensor(solution)
    assert not solution.requires_grad


def test_batch_size_one_preserves_batch_dimension():
    """Test that batch_size=1 is different from unbatched.

    When the input is explicitly batched with batch_size=1 (shape (1, n)),
    the gradients should also be batched with shape (1, n), not unbatched (n,).
    """
    n = 3
    x = cp.Variable(n)
    b = cp.Parameter(n)

    # Simple quadratic problem: minimize ||x - b||^2
    objective = cp.Minimize(cp.sum_squares(x - b))
    problem = cp.Problem(objective)

    cvxpylayer = CvxpyLayer(problem, parameters=[b], variables=[x])

    # Create explicitly batched input with batch_size=1
    b_batched = torch.randn(1, n, requires_grad=True)  # Shape: (1, n)

    # Solve
    (x_batched,) = cvxpylayer(b_batched)

    # Solution should be batched
    assert x_batched.shape == (1, n), f"Expected shape (1, {n}), got {x_batched.shape}"

    # Compute gradient
    loss = x_batched.sum()
    loss.backward()

    # Gradient should preserve batch dimension
    assert b_batched.grad is not None
    assert b_batched.grad.shape == (1, n), (
        f"Expected gradient shape (1, {n}), got {b_batched.grad.shape}. "
        "Batch dimension should be preserved for batch_size=1."
    )


def test_solver_args_actually_used():
    """Test that solver_args actually affect the solver's behavior.

    This verifies solver_args are truly passed to the solver by:
    1. Solving with very restrictive max_iters (should give suboptimal solution)
    2. Solving with normal settings (should give better solution)
    3. Verifying the solutions differ, proving solver_args were used
    """
    _ = set_seed(123)
    m, n = 50, 20

    A = cp.Parameter((m, n))
    b = cp.Parameter(m)
    x = cp.Variable(n)
    obj = cp.sum_squares(A @ x - b) + 0.01 * cp.sum_squares(x)
    prob = cp.Problem(cp.Minimize(obj))

    layer = CvxpyLayer(prob, [A, b], [x])

    A_th = torch.randn(m, n).double()
    b_th = torch.randn(m).double()

    # Solve with very restrictive iterations (should stop early, suboptimal)
    (x_restricted,) = layer(A_th, b_th, solver_args={"max_iters": 1})

    # Solve with proper iterations (should converge to optimal)
    (x_optimal,) = layer(A_th, b_th, solver_args={"max_iters": 10000, "eps": 1e-10})

    # The solutions should differ if solver_args were actually used
    # With only 1 iteration, the solution should be far from optimal
    diff = torch.norm(x_restricted - x_optimal).item()
    assert diff > 1e-3, (
        f"Solutions with max_iters=1 and max_iters=10000 are too similar (diff={diff}). "
        "This suggests solver_args are not being passed to the solver."
    )

    # The optimal solution should have much lower objective value
    obj_restricted = (
        torch.sum((A_th @ x_restricted - b_th) ** 2) + 0.01 * torch.sum(x_restricted**2)
    ).item()
    obj_optimal = (
        torch.sum((A_th @ x_optimal - b_th) ** 2) + 0.01 * torch.sum(x_optimal**2)
    ).item()

    assert obj_optimal < obj_restricted, (
        f"Optimal objective ({obj_optimal}) should be less than restricted ({obj_restricted}). "
        "This suggests solver_args are not being used properly."
    )


def test_nd_array_variable():
    _ = set_seed(123)
    m, n, k = 50, 20, 10

    A = cp.Parameter((m, n))
    b = cp.Parameter((m, k))
    x = cp.Variable((n, k))
    obj = cp.sum_squares(A @ x - b) + 0.01 * cp.sum_squares(x)
    prob = cp.Problem(cp.Minimize(obj))

    layer = CvxpyLayer(prob, [A, b], [x])

    A_th = torch.randn(m, n).double()
    b_th = torch.randn(m, k).double()

    # Solve with very restrictive iterations (should stop early, suboptimal)
    (x_th,) = layer(A_th, b_th)

    A.value = A_th.numpy()
    b.value = b_th.numpy()
    prob.solve()
    assert np.allclose(x.value, x_th.numpy(), atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------------------
# Parametric quad_form(x, P) tests (issue #136)
# ---------------------------------------------------------------------------

_moreau_available = importlib.util.find_spec("moreau") is not None
requires_moreau = pytest.mark.skipif(
    not _moreau_available, reason="moreau not installed"
)
SOLVER_ARGS = {"tol": 1e-12, "max_iters": 500}


@requires_moreau
def test_quad_form_psd_parameter_dpp():
    """quad_form(x, Q) with Q=Parameter(PSD=True) should pass DPP check in scope."""
    from cvxpy.utilities import scopes

    n = 3
    Q = cp.Parameter((n, n), PSD=True)
    q = cp.Parameter(n)
    x = cp.Variable(n)
    prob = cp.Problem(
        cp.Minimize(0.5 * cp.quad_form(x, Q) + q.T @ x),
        [x >= -1, x <= 1],
    )
    # Parametric quad_form requires quad_form_dpp_scope for DPP validation.
    # CvxpyLayer enters this scope automatically for QP-capable solvers.
    with scopes.quad_form_dpp_scope():
        assert prob.is_dcp(dpp=True)
    layer = CvxpyLayer(prob, parameters=[Q, q], variables=[x], solver="MOREAU")
    assert layer is not None


@requires_moreau
def test_quad_form_psd_forward():
    """Unconstrained QP: x* = -Q^{-1} q."""
    n = 4
    Q = cp.Parameter((n, n), PSD=True)
    q = cp.Parameter(n)
    x = cp.Variable(n)
    prob = cp.Problem(cp.Minimize(0.5 * cp.quad_form(x, Q) + q.T @ x), [])
    layer = CvxpyLayer(prob, parameters=[Q, q], variables=[x], solver="MOREAU")

    rng = set_seed(136)
    Q_np = np.eye(n) * 2.0
    q_np = rng.standard_normal(n)
    Q_t = torch.tensor(Q_np)
    q_t = torch.tensor(q_np)

    (y,) = layer(Q_t, q_t, solver_args=SOLVER_ARGS)
    expected = np.linalg.solve(Q_np, -q_np)
    assert np.allclose(y.detach().numpy(), expected, atol=1e-6)


@requires_moreau
def test_quad_form_psd_different_Q():
    """Different Q values should give different solutions."""
    n = 3
    Q = cp.Parameter((n, n), PSD=True)
    q = cp.Parameter(n)
    x = cp.Variable(n)
    prob = cp.Problem(cp.Minimize(0.5 * cp.quad_form(x, Q) + q.T @ x), [])
    layer = CvxpyLayer(prob, parameters=[Q, q], variables=[x], solver="MOREAU")

    q_t = torch.tensor([1.0, -1.0, 0.5])
    Q1 = torch.eye(n, dtype=torch.float64)
    Q2 = 3.0 * torch.eye(n, dtype=torch.float64)

    (y1,) = layer(Q1, q_t, solver_args=SOLVER_ARGS)
    (y2,) = layer(Q2, q_t, solver_args=SOLVER_ARGS)
    assert not torch.allclose(y1, y2, atol=1e-6)

    # Verify both are correct
    expected1 = np.linalg.solve(Q1.numpy(), -q_t.numpy())
    expected2 = np.linalg.solve(Q2.numpy(), -q_t.numpy())
    assert np.allclose(y1.detach().numpy(), expected1, atol=1e-6)
    assert np.allclose(y2.detach().numpy(), expected2, atol=1e-6)


@requires_moreau
def test_quad_form_psd_gradcheck_q():
    """Gradient w.r.t. linear parameter q should pass gradcheck."""
    n = 3
    Q = cp.Parameter((n, n), PSD=True)
    q = cp.Parameter(n)
    x = cp.Variable(n)
    prob = cp.Problem(cp.Minimize(0.5 * cp.quad_form(x, Q) + q.T @ x), [])
    layer = CvxpyLayer(prob, parameters=[Q, q], variables=[x], solver="MOREAU")

    Q_t = 2.0 * torch.eye(n, dtype=torch.float64)
    q_t = torch.tensor([1.0, -1.0, 0.5], requires_grad=True)

    def func(q_t):
        (y,) = layer(Q_t, q_t, solver_args=SOLVER_ARGS)
        return y.sum()

    assert torch.autograd.gradcheck(func, (q_t,), eps=1e-5, atol=1e-3)


@requires_moreau
def test_quad_form_psd_gradcheck_Q():
    """Gradient w.r.t. quadratic parameter Q should pass gradcheck."""
    n = 3
    Q = cp.Parameter((n, n), PSD=True)
    q = cp.Parameter(n)
    x = cp.Variable(n)
    prob = cp.Problem(cp.Minimize(0.5 * cp.quad_form(x, Q) + q.T @ x), [])
    layer = CvxpyLayer(prob, parameters=[Q, q], variables=[x], solver="MOREAU")

    q_t = torch.tensor([1.0, -1.0, 0.5])

    def func(Q_t):
        # Symmetrize so that gradcheck's per-entry perturbation affects both
        # Q[k,l] and Q[l,k], matching the analytical gradient.
        Q_sym = (Q_t + Q_t.T) / 2
        (y,) = layer(Q_sym, q_t, solver_args=SOLVER_ARGS)
        return y.sum()

    Q_t = 2.0 * torch.eye(n, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(func, (Q_t,), eps=1e-5, atol=1e-3)


@requires_moreau
def test_quad_form_psd_backward():
    """Backward pass should produce finite, non-zero gradients for both Q and q."""
    n = 4
    Q = cp.Parameter((n, n), PSD=True)
    q = cp.Parameter(n)
    x = cp.Variable(n)
    prob = cp.Problem(cp.Minimize(0.5 * cp.quad_form(x, Q) + q.T @ x), [])
    layer = CvxpyLayer(prob, parameters=[Q, q], variables=[x], solver="MOREAU")

    Q_t = torch.tensor(2.0 * np.eye(n), dtype=torch.float64, requires_grad=True)
    q_t = torch.randn(n, dtype=torch.float64, requires_grad=True)

    (y,) = layer(Q_t, q_t, solver_args=SOLVER_ARGS)
    y.sum().backward()

    assert Q_t.grad is not None
    assert q_t.grad is not None
    assert torch.all(torch.isfinite(Q_t.grad))
    assert torch.all(torch.isfinite(q_t.grad))
    assert q_t.grad.norm() > 0


@requires_moreau
def test_quad_form_psd_batched():
    """Batched Q and q should produce correct per-element solutions."""
    n, batch = 3, 4
    Q = cp.Parameter((n, n), PSD=True)
    q = cp.Parameter(n)
    x = cp.Variable(n)
    prob = cp.Problem(cp.Minimize(0.5 * cp.quad_form(x, Q) + q.T @ x), [])
    layer = CvxpyLayer(prob, parameters=[Q, q], variables=[x], solver="MOREAU")

    # Batched Q: diagonal with different scales
    scales = torch.tensor([1.0, 2.0, 3.0, 4.0])
    Q_batch = torch.eye(n, dtype=torch.float64).unsqueeze(0) * scales.unsqueeze(
        1
    ).unsqueeze(2)
    Q_batch.requires_grad_(True)
    torch.manual_seed(136)
    q_batch = torch.randn(batch, n, dtype=torch.float64, requires_grad=True)

    (y_batch,) = layer(Q_batch, q_batch, solver_args=SOLVER_ARGS)
    assert y_batch.shape == (batch, n)

    # Verify each element
    for i in range(batch):
        expected = np.linalg.solve(
            Q_batch[i].detach().numpy(), -q_batch[i].detach().numpy()
        )
        assert np.allclose(y_batch[i].detach().numpy(), expected, atol=1e-5)

    y_batch.sum().backward()
    assert Q_batch.grad is not None
    assert q_batch.grad is not None


@requires_moreau
def test_quad_form_psd_with_constraints():
    """Parametric Q with box constraints should clamp the solution."""
    n = 3
    Q = cp.Parameter((n, n), PSD=True)
    q = cp.Parameter(n)
    x = cp.Variable(n)
    prob = cp.Problem(
        cp.Minimize(0.5 * cp.quad_form(x, Q) + q.T @ x),
        [x >= -0.5, x <= 0.5],
    )
    layer = CvxpyLayer(prob, parameters=[Q, q], variables=[x], solver="MOREAU")

    Q_t = torch.eye(n, dtype=torch.float64)
    q_t = torch.tensor([3.0, -3.0, 0.1])  # Large q pushes x to bounds

    (y,) = layer(Q_t, q_t, solver_args=SOLVER_ARGS)
    # x* = clip(-Q^{-1}q, -0.5, 0.5) = clip([-3, 3, -0.1], -0.5, 0.5)
    assert np.allclose(y.detach().numpy(), [-0.5, 0.5, -0.1], atol=1e-4)


@requires_moreau
def test_quad_form_plus_constant_linear():
    """quad_form(x, P) + c @ x with constant c solves correctly.

    Regression test: ensures the q coefficient from the linear term is
    not overwritten when the quad_form dummy variable is processed after
    the true variable in extract_quadratic_coeffs.
    """
    n = 3
    P = cp.Parameter((n, n), PSD=True)
    c = np.array([2.0, -2.0, 0.5])
    x = cp.Variable(n)
    prob = cp.Problem(
        cp.Minimize(0.5 * cp.quad_form(x, P) + c @ x),
    )
    layer = CvxpyLayer(prob, parameters=[P], variables=[x], solver="MOREAU")

    P_t = torch.eye(n, dtype=torch.float64)
    (y,) = layer(P_t, solver_args=SOLVER_ARGS)
    # x* = -P^{-1} c = -c for P=I
    assert np.allclose(y.detach().numpy(), -c, atol=1e-5)

    # Re-solve with different P to test DPP caching
    P_t2 = 2.0 * torch.eye(n, dtype=torch.float64)
    (y2,) = layer(P_t2, solver_args=SOLVER_ARGS)
    # x* = -(2I)^{-1} c = -c/2
    assert np.allclose(y2.detach().numpy(), -c / 2, atol=1e-5)


def test_quad_form_psd_rejects_diffcp():
    """Parametric quad_form(x, Q) should fail with DIFFCP (non-QP solver)."""
    n = 3
    Q = cp.Parameter((n, n), PSD=True)
    q = cp.Parameter(n)
    x = cp.Variable(n)
    prob = cp.Problem(
        cp.Minimize(0.5 * cp.quad_form(x, Q) + q.T @ x),
        [x >= -1, x <= 1],
    )
    # DIFFCP can't handle parametric P — scope is not entered, so construction
    # fails (either DPP validation or canonicalization, depending on CVXPY version).
    with pytest.raises((ValueError, AssertionError)):
        CvxpyLayer(prob, parameters=[Q, q], variables=[x])


@requires_moreau
def test_quad_form_sum_of_parameters():
    """quad_form(x, P + Q) with P, Q both PSD Parameters — forward correctness.

    min 0.5*x'(P+Q)x + q'x  =>  x* = -(P+Q)^{-1} q
    """
    n = 3
    P = cp.Parameter((n, n), PSD=True)
    Q = cp.Parameter((n, n), PSD=True)
    q = cp.Parameter(n)
    x = cp.Variable(n)
    prob = cp.Problem(
        cp.Minimize(0.5 * cp.quad_form(x, P + Q) + q.T @ x),
    )
    layer = CvxpyLayer(prob, parameters=[P, Q, q], variables=[x], solver="MOREAU")

    P_t = torch.tensor(2.0 * np.eye(n), dtype=torch.float64)
    Q_t = torch.tensor(3.0 * np.eye(n), dtype=torch.float64)
    q_t = torch.tensor([1.0, -1.0, 0.5])

    (y,) = layer(P_t, Q_t, q_t, solver_args=SOLVER_ARGS)
    expected = np.linalg.solve(
        P_t.numpy() + Q_t.numpy(), -q_t.numpy()
    )
    assert np.allclose(y.detach().numpy(), expected, atol=1e-5)


@requires_moreau
def test_quad_form_negated_parameter():
    """quad_form(x, -P) with P NSD Parameter — forward correctness.

    min 0.5*x'(-P)x + q'x  =>  x* = -(-P)^{-1} q = P^{-1} q
    """
    n = 3
    P = cp.Parameter((n, n), NSD=True)
    q = cp.Parameter(n)
    x = cp.Variable(n)
    prob = cp.Problem(
        cp.Minimize(0.5 * cp.quad_form(x, -P) + q.T @ x),
    )
    layer = CvxpyLayer(prob, parameters=[P, q], variables=[x], solver="MOREAU")

    P_t = torch.tensor(-2.0 * np.eye(n), dtype=torch.float64)
    q_t = torch.tensor([1.0, -1.0, 0.5])

    (y,) = layer(P_t, q_t, solver_args=SOLVER_ARGS)
    # -P = 2I, so x* = -(2I)^{-1} q = -q/2
    expected = -q_t.numpy() / 2
    assert np.allclose(y.detach().numpy(), expected, atol=1e-5)


@requires_moreau
def test_quad_form_sum_of_parameters_gradcheck():
    """Gradient w.r.t. P, Q in quad_form(x, P + Q) should pass gradcheck."""
    n = 3
    P = cp.Parameter((n, n), PSD=True)
    Q = cp.Parameter((n, n), PSD=True)
    q = cp.Parameter(n)
    x = cp.Variable(n)
    prob = cp.Problem(
        cp.Minimize(0.5 * cp.quad_form(x, P + Q) + q.T @ x),
    )
    layer = CvxpyLayer(prob, parameters=[P, Q, q], variables=[x], solver="MOREAU")

    q_t = torch.tensor([1.0, -1.0, 0.5])

    def func(P_t, Q_t):
        P_sym = (P_t + P_t.T) / 2
        Q_sym = (Q_t + Q_t.T) / 2
        (y,) = layer(P_sym, Q_sym, q_t, solver_args=SOLVER_ARGS)
        return y.sum()

    P_t = 2.0 * torch.eye(n, dtype=torch.float64, requires_grad=True)
    Q_t = 3.0 * torch.eye(n, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(func, (P_t, Q_t), eps=1e-5, atol=1e-3)


@requires_moreau
def test_quad_form_negated_parameter_gradcheck():
    """Gradient w.r.t. P in quad_form(x, -P) should pass gradcheck."""
    n = 3
    P = cp.Parameter((n, n), NSD=True)
    q = cp.Parameter(n)
    x = cp.Variable(n)
    prob = cp.Problem(
        cp.Minimize(0.5 * cp.quad_form(x, -P) + q.T @ x),
    )
    layer = CvxpyLayer(prob, parameters=[P, q], variables=[x], solver="MOREAU")

    q_t = torch.tensor([1.0, -1.0, 0.5])

    def func(P_t):
        P_sym = (P_t + P_t.T) / 2
        (y,) = layer(P_sym, q_t, solver_args=SOLVER_ARGS)
        return y.sum()

    P_t = -2.0 * torch.eye(n, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(func, (P_t,), eps=1e-5, atol=1e-3)


@requires_moreau
def test_quad_form_sum_of_parameters_backward():
    """Backward pass for P+Q should produce finite non-zero grads for P, Q, q."""
    n = 4
    P = cp.Parameter((n, n), PSD=True)
    Q = cp.Parameter((n, n), PSD=True)
    q = cp.Parameter(n)
    x = cp.Variable(n)
    prob = cp.Problem(
        cp.Minimize(0.5 * cp.quad_form(x, P + Q) + q.T @ x),
    )
    layer = CvxpyLayer(prob, parameters=[P, Q, q], variables=[x], solver="MOREAU")

    P_t = torch.tensor(2.0 * np.eye(n), dtype=torch.float64, requires_grad=True)
    Q_t = torch.tensor(3.0 * np.eye(n), dtype=torch.float64, requires_grad=True)
    q_t = torch.randn(n, dtype=torch.float64, requires_grad=True)

    (y,) = layer(P_t, Q_t, q_t, solver_args=SOLVER_ARGS)
    y.sum().backward()

    for t, name in [(P_t, "P"), (Q_t, "Q"), (q_t, "q")]:
        assert t.grad is not None, f"{name}.grad is None"
        assert torch.all(torch.isfinite(t.grad)), f"{name}.grad has non-finite values"
        assert t.grad.norm() > 0, f"{name}.grad is all zeros"


@requires_moreau
def test_quad_form_rejects_symmetric_only_parameter():
    """quad_form(x, P) with P=Parameter(symmetric=True) but no PSD/NSD should be rejected.

    The convexity check requires P.is_psd() or P.is_nsd() to be True.
    """
    n = 3
    P = cp.Parameter((n, n), symmetric=True)
    q = cp.Parameter(n)
    x = cp.Variable(n)
    prob = cp.Problem(
        cp.Minimize(0.5 * cp.quad_form(x, P) + q.T @ x),
        [x >= -1, x <= 1],
    )
    with pytest.raises((ValueError, AssertionError)):
        CvxpyLayer(prob, parameters=[P, q], variables=[x], solver="MOREAU")


@requires_moreau
def test_quad_form_rejects_param_times_param():
    """quad_form(x, P @ P) is quadratic in params — not param-affine, rejected."""
    n = 2
    P = cp.Parameter((n, n), PSD=True)
    x = cp.Variable(n)
    prob = cp.Problem(
        cp.Minimize(cp.quad_form(x, P @ P, assume_PSD=True)),
        [cp.sum(x) == 1],
    )
    with pytest.raises((ValueError, AssertionError)):
        CvxpyLayer(prob, parameters=[P], variables=[x], solver="MOREAU")


@requires_moreau
def test_quad_form_rejects_param_in_x():
    """quad_form(p, Q) where x-argument contains a parameter should be rejected."""
    n = 2
    p = cp.Parameter(n)
    Q = cp.Parameter((n, n), PSD=True)
    prob = cp.Problem(
        cp.Minimize(cp.quad_form(p, Q)),
    )
    with pytest.raises((ValueError, AssertionError)):
        CvxpyLayer(prob, parameters=[p, Q], variables=[], solver="MOREAU")


@requires_moreau
def test_quad_form_rejects_x_plus_param():
    """quad_form(x + p, Q) should be rejected — x argument is not param-free."""
    n = 2
    x = cp.Variable(n)
    p = cp.Parameter(n)
    Q = cp.Parameter((n, n), PSD=True)
    prob = cp.Problem(
        cp.Minimize(cp.quad_form(x + p, Q)),
        [x >= -1, x <= 1],
    )
    with pytest.raises((ValueError, AssertionError)):
        CvxpyLayer(prob, parameters=[p, Q], variables=[x], solver="MOREAU")


@requires_moreau
def test_quad_form_dpp_detection_in_scope():
    """Parametric P expressions should be DPP only inside quad_form_dpp_scope."""
    from cvxpy.utilities import scopes

    n = 2
    P = cp.Parameter((n, n), PSD=True)
    Q = cp.Parameter((n, n), PSD=True)
    x = cp.Variable(n)

    # P+Q: DPP only in scope
    expr_sum = cp.quad_form(x, P + Q)
    assert not expr_sum.is_dpp()
    with scopes.quad_form_dpp_scope():
        assert expr_sum.is_dpp()

    # -P (with P NSD): DPP only in scope
    P_nsd = cp.Parameter((n, n), NSD=True)
    expr_neg = cp.quad_form(x, -P_nsd)
    assert not expr_neg.is_dpp()
    with scopes.quad_form_dpp_scope():
        assert expr_neg.is_dpp()

    # P@P: never DPP (not param-affine)
    with scopes.quad_form_dpp_scope():
        assert not cp.quad_form(x, P @ P, assume_PSD=True).is_dpp()


@requires_moreau
def test_quad_form_multiple_quad_forms():
    """quad_form(x, P1) + quad_form(x, P2) — two separate quad_forms.

    Equivalent to x'(P1+P2)x, so x* = -(P1+P2)^{-1} q.
    """
    n = 3
    P1 = cp.Parameter((n, n), PSD=True)
    P2 = cp.Parameter((n, n), PSD=True)
    q = cp.Parameter(n)
    x = cp.Variable(n)
    prob = cp.Problem(
        cp.Minimize(
            0.5 * cp.quad_form(x, P1) + 0.5 * cp.quad_form(x, P2) + q.T @ x
        ),
    )
    layer = CvxpyLayer(
        prob, parameters=[P1, P2, q], variables=[x], solver="MOREAU"
    )

    P1_t = torch.tensor(
        np.array([[2, 0.5, 0], [0.5, 1, 0], [0, 0, 3]], dtype=np.float64)
    )
    P2_t = torch.tensor(
        np.array([[1, 0, 0], [0, 2, 0.5], [0, 0.5, 1]], dtype=np.float64)
    )
    q_t = torch.tensor([1.0, -1.0, 0.5])

    (y,) = layer(P1_t, P2_t, q_t, solver_args=SOLVER_ARGS)
    expected = np.linalg.solve(
        P1_t.numpy() + P2_t.numpy(), -q_t.numpy()
    )
    assert np.allclose(y.detach().numpy(), expected, atol=1e-5)


@requires_moreau
def test_quad_form_sum_of_parameters_resolve():
    """Re-solving quad_form(x, P+Q) with different param values gives correct results."""
    n = 2
    P = cp.Parameter((n, n), PSD=True)
    Q = cp.Parameter((n, n), PSD=True)
    q = cp.Parameter(n)
    x = cp.Variable(n)
    prob = cp.Problem(
        cp.Minimize(0.5 * cp.quad_form(x, P + Q) + q.T @ x),
    )
    layer = CvxpyLayer(prob, parameters=[P, Q, q], variables=[x], solver="MOREAU")

    q_t = torch.tensor([1.0, -1.0])

    # First solve: P+Q = [[3,0],[0,2]]
    P1 = torch.tensor([[2.0, 0], [0, 1.0]])
    Q1 = torch.tensor([[1.0, 0], [0, 1.0]])
    (y1,) = layer(P1, Q1, q_t, solver_args=SOLVER_ARGS)
    expected1 = np.linalg.solve(P1.numpy() + Q1.numpy(), -q_t.numpy())
    assert np.allclose(y1.detach().numpy(), expected1, atol=1e-5)

    # Second solve: P+Q = [[2,0],[0,5]]
    P2 = torch.tensor([[1.0, 0], [0, 3.0]])
    Q2 = torch.tensor([[1.0, 0], [0, 2.0]])
    (y2,) = layer(P2, Q2, q_t, solver_args=SOLVER_ARGS)
    expected2 = np.linalg.solve(P2.numpy() + Q2.numpy(), -q_t.numpy())
    assert np.allclose(y2.detach().numpy(), expected2, atol=1e-5)

    # Solutions should differ
    assert not np.allclose(y1.detach().numpy(), y2.detach().numpy(), atol=1e-3)


@requires_moreau
def test_quad_form_nsd_maximize():
    """Maximize x'Qx + q'x with Q=NSD parameter and box constraints.

    max 0.5*x'Qx + q'x  s.t. -1 <= x <= 1
    For Q = -I, q = [2, -2, 0.5]:
        grad = Qx + q = 0 => x* = -Q^{-1}(-q) = q (unconstrained)
        clipped: clip([2, -2, 0.5], -1, 1) = [1, -1, 0.5]
    """
    n = 3
    Q = cp.Parameter((n, n), NSD=True)
    q = cp.Parameter(n)
    x = cp.Variable(n)
    prob = cp.Problem(
        cp.Maximize(0.5 * cp.quad_form(x, Q) + q.T @ x),
        [x >= -1, x <= 1],
    )
    layer = CvxpyLayer(prob, parameters=[Q, q], variables=[x], solver="MOREAU")

    Q_t = -torch.eye(n, dtype=torch.float64)
    q_t = torch.tensor([2.0, -2.0, 0.5])

    (y,) = layer(Q_t, q_t, solver_args=SOLVER_ARGS)
    assert np.allclose(y.detach().numpy(), [1.0, -1.0, 0.5], atol=1e-4)


@requires_moreau
def test_quad_form_nsd_maximize_backward():
    """Backward pass for NSD maximize should produce finite, non-zero gradients."""
    n = 4
    Q = cp.Parameter((n, n), NSD=True)
    q = cp.Parameter(n)
    x = cp.Variable(n)
    prob = cp.Problem(cp.Maximize(0.5 * cp.quad_form(x, Q) + q.T @ x), [])
    layer = CvxpyLayer(prob, parameters=[Q, q], variables=[x], solver="MOREAU")

    Q_t = torch.tensor(-2.0 * np.eye(n), dtype=torch.float64, requires_grad=True)
    q_t = torch.randn(n, dtype=torch.float64, requires_grad=True)

    (y,) = layer(Q_t, q_t, solver_args=SOLVER_ARGS)
    y.sum().backward()

    assert Q_t.grad is not None
    assert q_t.grad is not None
    assert torch.all(torch.isfinite(Q_t.grad))
    assert torch.all(torch.isfinite(q_t.grad))
    assert q_t.grad.norm() > 0
