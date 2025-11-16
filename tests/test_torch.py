"""Unit tests for cvxpylayers.torch."""

import cvxpy as cp
import diffcp
import numpy as np
import pytest
import torch

# from cvxpylayers.utils import backward_numpy, forward_numpy
from torch.autograd import grad

from cvxpylayers.torch import CvxpyLayer

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


@pytest.mark.skip
def test_sdp():
    _ = set_seed(243)

    n = 3
    p = 3
    C = cp.Parameter((n, n))
    A = [cp.Parameter((n, n)) for _ in range(p)]
    b = [cp.Parameter((1, 1)) for _ in range(p)]

    C_th = torch.randn(n, n).requires_grad_()
    A_th, b_th = [], []
    for _ in range(p):
        A_th.append(torch.randn(n, n).requires_grad_())
        b_th.append(torch.randn(1, 1).requires_grad_())

    X = cp.Variable((n, n), symmetric=True)
    constraints = [X >> 0]
    constraints += [cp.trace(A[i] @ X) == b[i] for i in range(p)]
    prob = cp.Problem(cp.Minimize(cp.trace(C @ X) + cp.sum_squares(X)), constraints)
    layer = CvxpyLayer(prob, [C] + A + b, [X])

    torch.autograd.gradcheck(layer, [C_th] + A_th + b_th)


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
    problem.solve(cp.SCS, gp=True)

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
