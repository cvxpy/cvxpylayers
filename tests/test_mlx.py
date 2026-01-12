import cvxpy as cp
import diffcp
import numpy as np
import pytest
import torch

mx = pytest.importorskip("mlx.core")

from cvxpylayers.mlx.cvxpylayer import CvxpyLayer
from cvxpylayers.torch import CvxpyLayer as TorchCvxpyLayer


def to_numpy(x):
    """Convert MLX array to numpy array."""
    return np.array(x, dtype=np.float32)


def _compare(a, b, atol=1e-4, rtol=1e-4):
    """Compare apple mlx and torch results."""
    a_np = to_numpy(a)
    b_np = b.detach().numpy() if isinstance(b, torch.Tensor) else np.asarray(b)
    assert np.allclose(a_np, b_np, atol=atol, rtol=rtol), f"Mismatch:\nmlx={a_np}\ntorch={b_np}"


def set_seed(x: int) -> np.random.Generator:
    """Set the random seed and return a numpy random generator.

    Parameters
    ----------
    x : int
        The seed value to use for random number generators.

    Returns
    -------
    np.random.Generator
        A numpy random number generator instance with the specified seed.
    """
    np.random.seed(x)
    return np.random.default_rng(x)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def test_example():
    """Basic example test."""
    n, m = 2, 3
    print(f"datatype of m is {type(m)}")
    print(f"datatype of n is {type(n)}")
    x = cp.Variable(n)
    A = cp.Parameter((m, n))
    b = cp.Parameter(m)
    constraints = [x >= 0]
    objective = cp.Minimize(0.5 * cp.pnorm(A @ x - b, p=1))
    problem = cp.Problem(objective, constraints)
    assert problem.is_dpp()

    cvxpylayer = CvxpyLayer(problem, parameters=[A, b], variables=[x])
    A_mx = mx.array(np.random.randn(m, n), dtype=mx.float32)
    b_mx = mx.array(np.random.randn(m), dtype=mx.float32)

    # solve the problem
    (solution,) = cvxpylayer(A_mx, b_mx)

    # compute the gradient of the sum of the solution with respect to A, b
    def sum_sol(A_mx, b_mx):
        (solution,) = cvxpylayer(A_mx, b_mx)
        return mx.sum(solution)

    grad_loss = mx.grad(lambda A_, b_: mx.sum(cvxpylayer(A_, b_)[0]), argnums=(0, 1))
    grad_A, grad_b = grad_loss(A_mx, b_mx)
    assert grad_A.shape == A_mx.shape
    assert grad_b.shape == b_mx.shape


def test_least_squares():
    """Forward-only test for least squares problem."""
    # TODO : Narasimhan
    # testing forward correctness only because of the following :
    # 1. mx linalg solve not supported on gpu, so have to use cpu stream.
    # 2. Even after using cpu streams, VJP is absent.
    # 3. If we try to emualte using np.linalg.solve, gradient tracking is
    # lost, hence no way to perform backprop
    #  Seed setup for reproducibility
    rng = set_seed(243)
    m, n = 100, 20

    #  Define CVXPY problem: minimize ||A x - b||² + ||x||²
    A = cp.Parameter((m, n))
    b = cp.Parameter(m)
    x = cp.Variable(n)
    obj = cp.sum_squares(A @ x - b) + cp.sum_squares(x)
    prob = cp.Problem(cp.Minimize(obj))
    prob_mx = CvxpyLayer(prob, [A, b], [x])

    #  Generate MLX data
    A_mx = mx.array(rng.standard_normal((m, n)), dtype=mx.float32)
    b_mx = mx.array(rng.standard_normal(m), dtype=mx.float32)

    #  Solve via CVXPYLayer (MLX backend)
    x_pred = prob_mx(A_mx, b_mx, solver_args={"eps": 1e-10})[0]

    #  Analytical least-squares solution
    def lstsq(A, b):
        ATA = A.T @ A + mx.eye(A.shape[1], dtype=mx.float32)
        ATb = A.T @ b
        x_mx = mx.linalg.solve(ATA, ATb, stream=mx.Device(mx.cpu))
        return x_mx

    x_ref = lstsq(A_mx, b_mx)

    #  Forward check only
    assert np.allclose(
        to_numpy(x_pred),
        to_numpy(x_ref),
        atol=1e-5,
    ), "Mismatch between cvxpylayer and mlx"


def test_logistic_regression():
    """Test logistic regression problem."""
    rng = set_seed(0)

    N, n = 5, 2

    X_np = rng.standard_normal((N, n))
    a_true = rng.standard_normal((n, 1))
    y_np = np.round(sigmoid(X_np.dot(a_true) + rng.standard_normal((N, 1)) * 0.5))

    X_mx = mx.array(X_np, dtype=mx.float32)
    lam_mx = mx.array([0.1], dtype=mx.float32)

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

    def loss_fn(X, lam):
        (a_sol,) = fit_logreg(X, lam)
        return mx.sum(a_sol)

    # Test that gradients can be computed
    grad_X = mx.grad(lambda X_: loss_fn(X_, lam_mx))(X_mx)
    grad_lam = mx.grad(lambda lam_: loss_fn(X_mx, lam_))(lam_mx)
    assert grad_X.shape == X_mx.shape
    assert grad_lam.shape == lam_mx.shape


def test_lml():
    """Test LML (log-maximum likelihood) problem."""
    k = 2
    x = cp.Parameter(4)
    y = cp.Variable(4)
    obj = -x @ y - cp.sum(cp.entr(y)) - cp.sum(cp.entr(1.0 - y))
    cons = [cp.sum(y) == k]
    prob = cp.Problem(cp.Minimize(obj), cons)
    lml = CvxpyLayer(prob, [x], [y])

    x_mx = mx.array([1.0, -1.0, -1.0, -1.0], dtype=mx.float32)

    def loss_fn(x):
        (y_sol,) = lml(x)
        return mx.sum(y_sol)

    # Test that gradients can be computed
    grad_x = mx.grad(loss_fn)(x_mx)
    assert grad_x.shape == x_mx.shape


def test_not_enough_parameters():
    """Test error when not enough parameters are provided."""
    x = cp.Variable(1)
    lam = cp.Parameter(1, nonneg=True)
    lam2 = cp.Parameter(1, nonneg=True)
    objective = lam * cp.norm(x, 1) + lam2 * cp.sum_squares(x)
    prob = cp.Problem(cp.Minimize(objective))
    with pytest.raises(ValueError, match="must exactly match problem.parameters"):
        layer = CvxpyLayer(prob, [lam], [x])  # noqa: F841


def test_not_enough_parameters_at_call_time():
    """Test error when not enough parameters are provided at call time."""
    x = cp.Variable(1)
    lam = cp.Parameter(1, nonneg=True)
    lam2 = cp.Parameter(1, nonneg=True)
    objective = lam * cp.norm(x, 1) + lam2 * cp.sum_squares(x)
    prob = cp.Problem(cp.Minimize(objective))
    layer = CvxpyLayer(prob, [lam, lam2], [x])
    lam_mx = mx.ones(1, dtype=mx.float32)
    with pytest.raises(
        ValueError,
        match="A tensor must be provided for each CVXPY parameter.*",
    ):
        layer(lam_mx)


def test_none_parameter_at_call_time():
    """Test that passing None as a parameter raises an appropriate error."""
    x = cp.Variable(1)
    lam = cp.Parameter(1, nonneg=True)
    lam2 = cp.Parameter(1, nonneg=True)
    objective = lam * cp.norm(x, 1) + lam2 * cp.sum_squares(x)
    prob = cp.Problem(cp.Minimize(objective))
    layer = CvxpyLayer(prob, [lam, lam2], [x])
    lam_mx = mx.ones(1, dtype=mx.float32)
    with pytest.raises(AttributeError):
        layer(lam_mx, None)


def test_too_many_variables():
    """Test error when too many variables are requested."""
    x = cp.Variable(1)
    y = cp.Variable(1)
    lam = cp.Parameter(1, nonneg=True)
    objective = lam * cp.norm(x, 1)
    prob = cp.Problem(cp.Minimize(objective))
    with pytest.raises(ValueError, match="must be a subset of problem.variables"):
        layer = CvxpyLayer(prob, [lam], [x, y])  # noqa: F841


def test_infeasible():
    """Test error handling for infeasible problems."""
    x = cp.Variable(1)
    param = cp.Parameter(1)
    prob = cp.Problem(cp.Minimize(param), [x >= 1, x <= -1])
    layer = CvxpyLayer(prob, [param], [x])
    param_mx = mx.ones(1, dtype=mx.float32)
    with pytest.raises(diffcp.SolverError):
        layer(param_mx)


def test_unbounded():
    """Test error handling for unbounded problems."""
    x = cp.Variable(1)
    param = cp.Parameter(1)
    prob = cp.Problem(cp.Minimize(x), [x <= param])
    layer = CvxpyLayer(prob, [param], [x])
    param_mx = mx.ones(1, dtype=mx.float32)
    with pytest.raises(diffcp.SolverError):
        layer(param_mx)


def test_incorrect_parameter_shape():
    """Test error handling for incorrect parameter shapes."""
    rng = set_seed(243)
    m, n = 100, 20

    A = cp.Parameter((m, n))
    b = cp.Parameter(m)
    x = cp.Variable(n)
    obj = cp.sum_squares(A @ x - b) + cp.sum_squares(x)
    prob = cp.Problem(cp.Minimize(obj))
    prob_mx = CvxpyLayer(prob, [A, b], [x])

    A_mx = mx.array(rng.standard_normal((32, m, n)), dtype=mx.float32)
    b_mx = mx.array(rng.standard_normal((20, m)), dtype=mx.float32)

    with pytest.raises(ValueError, match="Inconsistent batch sizes"):
        prob_mx(A_mx, b_mx)

    A_mx = mx.array(rng.standard_normal((32, m, n)), dtype=mx.float32)
    b_mx = mx.array(rng.standard_normal((32, 2 * m)), dtype=mx.float32)

    with pytest.raises(ValueError, match="Invalid parameter shape"):
        prob_mx(A_mx, b_mx)

    A_mx = mx.array(rng.standard_normal((m, n)), dtype=mx.float32)
    b_mx = mx.array(rng.standard_normal(2 * m), dtype=mx.float32)

    with pytest.raises(ValueError, match="Invalid parameter shape"):
        prob_mx(A_mx, b_mx)

    A_mx = mx.array(rng.standard_normal((32, m, n)), dtype=mx.float32)
    b_mx = mx.array(rng.standard_normal((32, 32, m)), dtype=mx.float32)

    with pytest.raises(ValueError, match="Invalid parameter dimensionality"):
        prob_mx(A_mx, b_mx)


def test_broadcasting():
    """Forward-only test for broadcasting correctness in least squares."""
    # TODO : Narasimhan
    # testing forward correctness only because of the following :
    # 1. mx linalg solve not supported on gpu, so have to use cpu stream.
    # 2. Even after using cpu streams, VJP is absent.
    # 3. If we try to emualte using np.linalg.solve, gradient tracking is
    # lost, hence no way to perform backprop
    #  Seed setup for reproducibility
    rng = set_seed(243)
    m, n = 100, 20

    A = cp.Parameter((m, n))
    b = cp.Parameter(m)
    x = cp.Variable(n)
    obj = cp.sum_squares(A @ x - b) + cp.sum_squares(x)
    prob = cp.Problem(cp.Minimize(obj))
    prob_mx = CvxpyLayer(prob, [A, b], [x])

    A_mx = mx.array(rng.standard_normal((m, n)), dtype=mx.float32)
    b_mx_0 = mx.array(rng.standard_normal(m), dtype=mx.float32)
    b_mx = mx.stack((b_mx_0, b_mx_0))  # shape (2, m)

    #  Batched solve via CvxpyLayer
    x_batched = prob_mx(A_mx, b_mx, solver_args={"eps": 1e-10})[0]

    #  Single-sample analytical solution
    def lstsq(A, b):
        ATA = A.T @ A + mx.eye(A.shape[1], dtype=mx.float32)
        ATb = A.T @ b
        x_mx = mx.linalg.solve(ATA, ATb, stream=mx.Device(mx.cpu))
        return x_mx

    x_single = lstsq(A_mx, b_mx_0)

    #  Forward match check: both batched outputs should equal the single one
    assert np.allclose(to_numpy(x_batched[0]), to_numpy(x_single), atol=1e-5), (
        "Broadcasted cvxpy layer 0 mismatched least-squares output"
    )

    assert np.allclose(to_numpy(x_batched[1]), to_numpy(x_single), atol=1e-5), (
        "Broadcasted cvxpy layer 1 mismatched least-squares output"
    )


def test_shared_parameter():
    """Test using the same parameter in multiple problems."""
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

    A_mx = mx.array(rng.standard_normal((m, n)), dtype=mx.float32)
    solver_args = {
        "eps": 1e-10,
        "acceleration_lookback": 0,
        "max_iters": 10000,
    }

    def f(A):
        (x1,) = layer1(A, solver_args=solver_args)
        (x2,) = layer2(A, solver_args=solver_args)
        return mx.sum(mx.concatenate((x1, x2)))

    # Test that gradients can be computed
    grad_A = mx.grad(f)(A_mx)
    assert grad_A.shape == A_mx.shape


def test_equality():
    """Test problem with equality constraints."""
    rng = set_seed(243)
    n = 10
    A = np.eye(n)
    x = cp.Variable(n)
    b = cp.Parameter(n)
    prob = cp.Problem(cp.Minimize(cp.sum_squares(x)), [A @ x == b])
    layer = CvxpyLayer(prob, parameters=[b], variables=[x])

    b_mx = mx.array(rng.standard_normal(n), dtype=mx.float32)

    def loss_fn(b):
        (x_sol,) = layer(b)
        return mx.sum(x_sol)

    # Test that gradients can be computed
    grad_b = mx.grad(loss_fn)(b_mx)
    assert grad_b.shape == b_mx.shape


def test_basic_gp():
    """Test basic geometric programming."""
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
    a_mx = mx.array(2.0, dtype=mx.float32)
    b_mx = mx.array(1.0, dtype=mx.float32)
    c_mx = mx.array(0.5, dtype=mx.float32)
    x_mx, y_mx, z_mx = layer(a_mx, b_mx, c_mx)

    assert np.allclose(np.array(x.value), np.array(x_mx), atol=1e-5)
    assert np.allclose(np.array(y.value), np.array(y_mx), atol=1e-5)
    assert np.allclose(np.array(z.value), np.array(z_mx), atol=1e-5)

    def f(a, b, c):
        res = layer(a, b, c, solver_args={"acceleration_lookback": 0})
        return mx.sum(res[0])

    # Test that gradients can be computed
    grad_a = mx.grad(lambda a_: f(a_, b_mx, c_mx))(a_mx)
    grad_b = mx.grad(lambda b_: f(a_mx, b_, c_mx))(b_mx)
    grad_c = mx.grad(lambda c_: f(a_mx, b_mx, c_))(c_mx)
    assert grad_a.shape == a_mx.shape
    assert grad_b.shape == b_mx.shape
    assert grad_c.shape == c_mx.shape


def test_batched_gp():
    """Test GP with batched parameters."""
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

    # Batched parameters - test with batch size 4
    # For scalar parameters, batching means 1D arrays
    batch_size = 4
    a_batch = mx.array([2.0, 1.5, 2.5, 1.8], dtype=mx.float32)
    b_batch = mx.array([1.0, 1.2, 0.8, 1.5], dtype=mx.float32)
    c_batch = mx.array([0.5, 0.6, 0.4, 0.5], dtype=mx.float32)

    # Forward pass
    x_batch, y_batch, z_batch = layer(a_batch, b_batch, c_batch)

    # Check shapes - batched results are (batch_size,) for scalar variables
    assert x_batch.shape == (batch_size,)
    assert y_batch.shape == (batch_size,)
    assert z_batch.shape == (batch_size,)

    # Verify each batch element by solving individually
    for i in range(batch_size):
        a.value = float(np.array(a_batch[i]))
        b.value = float(np.array(b_batch[i]))
        c.value = float(np.array(c_batch[i]))
        problem.solve(cp.CLARABEL, gp=True)

        assert np.allclose(np.array(x.value), np.array(x_batch[i]), atol=1e-4, rtol=1e-4), (
            f"Mismatch in batch {i} for x"
        )
        assert np.allclose(np.array(y.value), np.array(y_batch[i]), atol=1e-4, rtol=1e-4), (
            f"Mismatch in batch {i} for y"
        )
        assert np.allclose(np.array(z.value), np.array(z_batch[i]), atol=1e-4, rtol=1e-4), (
            f"Mismatch in batch {i} for z"
        )

    # Test gradients on batched problem
    def f_batch(a, b, c):
        res = layer(a, b, c, solver_args={"acceleration_lookback": 0})
        return mx.sum(res[0])

    # Test that gradients can be computed
    grad_a = mx.grad(lambda a_: f_batch(a_, b_batch, c_batch))(a_batch)
    grad_b = mx.grad(lambda b_: f_batch(a_batch, b_, c_batch))(b_batch)
    grad_c = mx.grad(lambda c_: f_batch(a_batch, b_batch, c_))(c_batch)
    assert grad_a.shape == a_batch.shape
    assert grad_b.shape == b_batch.shape
    assert grad_c.shape == c_batch.shape


def test_gp_without_param_values():
    """Test that GP layers can be created without setting parameter values."""
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
    a_mx = mx.array(2.0, dtype=mx.float32)
    b_mx = mx.array(1.0, dtype=mx.float32)
    c_mx = mx.array(0.5, dtype=mx.float32)

    # Forward pass
    x_mx, y_mx, z_mx = layer(a_mx, b_mx, c_mx)

    # Verify solution against CVXPY direct solve
    a.value = 2.0
    b.value = 1.0
    c.value = 0.5
    problem.solve(cp.CLARABEL, gp=True)

    assert np.allclose(np.array(x.value), np.array(x_mx), atol=1e-5)
    assert np.allclose(np.array(y.value), np.array(y_mx), atol=1e-5)
    assert np.allclose(np.array(z.value), np.array(z_mx), atol=1e-5)

    # Test gradients
    def f(a, b, c):
        res = layer(a, b, c, solver_args={"acceleration_lookback": 0})
        return mx.sum(res[0])

    # Test that gradients can be computed
    grad_a = mx.grad(lambda a_: f(a_, b_mx, c_mx))(a_mx)
    grad_b = mx.grad(lambda b_: f(a_mx, b_, c_mx))(b_mx)
    grad_c = mx.grad(lambda c_: f(a_mx, b_mx, c_))(c_mx)
    assert grad_a.shape == a_mx.shape
    assert grad_b.shape == b_mx.shape
    assert grad_c.shape == c_mx.shape


def test_batch_size_one_preserves_batch_dimension():
    """Test that batch_size=1 is different from unbatched.

    When the input is explicitly batched with batch_size=1 (shape (1, n)),
    the gradients should also be batched with shape (1, n), not unbatched (n,).
    """
    rng = set_seed(243)
    n = 3
    x = cp.Variable(n)
    b = cp.Parameter(n)

    # Simple quadratic problem: minimize ||x - b||^2
    objective = cp.Minimize(cp.sum_squares(x - b))
    problem = cp.Problem(objective)

    cvxpylayer = CvxpyLayer(problem, parameters=[b], variables=[x])

    # Create explicitly batched input with batch_size=1
    b_batched = mx.array(rng.standard_normal((1, n)), dtype=mx.float32)  # Shape: (1, n)

    # Solve
    (x_batched,) = cvxpylayer(b_batched)

    # Solution should be batched
    assert x_batched.shape == (1, n), f"Expected shape (1, {n}), got {x_batched.shape}"

    # Compute gradient
    def loss_fn(b):
        (x_sol,) = cvxpylayer(b)
        return mx.sum(x_sol)

    grad_b = mx.grad(loss_fn)(b_batched)

    # Gradient should preserve batch dimension
    assert grad_b.shape == (1, n), (
        f"Expected gradient shape (1, {n}), got {grad_b.shape}. "
        "Batch dimension should be preserved for batch_size=1."
    )


def test_solver_args_actually_used():
    """Test that solver_args actually affect the solver's behavior.

    This verifies solver_args are truly passed to the solver by:
    1. Solving with very restrictive max_iters
    (should give suboptimal solution)
    2. Solving with normal settings (should give better solution)
    3. Verifying the solutions differ, proving solver_args were used
    """
    rng = set_seed(123)
    m, n = 50, 20

    A = cp.Parameter((m, n))
    b = cp.Parameter(m)
    x = cp.Variable(n)
    obj = cp.sum_squares(A @ x - b) + 0.01 * cp.sum_squares(x)
    prob = cp.Problem(cp.Minimize(obj))

    layer = CvxpyLayer(prob, [A, b], [x])

    A_mx = mx.array(rng.standard_normal((m, n)), dtype=mx.float32)
    b_mx = mx.array(rng.standard_normal(m), dtype=mx.float32)

    # Solve with very restrictive iterations (should stop early, suboptimal)
    (x_restricted,) = layer(A_mx, b_mx, solver_args={"max_iters": 1})

    # Solve with proper iterations (should converge to optimal)
    (x_optimal,) = layer(A_mx, b_mx, solver_args={"max_iters": 10000, "eps": 1e-10})

    # The solutions should differ if solver_args were actually used
    # With only 1 iteration, the solution should be far from optimal
    diff = np.linalg.norm(np.array(x_restricted) - np.array(x_optimal))
    assert diff > 1e-3, (
        "Solutions with max_iters=1 and max_iters=10000"
        f"are too similar (diff={diff}). "
        "This suggests solver_args are not being passed to the solver."
    )

    # The optimal solution should have much lower objective value
    A_np = np.array(A_mx)
    b_np = np.array(b_mx)
    x_restricted_np = np.array(x_restricted)
    x_optimal_np = np.array(x_optimal)

    obj_restricted = np.sum((A_np @ x_restricted_np - b_np) ** 2) + 0.01 * np.sum(
        x_restricted_np**2
    )
    obj_optimal = np.sum((A_np @ x_optimal_np - b_np) ** 2) + 0.01 * np.sum(x_optimal_np**2)

    assert obj_optimal < obj_restricted, (
        f"Optimal objective ({obj_optimal}) should be less"
        "than restricted ({obj_restricted}). "
        "This suggests solver_args are not being used properly."
    )


def test_forward_method():
    """Test that forward() method is an alias for __call__."""
    rng = set_seed(243)
    n, m = 2, 3
    x = cp.Variable(n)
    A = cp.Parameter((m, n))
    b = cp.Parameter(m)
    constraints = [x >= 0]
    objective = cp.Minimize(0.5 * cp.pnorm(A @ x - b, p=1))
    problem = cp.Problem(objective, constraints)

    cvxpylayer = CvxpyLayer(problem, parameters=[A, b], variables=[x])
    A_mx = mx.array(rng.standard_normal((m, n)), dtype=mx.float32)
    b_mx = mx.array(rng.standard_normal(m), dtype=mx.float32)

    # Test both methods return the same result
    solution_call = cvxpylayer(A_mx, b_mx)
    solution_forward = cvxpylayer.forward(A_mx, b_mx)

    assert len(solution_call) == len(solution_forward)
    for s1, s2 in zip(solution_call, solution_forward):
        assert np.allclose(np.array(s1), np.array(s2))


# =====================All the following tests assume
# the torch cvxpylayer as golden reference/ ground truth
# =============


@pytest.mark.parametrize("n", [101])
def test_relu(n):
    """Test ReLU projection comparing MLX with PyTorch."""
    x_param = cp.Parameter(n)
    y_var = cp.Variable(n)
    prob = cp.Problem(cp.Minimize(cp.sum_squares(y_var - x_param)), [y_var >= 0])

    # Torch CVXPY layer
    torch_layer = TorchCvxpyLayer(prob, parameters=[x_param], variables=[y_var])
    mlx_layer = CvxpyLayer(prob, parameters=[x_param], variables=[y_var])

    # Input
    x_np = np.linspace(-5, 5, n).astype(np.float32)
    x_mx = mx.array(x_np, dtype=mx.float32)
    x_torch = torch.tensor(x_np, requires_grad=True)

    # Forward comparison
    y_mx = mlx_layer(x_mx)[0]
    (y_torch,) = torch_layer(x_torch)
    _compare(y_mx, y_torch)

    # Gradient comparison
    y_torch.sum().backward()  # scalar gradient hence sum
    grad_torch = x_torch.grad
    grad_loss = mx.grad(lambda x: mx.sum(mlx_layer(x)[0]))
    grad_mx = grad_loss(x_mx)
    _compare(grad_mx, grad_torch)


@pytest.mark.parametrize("n", [100])
def test_sigmoid(n):
    """Test sigmoid projection comparing MLX with PyTorch."""
    x_param = cp.Parameter(n)
    y_var = cp.Variable(n)
    obj = cp.Minimize(-x_param.T @ y_var - cp.sum(cp.entr(y_var) + cp.entr(1.0 - y_var)))
    prob = cp.Problem(obj)

    torch_layer = TorchCvxpyLayer(prob, parameters=[x_param], variables=[y_var])
    mlx_layer = CvxpyLayer(prob, parameters=[x_param], variables=[y_var])

    x_np = np.linspace(-5, 5, n).astype(np.float32)
    x_mx = mx.array(x_np, dtype=mx.float32)
    x_torch = torch.tensor(x_np, requires_grad=True)

    y_mx = mlx_layer(x_mx)[0]
    (y_torch,) = torch_layer(x_torch)
    _compare(y_mx, y_torch)

    y_torch.sum().backward()
    grad_torch = x_torch.grad
    grad_loss = mx.grad(lambda x: mx.sum(mlx_layer(x)[0]))
    grad_mx = grad_loss(x_mx)
    _compare(grad_mx, grad_torch)


@pytest.mark.parametrize("n", [4])
def test_sparsemax(n):
    """Test sparsemax projection comparing MLX with PyTorch."""
    x = cp.Parameter(n)
    y = cp.Variable(n)
    constraint = [cp.sum(y) == 1, 0 <= y, y <= 1]
    prob = cp.Problem(cp.Minimize(cp.sum_squares(x - y)), constraint)

    torch_layer = TorchCvxpyLayer(prob, parameters=[x], variables=[y])
    mlx_layer = CvxpyLayer(prob, parameters=[x], variables=[y])

    np.random.seed(0)
    x_np = np.random.randn(n).astype(np.float32)
    x_mx = mx.array(x_np, dtype=mx.float32)
    x_torch = torch.tensor(x_np, requires_grad=False)

    y_mx = mlx_layer(x_mx)[0]
    (y_torch,) = torch_layer(x_torch)
    _compare(y_mx, y_torch)


@pytest.mark.parametrize("n,k", [(4, 2)])
def test_csoftmax(n, k):
    """Test constrained softmax comparing MLX with PyTorch."""
    x = cp.Parameter(n)
    y = cp.Variable(n)
    u = np.full((n,), 1.0 / k)
    constraint = [cp.sum(y) == 1.0, y <= u]
    prob = cp.Problem(cp.Minimize(-x @ y - cp.sum(cp.entr(y))), constraint)

    torch_layer = TorchCvxpyLayer(prob, parameters=[x], variables=[y])
    mlx_layer = CvxpyLayer(prob, parameters=[x], variables=[y])

    np.random.seed(0)
    x_np = np.random.randn(n).astype(np.float32)
    x_mx = mx.array(x_np, dtype=mx.float32)
    x_torch = torch.tensor(x_np, requires_grad=False)

    y_mx = mlx_layer(x_mx)[0]
    (y_torch,) = torch_layer(x_torch)
    _compare(y_mx, y_torch)


@pytest.mark.parametrize("n,k", [(4, 2)])
def test_csparsemax(n, k):
    """Test constrained sparsemax comparing MLX with PyTorch."""
    x = cp.Parameter(n)
    y = cp.Variable(n)
    u = np.full((n,), 1.0 / k)
    obj = cp.sum_squares(x - y)
    constraint = [cp.sum(y) == 1.0, 0.0 <= y, y <= u]
    prob = cp.Problem(cp.Minimize(obj), constraint)

    torch_layer = TorchCvxpyLayer(prob, [x], [y])
    mlx_layer = CvxpyLayer(prob, [x], [y])

    x_np = np.random.randn(n).astype(np.float32)
    x_mx = mx.array(x_np, dtype=mx.float32)
    x_torch = torch.tensor(x_np, requires_grad=False)

    y_mx = mlx_layer(x_mx)[0]
    (y_torch,) = torch_layer(x_torch)
    _compare(y_mx, y_torch)


@pytest.mark.parametrize("n,k", [(4, 2)])
def test_limited_multilayer_proj(n, k):
    """Test limited multilayer projection comparing MLX with PyTorch."""
    x = cp.Parameter(n)
    y = cp.Variable(n)
    obj = -x @ y - cp.sum(cp.entr(y)) - cp.sum(cp.entr(1.0 - y))
    cons = [cp.sum(y) == k]
    prob = cp.Problem(cp.Minimize(obj), cons)

    torch_layer = TorchCvxpyLayer(prob, [x], [y])
    mlx_layer = CvxpyLayer(prob, [x], [y])

    x_np = np.random.randn(n).astype(np.float32)
    x_mx = mx.array(x_np, dtype=mx.float32)
    x_torch = torch.tensor(x_np, requires_grad=False)

    y_mx = mlx_layer(x_mx)[0]
    (y_torch,) = torch_layer(x_torch)
    _compare(y_mx, y_torch)


@pytest.mark.parametrize("n", [2])
def test_multiple_variables_vs_torch(n):
    """Test optimization with multiple variables comparing MLX with PyTorch."""
    x = cp.Variable(n)
    y = cp.Variable(n)
    c = cp.Parameter(n)

    problem = cp.Problem(cp.Minimize(cp.sum_squares(x) + cp.sum_squares(y)), [x + y == c])

    mlx_layer = CvxpyLayer(problem, parameters=[c], variables=[x, y])
    torch_layer = TorchCvxpyLayer(problem, parameters=[c], variables=[x, y])

    c_val_np = np.array([2.0, 4.0], dtype=np.float32)
    c_val_mx = mx.array(c_val_np, dtype=mx.float32)
    c_val_torch = torch.tensor(c_val_np, requires_grad=True)

    # Forward pass
    x_mx, y_mx = mlx_layer(c_val_mx)
    x_torch, y_torch = torch_layer(c_val_torch)

    # Compare forward outputs
    _compare(x_mx, x_torch)
    _compare(y_mx, y_torch)

    # Gradient comparison
    (x_torch.sum() + y_torch.sum()).backward()

    def loss_fn(c_):
        x_sol, y_sol = mlx_layer(c_)
        return mx.sum(x_sol + y_sol)

    grad_mx = mx.grad(loss_fn)(c_val_mx)
    _compare(grad_mx, c_val_torch.grad)


@pytest.mark.parametrize("batch_size,n,m", [(5, 3, 4)])
def test_batched_solver(batch_size, n, m):
    """Test batched problem solving using PyTorch as reference."""
    x = cp.Variable(n)
    A = cp.Parameter((m, n))
    b = cp.Parameter(m)
    problem = cp.Problem(cp.Minimize(cp.sum_squares(A @ x - b)), [x >= 0])

    mlx_layer = CvxpyLayer(problem, parameters=[A, b], variables=[x])
    torch_layer = TorchCvxpyLayer(problem, parameters=[A, b], variables=[x])

    np.random.seed(42)
    A_batch_np = np.random.randn(batch_size, m, n).astype(np.float32)
    b_batch_np = np.random.randn(batch_size, m).astype(np.float32)
    A_batch_mx = mx.array(A_batch_np, dtype=mx.float32)
    b_batch_mx = mx.array(b_batch_np, dtype=mx.float32)
    A_batch_torch = torch.tensor(A_batch_np, requires_grad=True)
    b_batch_torch = torch.tensor(b_batch_np, requires_grad=True)

    # Forward pass
    y_mx = mlx_layer(A_batch_mx, b_batch_mx)[0]
    (y_torch,) = torch_layer(A_batch_torch, b_batch_torch)
    _compare(y_mx, y_torch)

    # Gradient comparison
    y_torch.sum().backward()

    grad_loss_A = mx.grad(lambda A_: mx.sum(mlx_layer(A_, b_batch_mx)[0]))(A_batch_mx)
    grad_loss_b = mx.grad(lambda b_: mx.sum(mlx_layer(A_batch_mx, b_)[0]))(b_batch_mx)

    _compare(grad_loss_A, A_batch_torch.grad)
    _compare(grad_loss_b, b_batch_torch.grad)


@pytest.mark.parametrize("n", [4])
def test_ellipsoid_projection(n):
    """Test a QP with two variables and constraints, which is an
    ellipsoid projection."""
    # Define problem
    _A = cp.Parameter((n, n))
    _z = cp.Parameter(n)
    _x = cp.Parameter(n)
    _y = cp.Variable(n)
    _t = cp.Variable(n)

    obj = cp.Minimize(0.5 * cp.sum_squares(_x - _y))
    cons = [0.5 * cp.sum_squares(_A @ _t) <= 1, _t == (_y - _z)]
    prob = cp.Problem(obj, cons)

    # MLX and Torch layers
    mlx_layer = CvxpyLayer(prob, parameters=[_A, _z, _x], variables=[_y, _t])
    torch_layer = TorchCvxpyLayer(prob, parameters=[_A, _z, _x], variables=[_y, _t])

    # Random input
    torch.manual_seed(0)
    A_val = torch.randn(n, n, requires_grad=True)
    z_val = torch.randn(n, requires_grad=True)
    x_val = torch.randn(n, requires_grad=True)

    # Forward pass
    y_torch, t_torch = torch_layer(A_val, z_val, x_val)
    A_mx = mx.array(A_val.detach().numpy(), dtype=mx.float32)
    z_mx = mx.array(z_val.detach().numpy(), dtype=mx.float32)
    x_mx = mx.array(x_val.detach().numpy(), dtype=mx.float32)
    y_mx, t_mx = mlx_layer(A_mx, z_mx, x_mx)

    # Compare outputs
    _compare(y_mx, y_torch)
    _compare(t_mx, t_torch)

    # Gradients
    (y_torch.sum() + t_torch.sum()).backward()

    # MLX gradients per parameter
    def loss_A(A_):
        y_sol, t_sol = mlx_layer(A_, z_mx, x_mx)
        return mx.sum(y_sol + t_sol)

    def loss_z(z_):
        y_sol, t_sol = mlx_layer(A_mx, z_, x_mx)
        return mx.sum(y_sol + t_sol)

    def loss_x(x_):
        y_sol, t_sol = mlx_layer(A_mx, z_mx, x_)
        return mx.sum(y_sol + t_sol)

    grad_y_A = mx.grad(loss_A)(A_mx)
    grad_y_z = mx.grad(loss_z)(z_mx)
    grad_y_x = mx.grad(loss_x)(x_mx)

    # Compare gradients
    _compare(grad_y_A, A_val.grad)
    _compare(grad_y_z, z_val.grad)
    _compare(grad_y_x, x_val.grad)
