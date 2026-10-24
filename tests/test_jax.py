import numpy as np
import pytest

jax = pytest.importorskip("jax")

import cvxpy as cp  # noqa: E402
import diffcp  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from jax import random  # noqa: E402
from jax.test_util import check_grads  # noqa: E402

from cvxpylayers.jax import CvxpyLayer  # noqa: E402

# Tight solves keep finite-difference checks above the solver accuracy floor.
GRADCHECK_SOLVER_ARGS = {
    "ipm_settings": {"tol_gap_abs": 1e-12, "tol_gap_rel": 1e-12, "tol_feas": 1e-12},
}


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def test_example():
    key = random.PRNGKey(0)

    n, m = 2, 3
    x = cp.Variable(n)
    A = cp.Parameter((m, n))
    b = cp.Parameter(m)
    constraints = [x >= 0]
    objective = cp.Minimize(0.5 * cp.pnorm(A @ x - b, p=1))
    problem = cp.Problem(objective, constraints)
    assert problem.is_dpp()

    cvxpylayer = CvxpyLayer(problem, parameters=[A, b], variables=[x])

    key, k1, k2 = random.split(key, num=3)
    A_jax = random.normal(k1, shape=(m, n))
    b_jax = random.normal(k2, shape=(m,))

    # solve the problem
    (solution,) = cvxpylayer(A_jax, b_jax)

    # compute the gradient of the sum of the solution with respect to A, b
    def sum_sol(A_jax, b_jax):
        (solution,) = cvxpylayer(A_jax, b_jax)
        return solution.sum()

    dsum_sol = jax.grad(sum_sol)
    dsum_sol(A_jax, b_jax)


def test_simple_batch_socp():
    key = random.PRNGKey(0)
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

    prob_jax = CvxpyLayer(prob, [P_sqrt, q, A, b], [x])

    key, k1, k2, k3, k4 = random.split(key, num=5)
    P_sqrt_jax = random.normal(k1, shape=(batch_size, n, n))
    q_jax = random.normal(k2, shape=(batch_size, n, 1))
    A_jax = random.normal(k3, shape=(batch_size, m, n))
    b_jax = random.normal(k4, shape=(batch_size, m, 1))

    def f(*params):
        (sol,) = prob_jax(*params)
        return sum(sol)

    check_grads(f, (P_sqrt_jax, q_jax, A_jax, b_jax), order=1, modes=["rev"], atol=5e-3, rtol=5e-3)


def test_least_squares():
    key = random.PRNGKey(0)
    m, n = 100, 20

    A = cp.Parameter((m, n))
    b = cp.Parameter(m)
    x = cp.Variable(n)
    obj = cp.sum_squares(A @ x - b) + cp.sum_squares(x)
    prob = cp.Problem(cp.Minimize(obj))
    prob_jax = CvxpyLayer(prob, [A, b], [x])

    key, k1, k2 = random.split(key, num=3)
    A_jax = random.normal(k1, shape=(m, n))
    b_jax = random.normal(k2, shape=(m,))

    def lstsq_sum_cp(A_jax, b_jax):
        x = prob_jax(A_jax, b_jax, solver_args=GRADCHECK_SOLVER_ARGS)[0]
        return sum(x)

    def lstsq_sum_linalg(A_jax, b_jax):
        x = jnp.linalg.solve(A_jax.T @ A_jax + jnp.eye(n), A_jax.T @ b_jax)
        return sum(x)

    d_lstsq_sum_cp = jax.grad(lstsq_sum_cp, [0, 1])
    d_lstsq_sum_linalg = jax.grad(lstsq_sum_linalg, [0, 1])

    grad_A_cvxpy, grad_b_cvxpy = d_lstsq_sum_cp(A_jax, b_jax)
    # Run linalg on CPU (conflict between JAX and multiprocessing)
    cpu = jax.devices("cpu")[0]
    with jax.default_device(cpu):
        grad_A_lstsq, grad_b_lstsq = d_lstsq_sum_linalg(
            jax.device_put(A_jax, cpu), jax.device_put(b_jax, cpu)
        )

    assert np.isclose(
        jnp.linalg.norm(grad_A_cvxpy - grad_A_lstsq).item(),
        0.0,
        atol=1e-6,
    )
    assert np.isclose(
        jnp.linalg.norm(grad_b_cvxpy - grad_b_lstsq).item(),
        0.0,
        atol=1e-6,
    )


def test_logistic_regression():
    key = random.PRNGKey(0)

    N, n = 5, 2

    key, k1, k2, k3 = random.split(key, num=4)
    X_np = random.normal(k1, shape=(N, n))
    a_true = random.normal(k2, shape=(n, 1))
    y_np = jnp.round(sigmoid(X_np @ a_true + random.normal(k3, shape=(N, 1)) * 0.5))

    X_jax = jnp.array(X_np)
    lam_jax = 0.1 * jnp.ones(1)

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

    fit_logreg = CvxpyLayer(prob, [X, lam], [a], solver_args=GRADCHECK_SOLVER_ARGS)

    check_grads(
        fit_logreg, (X_jax, lam_jax), order=1, eps=1e-3, modes=["rev"], atol=1e-3, rtol=1e-3
    )


@pytest.mark.skip
def test_entropy_maximization():
    key = random.PRNGKey(0)
    n, m, p = 5, 3, 2

    key, k1, k2, k3, k4 = random.split(key, num=5)
    tmp = random.normal(k1, shape=(n,))
    A_np = random.normal(k2, shape=(m, n))
    b_np = A_np.dot(tmp)
    F_np = random.normal(k3, shape=(p, n))
    g_np = F_np.dot(tmp) + random.normal(k4, shape=(p,))

    x = cp.Variable(n)
    A = cp.Parameter((m, n))
    b = cp.Parameter(m)
    F = cp.Parameter((p, n))
    g = cp.Parameter(p)
    obj = cp.Maximize(cp.sum(cp.entr(x)) - 0.01 * cp.sum_squares(x))
    constraints = [A @ x == b, F @ x <= g]
    prob = cp.Problem(obj, constraints)
    layer = CvxpyLayer(prob, [A, b, F, g], [x])

    A_jax, b_jax, F_jax, g_jax = map(lambda x: jnp.array(x), [A_np, b_np, F_np, g_np])

    check_grads(layer, (A_jax, b_jax, F_jax, g_jax), order=1, modes=["rev"])


def test_lml():
    k = 2
    x = cp.Parameter(4)
    y = cp.Variable(4)
    obj = -x @ y - cp.sum(cp.entr(y)) - cp.sum(cp.entr(1.0 - y))
    cons = [cp.sum(y) == k]
    prob = cp.Problem(cp.Minimize(obj), cons)
    lml = CvxpyLayer(prob, [x], [y], solver_args=GRADCHECK_SOLVER_ARGS)

    x_th = jnp.array([1.0, -1.0, -1.0, -1.0])
    check_grads(lml, (x_th,), order=1, eps=1e-3, modes=["rev"], atol=1e-3, rtol=1e-3)


def test_sdp():
    """Test SDP with symmetric parameters."""
    n = 3
    X = cp.Variable((n, n), symmetric=True)
    C = cp.Parameter((n, n), symmetric=True)

    psd_con = X >> 0
    trace_con = cp.trace(X) == 1
    prob = cp.Problem(cp.Minimize(cp.trace(C @ X)), [psd_con, trace_con])

    layer = CvxpyLayer(prob, parameters=[C], variables=[X], solver_args=GRADCHECK_SOLVER_ARGS)

    # Use a well-conditioned symmetric matrix
    C_jax = jnp.array([[2.0, 0.5, 0.1], [0.5, 3.0, 0.2], [0.1, 0.2, 1.5]])

    def solve(C):
        return layer((C + C.T) / 2)[0]

    def reference(C):
        _, vectors = jnp.linalg.eigh((C + C.T) / 2)
        v = vectors[:, 0]
        return jnp.outer(v, v)

    actual, pullback = jax.vjp(solve, C_jax)
    expected, ref_pullback = jax.vjp(reference, C_jax)
    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)
    # Check every output coordinate using ordinary VJPs, as check_grads does.
    for cotangent in jnp.eye(n * n).reshape(n * n, n, n):
        np.testing.assert_allclose(
            pullback(cotangent)[0], ref_pullback(cotangent)[0], atol=1e-4, rtol=1e-3
        )


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
    lam_jax = jnp.ones(1)
    with pytest.raises(
        ValueError,
        match="A tensor must be provided for each CVXPY parameter.*",
    ):
        layer(lam_jax)


def test_none_parameter_at_call_time():
    """Test that passing None as a parameter raises an appropriate error."""
    x = cp.Variable(1)
    lam = cp.Parameter(1, nonneg=True)
    lam2 = cp.Parameter(1, nonneg=True)
    objective = lam * cp.norm(x, 1) + lam2 * cp.sum_squares(x)
    prob = cp.Problem(cp.Minimize(objective))
    layer = CvxpyLayer(prob, [lam, lam2], [x])
    lam_jax = jnp.ones(1)
    with pytest.raises(AttributeError):
        layer(lam_jax, None)


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
    layer = CvxpyLayer(prob, [param], [x], solver="DIFFCP")
    param_jax = jnp.ones(1)
    with pytest.raises(diffcp.SolverError):
        layer(param_jax)


def test_unbounded():
    x = cp.Variable(1)
    param = cp.Parameter(1)
    prob = cp.Problem(cp.Minimize(x), [x <= param])
    layer = CvxpyLayer(prob, [param], [x], solver="DIFFCP")
    param_jax = jnp.ones(1)
    with pytest.raises(diffcp.SolverError):
        layer(param_jax)


def test_incorrect_parameter_shape():
    key = random.PRNGKey(0)
    m, n = 100, 20

    A = cp.Parameter((m, n))
    b = cp.Parameter(m)
    x = cp.Variable(n)
    obj = cp.sum_squares(A @ x - b) + cp.sum_squares(x)
    prob = cp.Problem(cp.Minimize(obj))
    prob_th = CvxpyLayer(prob, [A, b], [x])

    key, k1, k2 = random.split(key, num=3)
    A_th = random.normal(k1, shape=(32, m, n))
    b_th = random.normal(k2, shape=(20, m))

    with pytest.raises(ValueError, match="Inconsistent batch sizes"):
        prob_th(A_th, b_th)

    key, k1, k2 = random.split(key, num=3)
    A_th = random.normal(k1, shape=(32, m, n))
    b_th = random.normal(k2, shape=(32, 2 * m))

    with pytest.raises(ValueError, match="Invalid parameter shape"):
        prob_th(A_th, b_th)

    key, k1, k2 = random.split(key, num=3)
    A_th = random.normal(k1, shape=(m, n))
    b_th = random.normal(k2, shape=(2 * m,))

    with pytest.raises(ValueError, match="Invalid parameter shape"):
        prob_th(A_th, b_th)

    key, k1, k2 = random.split(key, num=3)
    A_th = random.normal(k1, shape=(32, m, n))
    b_th = random.normal(k2, shape=(32, 32, m))

    with pytest.raises(ValueError, match="Invalid parameter dimensionality"):
        prob_th(A_th, b_th)


def test_broadcasting():
    key = random.PRNGKey(0)
    m, n = 100, 20

    A = cp.Parameter((m, n))
    b = cp.Parameter(m)
    x = cp.Variable(n)
    obj = cp.sum_squares(A @ x - b) + cp.sum_squares(x)
    prob = cp.Problem(cp.Minimize(obj))
    prob_jax = CvxpyLayer(prob, [A, b], [x])

    key, k1, k2 = random.split(key, num=3)
    A_jax = random.normal(k1, shape=(m, n))
    b_jax_0 = random.normal(k2, shape=(m,))
    b_jax = jnp.stack((b_jax_0, b_jax_0))

    def lstsq_sum_cp(A_jax, b_jax):
        x = prob_jax(A_jax, b_jax, solver_args=GRADCHECK_SOLVER_ARGS)[0]
        return jnp.sum(x)

    def lstsq_sum_linalg(A_jax, b_jax):
        x = jnp.linalg.solve(A_jax.T @ A_jax + jnp.eye(n), A_jax.T @ b_jax)
        return sum(x)

    d_lstsq_sum_cp = jax.grad(lstsq_sum_cp, [0, 1])
    d_lstsq_sum_linalg = jax.grad(lstsq_sum_linalg, [0, 1])

    grad_A_cvxpy, grad_b_cvxpy = d_lstsq_sum_cp(A_jax, b_jax)
    # Run linalg on CPU (conflict between JAX and multiprocessing)
    cpu = jax.devices("cpu")[0]
    with jax.default_device(cpu):
        grad_A_lstsq, grad_b_lstsq = d_lstsq_sum_linalg(
            jax.device_put(A_jax, cpu), jax.device_put(b_jax_0, cpu)
        )

    assert np.isclose(
        jnp.linalg.norm(grad_A_cvxpy / 2.0 - grad_A_lstsq).item(),
        0.0,
        atol=1e-6,
    )
    assert np.isclose(
        jnp.linalg.norm(grad_b_cvxpy[0] - grad_b_lstsq).item(),
        0.0,
        atol=1e-6,
    )


def test_shared_parameter():
    key = random.PRNGKey(0)
    m, n = 10, 5

    A = cp.Parameter((m, n))
    x = cp.Variable(n)
    key, k1, k2 = random.split(key, num=3)
    b1 = random.normal(k1, shape=(m,))
    b2 = random.normal(k2, shape=(m,))
    prob1 = cp.Problem(cp.Minimize(cp.sum_squares(A @ x - b1)))
    layer1 = CvxpyLayer(prob1, parameters=[A], variables=[x])
    prob2 = cp.Problem(cp.Minimize(cp.sum_squares(A @ x - b2)))
    layer2 = CvxpyLayer(prob2, parameters=[A], variables=[x])

    key, k1 = random.split(key, num=2)
    A_jax = random.normal(k1, shape=(m, n))
    solver_args = {
        "ipm_settings": {"tol_gap_abs": 1e-10, "tol_gap_rel": 1e-10, "tol_feas": 1e-10},
    }

    def f(A_jax):
        (x1,) = layer1(A_jax, solver_args=solver_args)
        (x2,) = layer2(A_jax, solver_args=solver_args)
        return jnp.concatenate((x1, x2))

    # Compare VJPs to dense least squares, avoiding cancellation from
    # finite differences of two separately solved optimization problems.
    def reference(A):
        return jnp.concatenate([
            jnp.linalg.solve(A.T @ A, A.T @ b) for b in (b1, b2)
        ])

    actual, pullback = jax.vjp(f, A_jax)
    expected, ref_pullback = jax.vjp(reference, A_jax)
    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)
    cotangent = jnp.arange(2 * n, dtype=A_jax.dtype) / n
    np.testing.assert_allclose(
        pullback(cotangent)[0], ref_pullback(cotangent)[0], atol=1e-5, rtol=1e-5
    )


def test_equality():
    key = random.PRNGKey(0)
    n = 10
    A = np.eye(n)
    x = cp.Variable(n)
    b = cp.Parameter(n)
    prob = cp.Problem(cp.Minimize(cp.sum_squares(x)), [A @ x == b])
    layer = CvxpyLayer(prob, parameters=[b], variables=[x])

    key, k1 = random.split(key, num=2)
    b_jax = random.normal(k1, shape=(n,))

    check_grads(layer, [b_jax], order=1, modes=["rev"])


def _gp_reference_x(a, b, c):
    """Solve the GP below on its active branch x = y**c, with 0 < c < 1.

    Eliminating z gives xyz = y**(c+1) * (b/a - y**(c+1)) / (y**c + y).
    Newton's method in log(y) gives a smooth, independent gradient reference.
    """
    log_y = 0.5 * jnp.log(b / (3 * a))
    for _ in range(20):
        r = jnp.exp((c + 1) * log_y) * a / b
        p = jax.nn.sigmoid((c - 1) * log_y)
        residual = (c + 1) * (1 - r / (1 - r)) - 1 - (c - 1) * p
        derivative = -(c + 1)**2 * r / (1 - r)**2 - (c - 1)**2 * p * (1 - p)
        log_y = log_y - residual / derivative
    return jnp.exp(c * log_y)


def _check_gp_gradients(layer, params, order=(0, 1, 2)):
    def solve(*args):
        return layer(*args)[0].sum()

    def reference(*args):
        return _gp_reference_x(*(args[i] for i in order)).sum()

    np.testing.assert_allclose(solve(*params), reference(*params), atol=1e-5, rtol=1e-5)
    actual = jax.grad(solve, argnums=(0, 1, 2))(*params)
    expected = jax.grad(reference, argnums=(0, 1, 2))(*params)
    for grad, ref in zip(actual, expected):
        np.testing.assert_allclose(grad, ref, atol=1e-5, rtol=1e-5)


def test_basic_gp():
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

    layer = CvxpyLayer(
        problem, parameters=[a, b, c], variables=[x, y, z], gp=True,
        solver_args=GRADCHECK_SOLVER_ARGS,
    )
    a_jax = jnp.array(2.0)
    b_jax = jnp.array(1.0)
    c_jax = jnp.array(0.5)
    x_jax, y_jax, z_jax = layer(a_jax, b_jax, c_jax)

    assert np.isclose(x.value, x_jax, atol=1e-5)
    assert np.isclose(y.value, y_jax, atol=1e-5)
    assert np.isclose(z.value, z_jax, atol=1e-5)

    _check_gp_gradients(layer, (a_jax, b_jax, c_jax))


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
    layer = CvxpyLayer(
        problem, parameters=[a, b, c], variables=[x, y, z], gp=True,
        solver_args=GRADCHECK_SOLVER_ARGS,
    )

    # Batched parameters - test with batch size 4
    # For scalar parameters, batching means 1D arrays
    batch_size = 4
    a_batch = jnp.array([2.0, 1.5, 2.5, 1.8])
    b_batch = jnp.array([1.0, 1.2, 0.8, 1.5])
    c_batch = jnp.array([0.5, 0.6, 0.4, 0.5])

    # Forward pass
    x_batch, y_batch, z_batch = layer(a_batch, b_batch, c_batch)

    # Check shapes - batched results are (batch_size, 1) for scalar variables
    assert x_batch.shape == (batch_size,)
    assert y_batch.shape == (batch_size,)
    assert z_batch.shape == (batch_size,)

    # Verify each batch element by solving individually
    for i in range(batch_size):
        a.value = float(a_batch[i])
        b.value = float(b_batch[i])
        c.value = float(c_batch[i])
        problem.solve(cp.CLARABEL, gp=True)

        assert np.allclose(x.value, x_batch[i], atol=1e-4, rtol=1e-4), (
            f"Mismatch in batch {i} for x"
        )
        assert np.allclose(y.value, y_batch[i], atol=1e-4, rtol=1e-4), (
            f"Mismatch in batch {i} for y"
        )
        assert np.allclose(z.value, z_batch[i], atol=1e-4, rtol=1e-4), (
            f"Mismatch in batch {i} for z"
        )

    # Test gradients on batched problem
    _check_gp_gradients(layer, (a_batch, b_batch, c_batch))


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
    layer = CvxpyLayer(
        problem, parameters=[a, b, c], variables=[x, y, z], gp=True,
        solver_args=GRADCHECK_SOLVER_ARGS,
    )

    # Now use the layer with actual parameter values
    a_jax = jnp.array(2.0)
    b_jax = jnp.array(1.0)
    c_jax = jnp.array(0.5)

    # Forward pass
    x_jax, y_jax, z_jax = layer(a_jax, b_jax, c_jax)

    # Verify solution against CVXPY direct solve
    a.value = 2.0
    b.value = 1.0
    c.value = 0.5
    problem.solve(cp.CLARABEL, gp=True)

    assert np.isclose(x.value, x_jax, atol=1e-5)
    assert np.isclose(y.value, y_jax, atol=1e-5)
    assert np.isclose(z.value, z_jax, atol=1e-5)

    # Test gradients
    _check_gp_gradients(layer, (a_jax, b_jax, c_jax))


def test_gp_reversed_parameter_order():
    """Test that GP layers produce correct results regardless of parameter order.

    Regression test for a bug where the GP path in _build_user_order_mapping
    did not sort by column position, causing parameters to be concatenated
    in wrong order when user declaration order differed from CVXPY's internal
    column assignment order.
    """
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
    layer_abc = CvxpyLayer(
        problem, parameters=[a, b, c], variables=[x, y, z], gp=True,
        solver_args=GRADCHECK_SOLVER_ARGS,
    )
    layer_cba = CvxpyLayer(
        problem, parameters=[c, b, a], variables=[x, y, z], gp=True,
        solver_args=GRADCHECK_SOLVER_ARGS,
    )

    a_jax = jnp.array(2.0)
    b_jax = jnp.array(1.0)
    c_jax = jnp.array(0.5)

    # Forward pass with both orderings
    x1, y1, z1 = layer_abc(a_jax, b_jax, c_jax)
    x2, y2, z2 = layer_cba(c_jax, b_jax, a_jax)

    # Both orderings should produce same results
    assert jnp.allclose(x1, x2, atol=1e-5), f"x mismatch: {x1} vs {x2}"
    assert jnp.allclose(y1, y2, atol=1e-5), f"y mismatch: {y1} vs {y2}"
    assert jnp.allclose(z1, z2, atol=1e-5), f"z mismatch: {z1} vs {z2}"

    # Verify against CVXPY ground truth
    a.value = 2.0
    b.value = 1.0
    c.value = 0.5
    problem.solve(cp.CLARABEL, gp=True)

    assert np.isclose(x.value, float(x1), atol=1e-5)
    assert np.isclose(y.value, float(y1), atol=1e-5)
    assert np.isclose(z.value, float(z1), atol=1e-5)

    # Test gradients for reversed order
    _check_gp_gradients(layer_cba, (c_jax, b_jax, a_jax), order=(2, 1, 0))


def test_batch_size_one_preserves_batch_dimension():
    """Test that batch_size=1 is different from unbatched.

    When the input is explicitly batched with batch_size=1 (shape (1, n)),
    the output should also be batched with shape (1, n), not unbatched (n,).
    """
    key = random.PRNGKey(0)
    n = 3
    x = cp.Variable(n)
    b = cp.Parameter(n)

    # Simple quadratic problem: minimize ||x - b||^2
    objective = cp.Minimize(cp.sum_squares(x - b))
    problem = cp.Problem(objective)

    cvxpylayer = CvxpyLayer(problem, parameters=[b], variables=[x])

    # Create explicitly batched input with batch_size=1
    key, k1 = random.split(key)
    b_batched = random.normal(k1, shape=(1, n))  # Shape: (1, n)

    # Solve
    (x_batched,) = cvxpylayer(b_batched)

    # Solution should be batched
    assert x_batched.shape == (1, n), f"Expected shape (1, {n}), got {x_batched.shape}"

    # Compute gradient
    def loss_fn(b):
        (x_sol,) = cvxpylayer(b)
        return jnp.sum(x_sol)

    grad_b = jax.grad(loss_fn)(b_batched)

    # Gradient should preserve batch dimension
    assert grad_b.shape == (1, n), (
        f"Expected gradient shape (1, {n}), got {grad_b.shape}. "
        "Batch dimension should be preserved for batch_size=1."
    )
