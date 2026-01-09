import numpy as np
import pytest

jax = pytest.importorskip("jax")

import cvxpy as cp  # noqa: E402
import diffcp  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from jax import random  # noqa: E402
from jax.test_util import check_grads  # noqa: E402

from cvxpylayers.jax import CvxpyLayer  # noqa: E402


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

    check_grads(f, (P_sqrt_jax, q_jax, A_jax, b_jax), order=1, modes=["rev"])


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
        x = prob_jax(A_jax, b_jax, solver_args={"eps": 1e-10})[0]
        return sum(x)

    def lstsq_sum_linalg(A_jax, b_jax):
        x = jnp.linalg.solve(A_jax.T @ A_jax + jnp.eye(n), A_jax.T @ b_jax)
        return sum(x)

    d_lstsq_sum_cp = jax.grad(lstsq_sum_cp, [0, 1])
    d_lstsq_sum_linalg = jax.grad(lstsq_sum_linalg, [0, 1])

    grad_A_cvxpy, grad_b_cvxpy = d_lstsq_sum_cp(A_jax, b_jax)
    grad_A_lstsq, grad_b_lstsq = d_lstsq_sum_linalg(A_jax, b_jax)

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

    fit_logreg = CvxpyLayer(prob, [X, lam], [a])

    check_grads(fit_logreg, (X_jax, lam_jax), order=1, modes=["rev"])


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
    lml = CvxpyLayer(prob, [x], [y])

    x_th = jnp.array([1.0, -1.0, -1.0, -1.0])
    check_grads(lml, (x_th,), order=1, modes=["rev"])


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
    C_jax = jnp.array([[2.0, 0.5, 0.1], [0.5, 3.0, 0.2], [0.1, 0.2, 1.5]])

    check_grads(layer, (C_jax,), order=1, modes=["rev"], atol=1e-4, rtol=1e-3)


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
    param_jax = jnp.ones(1)
    with pytest.raises(diffcp.SolverError):
        layer(param_jax)


def test_unbounded():
    x = cp.Variable(1)
    param = cp.Parameter(1)
    prob = cp.Problem(cp.Minimize(x), [x <= param])
    layer = CvxpyLayer(prob, [param], [x])
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
        x = prob_jax(A_jax, b_jax, solver_args={"eps": 1e-10})[0]
        return jnp.sum(x)

    def lstsq_sum_linalg(A_jax, b_jax):
        x = jnp.linalg.solve(A_jax.T @ A_jax + jnp.eye(n), A_jax.T @ b_jax)
        return sum(x)

    d_lstsq_sum_cp = jax.grad(lstsq_sum_cp, [0, 1])
    d_lstsq_sum_linalg = jax.grad(lstsq_sum_linalg, [0, 1])

    grad_A_lstsq, grad_b_lstsq = d_lstsq_sum_linalg(A_jax, b_jax_0)
    grad_A_cvxpy, grad_b_cvxpy = d_lstsq_sum_cp(A_jax, b_jax)

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
        "eps": 1e-10,
        "acceleration_lookback": 0,
        "max_iters": 10000,
    }

    def f(A_jax):
        (x1,) = layer1(A_jax, solver_args=solver_args)
        (x2,) = layer2(A_jax, solver_args=solver_args)
        return jnp.concatenate((x1, x2))

    check_grads(f, [A_jax], order=1, modes=["rev"])


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

    layer = CvxpyLayer(problem, parameters=[a, b, c], variables=[x, y, z], gp=True)
    a_jax = jnp.array(2.0)
    b_jax = jnp.array(1.0)
    c_jax = jnp.array(0.5)
    x_jax, y_jax, z_jax = layer(a_jax, b_jax, c_jax)

    assert np.isclose(x.value, x_jax, atol=1e-5)
    assert np.isclose(y.value, y_jax, atol=1e-5)
    assert np.isclose(z.value, z_jax, atol=1e-5)

    check_grads(
        lambda a, b, c: jnp.sum(
            layer(
                a,
                b,
                c,
                solver_args={"acceleration_lookback": 0},
            )[0],
        ),
        [a_jax, b_jax, c_jax],
        order=1,
        modes=["rev"],
    )


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
    check_grads(
        lambda a, b, c: jnp.sum(
            layer(a, b, c, solver_args={"acceleration_lookback": 0})[0],
        ),
        [a_batch, b_batch, c_batch],
        order=1,
        modes=["rev"],
    )


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
    check_grads(
        lambda a, b, c: jnp.sum(
            layer(
                a,
                b,
                c,
                solver_args={"acceleration_lookback": 0},
            )[0],
        ),
        [a_jax, b_jax, c_jax],
        order=1,
        modes=["rev"],
    )
