"""Unit tests for cvxpylayers.tensorflow."""

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

import cvxpy as cp  # noqa: E402
import diffcp  # noqa: E402

from cvxpylayers.tensorflow import CvxpyLayer  # noqa: E402


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def test_example():
    np.random.seed(0)

    n, m = 2, 3
    x = cp.Variable(n)
    A = cp.Parameter((m, n))
    b = cp.Parameter(m)
    constraints = [x >= 0]
    objective = cp.Minimize(0.5 * cp.pnorm(A @ x - b, p=1))
    problem = cp.Problem(objective, constraints)
    assert problem.is_dpp()

    cvxpylayer = CvxpyLayer(problem, parameters=[A, b], variables=[x])

    A_tf = tf.Variable(tf.random.normal((m, n), dtype=tf.float64), dtype=tf.float64)
    b_tf = tf.Variable(tf.random.normal((m,), dtype=tf.float64), dtype=tf.float64)

    # solve the problem
    (solution,) = cvxpylayer(A_tf, b_tf)

    # compute the gradient of the sum of the solution with respect to A, b
    with tf.GradientTape() as tape:
        (solution,) = cvxpylayer(A_tf, b_tf)
        loss = tf.reduce_sum(solution)

    grads = tape.gradient(loss, [A_tf, b_tf])
    assert grads[0] is not None
    assert grads[1] is not None


def test_simple_batch_socp():
    np.random.seed(0)
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

    prob_tf = CvxpyLayer(prob, [P_sqrt, q, A, b], [x])

    P_sqrt_tf = tf.Variable(
        tf.random.normal((batch_size, n, n), dtype=tf.float64), dtype=tf.float64
    )
    q_tf = tf.Variable(
        tf.random.normal((batch_size, n, 1), dtype=tf.float64), dtype=tf.float64
    )
    A_tf = tf.Variable(
        tf.random.normal((batch_size, m, n), dtype=tf.float64), dtype=tf.float64
    )
    b_tf = tf.Variable(
        tf.random.normal((batch_size, m, 1), dtype=tf.float64), dtype=tf.float64
    )

    with tf.GradientTape() as tape:
        (sol,) = prob_tf(P_sqrt_tf, q_tf, A_tf, b_tf)
        loss = tf.reduce_sum(sol)

    grads = tape.gradient(loss, [P_sqrt_tf, q_tf, A_tf, b_tf])
    for g in grads:
        assert g is not None


def test_least_squares():
    np.random.seed(0)
    m, n = 100, 20

    A = cp.Parameter((m, n))
    b = cp.Parameter(m)
    x = cp.Variable(n)
    obj = cp.sum_squares(A @ x - b) + cp.sum_squares(x)
    prob = cp.Problem(cp.Minimize(obj))
    prob_tf = CvxpyLayer(prob, [A, b], [x])

    A_tf = tf.Variable(tf.random.normal((m, n), dtype=tf.float64), dtype=tf.float64)
    b_tf = tf.Variable(tf.random.normal((m,), dtype=tf.float64), dtype=tf.float64)

    # Compute gradients via cvxpylayers
    with tf.GradientTape() as tape:
        (x_opt,) = prob_tf(A_tf, b_tf, solver_args={"eps": 1e-10})
        loss_cvxpy = tf.reduce_sum(x_opt)

    grad_A_cvxpy, grad_b_cvxpy = tape.gradient(loss_cvxpy, [A_tf, b_tf])

    # Compute gradients via closed-form least squares
    with tf.GradientTape() as tape:
        x_lstsq = tf.linalg.solve(
            tf.transpose(A_tf) @ A_tf + tf.eye(n, dtype=tf.float64),
            tf.expand_dims(tf.linalg.matvec(tf.transpose(A_tf), b_tf), 1),
        )
        loss_lstsq = tf.reduce_sum(x_lstsq)

    grad_A_lstsq, grad_b_lstsq = tape.gradient(loss_lstsq, [A_tf, b_tf])

    np.testing.assert_allclose(
        grad_A_cvxpy.numpy(), grad_A_lstsq.numpy(), atol=1e-6, rtol=1e-6
    )
    np.testing.assert_allclose(
        grad_b_cvxpy.numpy(), grad_b_lstsq.numpy(), atol=1e-6, rtol=1e-6
    )


def test_logistic_regression():
    np.random.seed(0)

    N, n = 5, 2

    X_np = np.random.randn(N, n)
    a_true = np.random.randn(n, 1)
    y_np = np.round(sigmoid(X_np @ a_true + np.random.randn(N, 1) * 0.5))

    X_tf = tf.Variable(X_np, dtype=tf.float64)
    lam_tf = tf.Variable(0.1 * np.ones(1), dtype=tf.float64)

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

    with tf.GradientTape() as tape:
        (a_opt,) = fit_logreg(X_tf, lam_tf)
        loss = tf.reduce_sum(a_opt)

    grads = tape.gradient(loss, [X_tf, lam_tf])
    assert grads[0] is not None
    assert grads[1] is not None


def test_lml():
    k = 2
    x = cp.Parameter(4)
    y = cp.Variable(4)
    obj = -x @ y - cp.sum(cp.entr(y)) - cp.sum(cp.entr(1.0 - y))
    cons = [cp.sum(y) == k]
    prob = cp.Problem(cp.Minimize(obj), cons)
    lml = CvxpyLayer(prob, [x], [y])

    x_tf = tf.Variable([1.0, -1.0, -1.0, -1.0], dtype=tf.float64)

    with tf.GradientTape() as tape:
        (y_opt,) = lml(x_tf)
        loss = tf.reduce_sum(y_opt)

    grad = tape.gradient(loss, x_tf)
    assert grad is not None


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
    lam_tf = tf.ones(1, dtype=tf.float64)
    with pytest.raises(
        ValueError,
        match="A tensor must be provided for each CVXPY parameter.*",
    ):
        layer(lam_tf)


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
    param_tf = tf.ones(1, dtype=tf.float64)
    with pytest.raises(diffcp.SolverError):
        layer(param_tf)


def test_unbounded():
    x = cp.Variable(1)
    param = cp.Parameter(1)
    prob = cp.Problem(cp.Minimize(x), [x <= param])
    layer = CvxpyLayer(prob, [param], [x])
    param_tf = tf.ones(1, dtype=tf.float64)
    with pytest.raises(diffcp.SolverError):
        layer(param_tf)


def test_incorrect_parameter_shape():
    np.random.seed(0)
    m, n = 100, 20

    A = cp.Parameter((m, n))
    b = cp.Parameter(m)
    x = cp.Variable(n)
    obj = cp.sum_squares(A @ x - b) + cp.sum_squares(x)
    prob = cp.Problem(cp.Minimize(obj))
    prob_tf = CvxpyLayer(prob, [A, b], [x])

    A_tf = tf.random.normal((32, m, n), dtype=tf.float64)
    b_tf = tf.random.normal((20, m), dtype=tf.float64)

    with pytest.raises(ValueError, match="Inconsistent batch sizes"):
        prob_tf(A_tf, b_tf)

    A_tf = tf.random.normal((32, m, n), dtype=tf.float64)
    b_tf = tf.random.normal((32, 2 * m), dtype=tf.float64)

    with pytest.raises(ValueError, match="Invalid parameter shape"):
        prob_tf(A_tf, b_tf)

    A_tf = tf.random.normal((m, n), dtype=tf.float64)
    b_tf = tf.random.normal((2 * m,), dtype=tf.float64)

    with pytest.raises(ValueError, match="Invalid parameter shape"):
        prob_tf(A_tf, b_tf)

    A_tf = tf.random.normal((32, m, n), dtype=tf.float64)
    b_tf = tf.random.normal((32, 32, m), dtype=tf.float64)

    with pytest.raises(ValueError, match="Invalid parameter dimensionality"):
        prob_tf(A_tf, b_tf)


def test_broadcasting():
    np.random.seed(0)
    m, n = 100, 20

    A = cp.Parameter((m, n))
    b = cp.Parameter(m)
    x = cp.Variable(n)
    obj = cp.sum_squares(A @ x - b) + cp.sum_squares(x)
    prob = cp.Problem(cp.Minimize(obj))
    prob_tf = CvxpyLayer(prob, [A, b], [x])

    A_tf = tf.Variable(tf.random.normal((m, n), dtype=tf.float64), dtype=tf.float64)
    b_tf_0 = tf.random.normal((m,), dtype=tf.float64)
    b_tf = tf.Variable(tf.stack([b_tf_0, b_tf_0]), dtype=tf.float64)

    # Compute gradient from cvxpylayers (batched)
    with tf.GradientTape() as tape:
        (x_opt,) = prob_tf(A_tf, b_tf, solver_args={"eps": 1e-10})
        loss_cvxpy = tf.reduce_sum(x_opt)

    grad_A_cvxpy, grad_b_cvxpy = tape.gradient(loss_cvxpy, [A_tf, b_tf])

    # Compute gradient from closed-form (single)
    with tf.GradientTape() as tape:
        x_lstsq = tf.linalg.solve(
            tf.transpose(A_tf) @ A_tf + tf.eye(n, dtype=tf.float64),
            tf.expand_dims(tf.linalg.matvec(tf.transpose(A_tf), b_tf_0), 1),
        )
        loss_lstsq = tf.reduce_sum(x_lstsq)

    grad_A_lstsq, _ = tape.gradient(loss_lstsq, [A_tf, b_tf])

    # Batched gradient should be 2x the single gradient
    np.testing.assert_allclose(
        (grad_A_cvxpy / 2.0).numpy(), grad_A_lstsq.numpy(), atol=1e-6, rtol=1e-6
    )


def test_shared_parameter():
    np.random.seed(0)
    m, n = 10, 5

    A = cp.Parameter((m, n))
    x = cp.Variable(n)
    b1 = np.random.randn(m)
    b2 = np.random.randn(m)
    prob1 = cp.Problem(cp.Minimize(cp.sum_squares(A @ x - b1)))
    layer1 = CvxpyLayer(prob1, parameters=[A], variables=[x])
    prob2 = cp.Problem(cp.Minimize(cp.sum_squares(A @ x - b2)))
    layer2 = CvxpyLayer(prob2, parameters=[A], variables=[x])

    A_tf = tf.Variable(tf.random.normal((m, n), dtype=tf.float64), dtype=tf.float64)
    solver_args = {
        "eps": 1e-10,
        "acceleration_lookback": 0,
        "max_iters": 10000,
    }

    with tf.GradientTape() as tape:
        (x1,) = layer1(A_tf, solver_args=solver_args)
        (x2,) = layer2(A_tf, solver_args=solver_args)
        loss = tf.reduce_sum(tf.concat([x1, x2], axis=0))

    grad = tape.gradient(loss, A_tf)
    assert grad is not None


def test_equality():
    np.random.seed(0)
    n = 10
    A = np.eye(n)
    x = cp.Variable(n)
    b = cp.Parameter(n)
    prob = cp.Problem(cp.Minimize(cp.sum_squares(x)), [A @ x == b])
    layer = CvxpyLayer(prob, parameters=[b], variables=[x])

    b_tf = tf.Variable(tf.random.normal((n,), dtype=tf.float64), dtype=tf.float64)

    with tf.GradientTape() as tape:
        (x_opt,) = layer(b_tf)
        loss = tf.reduce_sum(x_opt)

    grad = tape.gradient(loss, b_tf)
    assert grad is not None


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
    a_tf = tf.constant(2.0, dtype=tf.float64)
    b_tf = tf.constant(1.0, dtype=tf.float64)
    c_tf = tf.constant(0.5, dtype=tf.float64)
    x_tf, y_tf, z_tf = layer(a_tf, b_tf, c_tf)

    np.testing.assert_allclose(x.value, x_tf.numpy(), atol=1e-5)
    np.testing.assert_allclose(y.value, y_tf.numpy(), atol=1e-5)
    np.testing.assert_allclose(z.value, z_tf.numpy(), atol=1e-5)

    # Test gradient computation
    a_tf_var = tf.Variable(2.0, dtype=tf.float64)
    b_tf_var = tf.Variable(1.0, dtype=tf.float64)
    c_tf_var = tf.Variable(0.5, dtype=tf.float64)

    with tf.GradientTape() as tape:
        x_opt, y_opt, z_opt = layer(
            a_tf_var, b_tf_var, c_tf_var, solver_args={"acceleration_lookback": 0}
        )
        loss = tf.reduce_sum(x_opt)

    grads = tape.gradient(loss, [a_tf_var, b_tf_var, c_tf_var])
    for g in grads:
        assert g is not None


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
    batch_size = 4
    a_batch = tf.constant([2.0, 1.5, 2.5, 1.8], dtype=tf.float64)
    b_batch = tf.constant([1.0, 1.2, 0.8, 1.5], dtype=tf.float64)
    c_batch = tf.constant([0.5, 0.6, 0.4, 0.5], dtype=tf.float64)

    # Forward pass
    x_batch, y_batch, z_batch = layer(a_batch, b_batch, c_batch)

    # Check shapes - batched results are (batch_size,) for scalar variables
    assert x_batch.shape == (batch_size,)
    assert y_batch.shape == (batch_size,)
    assert z_batch.shape == (batch_size,)

    # Verify each batch element by solving individually
    for i in range(batch_size):
        a.value = float(a_batch[i])
        b.value = float(b_batch[i])
        c.value = float(c_batch[i])
        problem.solve(cp.CLARABEL, gp=True)

        np.testing.assert_allclose(x.value, x_batch[i].numpy(), atol=1e-4, rtol=1e-4)
        np.testing.assert_allclose(y.value, y_batch[i].numpy(), atol=1e-4, rtol=1e-4)
        np.testing.assert_allclose(z.value, z_batch[i].numpy(), atol=1e-4, rtol=1e-4)

    # Test gradients on batched problem
    a_batch_var = tf.Variable([2.0, 1.5, 2.5, 1.8], dtype=tf.float64)
    b_batch_var = tf.Variable([1.0, 1.2, 0.8, 1.5], dtype=tf.float64)
    c_batch_var = tf.Variable([0.5, 0.6, 0.4, 0.5], dtype=tf.float64)

    with tf.GradientTape() as tape:
        x_opt, y_opt, z_opt = layer(
            a_batch_var,
            b_batch_var,
            c_batch_var,
            solver_args={"acceleration_lookback": 0},
        )
        loss = tf.reduce_sum(x_opt)

    grads = tape.gradient(loss, [a_batch_var, b_batch_var, c_batch_var])
    for g in grads:
        assert g is not None


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
    a_tf = tf.constant(2.0, dtype=tf.float64)
    b_tf = tf.constant(1.0, dtype=tf.float64)
    c_tf = tf.constant(0.5, dtype=tf.float64)

    # Forward pass
    x_tf, y_tf, z_tf = layer(a_tf, b_tf, c_tf)

    # Verify solution against CVXPY direct solve
    a.value = 2.0
    b.value = 1.0
    c.value = 0.5
    problem.solve(cp.CLARABEL, gp=True)

    np.testing.assert_allclose(x.value, x_tf.numpy(), atol=1e-5)
    np.testing.assert_allclose(y.value, y_tf.numpy(), atol=1e-5)
    np.testing.assert_allclose(z.value, z_tf.numpy(), atol=1e-5)

    # Test gradients
    a_tf_var = tf.Variable(2.0, dtype=tf.float64)
    b_tf_var = tf.Variable(1.0, dtype=tf.float64)
    c_tf_var = tf.Variable(0.5, dtype=tf.float64)

    with tf.GradientTape() as tape:
        x_opt, y_opt, z_opt = layer(
            a_tf_var, b_tf_var, c_tf_var, solver_args={"acceleration_lookback": 0}
        )
        loss = tf.reduce_sum(x_opt)

    grads = tape.gradient(loss, [a_tf_var, b_tf_var, c_tf_var])
    for g in grads:
        assert g is not None


def test_nd_array_variable():
    """Test with multidimensional array variables."""
    np.random.seed(0)
    n, k = 5, 3

    X = cp.Variable((n, k))
    A = cp.Parameter((n, n))
    B = cp.Parameter((n, k))
    prob = cp.Problem(cp.Minimize(cp.sum_squares(A @ X - B)))
    layer = CvxpyLayer(prob, parameters=[A, B], variables=[X])

    A_tf = tf.Variable(tf.random.normal((n, n), dtype=tf.float64), dtype=tf.float64)
    B_tf = tf.Variable(tf.random.normal((n, k), dtype=tf.float64), dtype=tf.float64)

    with tf.GradientTape() as tape:
        (X_opt,) = layer(A_tf, B_tf)
        loss = tf.reduce_sum(X_opt)

    grads = tape.gradient(loss, [A_tf, B_tf])
    assert grads[0] is not None
    assert grads[1] is not None
    assert X_opt.shape == (n, k)


def test_batch_size_one_preserves_batch_dimension():
    """Test that explicitly batched input with batch_size=1 preserves output batch dim."""
    np.random.seed(0)
    n = 5

    x = cp.Variable(n)
    b = cp.Parameter(n)
    prob = cp.Problem(cp.Minimize(cp.sum_squares(x - b)))
    layer = CvxpyLayer(prob, parameters=[b], variables=[x])

    # Unbatched input
    b_unbatched = tf.constant(np.random.randn(n), dtype=tf.float64)
    (x_unbatched,) = layer(b_unbatched)
    assert x_unbatched.shape == (n,), f"Expected (n,), got {x_unbatched.shape}"

    # Explicitly batched with batch_size=1
    b_batched = tf.constant(np.random.randn(1, n), dtype=tf.float64)
    (x_batched,) = layer(b_batched)
    assert x_batched.shape == (1, n), f"Expected (1, n), got {x_batched.shape}"


def test_solver_args_actually_used():
    """Test that solver_args are actually passed to the solver."""
    x = cp.Variable(1)
    param = cp.Parameter(1)
    prob = cp.Problem(cp.Minimize(cp.sum_squares(x - param)))
    layer = CvxpyLayer(prob, [param], [x])

    param_tf = tf.constant([1.0], dtype=tf.float64)

    # Solve with default args
    (x_default,) = layer(param_tf)

    # Solve with very high eps (should still converge but potentially different)
    (x_high_eps,) = layer(param_tf, solver_args={"eps": 1e-2})

    # Solve with very low eps (should be more precise)
    (x_low_eps,) = layer(param_tf, solver_args={"eps": 1e-12})

    # All should be close to the target
    np.testing.assert_allclose(x_default.numpy(), param_tf.numpy(), atol=1e-3)
    np.testing.assert_allclose(x_low_eps.numpy(), param_tf.numpy(), atol=1e-10)
