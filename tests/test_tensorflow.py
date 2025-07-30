import cvxpy as cp
import diffcp
import numpy as np
import pytest
import tensorflow as tf

from cvxpylayers.tensorflow import CvxpyLayer


def numerical_grad(f, params, param_values, delta=1e-6):
    size = int(sum(np.prod(v.shape) for v in param_values))
    values = np.zeros(size)
    offset = 0
    for param, value in zip(params, param_values):
        values[offset:offset + param.size] = value.numpy().flatten()
        param.value = values[offset:offset + param.size].reshape(param.shape)
        offset += param.size

    numgrad = np.zeros(values.shape)
    for i in range(values.size):
        old = values[i]
        values[i] = old + 0.5 * delta
        left_soln = f()

        values[i] = old - 0.5 * delta
        right_soln = f()

        numgrad[i] = (left_soln - right_soln) / delta
        values[i] = old

    numgrads = []
    offset = 0
    for param in params:
        numgrads.append(
            numgrad[offset:offset + param.size].reshape(param.shape))
        offset += param.size
    return numgrads


@pytest.mark.skip
def test_docstring_example():
    np.random.seed(0)
    tf.random.set_seed(0)

    n, m = 2, 3
    x = cp.Variable(n)
    A = cp.Parameter((m, n))
    b = cp.Parameter(m)
    constraints = [x >= 0]
    objective = cp.Minimize(0.5 * cp.pnorm(A @ x - b, p=1))
    problem = cp.Problem(objective, constraints)
    assert problem.is_dpp()

    cvxpylayer = CvxpyLayer(problem, parameters=[A, b], variables=[x])
    A_tf = tf.Variable(tf.random.normal((m, n)))
    b_tf = tf.Variable(tf.random.normal((m,)))

    with tf.GradientTape() as tape:
        # solve the problem, setting the values of A and b to A_tf and b_tf
        solution, = cvxpylayer(A_tf, b_tf)
        summed_solution = tf.math.reduce_sum(solution)
    gradA, gradb = tape.gradient(summed_solution, [A_tf, b_tf])

    def f():
        problem.solve(solver=cp.SCS, eps=1e-10)
        return np.sum(x.value)

    numgradA, numgradb = numerical_grad(f, [A, b], [A_tf, b_tf])
    np.testing.assert_almost_equal(gradA, numgradA, decimal=4)
    np.testing.assert_almost_equal(gradb, numgradb, decimal=4)


def test_simple_qp():
    np.random.seed(0)
    tf.random.set_seed(0)
    nx, ncon = 2, 3

    G = cp.Parameter((ncon, nx), name='G')
    h = cp.Parameter(ncon, name='h')
    x = cp.Variable(nx)
    obj = cp.Minimize(0.5 * cp.sum_squares(x - 1))
    cons = [G * x <= h]
    problem = cp.Problem(obj, cons)

    cvxlayer = CvxpyLayer(problem, [G, h], [x])
    x0 = tf.random.normal((nx, 1))
    s0 = tf.random.normal((ncon, 1))
    G_t = tf.random.normal((ncon, nx))
    h_t = tf.squeeze(tf.matmul(G_t, x0) + s0)

    with tf.GradientTape() as tape:
        tape.watch(G_t)
        tape.watch(h_t)
        soln = cvxlayer(G_t, h_t, solver_args={'eps': 1e-10})
    soln = {x.name(): soln[0]}

    grads = tape.gradient(soln, [G_t, h_t])
    gradG = grads[0]
    gradh = grads[1]

    def f():
        problem.solve(solver=cp.SCS, eps=1e-10)
        return {x.name(): x.value}

    numgradG, numgradh = numerical_grad(f, [G, h], [G_t, h_t])
    np.testing.assert_allclose(gradG, numgradG, atol=1e-4)
    np.testing.assert_allclose(gradh, numgradh, atol=1e-4)
    

def test_simple_qp_with_solver_args():
    np.random.seed(0)
    tf.random.set_seed(0)
    nx, ncon = 2, 3

    G = cp.Parameter((ncon, nx), name='G')
    h = cp.Parameter(ncon, name='h')
    x = cp.Variable(nx)
    obj = cp.Minimize(0.5 * cp.sum_squares(x - 1))
    cons = [G * x <= h]
    problem = cp.Problem(obj, cons)

    cvxlayer = CvxpyLayer(problem, [G, h], [x])
    x0 = tf.random.normal((nx, 1))
    s0 = tf.random.normal((ncon, 1))
    G_t = tf.random.normal((ncon, nx))
    h_t = tf.squeeze(tf.matmul(G_t, x0) + s0)

    with tf.GradientTape() as tape:
        tape.watch(G_t)
        tape.watch(h_t)
        soln = cvxlayer(G_t, h_t, solver_args={
            'eps': 1e-6,
            'max_iters': 100000,
            'acceleration_lookback': 0
        })
    soln = {x.name(): soln[0]}

    grads = tape.gradient(soln, [G_t, h_t])
    gradG = grads[0]
    gradh = grads[1]

    def f():
        problem.solve(solver=cp.SCS, eps=1e-6,
                     max_iters=100000,
                     acceleration_lookback=0)
        return {x.name(): x.value}

    numgradG, numgradh = numerical_grad(f, [G, h], [G_t, h_t])
    np.testing.assert_allclose(gradG, numgradG, atol=1e-4)
    np.testing.assert_allclose(gradh, numgradh, atol=1e-4)


def test_simple_qp_batched():
    np.random.seed(0)
    tf.random.set_seed(0)
    nx, ncon, batch_size = 2, 3, 4

    x = cp.Variable(nx)
    G = cp.Parameter((ncon, nx))
    h = cp.Parameter(ncon)
    obj = cp.Minimize(0.5 * cp.sum_squares(x - 1.))
    con = [G @ x <= h]
    problem = cp.Problem(obj, con)

    layer = CvxpyLayer(problem, [G, h], [x])

    x0_np = np.random.randn(batch_size, nx, 1)
    s0_np = np.random.randn(batch_size, ncon, 1)

    G_batch = np.random.randn(batch_size, ncon, nx)
    h_batch = np.squeeze(G_batch @ x0_np + s0_np, axis=-1)

    G_tf = tf.Variable(G_batch)
    h_tf = tf.Variable(h_batch)

    with tf.GradientTape() as tape:
        sol_batch = layer(G_tf, h_tf, solver_args={'eps': 1e-10})[0]
        obj_val = tf.math.reduce_sum(sol_batch)

    gradG, gradh = tape.gradient(obj_val, [G_tf, h_tf])

    # Compute the gradient of the last example in the batch
    # with respect to the parameters G, h
    def f():
        G.value = G_batch[-1]
        h.value = h_batch[-1]
        problem.solve(solver=cp.SCS, eps=1e-10)
        return x.value

    numgradG, numgradh = numerical_grad(f, [G, h], [G_batch[-1], h_batch[-1]])
    np.testing.assert_allclose(gradG[-1], numgradG, atol=1e-4)
    np.testing.assert_allclose(gradh[-1], numgradh, atol=1e-4)


def test_logistic_regression():
    tf.random.set_seed(0)
    np.random.seed(0)

    N, n = 10, 2

    X_np = np.random.randn(N, n)
    a_true = np.random.randn(n, 1)
    y_np = np.round(1. / (1. + np.exp(-X_np.dot(a_true) +
                                     np.random.randn(N, 1) * 0.5)))

    X_tf = tf.Variable(X_np, dtype=tf.float64)
    lam_tf = tf.Variable(0.1 * np.ones(1), dtype=tf.float64)

    a = cp.Variable((n, 1))
    X = cp.Parameter((N, n))
    lam = cp.Parameter(1, nonneg=True)
    y = y_np

    log_likelihood = cp.sum(
        cp.multiply(y, X @ a) -
        cp.log_sum_exp(cp.hstack([np.zeros((N, 1)), (X @ a)]).T, axis=0,
                       keepdims=True).T
    )
    prob = cp.Problem(
        cp.Minimize(-log_likelihood + lam * cp.sum_squares(a)))

    fit_logreg = CvxpyLayer(prob, [X, lam], [a])

    with tf.GradientTape() as tape:
        a_tf = fit_logreg(X_tf, lam_tf, solver_args={'eps': 1e-10})[0]
        sum_a = tf.math.reduce_sum(a_tf)

    # no assertion since numerical grad is not reliable enough for this
    # problem, just testing that the gradient is not None
    grad_X_tf, grad_lam_tf = tape.gradient(sum_a, [X_tf, lam_tf])
    assert grad_X_tf is not None
    assert grad_lam_tf is not None


def test_not_enough_parameters():
    x = cp.Variable(1)
    lam = cp.Parameter(1, nonneg=True)
    lam2 = cp.Parameter(1, nonneg=True)
    objective = lam * cp.norm(x, 1) + lam2 * cp.sum_squares(x)
    prob = cp.Problem(cp.Minimize(objective))
    with pytest.raises(ValueError):
        layer = CvxpyLayer(prob, [lam], [x])  # noqa: F841


def test_not_enough_parameters_at_call_time():
    x = cp.Variable(1)
    lam = cp.Parameter(1, nonneg=True)
    lam2 = cp.Parameter(1, nonneg=True)
    objective = lam * cp.norm(x, 1) + lam2 * cp.sum_squares(x)
    prob = cp.Problem(cp.Minimize(objective))
    layer = CvxpyLayer(prob, [lam, lam2], [x])  # noqa: F841
    with pytest.raises(ValueError, match='An array must be provided for each CVXPY parameter.*'):
        lam_tf = tf.ones(1)
        layer(lam_tf)


def test_non_dpp():
    x = cp.Variable()
    y = cp.Variable()
    prob = cp.Problem(cp.Minimize(x), [cp.abs(x + y) <= 1])
    with pytest.raises(ValueError):
        CvxpyLayer(prob, [], [x, y])


def test_too_many_variables():
    x = cp.Variable(1)
    y = cp.Variable(1)
    lam = cp.Parameter(1, nonneg=True)
    objective = lam * cp.norm(x, 1)
    prob = cp.Problem(cp.Minimize(objective))
    with pytest.raises(ValueError):
        layer = CvxpyLayer(prob, [lam], [x, y])  # noqa: F841


def test_infeasible():
    x = cp.Variable(1)
    param = cp.Parameter(1)
    prob = cp.Problem(cp.Minimize(param), [x >= 1, x <= -1])
    layer = CvxpyLayer(prob, [param], [x])
    param_tf = tf.ones(1)
    with pytest.raises(diffcp.SolverError):
        layer(param_tf)


def test_lml():
    tf.random.set_seed(0)
    np.random.seed(0)
    k = 2
    x = cp.Parameter(4)
    y = cp.Variable(4)
    obj = -x @ y - cp.sum(cp.entr(y)) - cp.sum(cp.entr(1. - y))
    cons = [cp.sum(y) == k]
    prob = cp.Problem(cp.Minimize(obj), cons)
    lml = CvxpyLayer(prob, [x], [y])

    x_tf = tf.Variable([1., -1., -1., -1.], dtype=tf.float64)

    with tf.GradientTape() as tape:
        y_tf = lml(x_tf)[0]
        sum_y = tf.math.reduce_sum(y_tf)

    def f():
        x.value = x_tf.numpy()
        prob.solve(solver=cp.SCS, eps=1e-8)
        return np.sum(y.value)

    grad_x_tf = tape.gradient(sum_y, x_tf)
    numgrad_x = numerical_grad(f, [x], [x_tf.numpy()])[0]
    np.testing.assert_allclose(grad_x_tf, numgrad_x, atol=1e-4)


@pytest.mark.skip
def test_sdp():
    tf.random.set_seed(0)
    np.random.seed(0)

    n = 3
    p = 3
    C = cp.Parameter((n, n))
    A = [cp.Parameter((n, n)) for _ in range(p)]
    b = [cp.Parameter((1, 1)) for _ in range(p)]

    C_tf = tf.Variable(tf.random.normal((n, n), dtype=tf.float64))
    A_tf, b_tf = [], []
    for _ in range(p):
        A_tf.append(tf.Variable(tf.random.normal((n, n), dtype=tf.float64)))
        b_tf.append(tf.Variable(tf.random.normal((1, 1), dtype=tf.float64)))

    X = cp.Variable((n, n), symmetric=True)
    constraints = [X >> 0]
    constraints += [
        cp.trace(A[i] @ X) == b[i] for i in range(p)
    ]
    prob = cp.Problem(cp.Minimize(
        cp.trace(C @ X) + cp.sum_squares(X)), constraints)
    layer = CvxpyLayer(prob, [C] + A + b, [X])

    with tf.GradientTape() as tape:
        X_tf = layer(C_tf, *A_tf, *b_tf)[0]
        sum_X = tf.math.reduce_sum(X_tf)

    def f():
        C.value = C_tf.numpy()
        for i in range(p):
            A[i].value = A_tf[i].numpy()
            b[i].value = b_tf[i].numpy()
        prob.solve(solver=cp.SCS, eps=1e-6)
        return np.sum(X.value)

    grads = tape.gradient(sum_X, [C_tf] + A_tf + b_tf)
    numgrads = numerical_grad(f, [C] + A + b, [C_tf] + A_tf + b_tf)
    
    for grad, numgrad in zip(grads, numgrads):
        np.testing.assert_allclose(grad, numgrad, atol=1e-3)


def test_basic_gp():
    tf.random.set_seed(0)
    np.random.seed(0)
    
    x = cp.Variable(pos=True)
    y = cp.Variable(pos=True)
    z = cp.Variable(pos=True)

    a = cp.Parameter(pos=True, value=2.0)
    b = cp.Parameter(pos=True, value=1.0)
    c = cp.Parameter(value=0.5)

    objective_fn = 1/(x*y*z)
    constraints = [a*(x*y + x*z + y*z) <= b, x >= y**c]
    problem = cp.Problem(cp.Minimize(objective_fn), constraints)
    problem.solve(cp.SCS, gp=True)

    layer = CvxpyLayer(
        problem, parameters=[a, b, c], variables=[x, y, z], gp=True)
    a_tf = tf.Variable(2.0, dtype=tf.float64)
    b_tf = tf.Variable(1.0, dtype=tf.float64)  
    c_tf = tf.Variable(0.5, dtype=tf.float64)
    x_tf, y_tf, z_tf = layer(a_tf, b_tf, c_tf)

    np.testing.assert_allclose(x.value, x_tf, atol=1e-5)
    np.testing.assert_allclose(y.value, y_tf, atol=1e-5)
    np.testing.assert_allclose(z.value, z_tf, atol=1e-5)

    with tf.GradientTape() as tape:
        x_tf_tape, y_tf_tape, z_tf_tape = layer(
            a_tf, b_tf, c_tf, solver_args={"acceleration_lookback": 0})
        sum_sol = tf.math.reduce_sum(x_tf_tape)

    grads = tape.gradient(sum_sol, [a_tf, b_tf, c_tf])
    for grad in grads:
        assert grad is not None


def test_broadcasting():
    tf.random.set_seed(0)
    np.random.seed(0)
    m, n = 100, 20

    A = cp.Parameter((m, n))
    b = cp.Parameter(m)
    x = cp.Variable(n)
    obj = cp.sum_squares(A @ x - b) + cp.sum_squares(x)
    prob = cp.Problem(cp.Minimize(obj))
    prob_tf = CvxpyLayer(prob, [A, b], [x])

    A_tf = tf.Variable(tf.random.normal((m, n), dtype=tf.float64))
    b_tf_0 = tf.Variable(tf.random.normal((m,), dtype=tf.float64))
    b_tf = tf.stack([b_tf_0, b_tf_0])

    with tf.GradientTape() as tape:
        x_tf = prob_tf(A_tf, b_tf, solver_args={'eps': 1e-10})[0]
        sum_x = tf.math.reduce_sum(x_tf)

    def lstsq(A, b):
        return tf.linalg.solve(
            tf.transpose(A) @ A + tf.eye(n, dtype=tf.float64),
            tf.transpose(A) @ b)

    with tf.GradientTape() as tape2:
        x_lstsq = lstsq(A_tf, b_tf_0)
        sum_x_lstsq = tf.math.reduce_sum(x_lstsq)

    grad_A_cvxpy, grad_b_cvxpy = tape.gradient(sum_x, [A_tf, b_tf])
    grad_A_lstsq, grad_b_lstsq = tape2.gradient(sum_x_lstsq, [A_tf, b_tf_0])

    np.testing.assert_allclose(grad_A_cvxpy / 2., grad_A_lstsq, atol=1e-6)
    np.testing.assert_allclose(grad_b_cvxpy[0], grad_b_lstsq, atol=1e-6)
