import unittest
import cvxpy as cp
import diffcp
import numpy as np
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


class TestCvxpyLayer(unittest.TestCase):
    @unittest.skip
    def test_docstring_example(self):
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

    def test_simple_qp(self):
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

        G.value = G_t.numpy()
        h.value = h_t.numpy()
        problem.solve(solver=cp.SCS)
        self.assertEqual(len(soln.values()), len(problem.variables()))
        np.testing.assert_almost_equal(
            x.value, list(soln.values())[0], decimal=5)

        def f():
            problem.solve(solver=cp.SCS, eps=1e-10)
            return np.sum(x.value)

        numgradG, numgradh = numerical_grad(f, [G, h], [G_t, h_t])
        np.testing.assert_almost_equal(gradG, numgradG, decimal=3)
        np.testing.assert_almost_equal(gradh, numgradh, decimal=3)

    def test_simple_qp_with_solver_args(self):
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

        G.value = G_t.numpy()
        h.value = h_t.numpy()
        problem.solve(solver=cp.SCS)
        self.assertEqual(len(soln.values()), len(problem.variables()))
        np.testing.assert_almost_equal(
            x.value, list(soln.values())[0], decimal=5)

        def f():
            problem.solve(solver=cp.SCS, eps=1e-10)
            return np.sum(x.value)

        numgradG, numgradh = numerical_grad(f, [G, h], [G_t, h_t])
        np.testing.assert_almost_equal(gradG, numgradG, decimal=3)
        np.testing.assert_almost_equal(gradh, numgradh, decimal=3)

    def test_simple_qp_batched(self):
        np.random.seed(0)
        tf.random.set_seed(0)
        nbtch, nx, ncon = 4, 3, 2

        G = cp.Parameter((ncon, nx), name='G')
        h = cp.Parameter(ncon, name='h')
        x = cp.Variable(nx, name='x')
        obj = cp.Minimize(0.5 * cp.sum_squares(x - 1))
        cons = [G * x <= h]
        problem = cp.Problem(obj, cons)

        cvxlayer = CvxpyLayer(problem, [G, h], [x])
        x0 = tf.random.normal((nx, 1))
        s0 = tf.random.normal((ncon, 1))
        G_t = tf.random.normal((nbtch, ncon, nx))
        h_t = tf.squeeze(tf.tensordot(G_t, x0, axes=1) + s0)

        with tf.GradientTape() as tape:
            tape.watch(G_t)
            tape.watch(h_t)
            soln = cvxlayer(G_t, h_t, solver_args={'eps': 1e-10})
        soln = {x.name(): soln[0]}

        grads = tape.gradient(soln, [G_t, h_t])
        gradG = grads[0]
        gradh = grads[1]

        solns = [tf.squeeze(t).numpy() for t in tf.split(soln['x'], nbtch)]
        Gs = [tf.squeeze(t) for t in tf.split(G_t, nbtch)]
        hs = [tf.squeeze(t) for t in tf.split(h_t, nbtch)]
        gradGs = [tf.squeeze(t).numpy() for t in tf.split(gradG, nbtch)]
        gradhs = [tf.squeeze(t).numpy() for t in tf.split(gradh, nbtch)]

        for soln, G_t, h_t, gG, gh in zip(solns, Gs, hs, gradGs, gradhs):
            G.value = G_t.numpy()
            h.value = h_t.numpy()
            problem.solve(solver=cp.SCS)
            np.testing.assert_almost_equal(x.value, soln, decimal=5)

            def f():
                problem.solve(solver=cp.SCS, eps=1e-10)
                return np.sum(x.value)

            numgradG, numgradh = numerical_grad(f, [G, h], [G_t, h_t])
            np.testing.assert_almost_equal(gG, numgradG, decimal=2)
            np.testing.assert_almost_equal(gh, numgradh, decimal=2)

    def test_logistic_regression(self):
        np.random.seed(243)
        N, n = 10, 2

        def sigmoid(z):
            return 1 / (1 + np.exp(-z))

        X_np = np.random.randn(N, n)
        a_true = np.random.randn(n, 1)
        y_np = np.round(sigmoid(X_np @ a_true + np.random.randn(N, 1) * 0.5))

        X_tf = tf.Variable(X_np)
        lam_tf = tf.Variable(1.0 * tf.ones(1))

        a = cp.Variable((n, 1))
        X = cp.Parameter((N, n))
        lam = cp.Parameter(1, nonneg=True)
        y = y_np

        log_likelihood = cp.sum(
            cp.multiply(y, X @ a) -
            cp.log_sum_exp(cp.hstack([np.zeros((N, 1)), X @ a]).T, axis=0,
                           keepdims=True).T
        )
        prob = cp.Problem(
            cp.Minimize(-log_likelihood + lam * cp.sum_squares(a)))
        fit_logreg = CvxpyLayer(prob, [X, lam], [a])

        with tf.GradientTape(persistent=True) as tape:
            weights = fit_logreg(X_tf, lam_tf, solver_args={'eps': 1e-8})[0]
            summed = tf.math.reduce_sum(weights)
        grad_X_tf, grad_lam_tf = tape.gradient(summed, [X_tf, lam_tf])

        def f_train():
            prob.solve(solver=cp.SCS, eps=1e-8)
            return np.sum(a.value)

        numgrad_X_tf, numgrad_lam_tf = numerical_grad(
            f_train, [X, lam], [X_tf, lam_tf], delta=1e-6)
        np.testing.assert_allclose(grad_X_tf, numgrad_X_tf, atol=1e-2)
        np.testing.assert_allclose(grad_lam_tf, numgrad_lam_tf, atol=1e-2)

    def test_not_enough_parameters(self):
        x = cp.Variable(1)
        lam = cp.Parameter(1, nonneg=True)
        lam2 = cp.Parameter(1, nonneg=True)
        objective = lam * cp.norm(x, 1) + lam2 * cp.sum_squares(x)
        prob = cp.Problem(cp.Minimize(objective))
        with self.assertRaisesRegex(ValueError, "The layer's parameters.*"):
            CvxpyLayer(prob, [lam], [x])  # noqa: F841

    def test_not_enough_parameters_at_call_time(self):
        x = cp.Variable(1)
        lam = cp.Parameter(1, nonneg=True)
        lam2 = cp.Parameter(1, nonneg=True)
        objective = lam * cp.norm(x, 1) + lam2 * cp.sum_squares(x)
        prob = cp.Problem(cp.Minimize(objective))
        layer = CvxpyLayer(prob, [lam, lam2], [x])
        with self.assertRaisesRegex(
                ValueError,
                'A tensor must be provided for each CVXPY parameter.*'):
            layer(lam)

    def test_non_dpp(self):
        x = cp.Variable(1)
        y = cp.Variable(1)
        lam = cp.Parameter(1)
        objective = lam * cp.norm(x, 1)
        prob = cp.Problem(cp.Minimize(objective))
        with self.assertRaisesRegex(ValueError, 'Problem must be DPP.'):
            CvxpyLayer(prob, [lam], [x, y])  # noqa: F841

    def test_too_many_variables(self):
        x = cp.Variable(1)
        y = cp.Variable(1)
        lam = cp.Parameter(1, nonneg=True)
        objective = lam * cp.norm(x, 1)
        prob = cp.Problem(cp.Minimize(objective))
        with self.assertRaisesRegex(ValueError, 'Argument `variables`.*'):
            CvxpyLayer(prob, [lam], [x, y])  # noqa: F841

    def test_infeasible(self):
        x = cp.Variable(1)
        param = cp.Parameter(1)
        prob = cp.Problem(cp.Minimize(param), [x >= 1, x <= -1])
        layer = CvxpyLayer(prob, [param], [x])
        param_tf = tf.ones(1)
        with self.assertRaises(diffcp.SolverError):
            layer(param_tf)

    def test_lml(self):
        tf.random.set_seed(0)
        k = 2
        x = cp.Parameter(4)
        y = cp.Variable(4)
        obj = -x * y - cp.sum(cp.entr(y)) - cp.sum(cp.entr(1. - y))
        cons = [cp.sum(y) == k]
        problem = cp.Problem(cp.Minimize(obj), cons)
        lml = CvxpyLayer(problem, [x], [y])
        x_tf = tf.Variable([1., -1., -1., -1.], dtype=tf.float64)

        with tf.GradientTape() as tape:
            y_opt = lml(x_tf, solver_args={'eps': 1e-10})[0]
            loss = -tf.math.log(y_opt[1])

        def f():
            problem.solve(solver=cp.SCS, eps=1e-10)
            return -np.log(y.value[1])

        grad = tape.gradient(loss, [x_tf])
        numgrad = numerical_grad(f, [x], [x_tf])
        np.testing.assert_almost_equal(grad, numgrad, decimal=3)

    @unittest.skip
    def test_sdp(self):
        tf.random.set_seed(5)

        n = 3
        p = 3
        C = cp.Parameter((n, n))
        A = [cp.Parameter((n, n)) for _ in range(p)]
        b = [cp.Parameter((1, 1)) for _ in range(p)]

        C_tf = tf.Variable(tf.random.normal((n, n), dtype=tf.float64))
        A_tf = [tf.Variable(tf.random.normal((n, n), dtype=tf.float64))
                for _ in range(p)]
        b_tf = [tf.Variable(tf.random.normal((1, 1), dtype=tf.float64))
                for _ in range(p)]

        X = cp.Variable((n, n), symmetric=True)
        constraints = [X >> 0]
        constraints += [
            cp.trace(A[i]@X) == b[i] for i in range(p)
        ]
        problem = cp.Problem(cp.Minimize(
            cp.trace(C @ X) - cp.log_det(X) + cp.sum_squares(X)),
            constraints)
        layer = CvxpyLayer(problem, [C] + A + b, [X])
        values = [C_tf] + A_tf + b_tf
        with tf.GradientTape() as tape:
            soln = layer(*values,
                         solver_args={'eps': 1e-10, 'max_iters': 10000})[0]
            summed = tf.math.reduce_sum(soln)
        grads = tape.gradient(summed, values)

        def f():
            problem.solve(cp.SCS, eps=1e-10, max_iters=10000)
            return np.sum(X.value)

        numgrads = numerical_grad(f, [C] + A + b, values, delta=1e-4)
        for g, ng in zip(grads, numgrads):
            np.testing.assert_allclose(g, ng, atol=1e-1)

    @unittest.skip
    def test_basic_gp(self):
        tf.random.set_seed(243)

        x = cp.Variable(pos=True)
        y = cp.Variable(pos=True)
        z = cp.Variable(pos=True)

        a = cp.Parameter(pos=True, value=2.0)
        b = cp.Parameter(pos=True, value=1.0)
        c = cp.Parameter(value=0.5)

        objective_fn = 1/(x*y*z)
        constraints = [a*(x*y + x*z + y*z) <= b, x >= y**c]
        problem = cp.Problem(cp.Minimize(objective_fn), constraints)
        problem.solve(cp.SCS, gp=True, eps=1e-12)

        layer = CvxpyLayer(
            problem, parameters=[a, b, c], variables=[x, y, z], gp=True)
        a_tf = tf.Variable(2.0, dtype=tf.float64)
        b_tf = tf.Variable(1.0, dtype=tf.float64)
        c_tf = tf.Variable(0.5, dtype=tf.float64)
        with tf.GradientTape() as tape:
            x_tf, y_tf, z_tf = layer(a_tf, b_tf, c_tf)
            summed = x_tf + y_tf + z_tf
        grads = tape.gradient(summed, [a_tf, b_tf, c_tf])

        def f():
            problem.solve(cp.SCS, eps=1e-12, max_iters=10000, gp=True)
            return x.value + y.value + z.value

        numgrads = numerical_grad(f, [a, b, c], [a_tf, b_tf, c_tf])
        for g, ng in zip(grads, numgrads):
            np.testing.assert_allclose(g, ng, atol=1e-2)

    def test_broadcasting(self):
        tf.random.set_seed(243)
        n_batch, m, n = 2, 500, 20

        A = cp.Parameter((m, n))
        b = cp.Parameter(m)
        x = cp.Variable(n)
        obj = cp.sum_squares(A@x - b) + cp.sum_squares(x)
        prob = cp.Problem(cp.Minimize(obj))
        prob_tf = CvxpyLayer(prob, [A, b], [x])

        A_tf = tf.Variable(tf.random.normal((m, n), dtype=tf.float64))
        b_tf = tf.random.normal([m], dtype=tf.float64)
        b_tf = tf.Variable(tf.stack([b_tf for _ in range(n_batch)]))
        b_tf_0 = tf.Variable(b_tf[0])

        with tf.GradientTape() as tape:
            x = prob_tf(A_tf, b_tf, solver_args={"eps": 1e-12})[0]
        grad_A_cvxpy, grad_b_cvxpy = tape.gradient(x, [A_tf, b_tf])

        with tf.GradientTape() as tape:
            x_lstsq = tf.linalg.lstsq(A_tf, tf.expand_dims(b_tf_0, 1))
        grad_A_lstsq, grad_b_lstsq = tape.gradient(x_lstsq, [A_tf, b_tf_0])
        grad_A_lstsq = tf.cast(grad_A_lstsq, tf.float64)
        grad_b_lstsq = tf.cast(grad_b_lstsq, tf.float64)

        self.assertAlmostEqual(
            tf.linalg.norm(grad_A_cvxpy / n_batch - grad_A_lstsq).numpy(),
            0.0, places=2)
        self.assertAlmostEqual(
            tf.linalg.norm(grad_b_cvxpy[0] - grad_b_lstsq).numpy(), 0.0,
            places=2)


if __name__ == '__main__':
    unittest.main()
