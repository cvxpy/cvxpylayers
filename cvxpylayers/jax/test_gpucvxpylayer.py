#!/usr/bin/python3

import unittest

import cvxpy as cp
import numpy as np

import jax
import jax.numpy as jnp
from jax import random
from jax.test_util import check_grads

from cvxpylayers.jax import GpuCvxpyLayer
from cvxpylayers.utils import forward_numpy, backward_numpy
import diffcp


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

class TestGpuCvxpyLayer(unittest.TestCase):

    def test_example(self):
        key = random.PRNGKey(0)

        n, m = 2, 3
        x = cp.Variable(n)
        A = cp.Parameter((m, n))
        b = cp.Parameter(m)
        constraints = [x >= 0]
        objective = cp.Minimize(0.5 * cp.pnorm(A @ x - b, p=1))
        problem = cp.Problem(objective, constraints)
        assert problem.is_dpp()

        cvxpylayer = GpuCvxpyLayer(problem, parameters=[A, b], variables=[x])

        key, k1, k2 = random.split(key, num=3)
        A_jax = random.normal(k1, shape=(m, n))
        b_jax = random.normal(k2, shape=(m,))

        # solve the problem
        solution, = cvxpylayer(A_jax, b_jax)

        # compute the gradient of the sum of the solution with respect to A, b
        def sum_sol(A_jax, b_jax):
            solution, = cvxpylayer(A_jax, b_jax)
            return solution.sum()

        dsum_sol = jax.grad(sum_sol)
        dsum_sol(A_jax, b_jax)

    @unittest.skip
    def test_simple_batch_socp(self):
        key = random.PRNGKey(0)
        n = 5
        m = 1
        batch_size = 4

        P_sqrt = cp.Parameter((n, n), name='P_sqrt')
        q = cp.Parameter((n, 1), name='q')
        A = cp.Parameter((m, n), name='A')
        b = cp.Parameter((m, 1), name='b')

        x = cp.Variable((n, 1), name='x')

        objective = 0.5 * cp.sum_squares(P_sqrt @ x) + q.T @ x
        constraints = [A@x == b, cp.norm(x) <= 1]
        prob = cp.Problem(cp.Minimize(objective), constraints)

        prob_jax = GpuCvxpyLayer(prob, [P_sqrt, q, A, b], [x])

        key, k1, k2, k3, k4 = random.split(key, num=5)
        P_sqrt_jax = random.normal(k1, shape=(batch_size, n, n))
        q_jax = random.normal(k2, shape=(batch_size, n, 1))
        A_jax = random.normal(k3, shape=(batch_size, m, n))
        b_jax = random.normal(k4, shape=(batch_size, m, 1))

        def f(*params):
            sol, = prob_jax(*params)
            return sum(sol)

        check_grads(f, (P_sqrt_jax, q_jax, A_jax, b_jax),
                    order=1, modes=['rev'])

    @unittest.skip
    def test_least_squares(self):
        key = random.PRNGKey(0)
        m, n = 100, 20

        A = cp.Parameter((m, n))
        b = cp.Parameter(m)
        x = cp.Variable(n)
        obj = cp.sum_squares(A@x - b) + cp.sum_squares(x)
        prob = cp.Problem(cp.Minimize(obj))
        prob_jax = GpuCvxpyLayer(prob, [A, b], [x])

        key, k1, k2 = random.split(key, num=3)
        A_jax = random.normal(k1, shape=(m, n))
        b_jax = random.normal(k2, shape=(m,))

        def lstsq_sum_cp(A_jax, b_jax):
            x = prob_jax(A_jax, b_jax, solver_args={'eps': 1e-10})[0]
            return sum(x)

        def lstsq_sum_linalg(A_jax, b_jax):
            x = jnp.linalg.solve(
                A_jax.T @ A_jax + jnp.eye(n),
                A_jax.T @ b_jax)
            return sum(x)

        d_lstsq_sum_cp = jax.grad(lstsq_sum_cp, [0, 1])
        d_lstsq_sum_linalg = jax.grad(lstsq_sum_linalg, [0, 1])

        grad_A_cvxpy, grad_b_cvxpy = d_lstsq_sum_cp(A_jax, b_jax)
        grad_A_lstsq, grad_b_lstsq = d_lstsq_sum_linalg(A_jax, b_jax)

        self.assertAlmostEqual(
            jnp.linalg.norm(grad_A_cvxpy - grad_A_lstsq).item(),
            0.0,
            places=6)
        self.assertAlmostEqual(
            jnp.linalg.norm(grad_b_cvxpy - grad_b_lstsq).item(),
            0.0,
            places=6)

    @unittest.skip
    def test_logistic_regression(self):
        key = random.PRNGKey(0)

        N, n = 5, 2

        key, k1, k2, k3 = random.split(key, num=4)
        X_np = random.normal(k1, shape=(N, n))
        a_true = random.normal(k2, shape=(n, 1))
        y_np = jnp.round(sigmoid(
            X_np @ a_true + random.normal(k3, shape=(N, 1)) * 0.5))

        X_jax = jnp.array(X_np)
        lam_jax = 0.1 * jnp.ones(1)

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

        fit_logreg = GpuCvxpyLayer(prob, [X, lam], [a])

        check_grads(fit_logreg, (X_jax, lam_jax),
                    order=1, modes=['rev'])

    @unittest.skip
    def test_entropy_maximization(self):
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
        obj = cp.Maximize(cp.sum(cp.entr(x)) - .01 * cp.sum_squares(x))
        constraints = [A @ x == b,
                       F @ x <= g]
        prob = cp.Problem(obj, constraints)
        layer = GpuCvxpyLayer(prob, [A, b, F, g], [x])

        A_jax, b_jax, F_jax, g_jax = map(
            lambda x: jnp.array(x),
            [A_np, b_np, F_np, g_np])

        check_grads(layer, (A_jax, b_jax, F_jax, g_jax),
                    order=1, modes=['rev'])

    @unittest.skip
    def test_lml(self):
        k = 2
        x = cp.Parameter(4)
        y = cp.Variable(4)
        obj = -x @ y - cp.sum(cp.entr(y)) - cp.sum(cp.entr(1. - y))
        cons = [cp.sum(y) == k]
        prob = cp.Problem(cp.Minimize(obj), cons)
        lml = GpuCvxpyLayer(prob, [x], [y])

        x_th = jnp.array([1., -1., -1., -1.])
        check_grads(lml, (x_th,), order=1, modes=['rev'])

    @unittest.skip
    def test_sdp(self):
        key = random.PRNGKey(0)

        n = 3
        p = 3
        C = cp.Parameter((n, n))
        A = [cp.Parameter((n, n)) for _ in range(p)]
        b = [cp.Parameter((1, 1)) for _ in range(p)]

        key, k1 = random.split(key, num=2)
        C_jax = random.normal(k1, shape=(n, n))
        A_jax, b_jax = [], []
        for _ in range(p):
            key, k1, k2 = random.split(key, num=3)
            A_jax.append(random.normal(k1, shape=(n, n)))
            b_jax.append(random.normal(k2, shape=(1, 1)))

        X = cp.Variable((n, n), symmetric=True)
        constraints = [X >> 0]
        constraints += [
            cp.trace(A[i]@X) == b[i] for i in range(p)
        ]
        prob = cp.Problem(cp.Minimize(cp.trace(C@X) + cp.sum_squares(X)),
                          constraints)
        layer = GpuCvxpyLayer(prob, [C] + A + b, [X])

        check_grads(layer, [C_jax] + A_jax + b_jax,
                    order=1, modes=['rev'])

    def test_not_enough_parameters(self):
        x = cp.Variable(1)
        lam = cp.Parameter(1, nonneg=True)
        lam2 = cp.Parameter(1, nonneg=True)
        objective = lam * cp.norm(x, 1) + lam2 * cp.sum_squares(x)
        prob = cp.Problem(cp.Minimize(objective))
        with self.assertRaises(ValueError):
            layer = GpuCvxpyLayer(prob, [lam], [x])  # noqa: F841

    def test_not_enough_parameters_at_call_time(self):
        x = cp.Variable(1)
        lam = cp.Parameter(1, nonneg=True)
        lam2 = cp.Parameter(1, nonneg=True)
        objective = lam * cp.norm(x, 1) + lam2 * cp.sum_squares(x)
        prob = cp.Problem(cp.Minimize(objective))
        layer = GpuCvxpyLayer(prob, [lam, lam2], [x])  # noqa: F841
        with self.assertRaisesRegex(
                ValueError,
                'An array must be provided for each CVXPY parameter.*'):
            lam_jax = jnp.ones(1)
            layer(lam_jax)

    def test_too_many_variables(self):
        x = cp.Variable(1)
        y = cp.Variable(1)
        lam = cp.Parameter(1, nonneg=True)
        objective = lam * cp.norm(x, 1)
        prob = cp.Problem(cp.Minimize(objective))
        with self.assertRaises(ValueError):
            layer = GpuCvxpyLayer(prob, [lam], [x, y])  # noqa: F841

    def test_infeasible(self):
        x = cp.Variable(1)
        param = cp.Parameter(1)
        prob = cp.Problem(cp.Minimize(param), [x >= 1, x <= -1])
        layer = GpuCvxpyLayer(prob, [param], [x])
        param_jax = jnp.ones(1)
        assert layer(param_jax) is None, layer(param_jax)

    def test_unbounded(self):
        x = cp.Variable(1)
        param = cp.Parameter(1)
        prob = cp.Problem(cp.Minimize(x), [x <= param])
        layer = GpuCvxpyLayer(prob, [param], [x])
        param_jax = jnp.ones(1)
        assert layer(param_jax) is None

    def test_incorrect_parameter_shape(self):
        key = random.PRNGKey(0)
        m, n = 100, 20

        A = cp.Parameter((m, n))
        b = cp.Parameter(m)
        x = cp.Variable(n)
        obj = cp.sum_squares(A@x - b) + cp.sum_squares(x)
        prob = cp.Problem(cp.Minimize(obj))
        prob_th = GpuCvxpyLayer(prob, [A, b], [x])

        key, k1, k2 = random.split(key, num=3)
        A_th = random.normal(k1, shape=(32, m, n))
        b_th = random.normal(k2, shape=(20, m))

        with self.assertRaises(ValueError):
            prob_th(A_th, b_th)

        key, k1, k2 = random.split(key, num=3)
        A_th = random.normal(k1, shape=(32, m, n))
        b_th = random.normal(k2, shape=(32, 2 * m))

        with self.assertRaises(ValueError):
            prob_th(A_th, b_th)

        key, k1, k2 = random.split(key, num=3)
        A_th = random.normal(k1, shape=(m, n))
        b_th = random.normal(k2, shape=(2 * m, ))

        with self.assertRaises(ValueError):
            prob_th(A_th, b_th)

        key, k1, k2 = random.split(key, num=3)
        A_th = random.normal(k1, shape=(32, m, n))
        b_th = random.normal(k2, shape=(32, 32, m))

        with self.assertRaises(ValueError):
            prob_th(A_th, b_th)

    def test_broadcasting(self):
        key = random.PRNGKey(0)
        m, n = 100, 20

        A = cp.Parameter((m, n))
        b = cp.Parameter(m)
        x = cp.Variable(n)
        obj = cp.sum_squares(A@x - b) + cp.sum_squares(x)
        prob = cp.Problem(cp.Minimize(obj))
        prob_jax = GpuCvxpyLayer(prob, [A, b], [x])

        key, k1, k2 = random.split(key, num=3)
        A_jax = random.normal(k1, shape=(m, n))
        b_jax_0 = random.normal(k2, shape=(m,))
        b_jax = jnp.stack((b_jax_0, b_jax_0))

        def lstsq_sum_cp(A_jax, b_jax):
            x = prob_jax(A_jax, b_jax, solver_args={'eps': 1e-10})[0]
            return jnp.sum(x)

        def lstsq_sum_linalg(A_jax, b_jax):
            x = jnp.linalg.solve(
                A_jax.T @ A_jax + jnp.eye(n),
                A_jax.T @ b_jax)
            return sum(x)

        d_lstsq_sum_cp = jax.grad(lstsq_sum_cp, [0, 1])
        d_lstsq_sum_linalg = jax.grad(lstsq_sum_linalg, [0, 1])

        grad_A_lstsq, grad_b_lstsq = d_lstsq_sum_linalg(A_jax, b_jax_0)
        grad_A_cvxpy, grad_b_cvxpy = d_lstsq_sum_cp(A_jax, b_jax)

        self.assertAlmostEqual(
            jnp.linalg.norm(grad_A_cvxpy / 2. - grad_A_lstsq).item(),
            0.0,
            places=6)
        self.assertAlmostEqual(
            jnp.linalg.norm(grad_b_cvxpy[0] - grad_b_lstsq).item(),
            0.0,
            places=6)

    def test_shared_parameter(self):
        key = random.PRNGKey(0)
        m, n = 10, 5

        A = cp.Parameter((m, n))
        x = cp.Variable(n)
        key, k1, k2 = random.split(key, num=3)
        b1 = random.normal(k1, shape=(m,))
        b2 = random.normal(k2, shape=(m,))
        prob1 = cp.Problem(cp.Minimize(cp.sum_squares(A @ x - b1)))
        layer1 = GpuCvxpyLayer(prob1, parameters=[A], variables=[x])
        prob2 = cp.Problem(cp.Minimize(cp.sum_squares(A @ x - b2)))
        layer2 = GpuCvxpyLayer(prob2, parameters=[A], variables=[x])

        key, k1 = random.split(key, num=2)
        A_jax = random.normal(k1, shape=(m, n))
        solver_args = {
            "eps": 1e-10,
            "acceleration_lookback": 0,
            "max_iters": 10000
        }

        def f(A_jax):
            x1, = layer1(A_jax, solver_args=solver_args)
            x2, = layer2(A_jax, solver_args=solver_args)
            return jnp.concatenate((x1, x2))

        check_grads(f, [A_jax], order=1, modes=['rev'])

    def test_equality(self):
        key = random.PRNGKey(0)
        n = 10
        A = np.eye(n)
        x = cp.Variable(n)
        b = cp.Parameter(n)
        prob = cp.Problem(cp.Minimize(cp.sum_squares(x)), [A@x == b])
        layer = GpuCvxpyLayer(prob, parameters=[b], variables=[x])

        key, k1 = random.split(key, num=2)
        b_jax = random.normal(k1, shape=(n,))

        check_grads(layer, [b_jax], order=1, modes=['rev'])

    
    @unittest.skip
    def test_basic_gp(self):
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

        layer = GpuCvxpyLayer(
            problem, parameters=[a, b, c], variables=[x, y, z], gp=True)
        a_jax = jnp.array(2.0)
        b_jax = jnp.array(1.0)
        c_jax = jnp.array(0.5)
        x_jax, y_jax, z_jax = layer(a_jax, b_jax, c_jax)

        self.assertAlmostEqual(x.value, x_jax, places=5)
        self.assertAlmostEqual(y.value, y_jax, places=5)
        self.assertAlmostEqual(z.value, z_jax, places=5)

        check_grads(
            lambda a, b, c: jnp.sum(layer(
                a, b, c, solver_args={"acceleration_lookback": 0},
            )[0]),
            [a_jax, b_jax, c_jax],
            order=1, modes=['rev']
        )

if __name__ == '__main__':
    unittest.main()
