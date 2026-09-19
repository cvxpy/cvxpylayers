"""
Copyright, the CVXPY authors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import cvxpy as cp
import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
moreau = pytest.importorskip("moreau")

from cvxpylayers.mlx import CvxpyLayer


@pytest.fixture(autouse=True)
def cpu_stream():
    with mx.stream(mx.cpu):
        yield


@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("dtype", [mx.float16, mx.float32])
def test_qp_primal_dual_gradients(batched, dtype):
    """Check P/q/b gradients, broadcasting, objective offsets, and dtypes."""
    x = cp.Variable(2)
    p = cp.Parameter(nonneg=True)
    q = cp.Parameter(2)
    b = cp.Parameter(2)
    offset = cp.Parameter()
    constraint = x >= b
    problem = cp.Problem(cp.Minimize(0.5 * p * cp.sum_squares(x) + q @ x + offset), [constraint])
    layer = CvxpyLayer(
        problem, [q, b, p, offset], [x, constraint.dual_variables[0]], solver="MOREAU"
    )
    q_np = np.array([[-4.0, 3.0], [-6.0, 5.0]]) if batched else np.array([-4.0, 3.0])
    args = [mx.array(a, dtype=dtype) for a in (q_np, [0.5, 1.0], 2.0, 7.0)]
    primal, dual = layer(*args)
    assert primal.dtype == dual.dtype == dtype
    np.testing.assert_allclose(np.array(primal), np.maximum(-q_np / 2, [0.5, 1]), atol=2e-3)
    np.testing.assert_allclose(np.array(dual), np.maximum(q_np + [1, 2], 0), atol=2e-3)

    def loss(*params):
        primal, dual = layer(*params)
        return mx.sum(primal) + 2 * mx.sum(dual)

    value, (dq, db, dp, doffset) = mx.value_and_grad(loss, argnums=(0, 1, 2, 3))(*args)
    np.testing.assert_allclose(
        np.array(value), np.sum(np.array(primal) + 2 * np.array(dual)), atol=2e-3
    )
    count = 2 if batched else 1
    np.testing.assert_allclose(np.array(dq), np.broadcast_to([-0.5, 2], q_np.shape), atol=2e-3)
    np.testing.assert_allclose(np.array(db), [0, 5 * count], atol=2e-3)
    np.testing.assert_allclose(np.array(dp), np.sum(q_np[..., 0] / 4 + 2), atol=2e-3)
    np.testing.assert_array_equal(np.array(doffset), 0)
    assert all(g.dtype == dtype for g in (dq, db, dp, doffset))


def test_matrix_parameter_gradients():
    """Exercise off-diagonal P, A's sign/CSR order, and a sparse RHS."""
    x = cp.Variable(3)
    P = cp.Parameter((3, 3), PSD=True)
    q = cp.Parameter(3)
    A = cp.Parameter((2, 3))
    b = cp.Parameter()
    constraint = A @ x == cp.hstack([0.0, b])
    problem = cp.Problem(cp.Minimize(0.5 * cp.quad_form(x, P) + q @ x), [constraint])
    layer = CvxpyLayer(problem, [P, q, A, b], [x, constraint.dual_variables[0]], solver="MOREAU")
    values = [
        np.array([[3.0, 0.2, 0.4], [0.2, 2.0, 0.1], [0.4, 0.1, 4.0]]),
        np.array([-1.0, 2.0, -3.0]),
        np.array([[1.0, 2.0, 0.0], [0.0, 1.0, 3.0]]),
        np.array(1.5),
    ]
    args = [mx.array(a, dtype=mx.float32) for a in values]

    def reference(P, q, A, b):
        kkt = np.block([[P, A.T], [A, np.zeros((2, 2))]])
        return np.linalg.solve(kkt, np.concatenate((-q, [0.0, float(b)])))

    primal, dual = layer(*args)
    np.testing.assert_allclose(
        np.concatenate((np.array(primal), np.array(dual))), reference(*values), atol=2e-5
    )

    def loss(*params):
        primal, dual = layer(*params)
        return mx.sum(primal) + 0.3 * mx.sum(dual)

    grads = mx.grad(loss, argnums=(0, 1, 2, 3))(*args)
    rng = np.random.default_rng(19)
    for i, grad in enumerate(grads):
        direction = rng.normal(size=values[i].shape)
        if i == 0:
            direction = (direction + direction.T) / 2
        plus, minus = list(values), list(values)
        plus[i] = values[i] + 1e-4 * direction
        minus[i] = values[i] - 1e-4 * direction
        weights = np.array([1.0, 1.0, 1.0, 0.3, 0.3])
        expected = weights @ (reference(*plus) - reference(*minus)) / 2e-4
        np.testing.assert_allclose(
            np.sum(np.array(grad) * direction), expected, atol=2e-4, rtol=2e-4
        )


def test_lp_without_quadratic_term():
    x = cp.Variable(2)
    b = cp.Parameter(2)
    problem = cp.Problem(cp.Minimize(cp.sum(x)), [x >= b])
    layer = CvxpyLayer(problem, [b], [x], solver="MOREAU")
    value = mx.array([1.0, 2.0])
    np.testing.assert_allclose(np.array(layer(value)[0]), [1, 2], atol=1e-5)
    np.testing.assert_allclose(
        np.array(mx.grad(lambda b: mx.sum(layer(b)[0]))(value)), [1, 1], atol=1e-5
    )


def test_direct_lp_without_matrices():
    x = cp.Variable(2)
    q = cp.Parameter(2)
    constraint = x >= 0
    problem = cp.Problem(cp.Minimize(q @ x), [constraint])
    layer = CvxpyLayer(problem, [q], [x, constraint.dual_variables[0]], solver="MOREAU")
    value = mx.array([[1.0, 2.0]])
    primal, dual = layer(value)
    assert primal.shape == dual.shape == (1, 2)
    np.testing.assert_allclose(np.array(primal), 0, atol=1e-5)
    np.testing.assert_allclose(np.array(dual), [[1, 2]], atol=1e-5)
    grad = mx.grad(lambda q: mx.sum(layer(q)[1]))(value)
    np.testing.assert_allclose(np.array(grad), [[1, 1]], atol=1e-5)


def test_soc_primal_dual_gradients():
    x = cp.Variable(2)
    t = cp.Parameter(nonneg=True)
    constraint = cp.norm(x) <= t
    problem = cp.Problem(cp.Minimize(-cp.sum(x)), [constraint])
    layer = CvxpyLayer(problem, [t], [x, constraint.dual_variables[0]], solver="MOREAU")
    value = mx.array(2.0)
    primal, dual = layer(value)
    np.testing.assert_allclose(np.array(primal), np.sqrt(2), atol=1e-5)
    np.testing.assert_allclose(np.array(dual), np.sqrt(2), atol=1e-5)
    grad = mx.grad(lambda t: mx.sum(layer(t)[0]))(value)
    np.testing.assert_allclose(np.array(grad), np.sqrt(2), atol=1e-4)


def test_multiple_calls_preserve_backward_state():
    """An intervening solve (including another batch size) must not change the VJP."""
    x = cp.Variable(1)
    q = cp.Parameter(1)
    problem = cp.Problem(cp.Minimize(0.5 * cp.sum_squares(x) + q @ x), [x >= 0.1])
    layer = CvxpyLayer(problem, [q], [x], solver="MOREAU")

    def loss(q):
        first = layer(q)[0]
        other = layer(mx.stack((-q, -2 * q)))[0]
        return mx.sum(first) + mx.sum(other)

    np.testing.assert_allclose(np.array(mx.grad(loss)(mx.array([-2.0]))), [-1.0], atol=1e-5)
    np.testing.assert_allclose(np.array(mx.grad(loss)(mx.array([2.0]))), [3.0], atol=1e-5)


def test_cpu_settings_and_overrides(monkeypatch):
    x = cp.Variable(1)
    b = cp.Parameter(1)
    problem = cp.Problem(cp.Minimize(cp.sum_squares(x)), [x >= b])
    defaults = {"max_iter": 50, "ipm_settings": {"direct_solve_method": "qdldl"}}
    layer = CvxpyLayer(problem, [b], [x], solver="MOREAU", solver_args=defaults)
    settings_seen = []
    compiled_solver = moreau.CompiledSolver

    def record_settings(*args, **kwargs):
        settings = kwargs["settings"]
        settings_seen.append(
            (
                settings.device,
                settings.enable_grad,
                settings.batch_size,
                settings.max_iter,
                settings.ipm_settings.direct_solve_method,
            )
        )
        return compiled_solver(*args, **kwargs)

    monkeypatch.setattr(moreau, "CompiledSolver", record_settings)
    overrides = {"max_iter": 80, "device": "auto", "enable_grad": False, "batch_size": 9}
    layer(mx.array([[1.0], [2.0]]), solver_args=overrides)
    layer(mx.array([1.0]))
    assert settings_seen == [("cpu", True, 2, 80, "qdldl"), ("cpu", True, 1, 50, "qdldl")]
    assert defaults == {"max_iter": 50, "ipm_settings": {"direct_solve_method": "qdldl"}}
    assert overrides == {"max_iter": 80, "device": "auto", "enable_grad": False, "batch_size": 9}
    with pytest.raises(ValueError, match="only supports device='cpu'"):
        layer(mx.array([1.0]), solver_args={"device": "cuda"})
    with pytest.raises(ValueError, match="requires solver='ipm'"):
        layer(mx.array([1.0]), solver_args={"solver": "active_set"})


def test_infeasible_batch_raises():
    x = cp.Variable(1)
    b = cp.Parameter(1)
    problem = cp.Problem(cp.Minimize(cp.sum_squares(x)), [x >= b, x <= 1])
    layer = CvxpyLayer(problem, [b], [x], solver="MOREAU")
    with pytest.raises(RuntimeError, match="batch element 1.*PrimalInfeasible"):
        layer(mx.array([[0.5], [2.0]]))


def test_direct_nonnegative_cone_dual_gradients():
    x = cp.Variable(2)
    q = cp.Parameter(2)
    constraint = x >= 0
    problem = cp.Problem(cp.Minimize(0.5 * cp.sum_squares(x) + q @ x), [constraint])
    layer = CvxpyLayer(problem, [q], [x, constraint.dual_variables[0]], solver="MOREAU")
    value = mx.array([-2.0, 3.0])
    primal, dual = layer(value)
    np.testing.assert_allclose(np.array(primal), [2, 0], atol=1e-5)
    np.testing.assert_allclose(np.array(dual), [0, 3], atol=1e-5)

    def loss(q):
        primal, dual = layer(q)
        return mx.sum(primal) + 2 * mx.sum(dual)

    np.testing.assert_allclose(np.array(mx.grad(loss)(value)), [-1, 2], atol=1e-4)


def test_psd_primal_dual_gradients():
    X = cp.Variable((3, 3), symmetric=True)
    Q = cp.Parameter((3, 3), symmetric=True)
    constraint = X >> 0
    problem = cp.Problem(
        cp.Minimize(0.5 * cp.sum_squares(X) - cp.sum(cp.multiply(Q, X))), [constraint]
    )
    layer = CvxpyLayer(problem, [Q], [X, constraint.dual_variables[0]], solver="MOREAU")
    Q_np = np.array([[1.0, 0.4, 0.2], [0.4, -2.0, 0.3], [0.2, 0.3, 3.0]])
    value = mx.array(Q_np, dtype=mx.float32)

    def reference(Q):
        eig, vec = np.linalg.eigh(Q)
        primal = (vec * np.maximum(eig, 0)) @ vec.T
        return primal, primal - Q

    primal, dual = layer(value)
    expected_primal, expected_dual = reference(Q_np)
    np.testing.assert_allclose(np.array(primal), expected_primal, atol=2e-4)
    np.testing.assert_allclose(np.array(dual), expected_dual, atol=2e-4)

    def loss(Q):
        primal, dual = layer(Q)
        return mx.sum(primal) + 0.3 * mx.sum(dual)

    direction = np.array([[0.2, 0.1, -0.3], [0.1, 0.4, 0.2], [-0.3, 0.2, -0.1]])
    plus = reference(Q_np + 1e-4 * direction)
    minus = reference(Q_np - 1e-4 * direction)
    expected = (np.sum(plus[0] - minus[0]) + 0.3 * np.sum(plus[1] - minus[1])) / 2e-4
    grad = mx.grad(loss)(value)
    np.testing.assert_allclose(np.sum(np.array(grad) * direction), expected, atol=2e-4)


def test_geometric_program_gradients():
    x = cp.Variable(pos=True)
    b = cp.Parameter(pos=True, value=2.0)
    problem = cp.Problem(cp.Minimize(x), [x >= b])
    layer = CvxpyLayer(problem, [b], [x], solver="MOREAU", gp=True)
    value = mx.array(2.0)
    np.testing.assert_allclose(np.array(layer(value)[0]), 2, atol=1e-5)
    np.testing.assert_allclose(np.array(mx.grad(lambda b: layer(b)[0])(value)), 1, atol=1e-5)


def test_repeated_layer_construction():
    """A second layer must not permute CVXPY's already-cached data again."""
    x = cp.Variable(2)
    A = cp.Parameter((2, 2))
    b = cp.Parameter(2)
    problem = cp.Problem(cp.Minimize(cp.sum_squares(x)), [A @ x == b])
    first = CvxpyLayer(problem, [A, b], [x], solver="MOREAU")
    second = CvxpyLayer(problem, [A, b], [x], solver="MOREAU")
    args = (mx.array([[1.0, 2.0], [3.0, 5.0]]), mx.array([2.0, 1.0]))
    expected = np.linalg.solve(np.array(args[0]), np.array(args[1]))
    for layer in (first, second):
        np.testing.assert_allclose(np.array(layer(*args)[0]), expected, atol=1e-4)


@pytest.mark.parametrize("kind", ["soc", "exp", "power", "gen_power"])
def test_vectorized_cone_dual_ordering(kind):
    """Recover each dual argument from interleaved vectorized cone blocks."""
    n = 8 if kind == "gen_power" else 6
    x = cp.Variable(n)
    q = cp.Parameter(n)
    if kind == "soc":
        constraint = cp.SOC(x[:2], cp.reshape(x[2:], (2, 2), order="F"))
    elif kind == "exp":
        constraint = cp.ExpCone(x[:2], x[2:4], x[4:])
    elif kind == "power":
        constraint = cp.PowCone3D(x[:2], x[2:4], x[4:], [0.3, 0.65])
    else:
        constraint = cp.PowConeND(
            cp.reshape(x[:6], (3, 2), order="F"),
            x[6:],
            np.array([[0.2, 0.3], [0.3, 0.3], [0.5, 0.4]]),
        )
    problem = cp.Problem(cp.Minimize(0.5 * cp.sum_squares(x) + q @ x), [constraint])
    q_np = np.linspace(-1.7, 0.6, n)
    q.value = q_np
    variables = [x, *constraint.dual_variables]
    options = {"tol_gap_abs": 1e-10, "tol_gap_rel": 1e-10, "tol_feas": 1e-10}
    problem.solve(solver=cp.CLARABEL, **options)
    expected = [v.value.copy() for v in variables]
    layer = CvxpyLayer(
        problem, [q], variables, solver="MOREAU", solver_args={"ipm_settings": options}
    )
    value = mx.array(q_np, dtype=mx.float32)
    for actual, reference in zip(layer(value), expected):
        np.testing.assert_allclose(np.array(actual), reference, atol=2e-4)

    def loss(q):
        return sum((i + 1) * mx.sum(v) for i, v in enumerate(layer(q)))

    direction = np.linspace(0.5, -0.2, n)
    # A larger step avoids amplifying the reference conic solver's dual error.
    step = 1e-2
    references = []
    for sign in (1, -1):
        q.value = q_np + sign * step * direction
        problem.solve(solver=cp.CLARABEL, **options)
        references.append(sum((i + 1) * np.sum(v.value) for i, v in enumerate(variables)))
    grad = mx.grad(loss)(value)
    np.testing.assert_allclose(
        np.array(grad) @ direction, (references[0] - references[1]) / (2 * step), atol=5e-4
    )
