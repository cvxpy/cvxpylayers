"""Moreau with cones CVXPY places directly on variables (direct cones).

CVXPY's MOREAU interface moves constraints such as ``x >= 0``, ``nonneg=True``,
``PSD=True`` and SOC/exp/power cones on variables out of the rows of ``A`` into
``dir_cones``. These must be forwarded to Moreau, and ``psd_triangle`` direct
cones additionally need the sqrt(2) off-diagonal scaling (issue #252).
"""

import cvxpy as cp
import numpy as np
import pytest

torch = pytest.importorskip("torch")
moreau = pytest.importorskip("moreau")

from cvxpylayers.torch import CvxpyLayer  # noqa: E402

torch.set_default_dtype(torch.double)

ATOL = 1e-4
# Finite differences vs. implicit derivatives near curved (exp/power) cone boundaries
GRAD_ATOL = 5e-3


def _nonneg_constraint():
    p, x = cp.Parameter(3), cp.Variable(3)
    prob = cp.Problem(cp.Minimize(cp.sum_squares(x - p)), [x >= 0])
    return prob, [p], [x], [np.array([-1.0, 0.5, 2.0])]


def _nonneg_attr():
    p, x = cp.Parameter(3), cp.Variable(3, nonneg=True)
    prob = cp.Problem(cp.Minimize(cp.sum_squares(x - p)))
    return prob, [p], [x], [np.array([-1.0, 0.5, 2.0])]


def _soc():
    p, x, t = cp.Parameter(3), cp.Variable(3), cp.Variable()
    prob = cp.Problem(cp.Minimize(cp.sum_squares(x - p) + 2 * t), [cp.SOC(t, x)])
    return prob, [p], [x, t], [np.array([1.0, -2.0, 0.5])]


def _exp():
    p, x = cp.Parameter(3), cp.Variable(3)
    prob = cp.Problem(cp.Minimize(cp.sum_squares(x - p)), [cp.ExpCone(x[0], x[1], x[2])])
    return prob, [p], [x], [np.array([2.0, 1.0, 0.5])]


def _power():
    p, x = cp.Parameter(3), cp.Variable(3)
    prob = cp.Problem(
        cp.Minimize(cp.sum_squares(x - p)), [cp.PowCone3D(x[0], x[1], x[2], 0.3)]
    )
    return prob, [p], [x], [np.array([1.0, 2.0, 3.0])]


def _psd_quadratic():
    P, X = cp.Parameter((3, 3)), cp.Variable((3, 3), PSD=True)
    prob = cp.Problem(cp.Minimize(cp.sum_squares(X - P)))
    P_val = np.array([[1.0, 2.0, 0.0], [2.0, -1.0, 0.5], [0.0, 0.5, 0.3]])
    return prob, [P], [X], [P_val]


def _psd_param_A():
    # Parameter enters A, so the non-constant P/A path is used
    C, X = cp.Parameter((2, 2)), cp.Variable((2, 2), PSD=True)
    prob = cp.Problem(
        cp.Minimize(cp.sum_squares(X) + X[0, 1]),
        [cp.trace(C @ X) == 1],
    )
    return prob, [C], [X], [np.array([[1.0, 0.3], [0.3, 2.0]])]


PROBLEMS = {
    "nonneg_constraint": _nonneg_constraint,
    "nonneg_attr": _nonneg_attr,
    "soc": _soc,
    "exp": _exp,
    "power": _power,
    "psd_quadratic": _psd_quadratic,
    "psd_param_A": _psd_param_A,
}


def _expected(prob, params, variables, values):
    for param, val in zip(params, values):
        param.value = val
    # Tight tolerances so finite differences of the solution are accurate
    prob.solve(
        solver=cp.CLARABEL, tol_gap_abs=1e-10, tol_gap_rel=1e-10, tol_feas=1e-10
    )
    assert prob.status == cp.OPTIMAL
    return [v.value for v in variables]


def _finite_difference_grads(prob, params, variables, values, eps=1e-4):
    """Central differences of sum_i (i + 1) * sum(variables[i]), solved with CLARABEL."""

    def loss(vals):
        sols = _expected(prob, params, variables, vals)
        return sum((i + 1) * np.sum(s) for i, s in enumerate(sols))

    grads = []
    for k, v in enumerate(values):
        g = np.zeros_like(v)
        for idx in np.ndindex(v.shape):
            vals_p = [w.copy() for w in values]
            vals_m = [w.copy() for w in values]
            vals_p[k][idx] += eps
            vals_m[k][idx] -= eps
            g[idx] = (loss(vals_p) - loss(vals_m)) / (2 * eps)
        grads.append(g)
    return grads


@pytest.mark.parametrize("name", PROBLEMS)
def test_problem_has_direct_cones(name):
    prob, *_ = PROBLEMS[name]()
    assert prob.get_problem_data("MOREAU")[0]["dir_cones"]


@pytest.mark.parametrize("name", PROBLEMS)
def test_torch_direct_cones(name):
    prob, params, variables, values = PROBLEMS[name]()
    expected = _expected(prob, params, variables, values)

    layer = CvxpyLayer(prob, params, variables, solver="MOREAU", solver_args={"device": "cpu"})
    sols = layer(*[torch.tensor(v) for v in values])

    for sol, exp in zip(sols, expected):
        np.testing.assert_allclose(sol.detach().numpy(), exp, atol=ATOL)


@pytest.mark.parametrize("name", PROBLEMS)
def test_torch_direct_cones_batched(name):
    prob, params, variables, values = PROBLEMS[name]()
    batch = [np.stack([v, 0.5 * v]) for v in values]

    layer = CvxpyLayer(prob, params, variables, solver="MOREAU", solver_args={"device": "cpu"})
    sols = layer(*[torch.tensor(v) for v in batch])

    for i in range(2):
        expected = _expected(prob, params, variables, [v[i] for v in batch])
        for sol, exp in zip(sols, expected):
            np.testing.assert_allclose(sol[i].detach().numpy(), exp, atol=ATOL)


@pytest.mark.parametrize("name", PROBLEMS)
def test_torch_direct_cones_gradients(name):
    prob, params, variables, values = PROBLEMS[name]()

    layer = CvxpyLayer(prob, params, variables, solver="MOREAU", solver_args={"device": "cpu"})
    tensors = [torch.tensor(v, requires_grad=True) for v in values]
    sols = layer(*tensors)
    sum(((i + 1) * s).sum() for i, s in enumerate(sols)).backward()

    expected = _finite_difference_grads(prob, params, variables, values)
    for t, g in zip(tensors, expected):
        np.testing.assert_allclose(t.grad.numpy(), g, atol=GRAD_ATOL)


def test_torch_direct_cone_duals():
    """Duals of constraints moved into direct cones are recovered."""
    c, x = cp.Parameter(2), cp.Variable(2)
    X = cp.Variable((2, 2), symmetric=True)
    nonneg, psd = x >= 0, X >> 0
    prob = cp.Problem(
        cp.Minimize(c @ x + cp.sum_squares(x) + cp.sum_squares(X) + cp.trace(X)),
        [nonneg, psd, X[0, 1] == 1],
    )
    duals = [nonneg.dual_variables[0], psd.dual_variables[0]]
    assert len(prob.get_problem_data("MOREAU")[0]["dir_cones"]) == 2

    c_val = np.array([1.0, -1.0])
    layer = CvxpyLayer(
        prob, [c], [x, X, *duals], solver="MOREAU", solver_args={"device": "cpu"}
    )
    sols = layer(torch.tensor(c_val))

    c.value = c_val
    prob.solve(solver=cp.CLARABEL)
    expected = [x.value, X.value, nonneg.dual_value, psd.dual_value]
    for sol, exp in zip(sols, expected):
        np.testing.assert_allclose(sol.detach().numpy(), exp, atol=ATOL)


class TestJax:
    @pytest.fixture(autouse=True)
    def _jax(self):
        jax = pytest.importorskip("jax")
        prev = jax.config.jax_enable_x64
        jax.config.update("jax_enable_x64", True)
        yield
        jax.config.update("jax_enable_x64", prev)

    def _layer(self, prob, params, variables, **kwargs):
        from cvxpylayers.jax import CvxpyLayer as JaxCvxpyLayer

        return JaxCvxpyLayer(prob, params, variables, **kwargs)

    def test_issue_252(self):
        import jax.numpy as jnp

        p, x = cp.Parameter(), cp.Variable()
        prob = cp.Problem(cp.Minimize(cp.square(x - p)), [x >= 0])
        layer = self._layer(prob, [p], [x], solver="MOREAU", solver_args={"device": "cpu"})
        (sol,) = layer(jnp.array(-1.0))
        assert abs(float(sol)) < ATOL

    @pytest.mark.parametrize("name", PROBLEMS)
    def test_jax_direct_cones(self, name):
        import jax.numpy as jnp

        prob, params, variables, values = PROBLEMS[name]()
        expected = _expected(prob, params, variables, values)
        layer = self._layer(
            prob, params, variables, solver="MOREAU", solver_args={"device": "cpu"}
        )
        sols = layer(*[jnp.array(v) for v in values])
        for sol, exp in zip(sols, expected):
            np.testing.assert_allclose(np.asarray(sol), exp, atol=ATOL)

        # Batched, and warm-started from the (scaled) cached solution
        batch = [jnp.array(np.stack([v, 0.5 * v])) for v in values]
        layer(*batch)
        sols = layer(*batch, warm_start=True)
        for i in range(2):
            expected = _expected(prob, params, variables, [np.asarray(v[i]) for v in batch])
            for sol, exp in zip(sols, expected):
                np.testing.assert_allclose(np.asarray(sol[i]), exp, atol=ATOL)

    @pytest.mark.parametrize("name", PROBLEMS)
    def test_jax_direct_cones_gradients(self, name):
        import jax
        import jax.numpy as jnp

        prob, params, variables, values = PROBLEMS[name]()
        layer = self._layer(
            prob, params, variables, solver="MOREAU", solver_args={"device": "cpu"}
        )

        def loss(*ps):
            sols = layer(*ps)
            return sum(((i + 1) * s).sum() for i, s in enumerate(sols))

        grads_moreau = jax.grad(loss, argnums=tuple(range(len(values))))(
            *[jnp.array(v) for v in values]
        )

        expected = _finite_difference_grads(prob, params, variables, values)
        for g_moreau, g in zip(grads_moreau, expected):
            np.testing.assert_allclose(np.asarray(g_moreau), g, atol=GRAD_ATOL)
