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

import importlib.util

import cvxpy as cp
import numpy as np
import pytest

torch = pytest.importorskip("torch")
moreau = pytest.importorskip("moreau")

from cvxpylayers.torch import CvxpyLayer

HAS_DIRECT_CONES = (
    importlib.util.find_spec("cvxpy.reductions.cone2cone.extract_direct_cones") is not None
)
requires_direct_cones = pytest.mark.skipif(
    not HAS_DIRECT_CONES, reason="Requires CVXPY's direct-cone reduction"
)


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and not (
        torch.cuda.is_available() and moreau.device_available("cuda")
    ):
        pytest.skip("CUDA not available")
    return request.param


@pytest.mark.parametrize("framework", ["torch", "jax"])
@pytest.mark.parametrize("batched", [False, True])
def test_default_backend_primal_dual_gradients(framework, batched, device):
    x = cp.Variable(3)
    q = cp.Parameter(3)
    con = x >= 0
    problem = cp.Problem(cp.Minimize(cp.quad_form(x, np.diag([1.0, 1.5, 2.0])) + q @ x), [con])
    values = np.array([-3.0, 2.0, -1.0])
    expected = [np.array([1.5, 0.0, 0.25]), np.array([0.0, 2.0, 0.0])]
    derivative = np.array([-0.5, 1.0, -0.25])
    if batched:
        values = np.stack((values, 2 * values))
        expected = [np.stack((v, 2 * v)) for v in expected]
        derivative = np.stack((derivative, derivative))

    if framework == "torch":
        layer = CvxpyLayer(problem, [q], [x, con.dual_variables[0]], solver_args={"device": device})
        p = torch.tensor(values, dtype=torch.double, device=device, requires_grad=True)
        actual = layer(p)
        sum(v.sum() for v in actual).backward()
        np.testing.assert_allclose(p.grad.cpu(), derivative, atol=2e-5)
        actual = [v.detach().cpu().numpy() for v in actual]
        warm = layer(p, warm_start=True)
        warm = [v.detach().cpu().numpy() for v in warm]
    else:
        jax = pytest.importorskip("jax")
        jnp = pytest.importorskip("jax.numpy")
        from cvxpylayers.jax import CvxpyLayer as JaxLayer

        layer = JaxLayer(problem, [q], [x, con.dual_variables[0]], solver_args={"device": device})
        p = jax.device_put(
            jnp.asarray(values), jax.devices("gpu" if device == "cuda" else "cpu")[0]
        )
        actual = jax.jit(layer)(p)
        grad = jax.jit(jax.grad(lambda p: sum(v.sum() for v in layer(p))))(p)
        np.testing.assert_allclose(grad, derivative, atol=2e-5)
        layer(p)  # Populate the eager warm-start cache.
        warm = layer(p, warm_start=True)

    assert layer.ctx.solver == "MOREAU"
    if HAS_DIRECT_CONES:
        assert layer.ctx.solver_ctx.A_shape[0] == 0
        assert layer.ctx.solver_ctx.cones.dir_cones[0].kind == "nonneg"
    for result, warmed, reference in zip(actual, warm, expected):
        np.testing.assert_allclose(result, reference, atol=2e-5)
        np.testing.assert_allclose(warmed, reference, atol=2e-5)


@requires_direct_cones
@pytest.mark.parametrize("framework", ["torch", "jax"])
def test_direct_linear_problem_without_slack_rows(framework, device):
    x = cp.Variable(2)
    q = cp.Parameter(2)
    con = x >= 0
    problem = cp.Problem(cp.Minimize(q @ x), [con])
    if framework == "torch":
        layer = CvxpyLayer(problem, [q], [x, con.dual_variables[0]], solver_args={"device": device})
        p = torch.tensor([1.0, 2.0], dtype=torch.double, device=device, requires_grad=True)
        primal, dual = layer(p)
        dual.sum().backward()
        grad = p.grad.cpu()
        primal, dual = primal.detach().cpu(), dual.detach().cpu()
    else:
        jax = pytest.importorskip("jax")
        jnp = pytest.importorskip("jax.numpy")
        from cvxpylayers.jax import CvxpyLayer as JaxLayer

        layer = JaxLayer(problem, [q], [x, con.dual_variables[0]], solver_args={"device": device})
        p = jax.device_put(
            jnp.array([1.0, 2.0]), jax.devices("gpu" if device == "cuda" else "cpu")[0]
        )
        primal, dual = jax.jit(layer)(p)
        grad = jax.jit(jax.grad(lambda p: layer(p)[1].sum()))(p)
    np.testing.assert_allclose(primal, [0.0, 0.0], atol=1e-7)
    np.testing.assert_allclose(dual, [1.0, 2.0], atol=1e-7)
    np.testing.assert_allclose(grad, [1.0, 1.0], atol=1e-7)


@requires_direct_cones
def test_direct_quadratic_parameter_gradient(device):
    x = cp.Variable(2)
    P = cp.Parameter((2, 2), PSD=True)
    q = cp.Parameter(2)
    problem = cp.Problem(cp.Minimize(0.5 * cp.quad_form(x, P) + q @ x), [x >= 0])
    layer = CvxpyLayer(problem, [P, q], [x], solver_args={"device": device})
    H = torch.tensor(
        [[2.0, 0.4], [0.4, 3.0]], dtype=torch.double, device=device, requires_grad=True
    )
    p = torch.tensor([-2.0, -1.0], dtype=torch.double, device=device, requires_grad=True)
    (actual,) = layer(H, p)
    symmetric = H.triu() + H.triu(1).T
    expected = torch.linalg.solve(symmetric, -p)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    actual_grad = torch.autograd.grad(actual.sum(), (H, p))
    expected_grad = torch.autograd.grad(expected.sum(), (H, p))
    for a, b in zip(actual_grad, expected_grad):
        torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)


@requires_direct_cones
@pytest.mark.parametrize("kind", ["soc", "exp", "power", "gen_power"])
def test_vectorized_direct_cones_and_dual_gradients(kind, device):
    n = 8 if kind == "gen_power" else 6
    x = cp.Variable(n)
    q = cp.Parameter(n)
    if kind == "soc":
        con = cp.SOC(x[:2], cp.reshape(x[2:], (2, 2), order="F"))
    elif kind == "exp":
        con = cp.ExpCone(x[:2], x[2:4], x[4:])
    elif kind == "power":
        con = cp.PowCone3D(x[:2], x[2:4], x[4:], [0.3, 0.65])
    else:
        con = cp.PowConeND(
            cp.reshape(x[:6], (3, 2), order="F"),
            x[6:],
            np.array([[0.2, 0.3], [0.3, 0.3], [0.5, 0.4]]),
        )
    problem = cp.Problem(cp.Minimize(0.5 * cp.sum_squares(x) + q @ x), [con])
    q.value = np.linspace(-1.7, 0.6, n)
    problem.solve(solver=cp.CLARABEL, tol_gap_abs=1e-10, tol_feas=1e-10, tol_gap_rel=1e-10)
    variables = [x, *con.dual_variables]
    expected = [v.value.copy() for v in variables]
    layer = CvxpyLayer(problem, [q], variables, solver_args={"device": device})
    assert [c.kind for c in layer.ctx.solver_ctx.cones.dir_cones] == [kind, kind]
    p = torch.tensor(q.value, dtype=torch.double, device=device, requires_grad=True)
    for actual, reference in zip(layer(p), expected):
        np.testing.assert_allclose(actual.detach().cpu(), reference, atol=2e-4)
    # CUDA reductions differ across backward calls at double-precision roundoff.
    assert torch.autograd.gradcheck(layer, (p,), eps=1e-4, atol=5e-4, rtol=5e-3, nondet_tol=1e-12)


@requires_direct_cones
@pytest.mark.parametrize("framework", ["torch", "jax"])
@pytest.mark.parametrize("direct", [True, False])
def test_psd_scaling_and_duals(framework, direct, device):
    x = cp.Variable((3, 3), symmetric=True)
    q = cp.Parameter((3, 3))
    con = x >> 0 if direct else x + np.eye(3) >> 0
    problem = cp.Problem(cp.Minimize(0.5 * cp.sum_squares(x) + cp.sum(cp.multiply(q, x))), [con])
    rotation, _ = np.linalg.qr(np.array([[1.0, 2.0, 3.0], [3.0, -1.0, 1.0], [-1.0, 2.0, 1.0]]))
    q.value = rotation @ np.diag([-2.0, -0.5, 1.5]) @ rotation.T
    problem.solve(solver=cp.CLARABEL, tol_gap_abs=1e-10, tol_feas=1e-10, tol_gap_rel=1e-10)
    expected = [x.value.copy(), con.dual_value.copy()]
    options = {
        "device": device,
        "ipm_settings": {
            "tol_gap_abs": 1e-10,
            "tol_gap_rel": 1e-10,
            "tol_feas": 1e-10,
        },
    }

    if framework == "torch":
        layer = CvxpyLayer(problem, [q], [x, con.dual_variables[0]], solver_args=options)
        p = torch.tensor(q.value, dtype=torch.double, device=device, requires_grad=True)
        actual = [v.detach().cpu() for v in layer(p)]
        assert torch.autograd.gradcheck(
            layer, (p,), eps=1e-4, atol=5e-4, rtol=5e-3, nondet_tol=1e-12
        )
        # Reusing CVXPY's cached canonicalization must not rescale or permute it twice.
        other = CvxpyLayer(problem, [q], [x, con.dual_variables[0]], solver_args=options)
        for a, b in zip(other(p), actual):
            np.testing.assert_allclose(a.detach().cpu(), b, atol=1e-7)
    else:
        jax = pytest.importorskip("jax")
        jnp = pytest.importorskip("jax.numpy")
        from cvxpylayers.jax import CvxpyLayer as JaxLayer

        layer = JaxLayer(problem, [q], [x, con.dual_variables[0]], solver_args=options)
        p = jax.device_put(
            jnp.asarray(q.value), jax.devices("gpu" if device == "cuda" else "cpu")[0]
        )
        actual = jax.jit(layer)(p)

        def loss(p):
            return sum(v.sum() for v in layer(p))

        grad = jax.jit(jax.grad(loss))(p)
        direction = jnp.asarray(np.arange(9).reshape(3, 3) / 9)
        finite_diff = (loss(p + 1e-4 * direction) - loss(p - 1e-4 * direction)) / 2e-4
        np.testing.assert_allclose(jnp.sum(grad * direction), finite_diff, atol=5e-4)

    assert layer.ctx.solver_ctx.has_psd_scaling == direct
    assert bool(layer.ctx.solver_ctx.cones.psd_dims) != direct
    for result, reference in zip(actual, expected):
        np.testing.assert_allclose(result, reference, atol=2e-4)
