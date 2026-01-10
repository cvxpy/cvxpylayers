"""Unit tests for dual variable support in cvxpylayers (JAX backend)."""

import cvxpy as cp
import numpy as np
import pytest

jax = pytest.importorskip("jax")

import jax.numpy as jnp  # noqa: E402
from jax.test_util import check_grads  # noqa: E402

from cvxpylayers.jax import CvxpyLayer  # noqa: E402


def test_equality_constraint_dual():
    """Test returning dual variable for equality constraint."""
    n = 2
    x = cp.Variable(n)
    c = cp.Parameter(n)
    b = cp.Parameter()

    eq_con = cp.sum(x) == b
    prob = cp.Problem(cp.Minimize(c @ x), [eq_con, x >= 0])

    layer = CvxpyLayer(
        prob,
        parameters=[c, b],
        variables=[x, eq_con.dual_variables[0]],
    )

    c_jax = jnp.array([1.0, 2.0])
    b_jax = jnp.array(1.0)

    x_opt, eq_dual = layer(c_jax, b_jax)

    # Verify solution by solving with CVXPY directly
    c.value = np.array(c_jax)
    b.value = float(b_jax)
    prob.solve(solver=cp.CLARABEL)

    np.testing.assert_allclose(np.array(x_opt), x.value, rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(np.array(eq_dual), eq_con.dual_value, rtol=1e-3, atol=1e-4)


def test_inequality_constraint_dual():
    """Test returning dual variable for inequality constraint."""
    n = 2
    x = cp.Variable(n)
    c = cp.Parameter(n)

    ineq_con = x >= 0
    prob = cp.Problem(cp.Minimize(c @ x + cp.sum_squares(x)), [ineq_con])

    layer = CvxpyLayer(
        prob,
        parameters=[c],
        variables=[x, ineq_con.dual_variables[0]],
    )

    c_jax = jnp.array([1.0, -1.0])

    x_opt, ineq_dual = layer(c_jax)

    # Verify with CVXPY
    c.value = np.array(c_jax)
    prob.solve(solver=cp.CLARABEL)

    np.testing.assert_allclose(np.array(x_opt), x.value, rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(np.array(ineq_dual), ineq_con.dual_value, rtol=1e-3, atol=1e-4)


def test_multiple_dual_variables():
    """Test returning multiple dual variables from different constraints."""
    n = 2
    x = cp.Variable(n)
    c = cp.Parameter(n)
    b = cp.Parameter()

    eq_con = cp.sum(x) == b
    ineq_con = x >= 0
    prob = cp.Problem(cp.Minimize(c @ x), [eq_con, ineq_con])

    layer = CvxpyLayer(
        prob,
        parameters=[c, b],
        variables=[x, eq_con.dual_variables[0], ineq_con.dual_variables[0]],
    )

    c_jax = jnp.array([1.0, 2.0])
    b_jax = jnp.array(1.0)

    x_opt, eq_dual, ineq_dual = layer(c_jax, b_jax)

    # Verify with CVXPY
    c.value = np.array(c_jax)
    b.value = float(b_jax)
    prob.solve(solver=cp.CLARABEL)

    np.testing.assert_allclose(np.array(x_opt), x.value, rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(np.array(eq_dual), eq_con.dual_value, rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(np.array(ineq_dual), ineq_con.dual_value, rtol=1e-3, atol=1e-4)


def test_dual_only():
    """Test returning only dual variables (no primal)."""
    n = 2
    x = cp.Variable(n)
    c = cp.Parameter(n)
    b = cp.Parameter()

    eq_con = cp.sum(x) == b
    prob = cp.Problem(cp.Minimize(c @ x), [eq_con, x >= 0])

    # Only request dual variable
    layer = CvxpyLayer(
        prob,
        parameters=[c, b],
        variables=[eq_con.dual_variables[0]],
    )

    c_jax = jnp.array([1.0, 2.0])
    b_jax = jnp.array(1.0)

    (eq_dual,) = layer(c_jax, b_jax)

    # Verify with CVXPY
    c.value = np.array(c_jax)
    b.value = float(b_jax)
    prob.solve(solver=cp.CLARABEL)

    np.testing.assert_allclose(np.array(eq_dual), eq_con.dual_value, rtol=1e-3, atol=1e-4)


def test_batched_dual_variables():
    """Test dual variables with batched parameters."""
    n = 2
    batch_size = 3

    x = cp.Variable(n)
    c = cp.Parameter(n)
    b = cp.Parameter()

    eq_con = cp.sum(x) == b
    prob = cp.Problem(cp.Minimize(c @ x), [eq_con, x >= 0])

    layer = CvxpyLayer(
        prob,
        parameters=[c, b],
        variables=[x, eq_con.dual_variables[0]],
    )

    # Batched inputs
    key = jax.random.PRNGKey(42)
    c_jax = jax.random.normal(key, (batch_size, n))
    b_jax = jnp.ones(batch_size)

    x_opt, eq_dual = layer(c_jax, b_jax)

    assert x_opt.shape == (batch_size, n)
    assert eq_dual.shape == (batch_size,)

    # Verify each batch element
    for i in range(batch_size):
        c.value = np.array(c_jax[i])
        b.value = float(b_jax[i])
        prob.solve(solver=cp.CLARABEL)

        np.testing.assert_allclose(np.array(x_opt[i]), x.value, rtol=1e-2, atol=1e-3)
        np.testing.assert_allclose(np.array(eq_dual[i]), eq_con.dual_value, rtol=1e-2, atol=1e-3)


def test_dual_gradient():
    """Test gradient computation through dual variables."""
    n = 2
    x = cp.Variable(n)
    c = cp.Parameter(n)
    b = cp.Parameter()

    eq_con = cp.sum(x) == b
    prob = cp.Problem(cp.Minimize(c @ x), [eq_con, x >= 0])

    layer = CvxpyLayer(
        prob,
        parameters=[c, b],
        variables=[x, eq_con.dual_variables[0]],
    )

    c_jax = jnp.array([1.0, 2.0])
    b_jax = jnp.array(1.0)

    # Compute gradient of dual variable with respect to parameters
    def loss_fn(c, b):
        _, eq_dual = layer(c, b)
        return jnp.sum(eq_dual)

    grads = jax.grad(loss_fn, argnums=[0, 1])(c_jax, b_jax)

    # Check that gradients exist and are finite
    assert grads[0] is not None
    assert grads[1] is not None
    assert jnp.isfinite(grads[0]).all()
    assert jnp.isfinite(grads[1]).all()


def test_dual_gradcheck_equality():
    """Rigorous gradient check for equality constraint dual using check_grads."""
    n = 2
    x = cp.Variable(n)
    c = cp.Parameter(n)
    b = cp.Parameter()

    eq_con = cp.sum(x) == b
    prob = cp.Problem(cp.Minimize(c @ x + 0.5 * cp.sum_squares(x)), [eq_con])

    layer = CvxpyLayer(
        prob,
        parameters=[c, b],
        variables=[x, eq_con.dual_variables[0]],
    )

    # Function that returns dual variable for gradcheck
    def f(c_jax, b_jax):
        _, eq_dual = layer(c_jax, b_jax)
        return eq_dual

    c_jax = jnp.array([0.5, -0.3])
    b_jax = jnp.array(1.0)

    check_grads(f, (c_jax, b_jax), order=1, modes=["rev"], atol=1e-4, rtol=1e-3)


def test_dual_gradcheck_inequality():
    """Rigorous gradient check for inequality constraint dual using check_grads."""
    n = 2
    x = cp.Variable(n)
    c = cp.Parameter(n)

    ineq_con = x >= 0
    prob = cp.Problem(cp.Minimize(c @ x + 0.5 * cp.sum_squares(x)), [ineq_con])

    layer = CvxpyLayer(
        prob,
        parameters=[c],
        variables=[x, ineq_con.dual_variables[0]],
    )

    # Function that returns dual variable for gradcheck
    def f(c_jax):
        _, ineq_dual = layer(c_jax)
        return ineq_dual

    c_jax = jnp.array([1.0, -1.0])

    check_grads(f, (c_jax,), order=1, modes=["rev"], atol=1e-4, rtol=1e-3)


def test_soc_constraint_dual():
    """Test dual variable for second-order cone constraint."""
    n = 3
    x = cp.Variable(n)
    c = cp.Parameter(n)
    t = cp.Parameter(nonneg=True)

    soc_con = cp.norm(x) <= t
    prob = cp.Problem(cp.Minimize(c @ x), [soc_con])

    layer = CvxpyLayer(
        prob,
        parameters=[c, t],
        variables=[x, soc_con.dual_variables[0]],
    )

    c_jax = jnp.array([1.0, 0.5, -0.5])
    t_jax = jnp.array(2.0)

    x_opt, soc_dual = layer(c_jax, t_jax)

    # Verify with CVXPY
    c.value = np.array(c_jax)
    t.value = float(t_jax)
    prob.solve(solver=cp.CLARABEL)

    np.testing.assert_allclose(np.array(x_opt), x.value, rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(np.array(soc_dual), soc_con.dual_value, rtol=1e-3, atol=1e-4)


def test_psd_constraint_dual():
    """Test dual variable for PSD constraint."""
    n = 2
    X = cp.Variable((n, n), symmetric=True)
    C = cp.Parameter((n, n), symmetric=True)

    psd_con = X >> 0  # X is positive semidefinite
    trace_con = cp.trace(X) == 1
    prob = cp.Problem(cp.Minimize(cp.trace(C @ X)), [psd_con, trace_con])

    layer = CvxpyLayer(
        prob,
        parameters=[C],
        variables=[X, psd_con.dual_variables[0]],
    )

    C_np = np.array([[1.0, 0.5], [0.5, 2.0]])
    C_jax = jnp.array(C_np)

    X_opt, psd_dual = layer(C_jax)

    # Verify with CVXPY
    C.value = C_np
    prob.solve(solver=cp.CLARABEL)

    np.testing.assert_allclose(np.array(X_opt), X.value, rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(np.array(psd_dual), psd_con.dual_value, rtol=1e-3, atol=1e-4)


def test_gp_inequality_constraint_dual():
    """Test dual variable for inequality constraint in geometric program."""
    x = cp.Variable(pos=True)
    y = cp.Variable(pos=True)
    z = cp.Variable(pos=True)

    a = cp.Parameter(pos=True)
    b = cp.Parameter(pos=True)

    # GP constraint: a * (x*y + x*z + y*z) <= b
    ineq_con = a * (x * y + x * z + y * z) <= b
    prob = cp.Problem(cp.Minimize(1 / (x * y * z)), [ineq_con])
    assert prob.is_dgp(dpp=True)

    layer = CvxpyLayer(
        prob,
        parameters=[a, b],
        variables=[x, y, z, ineq_con.dual_variables[0]],
        gp=True,
    )

    a_jax = jnp.array(2.0)
    b_jax = jnp.array(1.0)

    x_opt, y_opt, z_opt, ineq_dual = layer(a_jax, b_jax)

    # Verify with CVXPY
    a.value = float(a_jax)
    b.value = float(b_jax)
    prob.solve(solver=cp.CLARABEL, gp=True)

    np.testing.assert_allclose(float(x_opt), x.value, rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(float(y_opt), y.value, rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(float(z_opt), z.value, rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(float(ineq_dual), ineq_con.dual_value, rtol=1e-3, atol=1e-4)
