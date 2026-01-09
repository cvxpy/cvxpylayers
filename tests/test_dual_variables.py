"""Unit tests for dual variable support in cvxpylayers."""

import cvxpy as cp
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from cvxpylayers.torch import CvxpyLayer  # noqa: E402

torch.set_default_dtype(torch.double)


def test_equality_constraint_dual():
    """Test returning dual variable for equality constraint."""
    n = 2
    x = cp.Variable(n)
    c = cp.Parameter(n)
    b = cp.Parameter()

    eq_con = cp.sum(x) == b
    prob = cp.Problem(cp.Minimize(c @ x), [eq_con, x >= 0])

    # Request both primal variable x and dual variable for equality constraint
    layer = CvxpyLayer(
        prob,
        parameters=[c, b],
        variables=[x, eq_con.dual_variables[0]],
    )

    c_t = torch.tensor([1.0, 2.0], requires_grad=True)
    b_t = torch.tensor(1.0, requires_grad=True)

    x_opt, eq_dual = layer(c_t, b_t)

    # Verify solution by solving with CVXPY directly
    c.value = c_t.detach().numpy()
    b.value = b_t.detach().numpy().item()
    prob.solve(solver=cp.CLARABEL)

    np.testing.assert_allclose(x_opt.detach().numpy(), x.value, rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(eq_dual.detach().numpy(), eq_con.dual_value, rtol=1e-3, atol=1e-4)


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

    c_t = torch.tensor([1.0, -1.0], requires_grad=True)

    x_opt, ineq_dual = layer(c_t)

    # Verify with CVXPY
    c.value = c_t.detach().numpy()
    prob.solve(solver=cp.CLARABEL)

    np.testing.assert_allclose(x_opt.detach().numpy(), x.value, rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(ineq_dual.detach().numpy(), ineq_con.dual_value, rtol=1e-3, atol=1e-4)


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

    c_t = torch.tensor([1.0, 2.0], requires_grad=True)
    b_t = torch.tensor(1.0, requires_grad=True)

    x_opt, eq_dual, ineq_dual = layer(c_t, b_t)

    # Verify with CVXPY
    c.value = c_t.detach().numpy()
    b.value = b_t.detach().numpy().item()
    prob.solve(solver=cp.CLARABEL)

    np.testing.assert_allclose(x_opt.detach().numpy(), x.value, rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(eq_dual.detach().numpy(), eq_con.dual_value, rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(ineq_dual.detach().numpy(), ineq_con.dual_value, rtol=1e-3, atol=1e-4)


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

    c_t = torch.tensor([1.0, 2.0])
    b_t = torch.tensor(1.0)

    (eq_dual,) = layer(c_t, b_t)

    # Verify with CVXPY
    c.value = c_t.numpy()
    b.value = b_t.numpy().item()
    prob.solve(solver=cp.CLARABEL)

    np.testing.assert_allclose(eq_dual.detach().numpy(), eq_con.dual_value, rtol=1e-3, atol=1e-4)


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
    c_t = torch.randn(batch_size, n, requires_grad=True)
    b_t = torch.ones(batch_size, requires_grad=True)

    x_opt, eq_dual = layer(c_t, b_t)

    assert x_opt.shape == (batch_size, n)
    assert eq_dual.shape == (batch_size,)

    # Verify each batch element (use looser tolerance for batched comparison)
    for i in range(batch_size):
        c.value = c_t[i].detach().numpy()
        b.value = b_t[i].detach().numpy().item()
        prob.solve(solver=cp.CLARABEL)

        np.testing.assert_allclose(x_opt[i].detach().numpy(), x.value, rtol=1e-2, atol=1e-3)
        np.testing.assert_allclose(eq_dual[i].detach().numpy(), eq_con.dual_value, rtol=1e-2, atol=1e-3)


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

    c_t = torch.tensor([1.0, 2.0], requires_grad=True)
    b_t = torch.tensor(1.0, requires_grad=True)

    x_opt, eq_dual = layer(c_t, b_t)

    # Compute gradient of dual variable with respect to parameters
    loss = eq_dual.sum()
    loss.backward()

    # Just check that gradients exist and are finite
    assert c_t.grad is not None
    assert b_t.grad is not None
    assert torch.isfinite(c_t.grad).all()
    assert torch.isfinite(b_t.grad).all()


def test_dual_gradcheck_equality():
    """Rigorous gradient check for equality constraint dual using gradcheck."""
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
    def f(c_t, b_t):
        x_opt, eq_dual = layer(c_t, b_t)
        return eq_dual

    c_t = torch.tensor([0.5, -0.3], requires_grad=True)
    b_t = torch.tensor(1.0, requires_grad=True)

    torch.autograd.gradcheck(f, (c_t, b_t), atol=1e-4, rtol=1e-3)


def test_dual_gradcheck_inequality():
    """Rigorous gradient check for inequality constraint dual using gradcheck."""
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
    def f(c_t):
        x_opt, ineq_dual = layer(c_t)
        return ineq_dual

    c_t = torch.tensor([1.0, -1.0], requires_grad=True)

    torch.autograd.gradcheck(f, (c_t,), atol=1e-4, rtol=1e-3)


def test_dual_gradcheck_mixed():
    """Rigorous gradient check for mixed primal and dual variables."""
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

    # Function that returns both primal and dual for gradcheck
    def f(c_t, b_t):
        x_opt, eq_dual = layer(c_t, b_t)
        # Return a scalar combining both
        return x_opt.sum() + eq_dual

    c_t = torch.tensor([0.5, -0.3], requires_grad=True)
    b_t = torch.tensor(1.0, requires_grad=True)

    torch.autograd.gradcheck(f, (c_t, b_t), atol=1e-4, rtol=1e-3)


def test_dual_gradcheck_vector_equality():
    """Rigorous gradient check for vector equality constraint dual."""
    n, m = 3, 2
    x = cp.Variable(n)
    A = cp.Parameter((m, n))
    b = cp.Parameter(m)

    eq_con = A @ x == b
    prob = cp.Problem(cp.Minimize(0.5 * cp.sum_squares(x)), [eq_con])

    layer = CvxpyLayer(
        prob,
        parameters=[A, b],
        variables=[x, eq_con.dual_variables[0]],
    )

    def f(A_t, b_t):
        x_opt, eq_dual = layer(A_t, b_t)
        return eq_dual.sum()

    torch.manual_seed(42)
    A_t = torch.randn(m, n, requires_grad=True)
    b_t = torch.randn(m, requires_grad=True)

    torch.autograd.gradcheck(f, (A_t, b_t), atol=1e-4, rtol=1e-3)


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

    c_t = torch.tensor([1.0, 0.5, -0.5], requires_grad=True)
    t_t = torch.tensor(2.0, requires_grad=True)

    x_opt, soc_dual = layer(c_t, t_t)

    # Verify with CVXPY
    c.value = c_t.detach().numpy()
    t.value = t_t.detach().numpy().item()
    prob.solve(solver=cp.CLARABEL)

    np.testing.assert_allclose(x_opt.detach().numpy(), x.value, rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(soc_dual.detach().numpy(), soc_con.dual_value, rtol=1e-3, atol=1e-4)


def test_soc_gradcheck():
    """Rigorous gradient check for SOC constraint dual."""
    n = 3
    x = cp.Variable(n)
    c = cp.Parameter(n)
    t = cp.Parameter(nonneg=True)

    soc_con = cp.norm(x) <= t
    prob = cp.Problem(cp.Minimize(c @ x + 0.1 * cp.sum_squares(x)), [soc_con])

    layer = CvxpyLayer(
        prob,
        parameters=[c, t],
        variables=[x, soc_con.dual_variables[0]],
    )

    def f(c_t, t_t):
        x_opt, soc_dual = layer(c_t, t_t)
        return soc_dual.sum()

    c_t = torch.tensor([0.5, 0.3, -0.2], requires_grad=True)
    t_t = torch.tensor(2.0, requires_grad=True)

    torch.autograd.gradcheck(f, (c_t, t_t), atol=1e-4, rtol=1e-3)


def test_exp_cone_constraint_dual():
    """Test dual variable for exponential cone constraint."""
    x = cp.Variable()
    y = cp.Variable()
    z = cp.Variable()
    t = cp.Parameter()

    # Exponential cone: (x, y, z) in K_exp means y * exp(x/y) <= z, y > 0
    # cp.exp(x) <= t is equivalent to (x, 1, t) in exp cone
    exp_con = cp.exp(x) <= t
    prob = cp.Problem(cp.Minimize(-x), [exp_con, x >= -5])

    layer = CvxpyLayer(
        prob,
        parameters=[t],
        variables=[x, exp_con.dual_variables[0]],
    )

    t_t = torch.tensor(2.0, requires_grad=True)

    x_opt, exp_dual = layer(t_t)

    # Verify with CVXPY
    t.value = t_t.detach().numpy().item()
    prob.solve(solver=cp.CLARABEL)

    np.testing.assert_allclose(x_opt.detach().numpy(), x.value, rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(exp_dual.detach().numpy(), exp_con.dual_value, rtol=1e-3, atol=1e-4)


def test_exp_cone_gradcheck():
    """Rigorous gradient check for exponential cone constraint dual."""
    x = cp.Variable()
    t = cp.Parameter(nonneg=True)

    exp_con = cp.exp(x) <= t
    prob = cp.Problem(cp.Minimize(-x + 0.1 * x**2), [exp_con, x >= -5])

    layer = CvxpyLayer(
        prob,
        parameters=[t],
        variables=[x, exp_con.dual_variables[0]],
    )

    def f(t_t):
        x_opt, exp_dual = layer(t_t)
        return exp_dual.sum()

    t_t = torch.tensor(2.0, requires_grad=True)

    torch.autograd.gradcheck(f, (t_t,), atol=1e-4, rtol=1e-3)


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

    torch.manual_seed(42)
    C_np = np.array([[1.0, 0.5], [0.5, 2.0]])
    C_t = torch.tensor(C_np, requires_grad=True)

    X_opt, psd_dual = layer(C_t)

    # Verify with CVXPY
    C.value = C_t.detach().numpy()
    prob.solve(solver=cp.CLARABEL)

    np.testing.assert_allclose(X_opt.detach().numpy(), X.value, rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(psd_dual.detach().numpy(), psd_con.dual_value, rtol=1e-3, atol=1e-4)


def test_psd_gradcheck():
    """Rigorous gradient check for PSD constraint dual."""
    n = 2
    X = cp.Variable((n, n), symmetric=True)
    C = cp.Parameter((n, n), symmetric=True)

    psd_con = X >> 0
    trace_con = cp.trace(X) == 1
    prob = cp.Problem(cp.Minimize(cp.trace(C @ X)), [psd_con, trace_con])

    layer = CvxpyLayer(
        prob,
        parameters=[C],
        variables=[X, psd_con.dual_variables[0]],
    )

    def f(C_t):
        X_opt, psd_dual = layer(C_t)
        return psd_dual.sum()

    C_t = torch.tensor([[1.0, 0.5], [0.5, 2.0]], requires_grad=True)

    torch.autograd.gradcheck(f, (C_t,), atol=1e-4, rtol=1e-3)


def test_invalid_dual_variable():
    """Test that invalid dual variables raise an error."""
    x = cp.Variable(2)
    c = cp.Parameter(2)

    # Create a constraint not in the problem
    other_con = x >= 1
    prob = cp.Problem(cp.Minimize(c @ x), [x >= 0])

    with pytest.raises(ValueError, match="must be a subset of problem.variables"):
        CvxpyLayer(
            prob,
            parameters=[c],
            variables=[x, other_con.dual_variables[0]],
        )


def test_vector_equality_dual():
    """Test dual variable for vector equality constraint."""
    n, m = 3, 2
    x = cp.Variable(n)
    A = cp.Parameter((m, n))
    b = cp.Parameter(m)

    eq_con = A @ x == b
    prob = cp.Problem(cp.Minimize(cp.sum_squares(x)), [eq_con])

    layer = CvxpyLayer(
        prob,
        parameters=[A, b],
        variables=[x, eq_con.dual_variables[0]],
    )

    A_t = torch.randn(m, n, requires_grad=True)
    b_t = torch.randn(m, requires_grad=True)

    x_opt, eq_dual = layer(A_t, b_t)

    # Dual should have shape (m,)
    assert eq_dual.shape == (m,)

    # Verify with CVXPY
    A.value = A_t.detach().numpy()
    b.value = b_t.detach().numpy()
    prob.solve(solver=cp.CLARABEL)

    np.testing.assert_allclose(x_opt.detach().numpy(), x.value, rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(eq_dual.detach().numpy(), eq_con.dual_value, rtol=1e-3, atol=1e-4)


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

    a_t = torch.tensor(2.0, requires_grad=True)
    b_t = torch.tensor(1.0, requires_grad=True)

    x_opt, y_opt, z_opt, ineq_dual = layer(a_t, b_t)

    # Verify with CVXPY
    a.value = a_t.detach().numpy().item()
    b.value = b_t.detach().numpy().item()
    prob.solve(solver=cp.CLARABEL, gp=True)

    np.testing.assert_allclose(x_opt.detach().numpy(), x.value, rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(y_opt.detach().numpy(), y.value, rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(z_opt.detach().numpy(), z.value, rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(
        ineq_dual.detach().numpy(), ineq_con.dual_value, rtol=1e-3, atol=1e-4
    )


def test_gp_multiple_constraint_duals():
    """Test multiple dual variables in geometric program."""
    x = cp.Variable(pos=True)
    y = cp.Variable(pos=True)

    a = cp.Parameter(pos=True)
    b = cp.Parameter(pos=True)

    # Two inequality constraints - both active at optimal
    ineq_con1 = a * (x * y) <= b
    ineq_con2 = x + y <= 2  # Linear constraint

    prob = cp.Problem(cp.Minimize(1 / (x * y)), [ineq_con1, ineq_con2])
    assert prob.is_dgp(dpp=True)

    layer = CvxpyLayer(
        prob,
        parameters=[a, b],
        variables=[x, y, ineq_con1.dual_variables[0], ineq_con2.dual_variables[0]],
        gp=True,
    )

    a_t = torch.tensor(2.0, requires_grad=True)
    b_t = torch.tensor(1.0, requires_grad=True)

    x_opt, y_opt, dual1, dual2 = layer(a_t, b_t)

    # Verify with CVXPY
    a.value = a_t.detach().numpy().item()
    b.value = b_t.detach().numpy().item()
    prob.solve(solver=cp.CLARABEL, gp=True)

    np.testing.assert_allclose(x_opt.detach().numpy(), x.value, rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(y_opt.detach().numpy(), y.value, rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(dual1.detach().numpy(), ineq_con1.dual_value, rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(dual2.detach().numpy(), ineq_con2.dual_value, rtol=1e-3, atol=1e-4)


def test_gp_dual_gradcheck():
    """Rigorous gradient check for GP dual variables."""
    x = cp.Variable(pos=True)
    y = cp.Variable(pos=True)

    a = cp.Parameter(pos=True)
    b = cp.Parameter(pos=True)

    ineq_con = a * (x * y) <= b
    prob = cp.Problem(cp.Minimize(1 / (x * y)), [ineq_con])

    layer = CvxpyLayer(
        prob,
        parameters=[a, b],
        variables=[x, y, ineq_con.dual_variables[0]],
        gp=True,
    )

    def f(a_t, b_t):
        x_opt, y_opt, dual = layer(a_t, b_t)
        return dual.sum()

    a_t = torch.tensor(2.0, requires_grad=True)
    b_t = torch.tensor(1.0, requires_grad=True)

    torch.autograd.gradcheck(f, (a_t, b_t), atol=1e-4, rtol=1e-3)
