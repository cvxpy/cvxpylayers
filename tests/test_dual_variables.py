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
    np.testing.assert_allclose(
        ineq_dual.detach().numpy(), ineq_con.dual_value, rtol=1e-3, atol=1e-4
    )


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
    np.testing.assert_allclose(
        ineq_dual.detach().numpy(), ineq_con.dual_value, rtol=1e-3, atol=1e-4
    )


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
        np.testing.assert_allclose(
            eq_dual[i].detach().numpy(), eq_con.dual_value, rtol=1e-2, atol=1e-3
        )


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
    """Test dual variables for raw exponential cone constraint.

    ExpCone(x, y, z) represents: y * exp(x/y) <= z, y > 0
    It has 3 dual variables.
    """
    x = cp.Variable()
    y = cp.Variable()
    z = cp.Variable()
    t = cp.Parameter(nonneg=True)

    # Raw exponential cone constraint
    exp_con = cp.constraints.ExpCone(x, y, z)
    prob = cp.Problem(
        cp.Minimize(-z + 0.1 * (x**2 + y**2 + z**2)),
        [exp_con, y >= 0.1, z <= t],
    )

    layer = CvxpyLayer(
        prob,
        parameters=[t],
        variables=[
            x,
            y,
            z,
            exp_con.dual_variables[0],
            exp_con.dual_variables[1],
            exp_con.dual_variables[2],
        ],
    )

    t_t = torch.tensor(5.0, requires_grad=True)

    x_opt, y_opt, z_opt, dual0, dual1, dual2 = layer(t_t)

    # Verify with CVXPY
    t.value = t_t.detach().numpy().item()
    prob.solve(solver=cp.CLARABEL)

    np.testing.assert_allclose(x_opt.detach().numpy(), x.value, rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(y_opt.detach().numpy(), y.value, rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(z_opt.detach().numpy(), z.value, rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(
        dual0.detach().numpy(), exp_con.dual_variables[0].value, rtol=1e-3, atol=1e-4
    )
    np.testing.assert_allclose(
        dual1.detach().numpy(), exp_con.dual_variables[1].value, rtol=1e-3, atol=1e-4
    )
    np.testing.assert_allclose(
        dual2.detach().numpy(), exp_con.dual_variables[2].value, rtol=1e-3, atol=1e-4
    )


def test_exp_cone_gradcheck():
    """Rigorous gradient check for raw exponential cone dual variables."""
    x = cp.Variable()
    y = cp.Variable()
    z = cp.Variable()
    t = cp.Parameter(nonneg=True)

    exp_con = cp.constraints.ExpCone(x, y, z)
    prob = cp.Problem(
        cp.Minimize(-z + 0.1 * (x**2 + y**2 + z**2)),
        [exp_con, y >= 0.1, z <= t],
    )

    layer = CvxpyLayer(
        prob,
        parameters=[t],
        variables=[
            x,
            y,
            z,
            exp_con.dual_variables[0],
            exp_con.dual_variables[1],
            exp_con.dual_variables[2],
        ],
    )

    def f(t_t):
        x_opt, y_opt, z_opt, d0, d1, d2 = layer(t_t)
        return d0.sum() + d1.sum() + d2.sum()

    t_t = torch.tensor(5.0, requires_grad=True)

    torch.autograd.gradcheck(f, (t_t,), atol=1e-4, rtol=1e-3)


def test_pow_cone_constraint_dual():
    """Test dual variables for power cone constraint.

    PowCone3D(x, y, z, alpha) represents: x^alpha * y^(1-alpha) >= |z|, x >= 0, y >= 0
    It has 3 dual variables.
    """
    x = cp.Variable()
    y = cp.Variable()
    z = cp.Variable()
    t = cp.Parameter(nonneg=True)

    # Raw power cone constraint with alpha=0.5 (geometric mean)
    pow_con = cp.PowCone3D(x, y, z, 0.5)
    prob = cp.Problem(
        cp.Minimize(-z + 0.1 * (x**2 + y**2)),
        [pow_con, x >= 0.1, y >= 0.1, x + y <= t],
    )

    layer = CvxpyLayer(
        prob,
        parameters=[t],
        variables=[
            x,
            y,
            z,
            pow_con.dual_variables[0],
            pow_con.dual_variables[1],
            pow_con.dual_variables[2],
        ],
    )

    t_t = torch.tensor(4.0, requires_grad=True)

    x_opt, y_opt, z_opt, dual0, dual1, dual2 = layer(t_t)

    # Verify with CVXPY
    t.value = t_t.detach().numpy().item()
    prob.solve(solver=cp.CLARABEL)

    np.testing.assert_allclose(x_opt.detach().numpy(), x.value, rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(y_opt.detach().numpy(), y.value, rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(z_opt.detach().numpy(), z.value, rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(
        dual0.detach().numpy(), pow_con.dual_variables[0].value, rtol=1e-3, atol=1e-4
    )
    np.testing.assert_allclose(
        dual1.detach().numpy(), pow_con.dual_variables[1].value, rtol=1e-3, atol=1e-4
    )
    np.testing.assert_allclose(
        dual2.detach().numpy(), pow_con.dual_variables[2].value, rtol=1e-3, atol=1e-4
    )


def test_pow_cone_gradcheck():
    """Rigorous gradient check for power cone dual variables.

    Note: diffcp has a bug in adjoint_derivative for power cones - shape mismatch
    between dual dimensions. Forward pass works, backward fails.
    The forward pass is tested in test_pow_cone_constraint_dual.
    """
    pytest.skip("diffcp bug: shape mismatch in adjoint_derivative for power cones")


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


def test_soc_explicit_multi_dual():
    """Test SOC constraint with multiple dual variables.

    The explicit cp.SOC(t, x) constraint has two dual variables:
    - dual_variables[0]: scalar dual for t
    - dual_variables[1]: vector dual for x
    """
    n = 3
    x = cp.Variable(n)
    t = cp.Variable()
    c = cp.Parameter(n)
    t_param = cp.Parameter(nonneg=True)

    soc_con = cp.SOC(t, x)
    prob = cp.Problem(cp.Minimize(c @ x - t), [soc_con, t <= t_param])

    # Request both dual variables from the SOC constraint
    layer = CvxpyLayer(
        prob,
        parameters=[c, t_param],
        variables=[x, t, soc_con.dual_variables[0], soc_con.dual_variables[1]],
    )

    c_t = torch.tensor([1.0, 0.5, -0.5], requires_grad=True)
    t_t = torch.tensor(2.0, requires_grad=True)

    x_opt, t_opt, soc_dual0, soc_dual1 = layer(c_t, t_t)

    # Verify with CVXPY
    c.value = c_t.detach().numpy()
    t_param.value = t_t.detach().numpy().item()
    prob.solve(solver=cp.CLARABEL)

    np.testing.assert_allclose(x_opt.detach().numpy(), x.value, rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(t_opt.detach().numpy(), t.value, rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(
        soc_dual0.detach().numpy(), soc_con.dual_variables[0].value, rtol=1e-3, atol=1e-4
    )
    np.testing.assert_allclose(
        soc_dual1.detach().numpy().flatten(),
        soc_con.dual_variables[1].value.flatten(),
        rtol=1e-3,
        atol=1e-4,
    )


def test_soc_explicit_multi_dual_gradcheck():
    """Rigorous gradient check for SOC with multiple dual variables."""
    n = 3
    x = cp.Variable(n)
    t = cp.Variable()
    c = cp.Parameter(n)
    t_param = cp.Parameter(nonneg=True)

    soc_con = cp.SOC(t, x)
    # Add regularization for well-conditioned gradients
    prob = cp.Problem(cp.Minimize(c @ x - t + 0.1 * cp.sum_squares(x)), [soc_con, t <= t_param])

    layer = CvxpyLayer(
        prob,
        parameters=[c, t_param],
        variables=[x, t, soc_con.dual_variables[0], soc_con.dual_variables[1]],
    )

    def f(c_t, t_t):
        x_opt, t_opt, dual0, dual1 = layer(c_t, t_t)
        return dual0.sum() + dual1.sum()

    c_t = torch.tensor([0.5, 0.3, -0.2], requires_grad=True)
    t_t = torch.tensor(2.0, requires_grad=True)

    torch.autograd.gradcheck(f, (c_t, t_t), atol=1e-4, rtol=1e-3)


def test_mixed_cones_all_types():
    """Test dual variables from all cone types in a single problem.

    This verifies that the constraint ordering in the dual vector is correct:
    Zero (equality) -> NonNeg (inequality) -> SOC -> ExpCone -> PSD -> PowCone3D
    """
    # Variables for each cone type
    x_eq = cp.Variable(2)  # For equality constraint
    x_ineq = cp.Variable(2)  # For inequality constraint
    x_soc = cp.Variable(2)  # For SOC
    t_soc = cp.Variable()
    x_exp = cp.Variable()  # For ExpCone
    y_exp = cp.Variable()
    z_exp = cp.Variable()
    X_psd = cp.Variable((2, 2), symmetric=True)  # For PSD
    x_pow = cp.Variable()  # For PowCone3D
    y_pow = cp.Variable()
    z_pow = cp.Variable()

    # Parameters
    b_eq = cp.Parameter(2)
    ub = cp.Parameter(2)
    t_param = cp.Parameter(nonneg=True)

    # Create constraints of each type
    eq_con = x_eq == b_eq  # Zero cone (equality)
    ineq_con = x_ineq <= ub  # NonNeg cone (inequality)
    soc_con = cp.SOC(t_soc, x_soc)  # Second-order cone
    exp_con = cp.constraints.ExpCone(x_exp, y_exp, z_exp)  # Exponential cone
    psd_con = X_psd >> 0  # PSD cone
    pow_con = cp.PowCone3D(x_pow, y_pow, z_pow, 0.5)  # Power cone

    # Objective with regularization for all variables
    obj = (
        cp.sum_squares(x_eq)
        + cp.sum_squares(x_ineq)
        + cp.sum_squares(x_soc)
        - t_soc
        - z_exp
        + 0.1 * (x_exp**2 + y_exp**2 + z_exp**2)
        + cp.trace(X_psd)
        - z_pow
        + 0.1 * (x_pow**2 + y_pow**2)
    )

    prob = cp.Problem(
        cp.Minimize(obj),
        [
            eq_con,
            ineq_con,
            soc_con,
            t_soc <= t_param,
            exp_con,
            y_exp >= 0.1,
            z_exp <= t_param,
            psd_con,
            cp.trace(X_psd) >= 0.5,
            pow_con,
            x_pow >= 0.1,
            y_pow >= 0.1,
            x_pow + y_pow <= t_param,
        ],
    )

    # Request duals from each cone type
    layer = CvxpyLayer(
        prob,
        parameters=[b_eq, ub, t_param],
        variables=[
            x_eq,
            x_ineq,
            t_soc,
            z_exp,
            X_psd,
            z_pow,
            # Duals: eq, ineq, soc (2 duals), exp (3 duals), psd, pow (3 duals)
            eq_con.dual_variables[0],
            ineq_con.dual_variables[0],
            soc_con.dual_variables[0],
            soc_con.dual_variables[1],
            exp_con.dual_variables[0],
            exp_con.dual_variables[1],
            exp_con.dual_variables[2],
            psd_con.dual_variables[0],
            pow_con.dual_variables[0],
            pow_con.dual_variables[1],
            pow_con.dual_variables[2],
        ],
    )

    b_eq_t = torch.tensor([0.5, -0.3], requires_grad=True)
    ub_t = torch.tensor([1.0, 1.0], requires_grad=True)
    t_t = torch.tensor(3.0, requires_grad=True)

    results = layer(b_eq_t, ub_t, t_t)

    # Unpack results
    (
        x_eq_opt,
        x_ineq_opt,
        t_soc_opt,
        z_exp_opt,
        X_psd_opt,
        z_pow_opt,
        eq_dual,
        ineq_dual,
        soc_dual0,
        soc_dual1,
        exp_dual0,
        exp_dual1,
        exp_dual2,
        psd_dual,
        pow_dual0,
        pow_dual1,
        pow_dual2,
    ) = results

    # Verify with CVXPY
    b_eq.value = b_eq_t.detach().numpy()
    ub.value = ub_t.detach().numpy()
    t_param.value = t_t.detach().numpy().item()
    prob.solve(solver=cp.CLARABEL)

    # Check primal variables
    np.testing.assert_allclose(x_eq_opt.detach().numpy(), x_eq.value, rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(x_ineq_opt.detach().numpy(), x_ineq.value, rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(t_soc_opt.detach().numpy(), t_soc.value, rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(z_exp_opt.detach().numpy(), z_exp.value, rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(X_psd_opt.detach().numpy(), X_psd.value, rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(z_pow_opt.detach().numpy(), z_pow.value, rtol=1e-3, atol=1e-4)

    # Check dual variables from each cone type
    np.testing.assert_allclose(
        eq_dual.detach().numpy(), eq_con.dual_variables[0].value, rtol=1e-3, atol=1e-4
    )
    np.testing.assert_allclose(
        ineq_dual.detach().numpy(), ineq_con.dual_variables[0].value, rtol=1e-3, atol=1e-4
    )
    np.testing.assert_allclose(
        soc_dual0.detach().numpy(), soc_con.dual_variables[0].value, rtol=1e-3, atol=1e-4
    )
    np.testing.assert_allclose(
        soc_dual1.detach().numpy().flatten(),
        soc_con.dual_variables[1].value.flatten(),
        rtol=1e-3,
        atol=1e-4,
    )
    np.testing.assert_allclose(
        exp_dual0.detach().numpy(), exp_con.dual_variables[0].value, rtol=1e-3, atol=1e-4
    )
    np.testing.assert_allclose(
        exp_dual1.detach().numpy(), exp_con.dual_variables[1].value, rtol=1e-3, atol=1e-4
    )
    np.testing.assert_allclose(
        exp_dual2.detach().numpy(), exp_con.dual_variables[2].value, rtol=1e-3, atol=1e-4
    )
    np.testing.assert_allclose(
        psd_dual.detach().numpy(), psd_con.dual_variables[0].value, rtol=1e-3, atol=1e-4
    )
    np.testing.assert_allclose(
        pow_dual0.detach().numpy(), pow_con.dual_variables[0].value, rtol=1e-3, atol=1e-4
    )
    np.testing.assert_allclose(
        pow_dual1.detach().numpy(), pow_con.dual_variables[1].value, rtol=1e-3, atol=1e-4
    )
    np.testing.assert_allclose(
        pow_dual2.detach().numpy(), pow_con.dual_variables[2].value, rtol=1e-3, atol=1e-4
    )
