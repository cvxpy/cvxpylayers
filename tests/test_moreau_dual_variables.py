"""Unit tests for dual variable support with Moreau solver."""

import cvxpy as cp
import numpy as np
import pytest

torch = pytest.importorskip("torch")
moreau = pytest.importorskip("moreau")

from cvxpylayers.torch import CvxpyLayer  # noqa: E402

torch.set_default_dtype(torch.double)

# Tolerance constants for solution comparisons
SOLUTION_RTOL = 1e-3
SOLUTION_ATOL = 1e-4

# Check for CUDA availability
HAS_CUDA = torch.cuda.is_available() and moreau.device_available("cuda")

# Device parametrization - runs tests on both CPU and CUDA (if available)
DEVICES = ["cpu", "cuda"] if HAS_CUDA else ["cpu"]


def get_device_params():
    """Return device parameters with appropriate skip markers."""
    params = [pytest.param("cpu", id="cpu")]
    if HAS_CUDA:
        params.append(pytest.param("cuda", id="cuda"))
    else:
        params.append(pytest.param("cuda", id="cuda", marks=pytest.mark.skip(reason="CUDA not available")))
    return params


# ============================================================================
# Forward Pass Tests (verify dual values match ground truth)
# ============================================================================


@pytest.mark.parametrize("device", get_device_params())
def test_equality_constraint_dual_moreau(device):
    """Test returning dual variable for equality constraint with Moreau solver."""
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
        solver="MOREAU",
    )

    c_t = torch.tensor([1.0, 2.0], requires_grad=True, device=device)
    b_t = torch.tensor(1.0, requires_grad=True, device=device)

    x_opt, eq_dual = layer(c_t, b_t)

    # Verify output device matches input
    assert x_opt.device.type == device
    assert eq_dual.device.type == device

    # Verify solution by solving with CVXPY directly
    c.value = c_t.detach().cpu().numpy()
    b.value = b_t.detach().cpu().numpy().item()
    prob.solve(solver=cp.CLARABEL)

    np.testing.assert_allclose(x_opt.detach().cpu().numpy(), x.value, rtol=SOLUTION_RTOL, atol=SOLUTION_ATOL)
    np.testing.assert_allclose(eq_dual.detach().cpu().numpy(), eq_con.dual_value, rtol=SOLUTION_RTOL, atol=SOLUTION_ATOL)


@pytest.mark.parametrize("device", get_device_params())
def test_inequality_constraint_dual_moreau(device):
    """Test returning dual variable for inequality constraint with Moreau solver."""
    n = 2
    x = cp.Variable(n)
    c = cp.Parameter(n)

    ineq_con = x >= 0
    prob = cp.Problem(cp.Minimize(c @ x + cp.sum_squares(x)), [ineq_con])

    layer = CvxpyLayer(
        prob,
        parameters=[c],
        variables=[x, ineq_con.dual_variables[0]],
        solver="MOREAU",
    )

    c_t = torch.tensor([1.0, -1.0], requires_grad=True, device=device)

    x_opt, ineq_dual = layer(c_t)

    assert x_opt.device.type == device
    assert ineq_dual.device.type == device

    # Verify with CVXPY
    c.value = c_t.detach().cpu().numpy()
    prob.solve(solver=cp.CLARABEL)

    np.testing.assert_allclose(x_opt.detach().cpu().numpy(), x.value, rtol=SOLUTION_RTOL, atol=SOLUTION_ATOL)
    np.testing.assert_allclose(
        ineq_dual.detach().cpu().numpy(), ineq_con.dual_value, rtol=SOLUTION_RTOL, atol=SOLUTION_ATOL
    )


@pytest.mark.parametrize("device", get_device_params())
def test_multiple_dual_variables_moreau(device):
    """Test returning multiple dual variables from different constraints with Moreau."""
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
        solver="MOREAU",
    )

    c_t = torch.tensor([1.0, 2.0], requires_grad=True, device=device)
    b_t = torch.tensor(1.0, requires_grad=True, device=device)

    x_opt, eq_dual, ineq_dual = layer(c_t, b_t)

    assert x_opt.device.type == device

    # Verify with CVXPY
    c.value = c_t.detach().cpu().numpy()
    b.value = b_t.detach().cpu().numpy().item()
    prob.solve(solver=cp.CLARABEL)

    np.testing.assert_allclose(x_opt.detach().cpu().numpy(), x.value, rtol=SOLUTION_RTOL, atol=SOLUTION_ATOL)
    np.testing.assert_allclose(eq_dual.detach().cpu().numpy(), eq_con.dual_value, rtol=SOLUTION_RTOL, atol=SOLUTION_ATOL)
    np.testing.assert_allclose(
        ineq_dual.detach().cpu().numpy(), ineq_con.dual_value, rtol=SOLUTION_RTOL, atol=SOLUTION_ATOL
    )


@pytest.mark.parametrize("device", get_device_params())
def test_dual_only_moreau(device):
    """Test returning only dual variables (no primal) with Moreau."""
    n = 2
    x = cp.Variable(n)
    c = cp.Parameter(n)
    b = cp.Parameter()

    eq_con = cp.sum(x) == b
    prob = cp.Problem(cp.Minimize(c @ x), [eq_con, x >= 0])

    layer = CvxpyLayer(
        prob,
        parameters=[c, b],
        variables=[eq_con.dual_variables[0]],
        solver="MOREAU",
    )

    c_t = torch.tensor([1.0, 2.0], device=device)
    b_t = torch.tensor(1.0, device=device)

    (eq_dual,) = layer(c_t, b_t)

    assert eq_dual.device.type == device

    # Verify with CVXPY
    c.value = c_t.cpu().numpy()
    b.value = b_t.cpu().numpy().item()
    prob.solve(solver=cp.CLARABEL)

    np.testing.assert_allclose(eq_dual.detach().cpu().numpy(), eq_con.dual_value, rtol=SOLUTION_RTOL, atol=SOLUTION_ATOL)


@pytest.mark.parametrize("device", get_device_params())
def test_batched_dual_variables_moreau(device):
    """Test dual variables with batched parameters using Moreau."""
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
        solver="MOREAU",
    )

    torch.manual_seed(42)
    c_t = torch.randn(batch_size, n, requires_grad=True, device=device)
    b_t = torch.ones(batch_size, requires_grad=True, device=device)

    x_opt, eq_dual = layer(c_t, b_t)

    assert x_opt.shape == (batch_size, n)
    assert eq_dual.shape == (batch_size,)
    assert x_opt.device.type == device

    # Verify each batch element
    for i in range(batch_size):
        c.value = c_t[i].detach().cpu().numpy()
        b.value = b_t[i].detach().cpu().numpy().item()
        prob.solve(solver=cp.CLARABEL)

        np.testing.assert_allclose(x_opt[i].detach().cpu().numpy(), x.value, rtol=1e-2, atol=1e-3)
        np.testing.assert_allclose(
            eq_dual[i].detach().cpu().numpy(), eq_con.dual_value, rtol=1e-2, atol=1e-3
        )


@pytest.mark.parametrize("device", get_device_params())
def test_vector_equality_dual_moreau(device):
    """Test dual variable for vector equality constraint with Moreau."""
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
        solver="MOREAU",
    )

    torch.manual_seed(42)
    A_t = torch.randn(m, n, requires_grad=True, device=device)
    b_t = torch.randn(m, requires_grad=True, device=device)

    x_opt, eq_dual = layer(A_t, b_t)

    assert eq_dual.shape == (m,)
    assert x_opt.device.type == device

    # Verify with CVXPY
    A.value = A_t.detach().cpu().numpy()
    b.value = b_t.detach().cpu().numpy()
    prob.solve(solver=cp.CLARABEL)

    np.testing.assert_allclose(x_opt.detach().cpu().numpy(), x.value, rtol=SOLUTION_RTOL, atol=SOLUTION_ATOL)
    np.testing.assert_allclose(eq_dual.detach().cpu().numpy(), eq_con.dual_value, rtol=SOLUTION_RTOL, atol=SOLUTION_ATOL)


# ============================================================================
# Backward Pass Tests (verify gradients through duals)
# ============================================================================


@pytest.mark.parametrize("device", get_device_params())
def test_dual_gradient_moreau(device):
    """Test gradient computation through dual variables with Moreau."""
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
        solver="MOREAU",
    )

    c_t = torch.tensor([1.0, 2.0], requires_grad=True, device=device)
    b_t = torch.tensor(1.0, requires_grad=True, device=device)

    x_opt, eq_dual = layer(c_t, b_t)

    loss = eq_dual.sum()
    loss.backward()

    assert c_t.grad is not None
    assert b_t.grad is not None
    assert torch.isfinite(c_t.grad).all()
    assert torch.isfinite(b_t.grad).all()


@pytest.mark.parametrize("device", get_device_params())
def test_dual_gradcheck_equality_moreau(device):
    """Rigorous gradient check for equality constraint dual using Moreau."""
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
        solver="MOREAU",
    )

    def f(c_t, b_t):
        x_opt, eq_dual = layer(c_t, b_t)
        return eq_dual

    c_t = torch.tensor([0.5, -0.3], requires_grad=True, device=device)
    b_t = torch.tensor(1.0, requires_grad=True, device=device)

    torch.autograd.gradcheck(f, (c_t, b_t), atol=1e-4, rtol=1e-3, nondet_tol=1e-5)


@pytest.mark.parametrize("device", get_device_params())
def test_dual_gradcheck_inequality_moreau(device):
    """Rigorous gradient check for inequality constraint dual using Moreau."""
    n = 2
    x = cp.Variable(n)
    c = cp.Parameter(n)

    ineq_con = x >= 0
    prob = cp.Problem(cp.Minimize(c @ x + 0.5 * cp.sum_squares(x)), [ineq_con])

    layer = CvxpyLayer(
        prob,
        parameters=[c],
        variables=[x, ineq_con.dual_variables[0]],
        solver="MOREAU",
    )

    def f(c_t):
        x_opt, ineq_dual = layer(c_t)
        return ineq_dual

    c_t = torch.tensor([1.0, -1.0], requires_grad=True, device=device)

    torch.autograd.gradcheck(f, (c_t,), atol=1e-4, rtol=1e-3, nondet_tol=1e-5)


@pytest.mark.parametrize("device", get_device_params())
def test_dual_gradcheck_mixed_moreau(device):
    """Rigorous gradient check for mixed primal and dual variables with Moreau."""
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
        solver="MOREAU",
    )

    def f(c_t, b_t):
        x_opt, eq_dual = layer(c_t, b_t)
        return x_opt.sum() + eq_dual

    c_t = torch.tensor([0.5, -0.3], requires_grad=True, device=device)
    b_t = torch.tensor(1.0, requires_grad=True, device=device)

    torch.autograd.gradcheck(f, (c_t, b_t), atol=1e-4, rtol=1e-3, nondet_tol=1e-5)


@pytest.mark.parametrize("device", get_device_params())
def test_dual_gradcheck_vector_equality_moreau(device):
    """Rigorous gradient check for vector equality constraint dual with Moreau."""
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
        solver="MOREAU",
    )

    def f(A_t, b_t):
        x_opt, eq_dual = layer(A_t, b_t)
        return eq_dual.sum()

    torch.manual_seed(42)
    A_t = torch.randn(m, n, requires_grad=True, device=device)
    b_t = torch.randn(m, requires_grad=True, device=device)

    torch.autograd.gradcheck(f, (A_t, b_t), atol=1e-4, rtol=1e-3, nondet_tol=1e-5)


# ============================================================================
# Cone-Specific Tests (based on Moreau's supported cones)
# ============================================================================


@pytest.mark.parametrize("device", get_device_params())
def test_exp_cone_constraint_dual_moreau(device):
    """Test dual variables for raw exponential cone constraint with Moreau."""
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
            x, y, z,
            exp_con.dual_variables[0],
            exp_con.dual_variables[1],
            exp_con.dual_variables[2],
        ],
        solver="MOREAU",
    )

    t_t = torch.tensor(5.0, requires_grad=True, device=device)

    x_opt, y_opt, z_opt, dual0, dual1, dual2 = layer(t_t)

    assert x_opt.device.type == device

    t.value = t_t.detach().cpu().numpy().item()
    prob.solve(solver=cp.CLARABEL)

    np.testing.assert_allclose(x_opt.detach().cpu().numpy(), x.value, rtol=SOLUTION_RTOL, atol=SOLUTION_ATOL)
    np.testing.assert_allclose(y_opt.detach().cpu().numpy(), y.value, rtol=SOLUTION_RTOL, atol=SOLUTION_ATOL)
    np.testing.assert_allclose(z_opt.detach().cpu().numpy(), z.value, rtol=SOLUTION_RTOL, atol=SOLUTION_ATOL)
    np.testing.assert_allclose(
        dual0.detach().cpu().numpy(), exp_con.dual_variables[0].value, rtol=SOLUTION_RTOL, atol=SOLUTION_ATOL
    )
    np.testing.assert_allclose(
        dual1.detach().cpu().numpy(), exp_con.dual_variables[1].value, rtol=SOLUTION_RTOL, atol=SOLUTION_ATOL
    )
    np.testing.assert_allclose(
        dual2.detach().cpu().numpy(), exp_con.dual_variables[2].value, rtol=SOLUTION_RTOL, atol=SOLUTION_ATOL
    )


@pytest.mark.parametrize("device", get_device_params())
def test_exp_cone_gradcheck_moreau(device):
    """Rigorous gradient check for raw exponential cone dual variables with Moreau."""
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
            x, y, z,
            exp_con.dual_variables[0],
            exp_con.dual_variables[1],
            exp_con.dual_variables[2],
        ],
        solver="MOREAU",
    )

    def f(t_t):
        x_opt, y_opt, z_opt, d0, d1, d2 = layer(t_t)
        return d0.sum() + d1.sum() + d2.sum()

    t_t = torch.tensor(5.0, requires_grad=True, device=device)

    torch.autograd.gradcheck(f, (t_t,), atol=1e-4, rtol=1e-3, nondet_tol=1e-5)


@pytest.mark.parametrize("device", get_device_params())
def test_pow_cone_constraint_dual_moreau(device):
    """Test dual variables for power cone constraint with Moreau."""
    x = cp.Variable()
    y = cp.Variable()
    z = cp.Variable()
    t = cp.Parameter(nonneg=True)

    pow_con = cp.PowCone3D(x, y, z, 0.5)
    prob = cp.Problem(
        cp.Minimize(-z + 0.1 * (x**2 + y**2)),
        [pow_con, x >= 0.1, y >= 0.1, x + y <= t],
    )

    layer = CvxpyLayer(
        prob,
        parameters=[t],
        variables=[
            x, y, z,
            pow_con.dual_variables[0],
            pow_con.dual_variables[1],
            pow_con.dual_variables[2],
        ],
        solver="MOREAU",
    )

    t_t = torch.tensor(4.0, requires_grad=True, device=device)

    x_opt, y_opt, z_opt, dual0, dual1, dual2 = layer(t_t)

    assert x_opt.device.type == device

    t.value = t_t.detach().cpu().numpy().item()
    prob.solve(solver=cp.CLARABEL)

    np.testing.assert_allclose(x_opt.detach().cpu().numpy(), x.value, rtol=SOLUTION_RTOL, atol=SOLUTION_ATOL)
    np.testing.assert_allclose(y_opt.detach().cpu().numpy(), y.value, rtol=SOLUTION_RTOL, atol=SOLUTION_ATOL)
    np.testing.assert_allclose(z_opt.detach().cpu().numpy(), z.value, rtol=SOLUTION_RTOL, atol=SOLUTION_ATOL)
    np.testing.assert_allclose(
        dual0.detach().cpu().numpy(), pow_con.dual_variables[0].value, rtol=SOLUTION_RTOL, atol=SOLUTION_ATOL
    )
    np.testing.assert_allclose(
        dual1.detach().cpu().numpy(), pow_con.dual_variables[1].value, rtol=SOLUTION_RTOL, atol=SOLUTION_ATOL
    )
    np.testing.assert_allclose(
        dual2.detach().cpu().numpy(), pow_con.dual_variables[2].value, rtol=SOLUTION_RTOL, atol=SOLUTION_ATOL
    )


@pytest.mark.parametrize("device", get_device_params())
def test_pow_cone_gradcheck_moreau(device):
    """Rigorous gradient check for power cone dual variables with Moreau.

    Uses a quadratic objective to find a natural point on the power cone boundary,
    avoiding degenerate duals that arise when variables are pinned by equality constraints.
    The problem projects a target point onto the power cone, giving well-defined
    and continuously varying primal and dual solutions.
    """
    x = cp.Variable()
    y = cp.Variable()
    z = cp.Variable()

    # Target point parameters - chosen to be outside the cone so solution is on boundary
    x_target = cp.Parameter(nonneg=True)
    y_target = cp.Parameter(nonneg=True)
    z_target = cp.Parameter()

    # Power cone: x^alpha * y^(1-alpha) >= |z|, x >= 0, y >= 0
    alpha = 0.4
    pow_con = cp.PowCone3D(x, y, z, alpha)

    # Project target point onto power cone (quadratic objective)
    prob = cp.Problem(
        cp.Minimize((x - x_target) ** 2 + (y - y_target) ** 2 + (z - z_target) ** 2),
        [pow_con],
    )

    layer = CvxpyLayer(
        prob,
        parameters=[x_target, y_target, z_target],
        variables=[
            x, y, z,
            pow_con.dual_variables[0],
            pow_con.dual_variables[1],
            pow_con.dual_variables[2],
        ],
        solver="MOREAU",
        solver_args={"eps_abs": 1e-10, "eps_rel": 1e-10},
    )

    def f(x_t, y_t, z_t):
        x_opt, y_opt, z_opt, d0, d1, d2 = layer(x_t, y_t, z_t)
        # Return weighted sum to test all dual gradients
        return 0.5 * d0.sum() + 0.3 * d1.sum() + 0.2 * d2.sum()

    # Target outside cone: z_target > x_target^alpha * y_target^(1-alpha)
    # With x=1, y=1, alpha=0.4: boundary is z = 1^0.4 * 1^0.6 = 1
    # So z_target=1.5 is outside, forcing projection onto boundary
    x_t = torch.tensor(1.0, requires_grad=True, dtype=torch.float64, device=device)
    y_t = torch.tensor(1.0, requires_grad=True, dtype=torch.float64, device=device)
    z_t = torch.tensor(1.5, requires_grad=True, dtype=torch.float64, device=device)

    # Use larger eps for finite differences since power cone solutions need high precision
    torch.autograd.gradcheck(f, (x_t, y_t, z_t), eps=1e-4, atol=1e-3, rtol=1e-2, nondet_tol=1e-5)


# ============================================================================
# SOC Tests (Moreau only supports 3D SOCs)
# ============================================================================


@pytest.mark.parametrize("device", get_device_params())
def test_soc_constraint_dual_moreau(device):
    """Test dual variable for second-order cone constraint with Moreau."""
    n = 2
    x = cp.Variable(n)
    c = cp.Parameter(n)
    t = cp.Parameter(nonneg=True)

    soc_con = cp.norm(x) <= t
    prob = cp.Problem(cp.Minimize(c @ x), [soc_con])

    layer = CvxpyLayer(
        prob,
        parameters=[c, t],
        variables=[x, soc_con.dual_variables[0]],
        solver="MOREAU",
    )

    c_t = torch.tensor([1.0, 0.5], requires_grad=True, device=device)
    t_t = torch.tensor(2.0, requires_grad=True, device=device)

    x_opt, soc_dual = layer(c_t, t_t)

    assert x_opt.device.type == device

    c.value = c_t.detach().cpu().numpy()
    t.value = t_t.detach().cpu().numpy().item()
    prob.solve(solver=cp.CLARABEL)

    np.testing.assert_allclose(x_opt.detach().cpu().numpy(), x.value, rtol=SOLUTION_RTOL, atol=SOLUTION_ATOL)
    np.testing.assert_allclose(soc_dual.detach().cpu().numpy(), soc_con.dual_value, rtol=SOLUTION_RTOL, atol=SOLUTION_ATOL)


@pytest.mark.parametrize("device", get_device_params())
def test_soc_gradcheck_moreau(device):
    """Rigorous gradient check for SOC constraint dual with Moreau."""
    n = 2
    x = cp.Variable(n)
    c = cp.Parameter(n)
    t = cp.Parameter(nonneg=True)

    soc_con = cp.norm(x) <= t
    prob = cp.Problem(cp.Minimize(c @ x + 0.1 * cp.sum_squares(x)), [soc_con])

    layer = CvxpyLayer(
        prob,
        parameters=[c, t],
        variables=[x, soc_con.dual_variables[0]],
        solver="MOREAU",
    )

    def f(c_t, t_t):
        x_opt, soc_dual = layer(c_t, t_t)
        return soc_dual.sum()

    c_t = torch.tensor([0.5, 0.3], requires_grad=True, device=device)
    t_t = torch.tensor(2.0, requires_grad=True, device=device)

    torch.autograd.gradcheck(f, (c_t, t_t), atol=1e-4, rtol=1e-3, nondet_tol=1e-5)


def test_soc_explicit_multi_dual_moreau():
    """Test that SOC dual variables raise ValueError with Moreau solver."""
    n = 2
    x = cp.Variable(n)
    t = cp.Variable()
    c = cp.Parameter(n)
    t_param = cp.Parameter(nonneg=True)

    soc_con = cp.SOC(t, x)
    prob = cp.Problem(cp.Minimize(c @ x - t), [soc_con, t <= t_param])

    with pytest.raises(ValueError, match="SOC dual variables are not supported with the Moreau"):
        CvxpyLayer(
            prob,
            parameters=[c, t_param],
            variables=[x, t, soc_con.dual_variables[0], soc_con.dual_variables[1]],
            solver="MOREAU",
        )


def test_soc_explicit_multi_dual_gradcheck_moreau():
    """Test that SOC dual variables raise ValueError with Moreau (gradcheck version)."""
    n = 2
    x = cp.Variable(n)
    t = cp.Variable()
    c = cp.Parameter(n)
    t_param = cp.Parameter(nonneg=True)

    soc_con = cp.SOC(t, x)
    prob = cp.Problem(cp.Minimize(c @ x - t + 0.1 * cp.sum_squares(x)), [soc_con, t <= t_param])

    with pytest.raises(ValueError, match="SOC dual variables are not supported with the Moreau"):
        CvxpyLayer(
            prob,
            parameters=[c, t_param],
            variables=[x, t, soc_con.dual_variables[0], soc_con.dual_variables[1]],
            solver="MOREAU",
        )


# ============================================================================
# Mixed Cone Tests
# ============================================================================


@pytest.mark.parametrize("device", get_device_params())
def test_mixed_cones_moreau(device):
    """Test dual variables from multiple cone types in a single problem with Moreau."""
    x_eq = cp.Variable(2)
    x_ineq = cp.Variable(2)
    x_exp = cp.Variable()
    y_exp = cp.Variable()
    z_exp = cp.Variable()
    x_pow = cp.Variable()
    y_pow = cp.Variable()
    z_pow = cp.Variable()

    b_eq = cp.Parameter(2)
    ub = cp.Parameter(2)
    t_param = cp.Parameter(nonneg=True)

    eq_con = x_eq == b_eq
    ineq_con = x_ineq <= ub
    exp_con = cp.constraints.ExpCone(x_exp, y_exp, z_exp)
    pow_con = cp.PowCone3D(x_pow, y_pow, z_pow, 0.5)

    obj = (
        cp.sum_squares(x_eq)
        + cp.sum_squares(x_ineq)
        - z_exp
        + 0.1 * (x_exp**2 + y_exp**2 + z_exp**2)
        - z_pow
        + 0.1 * (x_pow**2 + y_pow**2)
    )

    prob = cp.Problem(
        cp.Minimize(obj),
        [
            eq_con, ineq_con, exp_con,
            y_exp >= 0.1, z_exp <= t_param,
            pow_con, x_pow >= 0.1, y_pow >= 0.1, x_pow + y_pow <= t_param,
        ],
    )

    layer = CvxpyLayer(
        prob,
        parameters=[b_eq, ub, t_param],
        variables=[
            x_eq, x_ineq, z_exp, z_pow,
            eq_con.dual_variables[0],
            ineq_con.dual_variables[0],
            exp_con.dual_variables[0],
            exp_con.dual_variables[1],
            exp_con.dual_variables[2],
            pow_con.dual_variables[0],
            pow_con.dual_variables[1],
            pow_con.dual_variables[2],
        ],
        solver="MOREAU",
    )

    b_eq_t = torch.tensor([0.5, -0.3], requires_grad=True, device=device)
    ub_t = torch.tensor([1.0, 1.0], requires_grad=True, device=device)
    t_t = torch.tensor(3.0, requires_grad=True, device=device)

    results = layer(b_eq_t, ub_t, t_t)

    (
        x_eq_opt, x_ineq_opt, z_exp_opt, z_pow_opt,
        eq_dual, ineq_dual,
        exp_dual0, exp_dual1, exp_dual2,
        pow_dual0, pow_dual1, pow_dual2,
    ) = results

    assert x_eq_opt.device.type == device

    b_eq.value = b_eq_t.detach().cpu().numpy()
    ub.value = ub_t.detach().cpu().numpy()
    t_param.value = t_t.detach().cpu().numpy().item()
    prob.solve(solver=cp.CLARABEL)

    np.testing.assert_allclose(x_eq_opt.detach().cpu().numpy(), x_eq.value, rtol=SOLUTION_RTOL, atol=SOLUTION_ATOL)
    np.testing.assert_allclose(x_ineq_opt.detach().cpu().numpy(), x_ineq.value, rtol=SOLUTION_RTOL, atol=SOLUTION_ATOL)
    np.testing.assert_allclose(z_exp_opt.detach().cpu().numpy(), z_exp.value, rtol=SOLUTION_RTOL, atol=SOLUTION_ATOL)
    np.testing.assert_allclose(z_pow_opt.detach().cpu().numpy(), z_pow.value, rtol=SOLUTION_RTOL, atol=SOLUTION_ATOL)
    np.testing.assert_allclose(eq_dual.detach().cpu().numpy(), eq_con.dual_variables[0].value, rtol=SOLUTION_RTOL, atol=SOLUTION_ATOL)
    np.testing.assert_allclose(ineq_dual.detach().cpu().numpy(), ineq_con.dual_variables[0].value, rtol=SOLUTION_RTOL, atol=SOLUTION_ATOL)
    np.testing.assert_allclose(exp_dual0.detach().cpu().numpy(), exp_con.dual_variables[0].value, rtol=SOLUTION_RTOL, atol=SOLUTION_ATOL)
    np.testing.assert_allclose(exp_dual1.detach().cpu().numpy(), exp_con.dual_variables[1].value, rtol=SOLUTION_RTOL, atol=SOLUTION_ATOL)
    np.testing.assert_allclose(exp_dual2.detach().cpu().numpy(), exp_con.dual_variables[2].value, rtol=SOLUTION_RTOL, atol=SOLUTION_ATOL)
    np.testing.assert_allclose(pow_dual0.detach().cpu().numpy(), pow_con.dual_variables[0].value, rtol=SOLUTION_RTOL, atol=SOLUTION_ATOL)
    np.testing.assert_allclose(pow_dual1.detach().cpu().numpy(), pow_con.dual_variables[1].value, rtol=SOLUTION_RTOL, atol=SOLUTION_ATOL)
    np.testing.assert_allclose(pow_dual2.detach().cpu().numpy(), pow_con.dual_variables[2].value, rtol=SOLUTION_RTOL, atol=SOLUTION_ATOL)


# ============================================================================
# JAX Dual Tests
# ============================================================================


def test_jax_equality_dual_moreau():
    """Test dual variable for equality constraint with Moreau in JAX."""
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    from cvxpylayers.jax import CvxpyLayer as JaxCvxpyLayer

    n = 2
    x = cp.Variable(n)
    c = cp.Parameter(n)
    b = cp.Parameter()

    eq_con = cp.sum(x) == b
    prob = cp.Problem(cp.Minimize(c @ x + 0.5 * cp.sum_squares(x)), [eq_con])

    layer = JaxCvxpyLayer(
        prob,
        parameters=[c, b],
        variables=[x, eq_con.dual_variables[0]],
        solver="MOREAU",
    )

    c_val = jnp.array([1.0, 2.0])
    b_val = jnp.array(1.0)

    x_opt, eq_dual = layer(c_val, b_val)

    c.value = np.array(c_val)
    b.value = float(b_val)
    prob.solve(solver=cp.CLARABEL)

    np.testing.assert_allclose(np.array(x_opt), x.value, rtol=SOLUTION_RTOL, atol=SOLUTION_ATOL)
    np.testing.assert_allclose(np.array(eq_dual), eq_con.dual_value, rtol=SOLUTION_RTOL, atol=SOLUTION_ATOL)


def test_jax_dual_gradient_moreau():
    """Test JAX autodiff through dual variable with Moreau."""
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    from cvxpylayers.jax import CvxpyLayer as JaxCvxpyLayer

    n = 2
    x = cp.Variable(n)
    c = cp.Parameter(n)
    b = cp.Parameter()

    eq_con = cp.sum(x) == b
    prob = cp.Problem(cp.Minimize(c @ x + 0.5 * cp.sum_squares(x)), [eq_con])

    layer = JaxCvxpyLayer(
        prob,
        parameters=[c, b],
        variables=[x, eq_con.dual_variables[0]],
        solver="MOREAU",
    )

    def f(c_val, b_val):
        x_opt, eq_dual = layer(c_val, b_val)
        return jnp.sum(eq_dual)

    c_val = jnp.array([0.5, -0.3])
    b_val = jnp.array(1.0)

    grad_c, grad_b = jax.grad(f, argnums=(0, 1))(c_val, b_val)

    assert jnp.isfinite(grad_c).all(), f"grad_c is not finite: {grad_c}"
    assert jnp.isfinite(grad_b), f"grad_b is not finite: {grad_b}"


def test_jax_inequality_dual_moreau():
    """Test dual variable for inequality constraint with Moreau in JAX."""
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    from cvxpylayers.jax import CvxpyLayer as JaxCvxpyLayer

    n = 2
    x = cp.Variable(n)
    c = cp.Parameter(n)

    ineq_con = x >= 0
    prob = cp.Problem(cp.Minimize(c @ x + cp.sum_squares(x)), [ineq_con])

    layer = JaxCvxpyLayer(
        prob,
        parameters=[c],
        variables=[x, ineq_con.dual_variables[0]],
        solver="MOREAU",
    )

    c_val = jnp.array([1.0, -1.0])

    x_opt, ineq_dual = layer(c_val)

    c.value = np.array(c_val)
    prob.solve(solver=cp.CLARABEL)

    np.testing.assert_allclose(np.array(x_opt), x.value, rtol=SOLUTION_RTOL, atol=SOLUTION_ATOL)
    np.testing.assert_allclose(np.array(ineq_dual), ineq_con.dual_value, rtol=SOLUTION_RTOL, atol=SOLUTION_ATOL)


def test_jax_jit_dual_moreau():
    """Test jax.jit with dual variables using Moreau solver."""
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    from cvxpylayers.jax import CvxpyLayer as JaxCvxpyLayer

    n = 2
    x = cp.Variable(n)
    c = cp.Parameter(n)
    b = cp.Parameter()

    eq_con = cp.sum(x) == b
    prob = cp.Problem(cp.Minimize(c @ x + 0.5 * cp.sum_squares(x)), [eq_con])

    layer = JaxCvxpyLayer(
        prob,
        parameters=[c, b],
        variables=[x, eq_con.dual_variables[0]],
        solver="MOREAU",
    )

    @jax.jit
    def solve_and_sum_dual(c_val, b_val):
        x_opt, eq_dual = layer(c_val, b_val)
        return jnp.sum(eq_dual)

    c_val = jnp.array([0.5, -0.3])
    b_val = jnp.array(1.0)

    # Test JIT forward
    result = solve_and_sum_dual(c_val, b_val)
    assert jnp.isfinite(result)

    # Test JIT gradient
    grad_c, grad_b = jax.grad(solve_and_sum_dual, argnums=(0, 1))(c_val, b_val)
    assert jnp.isfinite(grad_c).all()
    assert jnp.isfinite(grad_b)


def test_jax_vmap_dual_moreau():
    """Test jax.vmap with dual variables using Moreau solver."""
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    from cvxpylayers.jax import CvxpyLayer as JaxCvxpyLayer

    n = 2
    batch_size = 3
    x = cp.Variable(n)
    c = cp.Parameter(n)
    b = cp.Parameter()

    eq_con = cp.sum(x) == b
    prob = cp.Problem(cp.Minimize(c @ x + 0.5 * cp.sum_squares(x)), [eq_con])

    layer = JaxCvxpyLayer(
        prob,
        parameters=[c, b],
        variables=[x, eq_con.dual_variables[0]],
        solver="MOREAU",
    )

    def solve_single(c_val, b_val):
        x_opt, eq_dual = layer(c_val, b_val)
        return x_opt, eq_dual

    # Batched inputs
    key = jax.random.PRNGKey(42)
    c_batch = jax.random.normal(key, (batch_size, n))
    b_batch = jnp.ones(batch_size)

    # Test vmap
    x_batch, dual_batch = jax.vmap(solve_single)(c_batch, b_batch)

    assert x_batch.shape == (batch_size, n)
    assert dual_batch.shape == (batch_size,)

    # Verify each element against CVXPY
    for i in range(batch_size):
        c.value = np.array(c_batch[i])
        b.value = float(b_batch[i])
        prob.solve(solver=cp.CLARABEL)

        np.testing.assert_allclose(np.array(x_batch[i]), x.value, rtol=1e-2, atol=1e-3)
        np.testing.assert_allclose(np.array(dual_batch[i]), eq_con.dual_value, rtol=1e-2, atol=1e-3)


# ============================================================================
# torch.compile Tests
# ============================================================================


@pytest.fixture
def reset_dynamo():
    """Reset torch.compile cache between tests."""
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


@pytest.mark.parametrize("device", get_device_params())
def test_torch_compile_dual_moreau(device, reset_dynamo):
    """Test torch.compile with dual variables using Moreau solver."""
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
        solver="MOREAU",
    )

    @torch.compile
    def solve_and_sum_dual(c_t, b_t):
        x_opt, eq_dual = layer(c_t, b_t)
        return eq_dual.sum()

    c_t = torch.tensor([0.5, -0.3], requires_grad=True, device=device)
    b_t = torch.tensor(1.0, requires_grad=True, device=device)

    # Test compiled forward
    result = solve_and_sum_dual(c_t, b_t)
    assert torch.isfinite(result)

    # Test compiled backward
    result.backward()
    assert c_t.grad is not None
    assert b_t.grad is not None
    assert torch.isfinite(c_t.grad).all()
    assert torch.isfinite(b_t.grad)


# ============================================================================
# Comparison with DIFFCP
# ============================================================================


@pytest.mark.parametrize("device", get_device_params())
def test_dual_values_match_diffcp(device):
    """Test that Moreau dual values match DIFFCP for the same problem."""
    n = 3
    x = cp.Variable(n)
    A = cp.Parameter((2, n))
    b = cp.Parameter(2)

    eq_con = A @ x == b
    prob = cp.Problem(cp.Minimize(cp.sum_squares(x)), [eq_con])

    torch.manual_seed(42)
    A_t = torch.randn(2, n, requires_grad=True, device=device)
    b_t = torch.randn(2, requires_grad=True, device=device)

    layer_moreau = CvxpyLayer(
        prob,
        parameters=[A, b],
        variables=[x, eq_con.dual_variables[0]],
        solver="MOREAU",
    )
    layer_diffcp = CvxpyLayer(
        prob,
        parameters=[A, b],
        variables=[x, eq_con.dual_variables[0]],
        solver="DIFFCP",
    )

    x_moreau, dual_moreau = layer_moreau(A_t, b_t)
    # DIFFCP only works on CPU
    x_diffcp, dual_diffcp = layer_diffcp(A_t.cpu().clone(), b_t.cpu().clone())

    np.testing.assert_allclose(
        x_moreau.detach().cpu().numpy(), x_diffcp.detach().numpy(), rtol=SOLUTION_RTOL, atol=SOLUTION_ATOL
    )
    np.testing.assert_allclose(
        dual_moreau.detach().cpu().numpy(), dual_diffcp.detach().numpy(), rtol=SOLUTION_RTOL, atol=SOLUTION_ATOL
    )


@pytest.mark.parametrize("device", get_device_params())
def test_dual_gradients_match_diffcp(device):
    """Test that Moreau dual gradients match DIFFCP for the same problem."""
    n = 2
    x = cp.Variable(n)
    c = cp.Parameter(n)
    b = cp.Parameter()

    eq_con = cp.sum(x) == b
    prob = cp.Problem(cp.Minimize(c @ x + 0.5 * cp.sum_squares(x)), [eq_con])

    c_moreau = torch.tensor([0.5, -0.3], requires_grad=True, device=device)
    b_moreau = torch.tensor(1.0, requires_grad=True, device=device)
    c_diffcp = torch.tensor([0.5, -0.3], requires_grad=True)  # DIFFCP is CPU-only
    b_diffcp = torch.tensor(1.0, requires_grad=True)

    layer_moreau = CvxpyLayer(
        prob,
        parameters=[c, b],
        variables=[x, eq_con.dual_variables[0]],
        solver="MOREAU",
    )
    layer_diffcp = CvxpyLayer(
        prob,
        parameters=[c, b],
        variables=[x, eq_con.dual_variables[0]],
        solver="DIFFCP",
    )

    _, dual_moreau = layer_moreau(c_moreau, b_moreau)
    dual_moreau.sum().backward()

    _, dual_diffcp = layer_diffcp(c_diffcp, b_diffcp)
    dual_diffcp.sum().backward()

    np.testing.assert_allclose(
        c_moreau.grad.cpu().numpy(), c_diffcp.grad.numpy(), rtol=SOLUTION_RTOL, atol=SOLUTION_ATOL
    )
    np.testing.assert_allclose(
        b_moreau.grad.cpu().numpy(), b_diffcp.grad.numpy(), rtol=SOLUTION_RTOL, atol=SOLUTION_ATOL
    )
