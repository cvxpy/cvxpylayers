"""Test suite for the cuOpt solver backend (LP only)."""

import cvxpy as cp
import numpy as np
import pytest
import torch
from cvxpy.error import SolverError

from cvxpylayers.torch import CvxpyLayer

pytest.importorskip("cuopt")

torch.set_default_dtype(torch.double)


def compare_solvers(problem, params, param_vals, variables, atol=1e-3):
    """Compare CUOPT vs DIFFCP and CVXPY direct solve."""
    for param, val in zip(params, param_vals, strict=True):
        param.value = val
    problem.solve()
    assert problem.status == "optimal"
    true_sol = [v.value for v in variables]

    layer_diffcp = CvxpyLayer(problem, params, variables, solver="DIFFCP")
    sols_diffcp = layer_diffcp(*[torch.tensor(v, requires_grad=True) for v in param_vals])

    layer_cuopt = CvxpyLayer(problem, params, variables, solver="CUOPT")
    sols_cuopt = layer_cuopt(*[torch.tensor(v, requires_grad=True) for v in param_vals])

    for i, (s_cuopt, s_diffcp, s_true) in enumerate(
        zip(sols_cuopt, sols_diffcp, true_sol, strict=True)
    ):
        cuopt_err = np.linalg.norm(s_cuopt.detach().cpu().numpy().squeeze() - s_true)
        diffcp_err = np.linalg.norm(s_diffcp.detach().cpu().numpy().squeeze() - s_true)
        assert cuopt_err < atol, f"var {i}: ||CUOPT - true|| = {cuopt_err:.6e}"
        assert diffcp_err < atol, f"var {i}: ||DIFFCP - true|| = {diffcp_err:.6e}"


def test_small_random_lp():
    n, m = 4, 3
    x = cp.Variable(n)
    c = cp.Parameter(n)
    A = cp.Parameter((m, n))
    b = cp.Parameter(m)
    problem = cp.Problem(cp.Minimize(c @ x), [A @ x <= b, x >= 0, x <= 10])
    rng = np.random.default_rng(0)
    compare_solvers(
        problem,
        [c, A, b],
        [np.abs(rng.standard_normal(n)), rng.standard_normal((m, n)), np.ones(m) * 5],
        [x],
    )


def test_equality_and_inequality_lp():
    n = 3
    x = cp.Variable(n)
    c = cp.Parameter(n)
    A = cp.Parameter((1, n))
    b = cp.Parameter(1)
    G = cp.Parameter((2, n))
    h = cp.Parameter(2)
    problem = cp.Problem(cp.Minimize(c @ x), [A @ x == b, G @ x >= h, x >= -5, x <= 5])
    compare_solvers(
        problem,
        [c, A, b, G, h],
        [
            np.array([1.0, -1.0, 0.5]),
            np.array([[1.0, 1.0, 1.0]]),
            np.array([3.0]),
            np.eye(2, 3),
            np.zeros(2),
        ],
        [x],
    )


def test_pure_inequality_lp():
    n = 3
    x = cp.Variable(n)
    c = cp.Parameter(n)
    problem = cp.Problem(cp.Minimize(c @ x), [x >= 0, x <= 1])
    compare_solvers(problem, [c], [np.array([-1.0, 0.5, -0.3])], [x])


def test_batched_lp_forward():
    n = 3
    batch = 3
    x = cp.Variable(n)
    c = cp.Parameter(n)
    b = cp.Parameter(2)
    A_const = np.array([[1.0, 1.0, 1.0], [1.0, -1.0, 0.0]])
    problem = cp.Problem(cp.Minimize(c @ x), [A_const @ x == b, x >= 0, x <= 10])

    rng = np.random.default_rng(5)
    c_b = np.abs(rng.standard_normal((batch, n)))
    b_b = np.tile(np.array([3.0, 0.0]), (batch, 1))

    layer = CvxpyLayer(problem, [c, b], [x], solver="CUOPT")
    layer_d = CvxpyLayer(problem, [c, b], [x], solver="DIFFCP")

    (x_c,) = layer(torch.tensor(c_b), torch.tensor(b_b))
    (x_d,) = layer_d(torch.tensor(c_b), torch.tensor(b_b))
    assert x_c.shape == (batch, n)
    diff = torch.norm(x_c.cpu() - x_d).item()
    assert diff < 1e-3


def test_mixed_batched_unbatched():
    n = 3
    batch = 3
    x = cp.Variable(n)
    c = cp.Parameter(n)
    b = cp.Parameter(2)
    A_const = np.array([[1.0, 1.0, 1.0], [1.0, -1.0, 0.0]])
    problem = cp.Problem(cp.Minimize(c @ x), [A_const @ x == b, x >= 0, x <= 10])

    c_val = np.array([0.2, 0.5, 0.8])
    b_b = np.tile(np.array([3.0, 0.0]), (batch, 1))

    layer = CvxpyLayer(problem, [c, b], [x], solver="CUOPT")
    (x_sol,) = layer(torch.tensor(c_val), torch.tensor(b_b))
    assert x_sol.shape == (batch, n)


def test_qp_rejected():
    # cuOpt's cvxpy conic interface does not support quadratic objectives,
    # so cvxpy rejects the problem at canonicalization time.
    n = 3
    x = cp.Variable(n)
    target = cp.Parameter(n)
    problem = cp.Problem(cp.Minimize(cp.sum_squares(x - target)), [x >= -1, x <= 1])
    with pytest.raises(SolverError):
        CvxpyLayer(problem, [target], [x], solver="CUOPT")


def test_socp_rejected():
    n = 3
    x = cp.Variable(n)
    bound = cp.Parameter(n)
    problem = cp.Problem(cp.Minimize(cp.norm(x, 2)), [x >= bound])
    with pytest.raises((ValueError, SolverError), match="(?i)cuopt|cone|nonneg|zero|lp"):
        CvxpyLayer(problem, [bound], [x], solver="CUOPT")


def test_exp_cone_rejected():
    x = cp.Variable()
    bound = cp.Parameter()
    problem = cp.Problem(cp.Minimize(-cp.log(x)), [x >= bound])
    with pytest.raises((ValueError, SolverError), match="(?i)cuopt|cone|lp"):
        CvxpyLayer(problem, [bound], [x], solver="CUOPT")


def test_psd_rejected():
    X = cp.Variable((3, 3), PSD=True)
    problem = cp.Problem(cp.Minimize(cp.trace(X)))
    with pytest.raises((ValueError, SolverError)):
        CvxpyLayer(problem, [], [X], solver="CUOPT")


def test_backward_raises_with_moreau_hint():
    """CUOPT is forward-only; backward must raise and mention MOREAU."""
    n = 3
    x = cp.Variable(n)
    c = cp.Parameter(n)
    problem = cp.Problem(cp.Minimize(c @ x), [x >= 0, x <= 1])
    layer = CvxpyLayer(problem, [c], [x], solver="CUOPT")
    c_v = torch.tensor([-1.0, 0.5, -0.3], requires_grad=True)
    (sol,) = layer(c_v)
    with pytest.raises(NotImplementedError, match="MOREAU"):
        sol.sum().backward()


def test_solver_args_plumbing():
    """solver_args reach cuOpt's SolverSettings without error."""
    n = 3
    x = cp.Variable(n)
    c = cp.Parameter(n)
    problem = cp.Problem(cp.Minimize(c @ x), [x >= 0, x <= 1])
    layer = CvxpyLayer(problem, [c], [x], solver="CUOPT")
    (sol,) = layer(
        torch.tensor([-1.0, 0.5, -0.3]),
        solver_args={"optimality_tolerance": 1e-6},
    )
    assert sol.shape == (n,)

    with pytest.raises(ValueError, match="Unknown CUOPT solver_args"):
        layer(torch.tensor([-1.0, 0.5, -0.3]), solver_args={"not_a_real_option": 1})


def test_equality_dual_value_matches_cvxpy():
    """Forward dual of an equality constraint matches cvxpy's solution."""
    n = 3
    x = cp.Variable(n)
    c = cp.Parameter(n)
    A_const = np.array([[1.0, 1.0, 1.0]])
    b_const = np.array([3.0])
    eq_con = A_const @ x == b_const
    problem = cp.Problem(cp.Minimize(c @ x), [eq_con, x >= 0, x <= 5])

    c_val = np.array([1.0, 2.0, 3.0])
    c.value = c_val
    problem.solve()
    true_x = x.value
    true_dual = eq_con.dual_variables[0].value

    layer = CvxpyLayer(problem, [c], [x, eq_con.dual_variables[0]], solver="CUOPT")
    x_out, d_out = layer(torch.tensor(c_val))
    assert np.allclose(x_out.detach().cpu().numpy(), true_x, atol=1e-4)
    assert np.allclose(d_out.detach().cpu().numpy(), true_dual, atol=1e-4)
