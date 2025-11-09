"""Comprehensive test suite for MPAX solver."""

import cvxpy as cp
import numpy as np
import pytest
import torch
from cvxpy.error import SolverError

from cvxpylayers.torch import CvxpyLayer

# Skip all tests in this module if mpax is not installed
pytest.importorskip("mpax")
pytest.importorskip("jax")

torch.set_default_dtype(torch.double)


def compare_solvers(problem, params, param_vals, variables):
    """Compare MPAX vs DIFFCP and CVXPY direct solve."""
    # Set parameter values for direct solve
    for param, val in zip(params, param_vals, strict=True):
        param.value = val

    # Ground truth: CVXPY direct solve
    problem.solve()
    assert problem.status == "optimal", f"CVXPY failed to solve: {problem.status}"

    true_sol = [v.value for v in variables]
    true_obj = problem.value

    # Convert to torch tensors
    param_tensors = [torch.tensor(v, requires_grad=True) for v in param_vals]

    # Test DIFFCP
    layer_diffcp = CvxpyLayer(problem, params, variables, solver="DIFFCP")
    sols_diffcp = layer_diffcp(*param_tensors)

    # Recompute objective using CVXPY's expression
    for param, val in zip(params, param_vals, strict=True):
        param.value = val
    for var, sol in zip(variables, sols_diffcp, strict=True):
        var.value = sol.detach().numpy()
    diffcp_obj = problem.objective.value

    # Test MPAX
    layer_mpax = CvxpyLayer(problem, params, variables, solver="MPAX")
    sols_mpax = layer_mpax(*[torch.tensor(v, requires_grad=True) for v in param_vals])

    # Recompute objective
    for param, val in zip(params, param_vals, strict=True):
        param.value = val
    for var, sol in zip(variables, sols_mpax, strict=True):
        var.value = sol.detach().numpy()
    mpax_obj = problem.objective.value

    # Compare objectives
    obj_err = abs(mpax_obj - true_obj)
    diffcp_vs_mpax = abs(mpax_obj - diffcp_obj)

    assert obj_err < 1e-3, f"MPAX error={obj_err:.6f}"
    assert diffcp_vs_mpax < 1e-3, f"diff from DIFFCP={diffcp_vs_mpax:.6f}"

    # Compare primal solutions
    for i, (sol_mpax, sol_diffcp, sol_true) in enumerate(
        zip(sols_mpax, sols_diffcp, true_sol, strict=True)
    ):
        # Compare DIFFCP vs ground truth
        diffcp_err = np.linalg.norm(sol_diffcp.detach().numpy() - sol_true)
        assert diffcp_err < 1e-3, f"DIFFCP var {i} error: ||DIFFCP - true|| = {diffcp_err:.6e}"

        # Compare MPAX vs ground truth
        mpax_err = np.linalg.norm(sol_mpax.detach().numpy() - sol_true)
        assert mpax_err < 1e-3, f"MPAX var {i} error: ||MPAX - true|| = {mpax_err:.6e}"

        # Compare MPAX vs DIFFCP
        primal_diff = torch.norm(sol_mpax - sol_diffcp).item()
        assert primal_diff < 1e-3, (
            f"Primal variable {i} differs: ||MPAX - DIFFCP|| = {primal_diff:.6e}"
        )


def test_equality_only():
    """Test with only equality constraints."""
    # minimize x^T x subject to Ax = b
    n, m = 5, 2
    x = cp.Variable(n)
    A = cp.Parameter((m, n))
    b = cp.Parameter(m)

    problem = cp.Problem(cp.Minimize(cp.sum_squares(x)), [A @ x == b])

    np.random.seed(100)
    A_val = np.random.randn(m, n)
    b_val = np.random.randn(m)

    compare_solvers(problem, [A, b], [A_val, b_val], [x])


def test_inequality_only():
    """Test with only inequality constraints."""
    # minimize (x-1)^2 subject to x >= a
    x = cp.Variable(1)
    a = cp.Parameter(1)

    problem = cp.Problem(cp.Minimize(cp.sum_squares(x - 1)), [x >= a])

    a_val = np.array([0.5])

    compare_solvers(problem, [a], [a_val], [x])


def test_mixed_constraints():
    """Test with both equality and inequality constraints."""
    # minimize x^T x subject to Ax = b, Gx >= h
    n = 3
    x = cp.Variable(n)
    A = cp.Parameter((1, n))
    b = cp.Parameter(1)
    G = cp.Parameter((2, n))
    h = cp.Parameter(2)

    problem = cp.Problem(cp.Minimize(cp.sum_squares(x)), [A @ x == b, G @ x >= h])

    np.random.seed(200)
    A_val = np.array([[1.0, 1.0, 1.0]])  # sum(x) = 3
    b_val = np.array([3.0])
    G_val = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])  # x[0] >= 0, x[1] >= 0
    h_val = np.array([0.0, 0.0])

    compare_solvers(problem, [A, b, G, h], [A_val, b_val, G_val, h_val], [x])


def test_box_constraints():
    """Test with box constraints (variable bounds)."""
    # minimize (x-2)^T(x-2) subject to 0 <= x <= 1
    n = 3
    x = cp.Variable(n)
    target = cp.Parameter(n)

    problem = cp.Problem(cp.Minimize(cp.sum_squares(x - target)), [x >= 0, x <= 1])

    target_val = np.array([2.0, 0.5, -1.0])

    compare_solvers(problem, [target], [target_val], [x])


def test_qp_with_linear_objective():
    """Test QP with linear + quadratic objective."""
    # minimize x^T P x + q^T x subject to Ax = b
    n = 4
    x = cp.Variable(n)
    P_const = np.eye(n) * 2  # Constant P matrix
    q = cp.Parameter(n)
    A = cp.Parameter((1, n))
    b = cp.Parameter(1)

    problem = cp.Problem(cp.Minimize(0.5 * cp.quad_form(x, P_const) + q @ x), [A @ x == b])

    np.random.seed(300)
    q_val = np.random.randn(n)
    A_val = np.ones((1, n))
    b_val = np.array([2.0])

    compare_solvers(problem, [q, A, b], [q_val, A_val, b_val], [x])


def test_least_squares_with_constraints():
    """Test least squares with constraints (original failing test)."""
    # minimize ||Ax - b||^2 + ||x||^2
    n, m = 3, 4
    x = cp.Variable(n)
    A = cp.Parameter((m, n))
    b = cp.Parameter(m)

    problem = cp.Problem(cp.Minimize(cp.sum_squares(A @ x - b) + cp.sum_squares(x)))

    np.random.seed(42)
    A_val = np.random.randn(m, n)
    b_val = np.random.randn(m)

    compare_solvers(problem, [A, b], [A_val, b_val], [x])


def test_soc_problem_rejected():
    """Test that MPAX rejects second-order cone problems."""
    # Problem with norm (SOC constraint)
    x = cp.Variable(3)
    problem = cp.Problem(cp.Minimize(cp.norm(x, 2)), [x >= 0])

    with pytest.raises(SolverError, match="could not be reduced to a QP"):
        CvxpyLayer(problem, [], [x], solver="MPAX")


def test_exponential_cone_rejected():
    """Test that MPAX rejects exponential cone problems."""
    # Problem with logarithm (exponential cone)
    x = cp.Variable()
    problem = cp.Problem(cp.Minimize(-cp.log(x)), [x >= 0.1])

    with pytest.raises(SolverError, match="could not be reduced to a QP"):
        CvxpyLayer(problem, [], [x], solver="MPAX")


def test_sdp_rejected():
    """Test that MPAX rejects semidefinite programming problems."""
    # Problem with PSD constraint
    X = cp.Variable((3, 3), PSD=True)
    problem = cp.Problem(cp.Minimize(cp.trace(X)))

    with pytest.raises(SolverError, match="could not be reduced to a QP"):
        CvxpyLayer(problem, [], [X], solver="MPAX")


# # ============================================================================
# # Gradient Tests
# # ============================================================================


# def compare_gradients(problem, params, param_vals, variables):
#     """Compare MPAX gradients vs DIFFCP gradients."""
#     # Create param tensors with gradients enabled
#     mpax_params = [torch.tensor(v, requires_grad=True) for v in param_vals]
#     diffcp_params = [torch.tensor(v, requires_grad=True) for v in param_vals]

#     # Create layers
#     layer_mpax = CvxpyLayer(problem, params, variables, solver="MPAX")
#     layer_diffcp = CvxpyLayer(problem, params, variables, solver="DIFFCP")

#     # Forward pass
#     mpax_sols = layer_mpax(*mpax_params)
#     diffcp_sols = layer_diffcp(*diffcp_params)

#     # Compute loss (sum of squared solutions)
#     mpax_loss = sum(torch.sum(sol**2) for sol in mpax_sols)
#     diffcp_loss = sum(torch.sum(sol**2) for sol in diffcp_sols)

#     # Backward pass
#     mpax_grads = torch.autograd.grad(mpax_loss, mpax_params, create_graph=False)
#     diffcp_grads = torch.autograd.grad(diffcp_loss, diffcp_params, create_graph=False)

#     # Compare gradients
#     for i, (mpax_grad, diffcp_grad) in enumerate(zip(mpax_grads, diffcp_grads, strict=True)):
#         grad_diff = torch.norm(mpax_grad - diffcp_grad).item()
#         assert grad_diff < 1e-5, (
#             f"Parameter {i} gradient differs: ||MPAX - DIFFCP|| = {grad_diff:.6e}"
#         )


# # Numerical gradient checking tests (torch.autograd.gradcheck)


# def test_gradient_equality_only():
#     """Test MPAX gradients for equality constraints only."""
#     n, m = 3, 2
#     x = cp.Variable(n)
#     A = cp.Parameter((m, n))
#     b = cp.Parameter(m)

#     problem = cp.Problem(cp.Minimize(cp.sum_squares(x)), [A @ x == b])

#     np.random.seed(100)
#     A_val = np.random.randn(m, n)
#     b_val = np.random.randn(m)

#     layer = CvxpyLayer(problem, [A, b], [x], solver="MPAX")

#     # gradcheck requires double precision
#     A_t = torch.tensor(A_val, requires_grad=True)
#     b_t = torch.tensor(b_val, requires_grad=True)

#     assert torch.autograd.gradcheck(layer, (A_t, b_t), eps=1e-3, atol=1e-4)


# def test_gradient_inequality_only():
#     """Test MPAX gradients for inequality constraints only."""
#     x = cp.Variable(1)
#     a = cp.Parameter(1)

#     problem = cp.Problem(cp.Minimize(cp.sum_squares(x - 1)), [x >= a])

#     a_val = np.array([0.5])

#     layer = CvxpyLayer(problem, [a], [x], solver="MPAX")
#     a_t = torch.tensor(a_val, requires_grad=True)

#     assert torch.autograd.gradcheck(layer, (a_t,), eps=1e-3, atol=1e-4)


# def test_gradient_mixed_constraints():
#     """Test MPAX gradients for mixed equality and inequality constraints."""
#     n = 3
#     x = cp.Variable(n)
#     A = cp.Parameter((1, n))
#     b = cp.Parameter(1)
#     G = cp.Parameter((2, n))
#     h = cp.Parameter(2)

#     problem = cp.Problem(cp.Minimize(cp.sum_squares(x)), [A @ x == b, G @ x >= h])

#     np.random.seed(200)
#     A_val = np.array([[1.0, 1.0, 1.0]])
#     b_val = np.array([3.0])
#     G_val = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
#     h_val = np.array([0.0, 0.0])

#     layer = CvxpyLayer(problem, [A, b, G, h], [x], solver="MPAX")

#     A_t = torch.tensor(A_val, requires_grad=True)
#     b_t = torch.tensor(b_val, requires_grad=True)
#     G_t = torch.tensor(G_val, requires_grad=True)
#     h_t = torch.tensor(h_val, requires_grad=True)

#     assert torch.autograd.gradcheck(layer, (A_t, b_t, G_t, h_t), eps=1e-3, atol=1e-4)


# def test_gradient_box_constraints():
#     """Test MPAX gradients for box constraints."""
#     n = 3
#     x = cp.Variable(n)
#     target = cp.Parameter(n)

#     problem = cp.Problem(cp.Minimize(cp.sum_squares(x - target)), [x >= 0, x <= 1])

#     target_val = np.array([2.0, 0.5, -1.0])

#     layer = CvxpyLayer(problem, [target], [x], solver="MPAX")
#     target_t = torch.tensor(target_val, requires_grad=True)

#     assert torch.autograd.gradcheck(layer, (target_t,), eps=1e-3, atol=1e-4)


# def test_gradient_qp_with_linear_objective():
#     """Test MPAX gradients for QP with linear objective."""
#     n = 4
#     x = cp.Variable(n)
#     P_const = np.eye(n) * 2
#     q = cp.Parameter(n)
#     A = cp.Parameter((1, n))
#     b = cp.Parameter(1)

#     problem = cp.Problem(cp.Minimize(0.5 * cp.quad_form(x, P_const) + q @ x), [A @ x == b])

#     np.random.seed(300)
#     q_val = np.random.randn(n)
#     A_val = np.ones((1, n))
#     b_val = np.array([2.0])

#     layer = CvxpyLayer(problem, [q, A, b], [x], solver="MPAX")

#     q_t = torch.tensor(q_val, requires_grad=True)
#     A_t = torch.tensor(A_val, requires_grad=True)
#     b_t = torch.tensor(b_val, requires_grad=True)

#     assert torch.autograd.gradcheck(layer, (q_t, A_t, b_t), eps=1e-3, atol=1e-4)


# def test_gradient_least_squares():
#     """Test MPAX gradients for least squares problem."""
#     n, m = 3, 4
#     x = cp.Variable(n)
#     A = cp.Parameter((m, n))
#     b = cp.Parameter(m)

#     problem = cp.Problem(cp.Minimize(cp.sum_squares(A @ x - b) + cp.sum_squares(x)))

#     np.random.seed(42)
#     A_val = np.random.randn(m, n)
#     b_val = np.random.randn(m)

#     layer = CvxpyLayer(problem, [A, b], [x], solver="MPAX")

#     A_t = torch.tensor(A_val, requires_grad=True)
#     b_t = torch.tensor(b_val, requires_grad=True)

#     assert torch.autograd.gradcheck(layer, (A_t, b_t), eps=1e-3, atol=1e-4)


# # MPAX vs DIFFCP gradient comparison tests


# def test_gradient_comparison_equality_only():
#     """Compare MPAX vs DIFFCP gradients for equality constraints."""
#     n, m = 3, 2
#     x = cp.Variable(n)
#     A = cp.Parameter((m, n))
#     b = cp.Parameter(m)

#     problem = cp.Problem(cp.Minimize(cp.sum_squares(x)), [A @ x == b])

#     np.random.seed(100)
#     A_val = np.random.randn(m, n)
#     b_val = np.random.randn(m)

#     compare_gradients(problem, [A, b], [A_val, b_val], [x])


# def test_gradient_comparison_inequality_only():
#     """Compare MPAX vs DIFFCP gradients for inequality constraints."""
#     x = cp.Variable(1)
#     a = cp.Parameter(1)

#     problem = cp.Problem(cp.Minimize(cp.sum_squares(x - 1)), [x >= a])
#     a_val = np.array([0.5])

#     compare_gradients(problem, [a], [a_val], [x])


# def test_gradient_comparison_mixed_constraints():
#     """Compare MPAX vs DIFFCP gradients for mixed constraints."""
#     n = 3
#     x = cp.Variable(n)
#     A = cp.Parameter((1, n))
#     b = cp.Parameter(1)
#     G = cp.Parameter((2, n))
#     h = cp.Parameter(2)

#     problem = cp.Problem(cp.Minimize(cp.sum_squares(x)), [A @ x == b, G @ x >= h])

#     A_val = np.array([[1.0, 1.0, 1.0]])
#     b_val = np.array([3.0])
#     G_val = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
#     h_val = np.array([0.0, 0.0])

#     compare_gradients(problem, [A, b, G, h], [A_val, b_val, G_val, h_val], [x])


# def test_gradient_comparison_box_constraints():
#     """Compare MPAX vs DIFFCP gradients for box constraints."""
#     n = 3
#     x = cp.Variable(n)
#     target = cp.Parameter(n)

#     problem = cp.Problem(cp.Minimize(cp.sum_squares(x - target)), [x >= 0, x <= 1])
#     target_val = np.array([2.0, 0.5, -1.0])

#     compare_gradients(problem, [target], [target_val], [x])


# def test_gradient_comparison_qp_with_linear_objective():
#     """Compare MPAX vs DIFFCP gradients for QP with linear objective."""
#     n = 4
#     x = cp.Variable(n)
#     P_const = np.eye(n) * 2
#     q = cp.Parameter(n)
#     A = cp.Parameter((1, n))
#     b = cp.Parameter(1)

#     problem = cp.Problem(cp.Minimize(0.5 * cp.quad_form(x, P_const) + q @ x), [A @ x == b])

#     np.random.seed(300)
#     q_val = np.random.randn(n)
#     A_val = np.ones((1, n))
#     b_val = np.array([2.0])

#     compare_gradients(problem, [q, A, b], [q_val, A_val, b_val], [x])


# def test_gradient_comparison_least_squares():
#     """Compare MPAX vs DIFFCP gradients for least squares."""
#     n, m = 3, 4
#     x = cp.Variable(n)
#     A = cp.Parameter((m, n))
#     b = cp.Parameter(m)

#     problem = cp.Problem(cp.Minimize(cp.sum_squares(A @ x - b) + cp.sum_squares(x)))

#     np.random.seed(42)
#     A_val = np.random.randn(m, n)
#     b_val = np.random.randn(m)

#     compare_gradients(problem, [A, b], [A_val, b_val], [x])
