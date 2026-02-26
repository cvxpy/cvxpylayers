"""test suite for Moreau solver."""

import cvxpy as cp
import numpy as np
import pytest
import torch

from cvxpylayers.torch import CvxpyLayer

# Skip all tests in this module if moreau is not installed
moreau = pytest.importorskip("moreau")

# Check for CUDA availability
HAS_CUDA = torch.cuda.is_available() and moreau.device_available("cuda")

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

torch.set_default_dtype(torch.double)


def compare_solvers(problem, params, param_vals, variables):
    """Compare Moreau vs DIFFCP and CVXPY direct solve."""
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
        var.value = sol.detach().cpu().numpy()
    diffcp_obj = problem.objective.value

    # Test Moreau
    layer_moreau = CvxpyLayer(problem, params, variables, solver="MOREAU")
    sols_moreau = layer_moreau(*[torch.tensor(v, requires_grad=True) for v in param_vals])

    # Recompute objective
    for param, val in zip(params, param_vals, strict=True):
        param.value = val
    for var, sol in zip(variables, sols_moreau, strict=True):
        var.value = sol.detach().cpu().numpy()
    moreau_obj = problem.objective.value

    # Compare objectives
    obj_err = abs(moreau_obj - true_obj)
    diffcp_vs_moreau = abs(moreau_obj - diffcp_obj)

    assert obj_err < 1e-3, f"Moreau error={obj_err:.6f}"
    assert diffcp_vs_moreau < 1e-3, f"diff from DIFFCP={diffcp_vs_moreau:.6f}"

    # Compare primal solutions
    for i, (sol_moreau, sol_diffcp, sol_true) in enumerate(
        zip(sols_moreau, sols_diffcp, true_sol, strict=True)
    ):
        # Compare DIFFCP vs ground truth
        diffcp_err = np.linalg.norm(sol_diffcp.detach().cpu().numpy() - sol_true)
        assert diffcp_err < 1e-3, f"DIFFCP var {i} error: ||DIFFCP - true|| = {diffcp_err:.6e}"

        # Compare Moreau vs ground truth
        moreau_err = np.linalg.norm(sol_moreau.detach().cpu().numpy() - sol_true)
        assert moreau_err < 1e-3, f"Moreau var {i} error: ||Moreau - true|| = {moreau_err:.6e}"

        # Compare Moreau vs DIFFCP
        primal_diff = torch.norm(sol_moreau.cpu() - sol_diffcp).item()
        assert primal_diff < 1e-3, (
            f"Primal variable {i} differs: ||Moreau - DIFFCP|| = {primal_diff:.6e}"
        )


def compare_solvers_batched(problem, params, param_vals_batch, variables):
    """Compare Moreau vs DIFFCP for batched inputs."""
    batch_size = param_vals_batch[0].shape[0]

    # Convert to torch tensors (with batch dimension)
    param_tensors = [torch.tensor(v, requires_grad=True) for v in param_vals_batch]

    # Test DIFFCP with batched inputs
    layer_diffcp = CvxpyLayer(problem, params, variables, solver="DIFFCP")
    sols_diffcp = layer_diffcp(*param_tensors)

    # Test Moreau with batched inputs
    layer_moreau = CvxpyLayer(problem, params, variables, solver="MOREAU")
    sols_moreau = layer_moreau(*[torch.tensor(v, requires_grad=True) for v in param_vals_batch])

    # Compare solutions for each batch element
    for batch_idx in range(batch_size):
        # Extract parameter values for this batch element
        param_vals_single = [v[batch_idx] for v in param_vals_batch]

        # Solve with CVXPY as ground truth
        for param, val in zip(params, param_vals_single, strict=True):
            param.value = val.numpy() if hasattr(val, "numpy") else val
        problem.solve()
        assert problem.status == "optimal", f"Batch {batch_idx}: CVXPY failed"

        true_sol = [v.value for v in variables]
        true_obj = problem.value

        # Compare DIFFCP for this batch element
        for var, sol in zip(variables, sols_diffcp, strict=True):
            var.value = sol[batch_idx].detach().cpu().numpy()
        diffcp_obj = problem.objective.value

        # Compare Moreau for this batch element
        for param, val in zip(params, param_vals_single, strict=True):
            param.value = val.numpy() if hasattr(val, "numpy") else val
        for var, sol in zip(variables, sols_moreau, strict=True):
            var.value = sol[batch_idx].detach().cpu().numpy()
        moreau_obj = problem.objective.value

        # Compare objectives
        obj_err = abs(moreau_obj - true_obj)
        diffcp_vs_moreau = abs(moreau_obj - diffcp_obj)

        assert obj_err < 1e-3, f"Batch {batch_idx}: Moreau error={obj_err:.6f}"
        assert diffcp_vs_moreau < 1e-3, f"Batch {batch_idx}: diff={diffcp_vs_moreau:.6f}"

        # Compare primal solutions
        for i, (sol_moreau, sol_diffcp, sol_true) in enumerate(
            zip(sols_moreau, sols_diffcp, true_sol, strict=True)
        ):
            moreau_err = np.linalg.norm(sol_moreau[batch_idx].detach().cpu().numpy() - sol_true)
            assert moreau_err < 1e-3, (
                f"Batch {batch_idx}, var {i}: ||Moreau - true|| = {moreau_err:.6e}"
            )

            primal_diff = torch.norm(sol_moreau[batch_idx].cpu() - sol_diffcp[batch_idx]).item()
            assert primal_diff < 1e-3, (
                f"Batch {batch_idx}, var {i}: ||Moreau - DIFFCP|| = {primal_diff:.6e}"
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


def test_unconstrained_torch():
    """Test unconstrained QP with PyTorch.

    minimize x^2 + c*x  =>  x* = -c/2
    """
    n = 3
    x = cp.Variable(n)
    c = cp.Parameter(n)

    problem = cp.Problem(cp.Minimize(cp.sum_squares(x) + c @ x))

    c_val = np.array([1.0, -2.0, 0.5])

    layer = CvxpyLayer(problem, [c], [x], solver="MOREAU")
    c_tensor = torch.tensor(c_val, requires_grad=True)
    (x_sol,) = layer(c_tensor)

    # Verify analytical solution: x* = -c/2
    expected = -c_val / 2
    error = np.linalg.norm(x_sol.detach().numpy() - expected)
    assert error < 1e-4, f"Solution error: {error:.6e}"

    # Verify gradient: d/dc sum(x*) = -1/2
    x_sol.sum().backward()
    expected_grad = torch.full((n,), -0.5, dtype=torch.float64)
    grad_error = torch.norm(c_tensor.grad - expected_grad).item()
    assert grad_error < 1e-4, f"Gradient error: {grad_error:.6e}"


def test_unconstrained_jax():
    """Test unconstrained QP with JAX (including JIT)."""
    from cvxpylayers.jax import CvxpyLayer as JaxCvxpyLayer

    n = 3
    x = cp.Variable(n)
    c = cp.Parameter(n)

    problem = cp.Problem(cp.Minimize(cp.sum_squares(x) + c @ x))

    layer = JaxCvxpyLayer(problem, parameters=[c], variables=[x], solver="MOREAU")
    c_val = jnp.array([1.0, -2.0, 0.5])

    # Test basic solve
    (x_sol,) = layer(c_val)
    expected = -c_val / 2
    assert jnp.linalg.norm(x_sol - expected) < 1e-4

    # Test JIT + gradient
    @jax.jit
    def solve_and_sum(c):
        (x,) = layer(c)
        return jnp.sum(x)

    grad = jax.grad(solve_and_sum)(c_val)
    expected_grad = jnp.full((n,), -0.5)
    assert jnp.linalg.norm(grad - expected_grad) < 1e-4


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


def test_least_squares_with_regularization():
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


def test_equality_only_batched():
    """Test batched inputs with only equality constraints."""
    # minimize x^T x subject to Ax = b
    n, m = 5, 2
    batch_size = 3
    x = cp.Variable(n)
    A = cp.Parameter((m, n))
    b = cp.Parameter(m)

    problem = cp.Problem(cp.Minimize(cp.sum_squares(x)), [A @ x == b])

    np.random.seed(100)
    # Create batched parameter values
    A_val_batch = np.random.randn(batch_size, m, n)
    b_val_batch = np.random.randn(batch_size, m)

    compare_solvers_batched(problem, [A, b], [A_val_batch, b_val_batch], [x])


def test_inequality_only_batched():
    """Test batched inputs with only inequality constraints."""
    # minimize (x-1)^2 subject to x >= a
    x = cp.Variable(1)
    a = cp.Parameter(1)

    problem = cp.Problem(cp.Minimize(cp.sum_squares(x - 1)), [x >= a])

    # Create batched parameter values (batch_size = 4)
    a_val_batch = np.array([[0.5], [0.2], [0.8], [-0.5]])

    compare_solvers_batched(problem, [a], [a_val_batch], [x])


def test_mixed_constraints_batched():
    """Test batched inputs with both equality and inequality constraints."""
    # minimize x^T x subject to Ax = b, Gx >= h
    n = 3
    batch_size = 2
    x = cp.Variable(n)
    A = cp.Parameter((1, n))
    b = cp.Parameter(1)
    G = cp.Parameter((2, n))
    h = cp.Parameter(2)

    problem = cp.Problem(cp.Minimize(cp.sum_squares(x)), [A @ x == b, G @ x >= h])

    np.random.seed(200)
    # Create batched parameter values
    A_val_batch = np.array([[[1.0, 1.0, 1.0]], [[1.0, 0.5, 0.5]]])
    b_val_batch = np.array([[3.0], [2.0]])
    G_val_batch = np.tile(np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]), (batch_size, 1, 1))
    h_val_batch = np.tile(np.array([0.0, 0.0]), (batch_size, 1))

    compare_solvers_batched(
        problem, [A, b, G, h], [A_val_batch, b_val_batch, G_val_batch, h_val_batch], [x]
    )


def test_box_constraints_batched():
    """Test batched inputs with box constraints (variable bounds)."""
    # minimize (x-target)^T(x-target) subject to 0 <= x <= 1
    n = 3
    x = cp.Variable(n)
    target = cp.Parameter(n)

    problem = cp.Problem(cp.Minimize(cp.sum_squares(x - target)), [x >= 0, x <= 1])

    # Create batched parameter values (batch_size = 3)
    target_val_batch = np.array([[2.0, 0.5, -1.0], [0.5, 0.5, 0.5], [1.5, -0.5, 2.0]])

    compare_solvers_batched(problem, [target], [target_val_batch], [x])


def test_batch_size_one_preserves_batch_dimension():
    """Test that batch_size=1 is different from unbatched.

    When the input is explicitly batched with batch_size=1 (shape (1, n)),
    the output should also be batched with shape (1, n), not unbatched (n,).
    """
    n = 3
    x = cp.Variable(n)
    b = cp.Parameter(n)

    # Simple quadratic problem: minimize ||x - b||^2
    objective = cp.Minimize(cp.sum_squares(x - b))
    problem = cp.Problem(objective)

    layer_moreau = CvxpyLayer(problem, parameters=[b], variables=[x], solver="MOREAU")

    # Create parameter value
    b_value = torch.randn(n)

    # Test with unbatched input
    b_unbatched = b_value.clone().requires_grad_(True)  # Shape: (n,)
    (x_unbatched,) = layer_moreau(b_unbatched)

    # Solution should be unbatched
    assert x_unbatched.shape == (n,), f"Expected unbatched shape ({n},), got {x_unbatched.shape}"

    # Test with explicitly batched input with batch_size=1 (same values)
    b_batched = b_value.unsqueeze(0).clone().requires_grad_(True)  # Shape: (1, n)
    (x_batched,) = layer_moreau(b_batched)

    # Solution should be batched
    assert x_batched.shape == (1, n), (
        f"Expected batched shape (1, {n}), got {x_batched.shape}. "
        "Batch dimension should be preserved for batch_size=1."
    )

    # Verify the actual solutions are numerically identical (just differ in shape)
    assert torch.allclose(x_unbatched, x_batched.squeeze(0), atol=1e-6), (
        "Solutions for unbatched and batch_size=1 should be numerically identical"
    )


def test_jax_interface_forward_pass():
    """Test JAX interface with Moreau solver (forward pass only)."""
    from cvxpylayers.jax import CvxpyLayer as JaxCvxpyLayer

    # minimize ||x||^2 subject to Ax = b, Gx >= h
    n = 3
    x = cp.Variable(n)
    A = cp.Parameter((1, n))
    b = cp.Parameter(1)
    G = cp.Parameter((2, n))
    h = cp.Parameter(2)

    problem = cp.Problem(cp.Minimize(cp.sum_squares(x)), [A @ x == b, G @ x >= h])

    A_val = np.array([[1.0, 1.0, 1.0]])  # sum(x) = 3
    b_val = np.array([3.0])
    G_val = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])  # x[0] >= 0, x[1] >= 0
    h_val = np.array([0.0, 0.0])

    # Get ground truth from CVXPY
    A.value = A_val
    b.value = b_val
    G.value = G_val
    h.value = h_val
    problem.solve()
    true_sol = x.value
    true_obj = problem.value

    # Test JAX interface with Moreau
    layer = JaxCvxpyLayer(problem, [A, b, G, h], [x], solver="MOREAU")

    A_jax = jnp.array(A_val)
    b_jax = jnp.array(b_val)
    G_jax = jnp.array(G_val)
    h_jax = jnp.array(h_val)

    (x_sol,) = layer(A_jax, b_jax, G_jax, h_jax)

    # Compare solutions
    x_np = np.array(x_sol)
    error = np.linalg.norm(x_np - true_sol)
    obj_value = np.sum(x_np**2)
    obj_error = abs(obj_value - true_obj)

    assert error < 1e-3, f"Solution error: ||JAX-Moreau - CVXPY|| = {error:.6e}"
    assert obj_error < 1e-3, f"Objective error: |JAX-Moreau - CVXPY| = {obj_error:.6e}"


def test_jax_l1_norm_gradient():
    """Test JAX gradient for L1 norm objective (introduces auxiliary constraints).

    L1 norm problems have auxiliary variables/constraints during canonicalization,
    so A_shape[0] > b_idx.size. This tests that the backward pass correctly
    gathers gradients from the expanded constraint space.
    """
    from cvxpylayers.jax import CvxpyLayer as JaxCvxpyLayer

    n, m = 2, 3
    x = cp.Variable(n)
    A = cp.Parameter((m, n))
    b = cp.Parameter(m)
    constraints = [x >= 0]
    objective = cp.Minimize(0.5 * cp.pnorm(A @ x - b, p=1))
    problem = cp.Problem(objective, constraints)
    assert problem.is_dpp()

    layer = JaxCvxpyLayer(problem, parameters=[A, b], variables=[x], solver="MOREAU")

    key = jax.random.PRNGKey(0)
    key, k1, k2 = jax.random.split(key, 3)
    A_jax = jax.random.normal(k1, shape=(m, n))
    b_jax = jax.random.normal(k2, shape=(m,))

    # Forward pass
    (solution,) = layer(A_jax, b_jax)
    assert solution.shape == (n,), f"Expected shape ({n},), got {solution.shape}"

    # Backward pass - this was failing before the fix
    dlayer = jax.grad(lambda A, b: sum(layer(A, b)[0]), argnums=[0, 1])
    gradA, gradb = dlayer(A_jax, b_jax)

    assert gradA.shape == (m, n), f"Expected gradA shape ({m}, {n}), got {gradA.shape}"
    assert gradb.shape == (m,), f"Expected gradb shape ({m},), got {gradb.shape}"
    assert jnp.isfinite(gradA).all(), f"gradA contains non-finite values: {gradA}"
    assert jnp.isfinite(gradb).all(), f"gradb contains non-finite values: {gradb}"


def test_torch_l1_norm_gradient():
    """Test PyTorch gradient for L1 norm objective (introduces auxiliary constraints).

    L1 norm problems have auxiliary variables/constraints during canonicalization,
    so A_shape[0] > b_idx.size. This tests that the backward pass correctly
    gathers gradients from the expanded constraint space.
    """
    n, m = 2, 3
    x = cp.Variable(n)
    A = cp.Parameter((m, n))
    b = cp.Parameter(m)
    constraints = [x >= 0]
    objective = cp.Minimize(0.5 * cp.pnorm(A @ x - b, p=1))
    problem = cp.Problem(objective, constraints)
    assert problem.is_dpp()

    layer = CvxpyLayer(problem, parameters=[A, b], variables=[x], solver="MOREAU")

    np.random.seed(42)
    A_val = torch.tensor(np.random.randn(m, n), requires_grad=True, dtype=torch.float64)
    b_val = torch.tensor(np.random.randn(m), requires_grad=True, dtype=torch.float64)

    # Forward pass
    (solution,) = layer(A_val, b_val)
    assert solution.shape == (n,), f"Expected shape ({n},), got {solution.shape}"

    # Backward pass
    loss = solution.sum()
    loss.backward()

    assert A_val.grad is not None, "gradA was not computed"
    assert b_val.grad is not None, "gradb was not computed"
    assert A_val.grad.shape == (m, n), f"Expected gradA shape ({m}, {n}), got {A_val.grad.shape}"
    assert b_val.grad.shape == (m,), f"Expected gradb shape ({m},), got {b_val.grad.shape}"
    assert torch.isfinite(A_val.grad).all(), f"gradA contains non-finite values"
    assert torch.isfinite(b_val.grad).all(), f"gradb contains non-finite values"


def test_jax_batch_size_one_gradient():
    """Test JAX gradient with explicitly batched batch_size=1 input.

    This tests a regression where batch_size=1 with explicitly batched
    input (shape (1, n)) wasn't handled correctly in the backward pass.
    The VJP expected unbatched shapes but received batched shapes.
    """
    from cvxpylayers.jax import CvxpyLayer as JaxCvxpyLayer

    n, m = 3, 4
    x = cp.Variable(n)
    A = cp.Parameter((m, n))
    b = cp.Parameter(m)

    # L1 norm introduces auxiliary constraints, making A_shape[0] > b_idx.size
    objective = cp.Minimize(cp.norm(A @ x - b, 1) + 0.1 * cp.sum_squares(x))
    problem = cp.Problem(objective)

    layer = JaxCvxpyLayer(problem, parameters=[A, b], variables=[x], solver="MOREAU")

    np.random.seed(42)
    # Explicitly batched with batch_size=1
    A_val = jnp.array(np.random.randn(1, m, n))
    b_val = jnp.array(np.random.randn(1, m))

    # Forward pass
    (solution,) = layer(A_val, b_val)
    assert solution.shape == (1, n), f"Expected shape (1, {n}), got {solution.shape}"

    # Backward pass - this was failing before the fix
    def loss_fn(A, b):
        (x,) = layer(A, b)
        return x.sum()

    gradA, gradb = jax.grad(loss_fn, argnums=(0, 1))(A_val, b_val)

    assert gradA.shape == (1, m, n), f"Expected gradA shape (1, {m}, {n}), got {gradA.shape}"
    assert gradb.shape == (1, m), f"Expected gradb shape (1, {m}), got {gradb.shape}"
    assert jnp.isfinite(gradA).all(), "gradA contains non-finite values"
    assert jnp.isfinite(gradb).all(), "gradb contains non-finite values"


def test_jax_interface_batched():
    """Test JAX interface with Moreau solver for batched inputs."""
    from cvxpylayers.jax import CvxpyLayer as JaxCvxpyLayer

    # minimize ||x||^2 subject to Ax = b
    n, m = 4, 2
    x = cp.Variable(n)
    A = cp.Parameter((m, n))
    b = cp.Parameter(m)

    problem = cp.Problem(cp.Minimize(cp.sum_squares(x)), [A @ x == b])

    np.random.seed(400)
    A_val = np.random.randn(m, n)
    b_val_batch = np.random.randn(3, m)  # batch size = 3

    # Test JAX interface with Moreau
    layer = JaxCvxpyLayer(problem, [A, b], [x], solver="MOREAU")

    A_jax = jnp.array(A_val)
    b_jax = jnp.array(b_val_batch)

    (x_sol,) = layer(A_jax, b_jax)

    # Verify batch dimension is correct
    assert x_sol.shape == (3, n), f"Expected shape (3, {n}), got {x_sol.shape}"

    # Check each batch element against CVXPY ground truth
    for i in range(b_val_batch.shape[0]):
        A.value = A_val
        b.value = b_val_batch[i]
        problem.solve()
        true_sol = x.value

        x_sol_i = np.array(x_sol[i])
        error = np.linalg.norm(x_sol_i - true_sol)
        assert error < 1e-3, f"Batch {i} error: {error:.6e}"


def test_backward_gradient():
    """Test that backward pass computes correct gradients.

    For minimize ||x - b||^2, the optimal x* = b, so dx*/db = I.
    Thus d/db sum(x*) = d/db sum(b) = [1, 1, 1].
    """
    n = 3
    x = cp.Variable(n)
    b = cp.Parameter(n)

    problem = cp.Problem(cp.Minimize(cp.sum_squares(x - b)))

    layer_moreau = CvxpyLayer(problem, parameters=[b], variables=[x], solver="MOREAU")

    # Create parameter with requires_grad=True
    b_val = torch.tensor([1.0, 2.0, 3.0], requires_grad=True, dtype=torch.float64)

    # Forward pass
    (x_sol,) = layer_moreau(b_val)

    # Verify forward pass is correct
    assert torch.allclose(x_sol, b_val.detach(), atol=1e-4), (
        f"Forward: expected {b_val}, got {x_sol}"
    )

    # Backward pass should compute correct gradients
    loss = x_sol.sum()
    loss.backward()

    # Gradient should be [1, 1, 1] since dx*/db = I
    expected_grad = torch.ones(n, dtype=torch.float64)
    assert b_val.grad is not None, "Gradient was not computed"
    assert torch.allclose(b_val.grad, expected_grad, atol=1e-4), (
        f"Backward: expected grad {expected_grad}, got {b_val.grad}"
    )


# ============================================================================
# CPU-specific tests
# ============================================================================


def test_cpu_equality_only():
    """Test CPU path with only equality constraints."""
    # minimize x^T x subject to Ax = b
    n, m = 5, 2
    x = cp.Variable(n)
    A = cp.Parameter((m, n))
    b = cp.Parameter(m)

    problem = cp.Problem(cp.Minimize(cp.sum_squares(x)), [A @ x == b])

    np.random.seed(100)
    A_val = np.random.randn(m, n)
    b_val = np.random.randn(m)

    # Set parameter values for direct solve
    A.value = A_val
    b.value = b_val
    problem.solve()
    assert problem.status == "optimal"
    true_sol = x.value

    # Test with CPU tensors (explicitly on CPU)
    layer_moreau = CvxpyLayer(problem, [A, b], [x], solver="MOREAU")
    A_tensor = torch.tensor(A_val, device="cpu")
    b_tensor = torch.tensor(b_val, device="cpu")

    (x_sol,) = layer_moreau(A_tensor, b_tensor)

    # Verify output is on CPU
    assert x_sol.device.type == "cpu", f"Expected CPU tensor, got {x_sol.device}"

    # Verify solution is correct (detach needed since output has grad_fn)
    error = np.linalg.norm(x_sol.detach().numpy() - true_sol)
    assert error < 1e-3, f"CPU solver error: {error:.6e}"


def test_cpu_mixed_constraints():
    """Test CPU path with both equality and inequality constraints."""
    n = 3
    x = cp.Variable(n)
    A = cp.Parameter((1, n))
    b = cp.Parameter(1)
    G = cp.Parameter((2, n))
    h = cp.Parameter(2)

    problem = cp.Problem(cp.Minimize(cp.sum_squares(x)), [A @ x == b, G @ x >= h])

    A_val = np.array([[1.0, 1.0, 1.0]])  # sum(x) = 3
    b_val = np.array([3.0])
    G_val = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])  # x[0] >= 0, x[1] >= 0
    h_val = np.array([0.0, 0.0])

    # Get ground truth
    A.value = A_val
    b.value = b_val
    G.value = G_val
    h.value = h_val
    problem.solve()
    true_sol = x.value

    # Test with CPU tensors
    layer_moreau = CvxpyLayer(problem, [A, b, G, h], [x], solver="MOREAU")

    A_tensor = torch.tensor(A_val, device="cpu")
    b_tensor = torch.tensor(b_val, device="cpu")
    G_tensor = torch.tensor(G_val, device="cpu")
    h_tensor = torch.tensor(h_val, device="cpu")

    (x_sol,) = layer_moreau(A_tensor, b_tensor, G_tensor, h_tensor)

    # Verify output is on CPU
    assert x_sol.device.type == "cpu"

    # Verify solution (detach needed since output has grad_fn)
    error = np.linalg.norm(x_sol.detach().numpy() - true_sol)
    assert error < 1e-3, f"CPU solver error: {error:.6e}"


def test_cpu_batched():
    """Test CPU path with batched inputs."""
    n, m = 4, 2
    batch_size = 3
    x = cp.Variable(n)
    A = cp.Parameter((m, n))
    b = cp.Parameter(m)

    problem = cp.Problem(cp.Minimize(cp.sum_squares(x)), [A @ x == b])

    np.random.seed(100)
    A_val = np.random.randn(m, n)
    b_val_batch = np.random.randn(batch_size, m)

    # Test with CPU tensors
    layer_moreau = CvxpyLayer(problem, [A, b], [x], solver="MOREAU")

    A_tensor = torch.tensor(A_val, device="cpu")
    b_tensor = torch.tensor(b_val_batch, device="cpu")

    (x_sol,) = layer_moreau(A_tensor, b_tensor)

    # Verify output shape and device
    assert x_sol.shape == (batch_size, n), f"Expected shape ({batch_size}, {n}), got {x_sol.shape}"
    assert x_sol.device.type == "cpu"

    # Verify each batch element
    for i in range(batch_size):
        A.value = A_val
        b.value = b_val_batch[i]
        problem.solve()
        true_sol = x.value

        error = np.linalg.norm(x_sol[i].detach().numpy() - true_sol)
        assert error < 1e-3, f"Batch {i} error: {error:.6e}"


def test_cpu_output_device_matches_input():
    """Test that output tensors are on the same device as input tensors."""
    n = 3
    x = cp.Variable(n)
    b = cp.Parameter(n)

    problem = cp.Problem(cp.Minimize(cp.sum_squares(x - b)))
    layer_moreau = CvxpyLayer(problem, [b], [x], solver="MOREAU")

    # Test CPU
    b_cpu = torch.randn(n, device="cpu")
    (x_cpu,) = layer_moreau(b_cpu)
    assert x_cpu.device.type == "cpu", f"Expected CPU output, got {x_cpu.device}"


@pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available or TorchSolver not built")
def test_cuda_output_device_matches_input():
    """Test that CUDA output tensors are on the same device as input tensors."""
    n = 3
    x = cp.Variable(n)
    b = cp.Parameter(n)

    problem = cp.Problem(cp.Minimize(cp.sum_squares(x - b)))
    layer_moreau = CvxpyLayer(problem, [b], [x], solver="MOREAU")

    # Test CUDA
    b_cuda = torch.randn(n, device="cuda")
    (x_cuda,) = layer_moreau(b_cuda)
    assert x_cuda.device.type == "cuda", f"Expected CUDA output, got {x_cuda.device}"


@pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available or TorchSolver not built")
def test_cpu_and_cuda_solutions_match():
    """Test that CPU and CUDA paths produce the same solution."""
    n = 4
    x = cp.Variable(n)
    A = cp.Parameter((2, n))
    b = cp.Parameter(2)

    problem = cp.Problem(cp.Minimize(cp.sum_squares(x)), [A @ x == b])

    np.random.seed(42)
    A_val = np.random.randn(2, n)
    b_val = np.random.randn(2)

    layer_moreau = CvxpyLayer(problem, [A, b], [x], solver="MOREAU")

    # Solve on CPU
    A_cpu = torch.tensor(A_val, device="cpu")
    b_cpu = torch.tensor(b_val, device="cpu")
    (x_cpu,) = layer_moreau(A_cpu, b_cpu)

    # Solve on CUDA
    A_cuda = torch.tensor(A_val, device="cuda")
    b_cuda = torch.tensor(b_val, device="cuda")
    (x_cuda,) = layer_moreau(A_cuda, b_cuda)

    # Compare solutions
    diff = torch.norm(x_cpu - x_cuda.cpu()).item()
    assert diff < 1e-5, f"CPU and CUDA solutions differ by {diff:.6e}"


# ============================================================================
# Solver options tests
# ============================================================================


def test_solver_args_actually_used():
    """Test that solver_args actually affect the solver's behavior.

    This verifies solver_args are truly passed to the solver by checking
    that the max_iter setting is respected.

    Note: Moreau sets options at construction time, so we check the stored options.
    """
    n = 5
    x = cp.Variable(n)
    b = cp.Parameter(n)

    problem = cp.Problem(cp.Minimize(cp.sum_squares(x - b)))

    # Create layer with custom max_iter
    layer = CvxpyLayer(
        problem,
        parameters=[b],
        variables=[x],
        solver="MOREAU",
        solver_args={"max_iter": 42},
    )

    # Verify options are stored correctly
    assert layer.ctx.solver_ctx.options.get("max_iter") == 42, (
        "max_iter option was not stored correctly"
    )

    # Verify options are applied to solver settings
    settings = layer.ctx.solver_ctx._get_settings(enable_grad=True)
    assert settings.max_iter == 42, (
        f"max_iter not applied to settings: expected 42, got {settings.max_iter}"
    )


# ============================================================================
# Setup caching tests (constant P/A optimization)
# ============================================================================


def test_constant_PA_detection():
    """Test that constant P/A is detected and setup is cached.

    When only q (linear objective) depends on parameters, and P and A are constant,
    setup() should be called once during solver creation, not on every forward pass.

    Problem: minimize c'x subject to x >= 0, sum(x) = 1
    Here P=None, A is constant (constraint structure), only q=c depends on parameter.
    """
    n = 4
    x = cp.Variable(n)
    c = cp.Parameter(n)

    # Only c (linear cost) is parametrized; P=None, A is constant
    problem = cp.Problem(cp.Minimize(c @ x), [x >= 0, cp.sum(x) == 1])

    layer = CvxpyLayer(problem, parameters=[c], variables=[x], solver="MOREAU")

    # Verify PA_is_constant is True
    assert layer.ctx.solver_ctx.PA_is_constant, (
        "Expected PA_is_constant=True for problem with only linear cost parametrized"
    )

    # Verify solutions are correct
    # For minimize c'x s.t. x >= 0, sum(x) = 1: optimal puts all weight on smallest c_i
    c_val = torch.tensor([3.0, 1.0, 4.0, 2.0], requires_grad=True)
    (x_sol,) = layer(c_val)

    # Optimal solution: x = e_i where i = argmin(c)
    expected = torch.zeros(n)
    expected[1] = 1.0  # c[1] = 1.0 is minimum
    assert torch.allclose(x_sol, expected, atol=1e-4), f"Expected {expected}, got {x_sol}"


def test_constant_PA_multiple_forward_passes():
    """Test that cached setup works correctly across multiple forward passes.

    Problem: minimize c'x subject to x >= 0, sum(x) = 1
    """
    n = 3
    x = cp.Variable(n)
    c = cp.Parameter(n)

    problem = cp.Problem(cp.Minimize(c @ x), [x >= 0, cp.sum(x) == 1])
    layer = CvxpyLayer(problem, parameters=[c], variables=[x], solver="MOREAU")

    # Verify PA_is_constant
    assert layer.ctx.solver_ctx.PA_is_constant

    # Run multiple forward passes with different c values
    test_cases = [
        torch.tensor([1.0, 2.0, 3.0]),  # min at index 0
        torch.tensor([3.0, 1.0, 2.0]),  # min at index 1
        torch.tensor([2.0, 3.0, 1.0]),  # min at index 2
    ]
    for i, c_val in enumerate(test_cases):
        c_val = c_val.clone().requires_grad_(True)
        (x_sol,) = layer(c_val)

        # Optimal solution puts all weight on minimum c
        expected = torch.zeros(n)
        expected[c_val.detach().argmin()] = 1.0
        assert torch.allclose(x_sol, expected, atol=1e-4), (
            f"Pass {i}: Expected {expected}, got {x_sol}"
        )


def test_constant_PA_gradients():
    """Test that gradients work correctly when P/A are constant.

    Problem: minimize c'x subject to x >= 0, sum(x) = 1
    For c = [1, 2, 3], optimal x* = [1, 0, 0], so objective = c[0] = 1.
    Gradient of objective w.r.t. c is x* = [1, 0, 0].
    """
    n = 3
    x = cp.Variable(n)
    c = cp.Parameter(n)

    problem = cp.Problem(cp.Minimize(c @ x), [x >= 0, cp.sum(x) == 1])
    layer = CvxpyLayer(problem, parameters=[c], variables=[x], solver="MOREAU")

    assert layer.ctx.solver_ctx.PA_is_constant

    c_val = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    (x_sol,) = layer(c_val)

    # Forward pass: optimal x = [1, 0, 0]
    expected_x = torch.tensor([1.0, 0.0, 0.0])
    assert torch.allclose(x_sol, expected_x, atol=1e-4), f"Expected {expected_x}, got {x_sol}"

    # Backward pass: d(c'x)/dc = x
    loss = (c_val * x_sol).sum()  # This is c'x
    loss.backward()

    # Gradient should be x* = [1, 0, 0]
    expected_grad = expected_x
    assert c_val.grad is not None
    assert torch.allclose(c_val.grad, expected_grad, atol=1e-4), (
        f"Expected grad {expected_grad}, got {c_val.grad}"
    )


def test_parametrized_PA_not_cached():
    """Test that when P or A depend on parameters, setup is not cached."""
    n = 3
    x = cp.Variable(n)
    A = cp.Parameter((1, n))  # Constraint matrix is parametrized
    b = cp.Parameter(1)

    problem = cp.Problem(cp.Minimize(cp.sum_squares(x)), [A @ x == b])
    layer = CvxpyLayer(problem, parameters=[A, b], variables=[x], solver="MOREAU")

    # PA_is_constant should be False since A is parametrized
    assert not layer.ctx.solver_ctx.PA_is_constant, (
        "Expected PA_is_constant=False when A is parametrized"
    )

    # Verify solutions are still correct
    A_val = torch.tensor([[1.0, 1.0, 1.0]])
    b_val = torch.tensor([3.0])

    (x_sol,) = layer(A_val, b_val)

    # Check constraint satisfaction: sum(x) = 3
    constraint_value = (A_val @ x_sol.unsqueeze(-1)).squeeze()
    assert torch.allclose(constraint_value, b_val, atol=1e-4)


# ============================================================================
# JIT/compile compatibility tests
# ============================================================================


@pytest.fixture
def reset_dynamo():
    """Reset torch.compile cache between tests to avoid cross-test pollution."""
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_torch_compile_unbatched(device, reset_dynamo):
    """Test that torch.compile works with unbatched inputs.

    This verifies that the batch-conditional code paths in cvxpylayer.py
    don't break torch.compile tracing.
    """
    if device == "cuda" and not HAS_CUDA:
        pytest.skip("CUDA not available")

    n = 3
    x = cp.Variable(n)
    b = cp.Parameter(n)

    problem = cp.Problem(cp.Minimize(cp.sum_squares(x - b)))
    layer = CvxpyLayer(problem, parameters=[b], variables=[x], solver="MOREAU")

    # Compile the forward pass
    compiled_forward = torch.compile(layer.forward, fullgraph=False)

    # Test unbatched
    b_val = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64, device=device, requires_grad=True)
    (x_sol,) = compiled_forward(b_val)

    # Verify shape (unbatched should have no batch dim)
    assert x_sol.shape == (n,), f"Expected shape ({n},), got {x_sol.shape}"

    # Verify solution is correct (x* = b for this problem)
    assert torch.allclose(x_sol, b_val.detach(), atol=1e-4), (
        f"Expected {b_val.detach()}, got {x_sol}"
    )

    # Verify gradients work through compiled function
    loss = x_sol.sum()
    loss.backward()
    expected_grad = torch.ones(n, dtype=torch.float64, device=device)
    assert b_val.grad is not None, "Gradient was not computed"
    assert torch.allclose(b_val.grad, expected_grad, atol=1e-4), (
        f"Expected grad {expected_grad}, got {b_val.grad}"
    )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_torch_compile_batched(device, reset_dynamo):
    """Test that torch.compile works with batched inputs.

    This verifies that the batch-conditional code paths in cvxpylayer.py
    don't break torch.compile tracing for batched inputs.
    """
    if device == "cuda" and not HAS_CUDA:
        pytest.skip("CUDA not available")

    n = 4
    batch_size = 4
    x = cp.Variable(n)
    b = cp.Parameter(n)

    problem = cp.Problem(cp.Minimize(cp.sum_squares(x - b)))
    layer = CvxpyLayer(problem, parameters=[b], variables=[x], solver="MOREAU")

    # Compile the forward pass
    compiled_forward = torch.compile(layer.forward, fullgraph=False)

    # Test batched
    b_val = torch.randn(batch_size, n, dtype=torch.float64, device=device, requires_grad=True)
    (x_sol,) = compiled_forward(b_val)

    # Verify shape (batched should have batch dim)
    assert x_sol.shape == (batch_size, n), (
        f"Expected shape ({batch_size}, {n}), got {x_sol.shape}"
    )

    # Verify solution is correct (x* = b for this problem)
    assert torch.allclose(x_sol, b_val.detach(), atol=1e-4), (
        f"Expected {b_val.detach()}, got {x_sol}"
    )

    # Verify gradients work through compiled function
    loss = x_sol.sum()
    loss.backward()
    expected_grad = torch.ones(batch_size, n, dtype=torch.float64, device=device)
    assert b_val.grad is not None, "Gradient was not computed"
    assert torch.allclose(b_val.grad, expected_grad, atol=1e-4), (
        f"Expected grad {expected_grad}, got {b_val.grad}"
    )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_torch_compile_batch_size_one(device, reset_dynamo):
    """Test that torch.compile preserves batch dimension for batch_size=1.

    This is a regression test: batch_size=1 with explicit batch dim (1, n)
    should NOT be squeezed to (n,).
    """
    if device == "cuda" and not HAS_CUDA:
        pytest.skip("CUDA not available")

    n = 5
    x = cp.Variable(n)
    b = cp.Parameter(n)

    problem = cp.Problem(cp.Minimize(cp.sum_squares(x - b)))
    layer = CvxpyLayer(problem, parameters=[b], variables=[x], solver="MOREAU")

    compiled_forward = torch.compile(layer.forward, fullgraph=False)

    # Explicitly batched with batch_size=1
    b_val = torch.randn(1, n, dtype=torch.float64, device=device, requires_grad=True)
    (x_sol,) = compiled_forward(b_val)

    # Should preserve batch dimension
    assert x_sol.shape == (1, n), (
        f"Expected shape (1, {n}), got {x_sol.shape}. "
        "Batch dimension should be preserved for batch_size=1."
    )

    # Verify gradients
    loss = x_sol.sum()
    loss.backward()
    assert b_val.grad is not None
    assert b_val.grad.shape == (1, n)


# ========== JAX JIT Tests ==========
# These tests verify JIT/vmap compatibility with the Moreau solver.
# Moreau's JAX solver uses custom_vjp with pure_callback and vmap_method="broadcast_all",
# making it fully compatible with jax.jit, jax.vmap, and jax.pmap.


@pytest.fixture
def clear_jax_cache():
    """Clear JAX compilation cache between tests."""
    jax = pytest.importorskip("jax")
    yield
    jax.clear_caches()


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_jax_jit_unbatched(clear_jax_cache, device):
    """Test that jax.jit works with unbatched inputs."""
    if device == "cuda" and not HAS_CUDA:
        pytest.skip("CUDA not available")
    jax = pytest.importorskip("jax")
    jnp = jax.numpy
    from cvxpylayers.jax import CvxpyLayer as JaxCvxpyLayer

    n = 3
    x = cp.Variable(n)
    b = cp.Parameter(n)
    problem = cp.Problem(cp.Minimize(cp.sum_squares(x - b)))
    layer = JaxCvxpyLayer(problem, parameters=[b], variables=[x], solver="MOREAU")

    jitted_layer = jax.jit(lambda b: layer(b, solver_args={"device": device}))
    b_val = jnp.array([1.0, 2.0, 3.0])
    (x_sol,) = jitted_layer(b_val)

    assert x_sol.shape == (n,)
    assert jnp.allclose(x_sol, b_val, atol=1e-4)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_jax_jit_batched(clear_jax_cache, device):
    """Test that jax.jit works with batched inputs."""
    if device == "cuda" and not HAS_CUDA:
        pytest.skip("CUDA not available")
    jax = pytest.importorskip("jax")
    from cvxpylayers.jax import CvxpyLayer as JaxCvxpyLayer

    n, batch_size = 4, 4
    x = cp.Variable(n)
    b = cp.Parameter(n)
    problem = cp.Problem(cp.Minimize(cp.sum_squares(x - b)))
    layer = JaxCvxpyLayer(problem, parameters=[b], variables=[x], solver="MOREAU")

    jitted_layer = jax.jit(lambda b: layer(b, solver_args={"device": device}))
    b_val = jax.random.normal(jax.random.PRNGKey(0), (batch_size, n))
    (x_sol,) = jitted_layer(b_val)

    assert x_sol.shape == (batch_size, n)
    assert jax.numpy.allclose(x_sol, b_val, atol=1e-4)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_jax_jit_batch_size_one(clear_jax_cache, device):
    """Regression: batch_size=1 should preserve batch dim."""
    if device == "cuda" and not HAS_CUDA:
        pytest.skip("CUDA not available")
    jax = pytest.importorskip("jax")
    from cvxpylayers.jax import CvxpyLayer as JaxCvxpyLayer

    n = 5
    x = cp.Variable(n)
    b = cp.Parameter(n)
    problem = cp.Problem(cp.Minimize(cp.sum_squares(x - b)))
    layer = JaxCvxpyLayer(problem, parameters=[b], variables=[x], solver="MOREAU")

    jitted_layer = jax.jit(lambda b: layer(b, solver_args={"device": device}))
    b_val = jax.random.normal(jax.random.PRNGKey(0), (1, n))
    (x_sol,) = jitted_layer(b_val)

    assert x_sol.shape == (1, n), f"Expected (1, {n}), got {x_sol.shape}"


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_jax_jit_gradient(clear_jax_cache, device):
    """Test gradients through jax.jit."""
    if device == "cuda" and not HAS_CUDA:
        pytest.skip("CUDA not available")
    jax = pytest.importorskip("jax")
    jnp = jax.numpy
    from cvxpylayers.jax import CvxpyLayer as JaxCvxpyLayer

    n = 3
    x = cp.Variable(n)
    b = cp.Parameter(n)
    problem = cp.Problem(cp.Minimize(cp.sum_squares(x - b)))
    layer = JaxCvxpyLayer(problem, parameters=[b], variables=[x], solver="MOREAU")

    @jax.jit
    def loss_fn(b_val):
        (x,) = layer(b_val, solver_args={"device": device})
        return jnp.sum(x)

    grad_fn = jax.jit(jax.grad(loss_fn))
    b_val = jnp.array([1.0, 2.0, 3.0])
    grad = grad_fn(b_val)

    # For min ||x-b||^2, x*=b, so dx*/db = I, d/db sum(x*) = [1,1,1]
    expected_grad = jnp.ones(n)
    assert jnp.allclose(grad, expected_grad, atol=1e-4)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_jax_vmap_external(clear_jax_cache, device):
    """Test using jax.vmap externally on the layer."""
    if device == "cuda" and not HAS_CUDA:
        pytest.skip("CUDA not available")
    jax = pytest.importorskip("jax")
    jnp = jax.numpy
    from cvxpylayers.jax import CvxpyLayer as JaxCvxpyLayer

    n, batch_size = 3, 4
    x = cp.Variable(n)
    b = cp.Parameter(n)
    problem = cp.Problem(cp.Minimize(cp.sum_squares(x - b)))
    layer = JaxCvxpyLayer(problem, parameters=[b], variables=[x], solver="MOREAU")

    def solve_single(b_val):
        (x,) = layer(b_val, solver_args={"device": device})
        return x

    vmapped_solve = jax.vmap(solve_single)
    b_batch = jax.random.normal(jax.random.PRNGKey(0), (batch_size, n))
    x_batch = vmapped_solve(b_batch)

    assert x_batch.shape == (batch_size, n)
    assert jnp.allclose(x_batch, b_batch, atol=1e-4)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_jax_jit_vmap_composition(clear_jax_cache, device):
    """Test jit(vmap(...)) composition."""
    if device == "cuda" and not HAS_CUDA:
        pytest.skip("CUDA not available")
    jax = pytest.importorskip("jax")
    jnp = jax.numpy
    from cvxpylayers.jax import CvxpyLayer as JaxCvxpyLayer

    n, batch_size = 3, 4
    x = cp.Variable(n)
    b = cp.Parameter(n)
    problem = cp.Problem(cp.Minimize(cp.sum_squares(x - b)))
    layer = JaxCvxpyLayer(problem, parameters=[b], variables=[x], solver="MOREAU")

    def solve_single(b_val):
        (x,) = layer(b_val, solver_args={"device": device})
        return x

    jit_vmapped = jax.jit(jax.vmap(solve_single))
    b_batch = jax.random.normal(jax.random.PRNGKey(0), (batch_size, n))
    x_batch = jit_vmapped(b_batch)

    assert x_batch.shape == (batch_size, n)
    assert jnp.allclose(x_batch, b_batch, atol=1e-4)


# ============================================================================
# Warm-starting tests
# ============================================================================


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_warm_start_basic(device):
    """Test that warm-started solve produces the same solution as cold solve."""
    if device == "cuda" and not HAS_CUDA:
        pytest.skip("CUDA not available")

    n = 5
    x = cp.Variable(n)
    b = cp.Parameter(n)

    problem = cp.Problem(cp.Minimize(cp.sum_squares(x - b)))
    layer = CvxpyLayer(problem, parameters=[b], variables=[x], solver="MOREAU")

    b_val = torch.randn(n, device=device)

    # First call (cold, populates cache)
    (x_cold,) = layer(b_val)

    # Second call with warm start (uses cached solution)
    b_val2 = b_val + 0.01 * torch.randn(n, device=device)
    (x_warm,) = layer(b_val2, warm_start=True)

    # Also solve cold for comparison
    (x_cold2,) = layer(b_val2)

    # Warm and cold should produce same solution
    assert torch.allclose(x_warm, x_cold2, atol=1e-5), (
        f"Warm-started solve should match cold: diff={torch.norm(x_warm - x_cold2)}"
    )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_warm_start_gradients(device):
    """Test that backward() works correctly with warm_start=True."""
    if device == "cuda" and not HAS_CUDA:
        pytest.skip("CUDA not available")

    n = 4
    x = cp.Variable(n)
    b = cp.Parameter(n)

    problem = cp.Problem(cp.Minimize(cp.sum_squares(x - b)))
    layer = CvxpyLayer(problem, parameters=[b], variables=[x], solver="MOREAU")

    # First call to populate cache
    b_val1 = torch.randn(n, device=device, requires_grad=True)
    (x1,) = layer(b_val1, warm_start=True)
    x1.sum().backward()
    grad1 = b_val1.grad.clone()

    # Second call with warm start
    b_val2 = torch.randn(n, device=device, requires_grad=True)
    (x2,) = layer(b_val2, warm_start=True)
    x2.sum().backward()
    grad2 = b_val2.grad.clone()

    # For min ||x-b||^2, x*=b, so dx*/db = I, d/db sum(x*) = [1,1,...,1]
    expected_grad = torch.ones(n, dtype=torch.float64, device=device)
    assert torch.allclose(grad1, expected_grad, atol=1e-4), (
        f"Expected grad {expected_grad}, got {grad1}"
    )
    assert torch.allclose(grad2, expected_grad, atol=1e-4), (
        f"Expected grad {expected_grad}, got {grad2}"
    )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_warm_start_batched(device):
    """Test warm-starting with batched problems."""
    if device == "cuda" and not HAS_CUDA:
        pytest.skip("CUDA not available")

    n = 3
    batch_size = 4
    x = cp.Variable(n)
    b = cp.Parameter(n)

    problem = cp.Problem(cp.Minimize(cp.sum_squares(x - b)))
    layer = CvxpyLayer(problem, parameters=[b], variables=[x], solver="MOREAU")

    # First batched call
    b_val1 = torch.randn(batch_size, n, device=device)
    (x1,) = layer(b_val1, warm_start=True)

    assert x1.shape == (batch_size, n)
    assert torch.allclose(x1, b_val1, atol=1e-4)

    # Second batched call with warm start
    b_val2 = b_val1 + 0.01 * torch.randn(batch_size, n, device=device)
    (x2,) = layer(b_val2, warm_start=True)

    assert x2.shape == (batch_size, n)
    assert torch.allclose(x2, b_val2, atol=1e-4)


def test_warm_start_non_moreau_raises():
    """Test that warm_start=True raises ValueError with non-Moreau solver."""
    n = 3
    x = cp.Variable(n)
    b = cp.Parameter(n)

    problem = cp.Problem(cp.Minimize(cp.sum_squares(x - b)))
    layer = CvxpyLayer(problem, parameters=[b], variables=[x], solver="DIFFCP")

    b_val = torch.randn(n)
    with pytest.raises(ValueError, match="warm_start=True is only supported with solver='MOREAU'"):
        layer(b_val, warm_start=True)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_warm_start_training_loop(device):
    """Test warm-starting in a multi-step training loop."""
    if device == "cuda" and not HAS_CUDA:
        pytest.skip("CUDA not available")

    n = 3
    x = cp.Variable(n)
    b = cp.Parameter(n)

    problem = cp.Problem(cp.Minimize(cp.sum_squares(x - b)))
    layer = CvxpyLayer(problem, parameters=[b], variables=[x], solver="MOREAU")

    # Simulate a training loop where parameters change gradually
    b_val = torch.randn(n, device=device, requires_grad=True)
    losses = []

    for _ in range(5):
        (x_sol,) = layer(b_val, warm_start=True)
        loss = x_sol.sum()
        losses.append(loss.item())

        # Check gradient
        loss.backward()
        assert b_val.grad is not None

        # Simulate parameter update (small step)
        with torch.no_grad():
            b_val = (b_val - 0.01 * b_val.grad).requires_grad_(True)

    # Verify the layer's warm start cache was updated
    assert layer._warm_start_cache is not None


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_warm_start_cache_updated_without_flag(device):
    """Test that warm start cache is updated even when warm_start=False."""
    if device == "cuda" and not HAS_CUDA:
        pytest.skip("CUDA not available")

    n = 3
    x = cp.Variable(n)
    b = cp.Parameter(n)

    problem = cp.Problem(cp.Minimize(cp.sum_squares(x - b)))
    layer = CvxpyLayer(problem, parameters=[b], variables=[x], solver="MOREAU")

    # Call without warm_start flag
    b_val = torch.randn(n, device=device)
    (x1,) = layer(b_val)

    # Cache should be populated
    assert layer._warm_start_cache is not None

    # Now call with warm_start=True — should use the cached solution
    b_val2 = b_val + 0.01 * torch.randn(n, device=device)
    (x2,) = layer(b_val2, warm_start=True)
    assert torch.allclose(x2, b_val2, atol=1e-4)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_warm_start_jax_basic(device):
    """Test JAX warm-starting produces correct solutions."""
    if device == "cuda" and not HAS_CUDA:
        pytest.skip("CUDA not available")
    from cvxpylayers.jax import CvxpyLayer as JaxCvxpyLayer

    n = 5
    x = cp.Variable(n)
    b = cp.Parameter(n)

    problem = cp.Problem(cp.Minimize(cp.sum_squares(x - b)))
    layer = JaxCvxpyLayer(problem, parameters=[b], variables=[x], solver="MOREAU")

    b_val = jnp.array(np.random.randn(n))

    # First call (cold, populates cache)
    (x_cold,) = layer(b_val, solver_args={"device": device})

    # Verify solution is correct
    assert jnp.allclose(x_cold, b_val, atol=1e-4)

    # Second call with warm start (uses cached solution)
    b_val2 = b_val + 0.01 * jnp.array(np.random.randn(n))
    (x_warm,) = layer(b_val2, warm_start=True, solver_args={"device": device})

    assert jnp.allclose(x_warm, b_val2, atol=1e-4), (
        f"Warm-started solve incorrect: diff={jnp.linalg.norm(x_warm - b_val2)}"
    )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_warm_start_jax_gradients(device):
    """Test JAX gradients work with warm_start=True."""
    if device == "cuda" and not HAS_CUDA:
        pytest.skip("CUDA not available")
    from cvxpylayers.jax import CvxpyLayer as JaxCvxpyLayer

    n = 5
    x = cp.Variable(n)
    b = cp.Parameter(n)

    problem = cp.Problem(cp.Minimize(cp.sum_squares(x - b)))
    layer = JaxCvxpyLayer(problem, parameters=[b], variables=[x], solver="MOREAU")

    def solve_and_sum(b_val):
        (x_sol,) = layer(b_val, warm_start=True, solver_args={"device": device})
        return jnp.sum(x_sol)

    b_val = jnp.array(np.random.randn(n))

    # First call to populate cache
    grad1 = jax.grad(solve_and_sum)(b_val)

    # For min ||x-b||^2, x*=b, so dx*/db = I, d/db sum(x*) = [1,1,...,1]
    expected_grad = jnp.ones(n)
    assert jnp.allclose(grad1, expected_grad, atol=1e-4), (
        f"Expected grad {expected_grad}, got {grad1}"
    )

    # Second call with warm start
    b_val2 = b_val + 0.01 * jnp.array(np.random.randn(n))
    grad2 = jax.grad(solve_and_sum)(b_val2)
    assert jnp.allclose(grad2, expected_grad, atol=1e-4), (
        f"Expected grad {expected_grad}, got {grad2}"
    )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_warm_start_jax_batched(device):
    """Test JAX warm-starting with batched problems."""
    if device == "cuda" and not HAS_CUDA:
        pytest.skip("CUDA not available")
    from cvxpylayers.jax import CvxpyLayer as JaxCvxpyLayer

    n = 3
    batch_size = 4
    x = cp.Variable(n)
    b = cp.Parameter(n)

    problem = cp.Problem(cp.Minimize(cp.sum_squares(x - b)))
    layer = JaxCvxpyLayer(problem, parameters=[b], variables=[x], solver="MOREAU")

    # First batched call
    b_val1 = jnp.array(np.random.randn(batch_size, n))
    (x1,) = layer(b_val1, solver_args={"device": device})

    assert x1.shape == (batch_size, n)
    assert jnp.allclose(x1, b_val1, atol=1e-4)

    # Second batched call with warm start
    b_val2 = b_val1 + 0.01 * jnp.array(np.random.randn(batch_size, n))
    (x2,) = layer(b_val2, warm_start=True, solver_args={"device": device})

    assert x2.shape == (batch_size, n)
    assert jnp.allclose(x2, b_val2, atol=1e-4)


def test_warm_start_jax_non_moreau_raises():
    """Test that JAX warm_start=True raises ValueError with non-Moreau solver."""
    from cvxpylayers.jax import CvxpyLayer as JaxCvxpyLayer

    n = 3
    x = cp.Variable(n)
    b = cp.Parameter(n)

    problem = cp.Problem(cp.Minimize(cp.sum_squares(x - b)))
    layer = JaxCvxpyLayer(problem, parameters=[b], variables=[x])

    b_val = jnp.array(np.random.randn(n))
    with pytest.raises(ValueError, match="warm_start=True is only supported with solver='MOREAU'"):
        layer(b_val, warm_start=True)
