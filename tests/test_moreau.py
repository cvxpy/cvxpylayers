"""test suite for Moreau solver."""

import cvxpy as cp
import numpy as np
import pytest
import torch

from cvxpylayers.torch import CvxpyLayer

# Skip all tests in this module if moreau is not installed
moreau = pytest.importorskip("moreau")
import moreau.torch as moreau_torch

# Check for CUDA availability
HAS_CUDA = torch.cuda.is_available() and moreau_torch.cuda_available()

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
        assert moreau_err < 1e-3, (
            f"Moreau var {i} error: ||Moreau - true|| = {moreau_err:.6e}"
        )

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
    sols_moreau = layer_moreau(
        *[torch.tensor(v, requires_grad=True) for v in param_vals_batch]
    )

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
            moreau_err = np.linalg.norm(
                sol_moreau[batch_idx].detach().cpu().numpy() - sol_true
            )
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


def test_backward_not_implemented():
    """Test that backward pass raises NotImplementedError."""
    n = 3
    x = cp.Variable(n)
    b = cp.Parameter(n)

    problem = cp.Problem(cp.Minimize(cp.sum_squares(x - b)))

    layer_moreau = CvxpyLayer(problem, parameters=[b], variables=[x], solver="MOREAU")

    # Create parameter with requires_grad=True
    b_val = torch.randn(n, requires_grad=True)

    # Forward pass should work
    (x_sol,) = layer_moreau(b_val)

    # Backward pass should raise NotImplementedError
    with pytest.raises(NotImplementedError, match="not yet implemented"):
        x_sol.sum().backward()


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

    # Verify solution is correct
    error = np.linalg.norm(x_sol.numpy() - true_sol)
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

    # Verify solution
    error = np.linalg.norm(x_sol.numpy() - true_sol)
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

        error = np.linalg.norm(x_sol[i].numpy() - true_sol)
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
