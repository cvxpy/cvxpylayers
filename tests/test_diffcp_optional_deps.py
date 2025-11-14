"""Test that DIFFCP works with torch-only or jax-only (not requiring both)."""

import sys
from unittest import mock

import cvxpy as cp


def test_diffcp_torch_without_jax():
    """Test that PyTorch CvxpyLayer with DIFFCP works without JAX installed."""
    print("Test 1: PyTorch CvxpyLayer with DIFFCP (no JAX)")
    print("=" * 60)

    # Simulate jax not being installed
    with mock.patch.dict(sys.modules, {"jax": None, "jax.numpy": None}):
        # Reload to pick up the mocked imports
        import importlib

        import cvxpylayers.interfaces.diffcp_if as diffcp_if

        importlib.reload(diffcp_if)

        # Now try to use PyTorch layer
        import torch

        from cvxpylayers.torch import CvxpyLayer

        # Simple QP: minimize ||x||^2 subject to Ax = b
        x = cp.Variable(2)
        A = cp.Parameter((1, 2))
        b = cp.Parameter(1)
        problem = cp.Problem(cp.Minimize(cp.sum_squares(x)), [A @ x == b])

        import numpy as np

        A_val = torch.tensor([[1.0, 1.0]], dtype=torch.float64, requires_grad=True)
        b_val = torch.tensor([2.0], dtype=torch.float64, requires_grad=True)

        try:
            # Create layer with DIFFCP solver
            layer = CvxpyLayer(problem, [A, b], [x], solver="DIFFCP")
            print("✓ Layer created successfully (JAX not required)")

            # Forward pass
            (x_sol,) = layer(A_val, b_val)
            print(f"✓ Forward pass works: x = {x_sol.detach().numpy()}")

            # Verify solution is correct (should be [1.0, 1.0])
            expected = np.array([1.0, 1.0])
            actual = x_sol.detach().numpy()
            error = np.linalg.norm(actual - expected)

            if error > 1e-3:
                print(f"✗ Solution incorrect (error = {error:.6e})")
                return False
            print(f"✓ Solution correct (error = {error:.6e})")

            # Backward pass
            loss = x_sol.sum()
            loss.backward()
            print("✓ Backward pass works")

            # Check gradients exist
            if A_val.grad is not None and b_val.grad is not None:
                print(f"✓ Gradients computed: dL/dA norm = {A_val.grad.norm():.6e}")
                print("✓ Test PASSED\n")
                return True
            else:
                print("✗ Gradients not computed")
                return False

        except ImportError as e:
            if "jax" in str(e).lower():
                print(f"✗ FAILED: Incorrectly requires JAX: {e}")
                return False
            else:
                # Some other import error
                raise
        except Exception as e:
            print(f"✗ FAILED: {e}")
            import traceback

            traceback.print_exc()
            return False


def test_diffcp_jax_without_torch():
    """Test that JAX CvxpyLayer with DIFFCP works without PyTorch installed."""
    print("Test 2: JAX CvxpyLayer with DIFFCP (no PyTorch)")
    print("=" * 60)

    # Simulate torch not being installed
    with mock.patch.dict(sys.modules, {"torch": None}):
        # Reload to pick up the mocked imports
        import importlib

        import cvxpylayers.interfaces.diffcp_if as diffcp_if

        importlib.reload(diffcp_if)

        # Now try to use JAX layer
        import jax
        import jax.numpy as jnp

        from cvxpylayers.jax import CvxpyLayer

        # Simple QP: minimize ||x||^2 subject to Ax = b
        x = cp.Variable(2)
        A = cp.Parameter((1, 2))
        b = cp.Parameter(1)
        problem = cp.Problem(cp.Minimize(cp.sum_squares(x)), [A @ x == b])

        A_val = jnp.array([[1.0, 1.0]])
        b_val = jnp.array([2.0])

        try:
            # Create layer with DIFFCP solver
            layer = CvxpyLayer(problem, [A, b], [x], solver="DIFFCP")
            print("✓ Layer created successfully (PyTorch not required)")

            # Forward pass
            (x_sol,) = layer(A_val, b_val)
            print(f"✓ Forward pass works: x = {x_sol}")

            # Verify solution is correct (should be [1.0, 1.0])
            expected = jnp.array([1.0, 1.0])
            error = jnp.linalg.norm(x_sol - expected)

            if error > 1e-3:
                print(f"✗ Solution incorrect (error = {error:.6e})")
                return False
            print(f"✓ Solution correct (error = {error:.6e})")

            # Backward pass - define a loss function and compute gradients
            def loss_fn(A, b):
                (x_sol,) = layer(A, b)
                return jnp.sum(x_sol)

            # Compute gradients
            grads = jax.grad(loss_fn, argnums=(0, 1))(A_val, b_val)
            dA, db = grads
            print("✓ Backward pass works")
            print(f"✓ Gradients computed: dL/dA norm = {jnp.linalg.norm(dA):.6e}")
            print("✓ Test PASSED\n")
            return True

        except ImportError as e:
            if "torch" in str(e).lower():
                print(f"✗ FAILED: Incorrectly requires PyTorch: {e}")
                return False
            else:
                # Some other import error
                raise
        except Exception as e:
            print(f"✗ FAILED: {e}")
            import traceback

            traceback.print_exc()
            return False


if __name__ == "__main__":
    print("Testing DIFFCP with optional torch/jax dependencies\n")
    print(
        "NOTE: These tests simulate missing dependencies using mocks.\n"
        "Full isolation would require separate Python processes.\n"
    )

    results = []

    # Test 1: PyTorch without JAX
    results.append(test_diffcp_torch_without_jax())

    # Test 2: JAX without PyTorch
    results.append(test_diffcp_jax_without_torch())

    # Summary
    print("=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Results: {passed}/{total} tests passed")

    if all(results):
        print("✓ All tests PASSED!")
        print("\nConclusion: DIFFCP works with:")
        print("  • PyTorch-only (no JAX required) - forward + backward")
        print("  • JAX-only (no PyTorch required) - forward + backward")
    else:
        print("✗ Some tests FAILED")
        exit(1)
