"""Tests for the custom SolverInterface plumbing.

Three solver implementations are tested:
  (a) NumpySingleMock  — implements solve_numpy / derivative_numpy (single-problem)
  (b) TorchBatchMock   — implements solve_torch_batch / derivative_torch_batch
  (c) CvxpygenSolverInterface — wraps a CVXPYgen-generated C solver (requires
      cvxpygen + cmake; skipped when not available)

Mocks (a) and (b) delegate to diffcp via the structure stored in a
DiffcpSolverContext so that we can compare the output with the reference
diffcp-based CvxpyLayer.
"""
import os
import subprocess
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import cvxpy as cp  # noqa: E402
import diffcp  # noqa: E402

from cvxpylayers.interfaces.base import SolverInterface  # noqa: E402
from cvxpylayers.interfaces.diffcp_if import (  # noqa: E402
    _build_diffcp_matrices,
    _compute_gradients,
    dims_to_solver_dict,
)
from cvxpylayers.torch import CvxpyLayer  # noqa: E402

torch.set_default_dtype(torch.double)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_problem():
    """Simple LP: min 0.5*||Ax - b||_1  s.t. x >= 0."""
    n, m = 2, 3
    x = cp.Variable(n)
    A = cp.Parameter((m, n))
    b = cp.Parameter(m)
    objective = cp.Minimize(0.5 * cp.pnorm(A @ x - b, p=1))
    problem = cp.Problem(objective, [x >= 0])
    assert problem.is_dpp()
    return problem, A, b, x


def _reference_layer(problem, A_param, b_param, x_var):
    return CvxpyLayer(problem, parameters=[A_param, b_param], variables=[x_var])


def _random_inputs(m, n, seed=0):
    torch.manual_seed(seed)
    A_t = torch.randn(m, n, requires_grad=True, dtype=torch.double)
    b_t = torch.randn(m, requires_grad=True, dtype=torch.double)
    return A_t, b_t


def _random_inputs_batched(m, n, B, seed=1):
    torch.manual_seed(seed)
    A_t = torch.randn(B, m, n, requires_grad=True, dtype=torch.double)
    b_t = torch.randn(B, m, requires_grad=True, dtype=torch.double)
    return A_t, b_t


# ---------------------------------------------------------------------------
# Mock solver (a): numpy single-problem, wraps diffcp
# ---------------------------------------------------------------------------

class NumpySingleMock(SolverInterface):
    """Wraps diffcp in the ``solve_numpy`` / ``derivative_numpy`` interface.

    Stores the DiffcpSolverContext so it can reconstruct the full A/b/c
    matrices from the non-zero value vectors handed to solve_numpy.
    """

    canon_solver: str = "DIFFCP"

    def __init__(self, solver_ctx):
        self._ctx = solver_ctx

    def solve_numpy(self, P, q, A, dims, solver_args, needs_grad):
        ctx = self._ctx
        # Reconstruct diffcp's As, bs, cs from non-zero values (batch size 1)
        q_t = torch.from_numpy(q).unsqueeze(1)   # (nnz_q, 1)
        A_t = torch.from_numpy(A).unsqueeze(1)   # (nnz_A, 1)
        As, bs, cs, b_idxs = _build_diffcp_matrices(
            A_t, q_t, ctx.A_structure, ctx.A_shape, ctx.b_idx, 1,
        )
        cone_dict = dims_to_solver_dict(ctx.dims)
        if needs_grad:
            xs, ys, _, _, adj = diffcp.solve_and_derivative_batch(
                As, bs, cs, [cone_dict], **(solver_args or {}),
            )
        else:
            xs, ys, _ = diffcp.solve_only_batch(
                As, bs, cs, [cone_dict], **(solver_args or {}),
            )
            adj = None
        primal = xs[0].astype(np.float64)
        dual = ys[0].astype(np.float64)
        # Pack the data needed for derivative
        adjoint_data = (adj, bs, [b_idxs[0]], 1) if adj is not None else None
        return primal, dual, adjoint_data

    def derivative_numpy(self, dprimal, ddual, adjoint_data):
        if adjoint_data is None:
            raise RuntimeError("adjoint_data is None; solve was called with needs_grad=False")
        adj, bs, b_idxs, batch_size = adjoint_data
        dp = dprimal[np.newaxis, :]
        dd = ddual[np.newaxis, :]
        dq_list, dA_list = _compute_gradients(adj, dp, dd, bs, b_idxs, batch_size)
        dq = dq_list[0].astype(np.float64)
        dA = dA_list[0].astype(np.float64)
        return None, dq, dA  # dP=None (LP)


# ---------------------------------------------------------------------------
# Mock solver (b): torch batch, wraps diffcp
# ---------------------------------------------------------------------------

class TorchBatchMock(SolverInterface):
    """Wraps diffcp in the ``solve_torch_batch`` / ``derivative_torch_batch`` interface."""

    canon_solver: str = "DIFFCP"

    def __init__(self, solver_ctx):
        self._ctx = solver_ctx

    def solve_torch_batch(self, P, q, A, dims, solver_args, needs_grad):
        ctx = self._ctx
        batch_size = q.shape[0]
        # Inputs are (B, nnz); diffcp helpers expect (nnz, B) for _build_diffcp_matrices
        q_bl = q.T.contiguous()   # (nnz_q, B)
        A_bl = A.T.contiguous()   # (nnz_A, B)
        As, bs, cs, b_idxs = _build_diffcp_matrices(
            A_bl, q_bl, ctx.A_structure, ctx.A_shape, ctx.b_idx, batch_size,
        )
        cone_dicts = [dims_to_solver_dict(ctx.dims)] * batch_size
        if needs_grad:
            xs, ys, _, _, adj = diffcp.solve_and_derivative_batch(
                As, bs, cs, cone_dicts, **(solver_args or {}),
            )
        else:
            xs, ys, _ = diffcp.solve_only_batch(
                As, bs, cs, cone_dicts, **(solver_args or {}),
            )
            adj = None
        primal = torch.stack([torch.from_numpy(x.astype(np.float64)) for x in xs])
        dual = torch.stack([torch.from_numpy(y.astype(np.float64)) for y in ys])
        adjoint_data = (adj, bs, b_idxs, batch_size) if adj is not None else None
        return primal, dual, adjoint_data

    def derivative_torch_batch(self, dprimal, ddual, adjoint_data):
        if adjoint_data is None:
            raise RuntimeError("adjoint_data is None; solve was called with needs_grad=False")
        adj, bs, b_idxs, batch_size = adjoint_data
        dp_np = dprimal.detach().cpu().numpy()
        dd_np = ddual.detach().cpu().numpy()
        dq_list, dA_list = _compute_gradients(adj, dp_np, dd_np, bs, b_idxs, batch_size)
        ref = dprimal
        dtype, device = ref.dtype, ref.device
        # _compute_gradients returns lists of 1-D arrays; stack → (B, nnz)
        dq = torch.stack(
            [torch.from_numpy(g.astype(np.float64)) for g in dq_list]
        ).to(dtype=dtype, device=device)
        dA = torch.stack(
            [torch.from_numpy(g.astype(np.float64)) for g in dA_list]
        ).to(dtype=dtype, device=device)
        return None, dq, dA  # dP=None (LP)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def problem_and_ref():
    problem, A_param, b_param, x_var = _make_problem()
    ref = _reference_layer(problem, A_param, b_param, x_var)
    return problem, A_param, b_param, x_var, ref


# ---------------------------------------------------------------------------
# Tests: @require_one_of enforcement
# ---------------------------------------------------------------------------

def test_require_one_of_solve_raises():
    """A SolverInterface subclass with no solve override must be rejected."""
    with pytest.raises(TypeError, match="must override at least one of"):
        class BadSolver(SolverInterface):
            def derivative_numpy(self, dp, dd, adj):
                return None, dp, dd


def test_require_one_of_derivative_raises():
    """A SolverInterface subclass with no derivative override must be rejected."""
    with pytest.raises(TypeError, match="must override at least one of"):
        class BadSolver(SolverInterface):
            def solve_numpy(self, P, q, A, dims, sa, ng):
                return q, q, None


def test_require_one_of_both_ok():
    """A SolverInterface subclass with at least one of each passes."""
    class OkSolver(SolverInterface):
        def solve_numpy(self, P, q, A, dims, sa, ng):
            return q, q, None

        def derivative_numpy(self, dp, dd, adj):
            return None, dp, dd

    assert OkSolver()  # no error


# ---------------------------------------------------------------------------
# Tests: CvxpyLayer construction with custom solver
# ---------------------------------------------------------------------------

def test_layer_construction_numpy_mock(problem_and_ref):
    problem, A_param, b_param, x_var, ref = problem_and_ref
    mock = NumpySingleMock(ref.ctx.solver_ctx)
    layer = CvxpyLayer(problem, parameters=[A_param, b_param], variables=[x_var],
                       solver=mock)
    assert isinstance(layer.ctx.solver, SolverInterface)
    assert layer.ctx.solver is mock


def test_layer_construction_torch_mock(problem_and_ref):
    problem, A_param, b_param, x_var, ref = problem_and_ref
    mock = TorchBatchMock(ref.ctx.solver_ctx)
    layer = CvxpyLayer(problem, parameters=[A_param, b_param], variables=[x_var],
                       solver=mock)
    assert isinstance(layer.ctx.solver, SolverInterface)
    assert layer.ctx.solver is mock


# ---------------------------------------------------------------------------
# Tests: forward pass correctness (unbatched)
# ---------------------------------------------------------------------------

def test_forward_numpy_mock_matches_reference(problem_and_ref):
    n, m = 2, 3
    problem, A_param, b_param, x_var, ref = problem_and_ref
    mock = NumpySingleMock(ref.ctx.solver_ctx)
    custom_layer = CvxpyLayer(problem, parameters=[A_param, b_param],
                              variables=[x_var], solver=mock)

    A_t, b_t = _random_inputs(m, n)
    (x_ref,) = ref(A_t, b_t)
    (x_custom,) = custom_layer(A_t, b_t)

    assert x_custom.shape == x_ref.shape
    assert torch.allclose(x_custom, x_ref, atol=1e-5)


def test_forward_torch_mock_matches_reference(problem_and_ref):
    n, m = 2, 3
    problem, A_param, b_param, x_var, ref = problem_and_ref
    mock = TorchBatchMock(ref.ctx.solver_ctx)
    custom_layer = CvxpyLayer(problem, parameters=[A_param, b_param],
                              variables=[x_var], solver=mock)

    A_t, b_t = _random_inputs(m, n)
    (x_ref,) = ref(A_t, b_t)
    (x_custom,) = custom_layer(A_t, b_t)

    assert x_custom.shape == x_ref.shape
    assert torch.allclose(x_custom, x_ref, atol=1e-5)


# ---------------------------------------------------------------------------
# Tests: forward pass — batched inputs
# ---------------------------------------------------------------------------

def test_forward_torch_mock_batched(problem_and_ref):
    n, m, B = 2, 3, 4
    problem, A_param, b_param, x_var, ref = problem_and_ref
    mock = TorchBatchMock(ref.ctx.solver_ctx)
    custom_layer = CvxpyLayer(problem, parameters=[A_param, b_param],
                              variables=[x_var], solver=mock)

    A_t, b_t = _random_inputs_batched(m, n, B)
    (x_ref,) = ref(A_t, b_t)
    (x_custom,) = custom_layer(A_t, b_t)

    assert x_custom.shape == x_ref.shape
    assert torch.allclose(x_custom, x_ref, atol=1e-5)


# ---------------------------------------------------------------------------
# Tests: backward pass (gradient flow)
# ---------------------------------------------------------------------------

def test_backward_numpy_mock(problem_and_ref):
    n, m = 2, 3
    problem, A_param, b_param, x_var, ref = problem_and_ref
    mock = NumpySingleMock(ref.ctx.solver_ctx)
    custom_layer = CvxpyLayer(problem, parameters=[A_param, b_param],
                              variables=[x_var], solver=mock)

    A_t, b_t = _random_inputs(m, n)
    (x_ref,) = ref(A_t, b_t)
    x_ref.sum().backward()
    dA_ref, db_ref = A_t.grad.clone(), b_t.grad.clone()

    A_t2, b_t2 = _random_inputs(m, n)  # fresh tensors, same values (same seed)
    (x_custom,) = custom_layer(A_t2, b_t2)
    x_custom.sum().backward()

    assert torch.allclose(A_t2.grad, dA_ref, atol=1e-5)
    assert torch.allclose(b_t2.grad, db_ref, atol=1e-5)


def test_backward_torch_mock(problem_and_ref):
    n, m = 2, 3
    problem, A_param, b_param, x_var, ref = problem_and_ref
    mock = TorchBatchMock(ref.ctx.solver_ctx)
    custom_layer = CvxpyLayer(problem, parameters=[A_param, b_param],
                              variables=[x_var], solver=mock)

    A_t, b_t = _random_inputs(m, n)
    (x_ref,) = ref(A_t, b_t)
    x_ref.sum().backward()
    dA_ref, db_ref = A_t.grad.clone(), b_t.grad.clone()

    A_t2, b_t2 = _random_inputs(m, n)
    (x_custom,) = custom_layer(A_t2, b_t2)
    x_custom.sum().backward()

    assert torch.allclose(A_t2.grad, dA_ref, atol=1e-5)
    assert torch.allclose(b_t2.grad, db_ref, atol=1e-5)


def test_backward_torch_mock_batched(problem_and_ref):
    n, m, B = 2, 3, 4
    problem, A_param, b_param, x_var, ref = problem_and_ref
    mock = TorchBatchMock(ref.ctx.solver_ctx)
    custom_layer = CvxpyLayer(problem, parameters=[A_param, b_param],
                              variables=[x_var], solver=mock)

    A_t, b_t = _random_inputs_batched(m, n, B)
    (x_ref,) = ref(A_t, b_t)
    x_ref.sum().backward()
    dA_ref, db_ref = A_t.grad.clone(), b_t.grad.clone()

    A_t2, b_t2 = _random_inputs_batched(m, n, B)
    (x_custom,) = custom_layer(A_t2, b_t2)
    x_custom.sum().backward()

    assert torch.allclose(A_t2.grad, dA_ref, atol=1e-5)
    assert torch.allclose(b_t2.grad, db_ref, atol=1e-5)


# ---------------------------------------------------------------------------
# Tests: ring auto-conversion (numpy single → torch layer)
# ---------------------------------------------------------------------------

def test_ring_numpy_single_to_torch_layer(problem_and_ref):
    """A solver that only implements solve_numpy / derivative_numpy should work
    when called through the Torch CvxpyLayer (ring converts automatically)."""
    n, m = 2, 3
    problem, A_param, b_param, x_var, ref = problem_and_ref
    mock = NumpySingleMock(ref.ctx.solver_ctx)

    # Explicitly use the torch layer — ring will call solve_torch_batch →
    # solve_torch → solve_numpy
    custom_layer = CvxpyLayer(problem, parameters=[A_param, b_param],
                              variables=[x_var], solver=mock)
    A_t, b_t = _random_inputs(m, n)
    (x_custom,) = custom_layer(A_t, b_t)
    (x_ref,) = ref(A_t, b_t)
    assert torch.allclose(x_custom, x_ref, atol=1e-5)


# ===========================================================================
# CVXPYgen integration
# ===========================================================================
# Requires:  pip install cvxpygen
#            cmake >= 3.5 (to compile the generated C extension)
#
# Pattern demonstrated
# --------------------
# * Forward  : CVXPYgen's compiled C solver (cpg_solve) — fast hardware-level solve
# * Backward : CVXPYgen's cpg_gradient for param-level grads, then converted to
#              dq/dA (what _ScipySparseMatmul.backward expects) via pseudoinverse.
#
# The two lifecycle hooks used here are:
#   setup(ctx)         — called once in CvxpyLayer.__init__ after canonicalization
#   set_params(params) — called each CvxpyLayer.forward() with raw numpy param arrays
# ===========================================================================

def _cmake_available() -> bool:
    try:
        subprocess.run(["cmake", "--version"], capture_output=True, check=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


pytestmark_cmake = pytest.mark.skipif(
    not _cmake_available(), reason="cmake required to compile CVXPYgen C code"
)


class CvxpygenSolverInterface(SolverInterface):
    """Hybrid CVXPYgen + diffcp solver interface.

    **Forward**  : CVXPYgen's compiled C solver is called for the primal/dual
                   solution.  A parallel diffcp solve is run to produce the cone
                   primal/dual vectors in the format expected by
                   ``_recover_results``, and to capture the diffcp adjoint for
                   the backward pass.  (In a production use-case you would store
                   the diffcp SolverContext once and skip the re-canonicalization,
                   or use CVXPYgen's own intermediate gradient data exclusively.)

    **Backward** : CVXPYgen's ``cpg_gradient`` computes parameter-level gradients
                   ``d(loss)/d(params_i)``.  These are converted to the
                   ``(dq_eval, dA_eval)`` format that the sparse-matmul autograd
                   chain expects via the minimum-norm pseudoinverse:

                       scipy_q.T @ dq + scipy_A.T @ dA = dparam_flat  (+ 0 for const)

    **Lifecycle hooks used**

    ``setup(ctx)``
        Stores the ``LayersContext``, pre-computes the pseudoinverse factor
        ``M = scipy_q.T @ scipy_q + scipy_A.T @ scipy_A`` for fast backward
        computation, and imports the CVXPYgen module.

    ``set_params(params)``
        Receives the raw numpy parameter arrays and sets them directly on the
        CVXPY Parameter objects so that ``cpg_solve`` and ``cpg_gradient``
        can read them via the standard CVXPY ``param.value`` interface.
    """

    canon_solver: str = "DIFFCP"

    def __init__(
        self,
        problem: cp.Problem,
        requested_vars: list,
        requested_params: list,
        diffcp_solver_ctx,          # from reference CvxpyLayer.ctx.solver_ctx
        cpg_solve_fn,               # cpg_solve from cpg_solver.py
        cpg_solve_and_grad_fn,      # cpg_solve_and_gradient_info from cpg_solver.py
        cpg_gradient_fn,            # cpg_gradient from cpg_solver.py
    ) -> None:
        self._problem = problem
        self._requested_vars = requested_vars
        self._requested_params = requested_params
        self._diffcp_ctx = diffcp_solver_ctx
        self._cpg_solve = cpg_solve_fn
        self._cpg_solve_and_grad = cpg_solve_and_grad_fn
        self._cpg_gradient = cpg_gradient_fn
        # Filled in setup():
        self._ctx = None
        self._scipy_q = None
        self._scipy_A = None
        self._pinv_factor = None   # pre-factored (K @ K.T) for pseudoinverse
        self._K = None             # [scipy_q.T | scipy_A.T]

    # ------------------------------------------------------------------
    # Lifecycle hooks
    # ------------------------------------------------------------------

    def setup(self, ctx) -> None:
        """Store context and pre-compute pseudoinverse factor for the backward."""
        self._ctx = ctx
        self._scipy_q = ctx.q.tocsr()                              # (n_c, n_params+1)
        self._scipy_A = ctx.reduced_A.reduced_mat.tocsr()          # (nnz_A, n_params+1)
        q_dense = self._scipy_q.toarray()
        A_dense = self._scipy_A.toarray()
        # K has shape (n_params+1, n_c + nnz_A)
        self._K = np.hstack([q_dense.T, A_dense.T])
        # Pre-factor M = K @ K.T  (n_params+1, n_params+1)
        M = self._K @ self._K.T
        # Store as (M, K) — use lstsq for stability
        self._M = M

    def set_params(self, params: list) -> None:
        """Set current parameter values on the CVXPY problem for cpg_solve."""
        for param_obj, val in zip(self._requested_params, params):
            param_obj.value = val

    # ------------------------------------------------------------------
    # Solve / derivative
    # ------------------------------------------------------------------

    def solve_numpy(self, P, q, A, dims, solver_args, needs_grad):
        """Forward: CVXPYgen solve + parallel diffcp for primal/dual extraction."""
        # 1. CVXPYgen forward solve (fast C extension)
        if needs_grad:
            _, cpg_grad_primal, cpg_grad_dual = self._cpg_solve_and_grad(self._problem)
        else:
            self._cpg_solve(self._problem)
            cpg_grad_primal, cpg_grad_dual = None, None

        # 2. Parallel diffcp solve to get cone primal/dual for _recover_results
        #    (This is where a production wrapper would skip diffcp and instead
        #     reconstruct the cone vector from CVXPYgen's variable values.)
        ctx = self._diffcp_ctx
        q_t = torch.from_numpy(q).unsqueeze(1)
        A_t = torch.from_numpy(A).unsqueeze(1)
        As, bs, cs, b_idxs = _build_diffcp_matrices(
            A_t, q_t, ctx.A_structure, ctx.A_shape, ctx.b_idx, 1,
        )
        cone_dict = dims_to_solver_dict(ctx.dims)
        if needs_grad:
            xs, ys, _, _, adj = diffcp.solve_and_derivative_batch(
                As, bs, cs, [cone_dict], **(solver_args or {}),
            )
        else:
            xs, ys, _ = diffcp.solve_only_batch(
                As, bs, cs, [cone_dict], **(solver_args or {}),
            )
            adj = None

        primal = xs[0].astype(np.float64)
        dual = ys[0].astype(np.float64)

        # Verify CVXPYgen solution matches diffcp (optional sanity check)
        for var_info, cvxpy_var in zip(self._ctx.var_recover, self._requested_vars):
            if var_info.source == "primal" and var_info.primal is not None:
                cpg_val = np.asarray(cvxpy_var.value).flatten(order="F")
                diffcp_val = primal[var_info.primal]
                assert np.allclose(cpg_val, diffcp_val, atol=1e-4), (
                    f"CVXPYgen and diffcp solutions differ for {cvxpy_var.name()}: "
                    f"max|diff| = {np.max(np.abs(cpg_val - diffcp_val)):.2e}"
                )

        adjoint_data = (
            (adj, bs, [b_idxs[0]], 1, cpg_grad_primal, cpg_grad_dual)
            if needs_grad else None
        )
        return primal, dual, adjoint_data

    def derivative_numpy(self, dprimal, ddual, adjoint_data):
        """Backward: CVXPYgen cpg_gradient → pseudoinverse → (dq, dA)."""
        adj, bs, b_idxs, batch_size, cpg_grad_primal, cpg_grad_dual = adjoint_data

        # 1. Set variable .gradient from cone dprimal (via var_recover slices)
        for var_info, cvxpy_var in zip(self._ctx.var_recover, self._requested_vars):
            if var_info.source == "primal" and var_info.primal is not None:
                cvxpy_var.gradient = (
                    dprimal[var_info.primal].reshape(cvxpy_var.shape, order="F")
                )
            elif var_info.source == "dual" and var_info.dual is not None:
                cvxpy_var.gradient = (
                    ddual[var_info.dual].reshape(cvxpy_var.shape, order="F")
                )

        # 2. CVXPYgen backward → param.gradient for each CVXPY parameter
        self._cpg_gradient(self._problem, cpg_grad_primal, cpg_grad_dual)

        # 3. Collect and flatten param gradients in user parameter order
        dparam_list = []
        for param_obj in self._requested_params:
            g = np.asarray(param_obj.gradient)
            dparam_list.append(g.flatten(order="F"))
        dparam_flat = np.concatenate(dparam_list)  # (n_params_total,)

        # 4. Convert dparam_flat → (dq, dA) via minimum-norm pseudoinverse.
        #
        #    The sparse-matmul autograd chain computes:
        #        d(loss)/d(p_stack) = scipy_q.T @ dq + scipy_A.T @ dA
        #
        #    CVXPYgen gives us d(loss)/d(p_stack)[:-1] = dparam_flat
        #    (the last element is the constant term, gradient = 0).
        #
        #    We solve the underdetermined system  K @ d = target  for minimum-
        #    norm d, where  K = [scipy_q.T | scipy_A.T]  and
        #    target = [dparam_flat; 0].
        #
        target = np.append(dparam_flat, 0.0)          # (n_params+1,)
        alpha, *_ = np.linalg.lstsq(self._M, target, rcond=None)
        d_combined = self._K.T @ alpha                # (n_c + nnz_A,)

        n_c = self._scipy_q.shape[0]
        dq = d_combined[:n_c]
        dA = d_combined[n_c:]
        return None, dq, dA                           # dP=None (LP)


# ---------------------------------------------------------------------------
# CVXPYgen fixtures and helper
# ---------------------------------------------------------------------------

def _generate_cpg_code(problem, code_dir, solver="SCS"):
    """Generate CVXPYgen code and import the resulting module."""
    cpg = pytest.importorskip("cvxpygen").cpg
    cpg.generate_code(problem, code_dir=code_dir, solver=solver, gradient=True)
    # Compile the Python C extension
    import subprocess
    result = subprocess.run(
        [sys.executable, "setup.py", "--quiet", "build_ext", "--inplace"],
        cwd=code_dir, capture_output=True, text=True,
    )
    if result.returncode != 0:
        pytest.skip(f"CVXPYgen compilation failed: {result.stderr}")

    # Import the generated module
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "cpg_solver", os.path.join(code_dir, "cpg_solver.py")
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["cpg_solver"] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# CVXPYgen tests
# ---------------------------------------------------------------------------

@pytestmark_cmake
def test_cvxpygen_forward(problem_and_ref, tmp_path):
    """CVXPYgen forward solution matches the reference diffcp solution."""
    problem, A_param, b_param, x_var, ref = problem_and_ref
    n, m = 2, 3
    code_dir = str(tmp_path / "cpg_nonneg_ls")

    cpg_mod_loaded = _generate_cpg_code(problem, code_dir, solver="SCS")
    cpg_solve_fn = cpg_mod_loaded.cpg_solve
    cpg_solve_and_grad_fn = cpg_mod_loaded.cpg_solve_and_gradient_info
    cpg_gradient_fn = cpg_mod_loaded.cpg_gradient

    solver = CvxpygenSolverInterface(
        problem, [x_var], [A_param, b_param],
        ref.ctx.solver_ctx,
        cpg_solve_fn, cpg_solve_and_grad_fn, cpg_gradient_fn,
    )
    layer = CvxpyLayer(
        problem, parameters=[A_param, b_param], variables=[x_var], solver=solver,
    )

    A_t, b_t = _random_inputs(m, n)
    (x_ref,) = ref(A_t, b_t)
    (x_custom,) = layer(A_t, b_t)

    assert x_custom.shape == x_ref.shape
    # CVXPYgen solution should closely match reference diffcp solution
    assert torch.allclose(x_custom, x_ref, atol=1e-4)


@pytestmark_cmake
def test_cvxpygen_backward(problem_and_ref, tmp_path):
    """CVXPYgen backward gradients match the reference diffcp gradients."""
    problem, A_param, b_param, x_var, ref = problem_and_ref
    n, m = 2, 3
    code_dir = str(tmp_path / "cpg_nonneg_ls_grad")

    cpg_mod_loaded = _generate_cpg_code(problem, code_dir, solver="SCS")
    cpg_solve_fn = cpg_mod_loaded.cpg_solve
    cpg_solve_and_grad_fn = cpg_mod_loaded.cpg_solve_and_gradient_info
    cpg_gradient_fn = cpg_mod_loaded.cpg_gradient

    solver = CvxpygenSolverInterface(
        problem, [x_var], [A_param, b_param],
        ref.ctx.solver_ctx,
        cpg_solve_fn, cpg_solve_and_grad_fn, cpg_gradient_fn,
    )
    layer = CvxpyLayer(
        problem, parameters=[A_param, b_param], variables=[x_var], solver=solver,
    )

    # Reference gradients
    A_t, b_t = _random_inputs(m, n)
    (x_ref,) = ref(A_t, b_t)
    x_ref.sum().backward()
    dA_ref, db_ref = A_t.grad.clone(), b_t.grad.clone()

    # CVXPYgen-backed gradients
    A_t2, b_t2 = _random_inputs(m, n)
    (x_custom,) = layer(A_t2, b_t2)
    x_custom.sum().backward()

    assert torch.allclose(A_t2.grad, dA_ref, atol=1e-4)
    assert torch.allclose(b_t2.grad, db_ref, atol=1e-4)
