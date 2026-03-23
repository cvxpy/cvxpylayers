"""Tests for the custom SolverInterface plumbing.

Two mock solver implementations are tested:
  (a) NumpySingleMock  — implements solve_numpy / derivative_numpy (single-problem)
  (b) TorchBatchMock   — implements solve_torch_batch / derivative_torch_batch

Both mocks delegate to diffcp via the structure stored in a DiffcpSolverContext so
that we can compare the output with the reference diffcp-based CvxpyLayer.
"""
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
    assert layer.ctx.solver == "CUSTOM"
    assert layer.ctx.custom_solver is mock


def test_layer_construction_torch_mock(problem_and_ref):
    problem, A_param, b_param, x_var, ref = problem_and_ref
    mock = TorchBatchMock(ref.ctx.solver_ctx)
    layer = CvxpyLayer(problem, parameters=[A_param, b_param], variables=[x_var],
                       solver=mock)
    assert layer.ctx.solver == "CUSTOM"
    assert layer.ctx.custom_solver is mock


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
