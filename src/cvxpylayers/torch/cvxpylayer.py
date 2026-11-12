import math
import warnings
from typing import Any, cast

import cvxpy as cp
import numpy as np
import scipy.sparse
import torch
from cvxpy.reductions.canonical_maps import ParameterTransform, ParameterTransformKind

import cvxpylayers.utils.parse_args as pa
from cvxpylayers.interfaces.base import SolverInterface


class _ScipySparseMatmul(torch.autograd.Function):
    """Sparse matrix-vector multiply using scipy on CPU with autograd support.

    PyTorch sparse CSR matmul on CPU is 80-200x slower than scipy due to
    MKL/OpenBLAS thread-spawning overhead. This uses scipy for the forward
    pass and computes gradients via the transpose multiply.
    """

    @staticmethod
    def forward(scipy_csr: scipy.sparse.csr_array, x: torch.Tensor) -> torch.Tensor:
        x_np = x.detach().numpy()
        result = scipy_csr @ x_np
        return torch.from_numpy(np.asarray(result))

    @staticmethod
    def setup_context(ctx: Any, inputs: tuple, output: torch.Tensor) -> None:
        scipy_csr, x = inputs
        ctx.scipy_csr_T = scipy_csr.T.tocsr()
        ctx.save_for_backward(x)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[None, torch.Tensor]:
        grad_np = grad_output.numpy()
        result = ctx.scipy_csr_T @ grad_np
        return None, torch.from_numpy(np.asarray(result))


def _reshape_fortran(array: torch.Tensor, shape: tuple) -> torch.Tensor:
    """Reshape array using Fortran (column-major) order.

    PyTorch doesn't support order='F' directly, so we use permutation.

    Args:
        array: Input tensor to reshape
        shape: Target shape tuple

    Returns:
        Reshaped tensor in Fortran order
    """
    if len(array.shape) == 0:
        return array.reshape(shape)
    x = array.permute(*reversed(range(len(array.shape))))
    return x.reshape(*reversed(shape)).permute(*reversed(range(len(shape))))


def _apply_transform(
    transform: ParameterTransform, value: torch.Tensor,
) -> torch.Tensor:
    """Evaluate one of CVXPY's parameter transformations on flat values."""
    kind = transform.kind
    if kind == ParameterTransformKind.GATHER:
        indices = torch.as_tensor(transform.indices, device=value.device)
        return value[..., indices]
    if kind == ParameterTransformKind.LOG:
        return torch.log(value)
    if kind == ParameterTransformKind.REAL:
        return torch.real(value)
    if kind == ParameterTransformKind.IMAG:
        return torch.imag(value)
    raise ValueError(f"Unknown parameter transform {kind}")


def _prepare_param_stack(
    params: tuple[torch.Tensor, ...],
    ctx: pa.LayersContext,
    batch: tuple,
) -> torch.Tensor:
    """Turn user parameter values into the stacked compiled-parameter vector.

    Evaluates CVXPY's parameter transformations (e.g. ``log`` for geometric
    programs, gathers of the free entries of structured parameters) and packs
    the compiled parameters in column order, followed by a constant 1.

    Returns:
        Tensor of shape (num_params + 1,) or (num_params + 1, batch_size).
    """
    # Always work with batch_size >= 1 for uniform processing.
    effective_batch = batch if batch else (1,)
    values: dict[int, torch.Tensor] = {}
    for i, (param, value) in enumerate(zip(ctx.parameters, params)):
        if ctx.batch_sizes[i] == 0:  # type: ignore[index]
            # Unbatched parameter - expand to match batch size
            value = value.unsqueeze(0).expand(effective_batch + value.shape)
        flat = _reshape_fortran(value, effective_batch + (-1,))
        source = ctx.param_sources[i]
        if source is not None:
            # Sparse parameters enter the transforms as their stored entries.
            flat = flat[..., torch.as_tensor(source, device=flat.device)]
        values[param.id] = flat
    for transform in ctx.param_transforms:
        values[transform.target_id] = _apply_transform(
            transform, values[transform.source_id])

    columns = [values[pid] for pid, _ in ctx.param_layout]
    dtype = columns[0].dtype if columns else params[0].dtype
    columns.append(torch.ones(effective_batch + (1,), dtype=dtype, device=params[0].device))
    # (batch_size, num_params) -> (num_params,) or (num_params, batch_size)
    p_stack = torch.cat(columns, -1).T
    return p_stack.reshape((p_stack.shape[0],) + batch)


def _recover_output(
    out: pa.OutputRecovery, vec: torch.Tensor, batch: tuple,
) -> torch.Tensor:
    """Read one output from a (batch, n) solution tensor; see ``OutputRecovery``."""
    device = vec.device
    if out.indices is not None:
        data = vec[..., torch.as_tensor(out.indices, device=device)]
    else:
        rows = torch.as_tensor(out.rows, device=device)
        cols = torch.as_tensor(out.cols, device=device)
        vals = torch.as_tensor(out.vals, dtype=vec.dtype, device=device)
        data = torch.zeros(batch + (out.size,), dtype=vec.dtype, device=device)
        data = data.index_add(-1, rows, vec[..., cols] * vals)
    if out.exp:
        data = torch.exp(data)
    if out.scatter is not None:
        dense = torch.zeros(batch + (math.prod(out.shape),), dtype=vec.dtype, device=device)
        data = dense.index_copy(-1, torch.as_tensor(out.scatter, device=device), data)
    return _reshape_fortran(data, batch + out.shape)


def _recover_results(
    primal: torch.Tensor,
    dual: torch.Tensor,
    ctx: pa.LayersContext,
    batch: tuple,
) -> tuple[torch.Tensor, ...]:
    """Recover the requested variables from the solver's primal/dual solutions.

    The solver always returns 2D tensors (batch_size, n), even for unbatched
    inputs (batch_size=1); the dummy batch dimension is removed at the end.
    """
    internal_batch = tuple(primal.shape[:-1])
    return tuple(
        _recover_output(out, primal if out.source == "primal" else dual, internal_batch)
        .reshape(batch + out.shape)
        for out in ctx.outputs
    )


def _split_outputs(
    values: torch.Tensor, ctx: pa.LayersContext, batch: tuple,
) -> tuple[torch.Tensor, ...]:
    """Split concatenated variable values (parameter-space solvers)."""
    offsets = ctx.output_offsets
    assert ctx.variables is not None
    return tuple(
        _reshape_fortran(values[..., offsets[i]:offsets[i + 1]],
                         tuple(values.shape[:-1]) + var.shape).reshape(batch + var.shape)
        for i, var in enumerate(ctx.variables)
    )


class CvxpyLayer(torch.nn.Module):
    """A differentiable convex optimization layer for PyTorch.

    This layer wraps a parametrized CVXPY problem, solving it in the forward pass
    and computing gradients via implicit differentiation in the backward pass.

    Example:
        >>> import cvxpy as cp
        >>> import torch
        >>> from cvxpylayers.torch import CvxpyLayer
        >>>
        >>> # Define a simple QP
        >>> x = cp.Variable(2)
        >>> A = cp.Parameter((3, 2))
        >>> b = cp.Parameter(3)
        >>> problem = cp.Problem(cp.Minimize(cp.sum_squares(A @ x - b)), [x >= 0])
        >>>
        >>> # Create the layer
        >>> layer = CvxpyLayer(problem, parameters=[A, b], variables=[x])
        >>>
        >>> # Solve with gradients
        >>> A_t = torch.randn(3, 2, requires_grad=True)
        >>> b_t = torch.randn(3, requires_grad=True)
        >>> (solution,) = layer(A_t, b_t)
        >>> solution.sum().backward()
    """

    def __init__(
        self,
        problem: cp.Problem,
        parameters: list[cp.Parameter],
        variables: list[cp.Variable],
        solver: str | SolverInterface | None = None,
        gp: bool = False,
        verbose: bool = False,
        canon_backend: str | None = None,
        solver_args: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the differentiable optimization layer.

        Args:
            problem: A CVXPY Problem. Must be DPP-compliant (``problem.is_dpp()``
                must return True).
            parameters: List of CVXPY Parameters that will be filled with values
                at runtime. Order must match the order of tensors passed to forward().
            variables: List of CVXPY Variables whose optimal values will be returned
                by forward(). Order determines the order of returned tensors.
            solver: CVXPY solver string (e.g., ``cp.CLARABEL``, ``cp.SCS``),
                a :class:`~cvxpylayers.interfaces.base.SolverInterface` instance
                for a custom solver, or ``None`` (uses diffcp by default).
                When a ``SolverInterface`` is passed, its ``canon_solver`` attribute
                selects the CVXPY canonicalization solver; the instance itself is
                called during the forward/backward pass instead of diffcp.
            gp: If True, problem is a geometric program. Parameters will be
                log-transformed before solving.
            verbose: If True, print solver output.
            canon_backend: Backend for canonicalization. Options are 'diffcp',
                'cuclarabel', or None (auto-select).
            solver_args: Default keyword arguments passed to the solver.
                Can be overridden per-call in forward().

        Raises:
            AssertionError: If problem is not DPP-compliant.
            ValueError: If parameters or variables are not part of the problem.
        """
        super().__init__()
        if solver_args is None:
            solver_args = {}

        # When solver is a SolverInterface, split off the CVXPY canonicalization
        # name (solver.canon_solver) and pass the object through to parse_args.
        canon_solver_str: str | None = (
            None if isinstance(solver, SolverInterface) else solver
        )
        custom_solver: SolverInterface | None = (
            solver if isinstance(solver, SolverInterface) else None
        )

        self.ctx = pa.parse_args(
            problem,
            variables,
            parameters,
            canon_solver_str,
            gp=gp,
            verbose=verbose,
            canon_backend=canon_backend,
            solver_args=solver_args,
            custom_solver=custom_solver,
        )
        if self.ctx.P_mat is not None:
            self.register_buffer("P", scipy_csr_to_torch_csr(self.ctx.P_mat))
        else:
            self.P = None
        self._P_scipy: scipy.sparse.csr_array | None = self.ctx.P_mat
        self.register_buffer("q", scipy_csr_to_torch_csr(self.ctx.q_mat))
        self._q_scipy: scipy.sparse.csr_array = self.ctx.q_mat
        self.register_buffer("A", scipy_csr_to_torch_csr(self.ctx.A_mat))
        self._A_scipy: scipy.sparse.csr_array = self.ctx.A_mat
        self._warm_start_cache = None
        if isinstance(self.ctx.solver, SolverInterface):
            self.ctx.solver.setup(self.ctx)

    def forward(
        self,
        *params: torch.Tensor,
        solver_args: dict[str, Any] | None = None,
        warm_start: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        """Solve the optimization problem and return optimal variable values.

        Args:
            *params: Tensor values for each CVXPY Parameter, in the same order
                as the ``parameters`` argument to __init__. Each tensor shape must
                match the corresponding Parameter shape, optionally with a batch
                dimension prepended. Batched and unbatched parameters can be mixed;
                unbatched parameters are broadcast.
            solver_args: Keyword arguments passed to the solver, overriding any
                defaults set in __init__.
            warm_start: If True, use the cached solution from the previous solve
                as a warm start for the solver. Only supported with solver="MOREAU".

        Returns:
            Tuple of tensors containing optimal values for each CVXPY Variable
            specified in the ``variables`` argument to __init__. If inputs are
            batched, outputs will have matching batch dimensions.

        Raises:
            RuntimeError: If the solver fails to find a solution.
            ValueError: If warm_start=True with a non-Moreau solver.

        Example:
            >>> # Single problem
            >>> (x_opt,) = layer(A_tensor, b_tensor)
            >>>
            >>> # Batched: solve 10 problems in parallel
            >>> A_batch = torch.randn(10, 3, 2)
            >>> b_batch = torch.randn(10, 3)
            >>> (x_batch,) = layer(A_batch, b_batch)  # x_batch.shape = (10, 2)
        """
        if solver_args is None:
            solver_args = {}
        if warm_start and self.ctx.solver != "MOREAU":
            raise ValueError(
                "warm_start=True is only supported with solver='MOREAU'. "
                f"Current solver is '{self.ctx.solver}'."
            )
        batch = self.ctx.validate_params(list(params))

        # Fast path: parameter-space solver (is_parametric=True).
        # Skip the canonical q @ p_stack / A @ p_stack evaluation entirely;
        # _CvxpyLayer propagates gradients directly to the param tensors.
        if isinstance(self.ctx.solver, SolverInterface) and self.ctx.solver.is_parametric:
            from cvxpylayers.interfaces.custom_if import _CvxpyLayer as _CustomCvxpyLayer

            # Set param.value on the CVXPY Problem directly — CvxpyLayer owns
            # both the Parameter objects and the current numpy values.
            for param_obj, p in zip(self.ctx.parameters, params):
                param_obj.value = p.detach().cpu().numpy()
            needs_grad = torch.is_grad_enabled() and any(
                p.requires_grad for p in params
            )
            values, _, _, _ = _CustomCvxpyLayer.apply(  # type: ignore[misc]
                None, None, None, self.ctx, solver_args, needs_grad, None, *params
            )
            return _split_outputs(values, self.ctx, batch)

        p_stack = _prepare_param_stack(params, self.ctx, batch)

        # Get dtype and device from input parameters to ensure type/device matching
        param_dtype = p_stack.dtype
        param_device = p_stack.device

        # Evaluate parametrized matrices
        if param_device.type == "cpu":
            # Use scipy sparse matmul on CPU (80-200x faster than torch sparse CSR)
            P_eval = (
                _ScipySparseMatmul.apply(self._P_scipy, p_stack)
                if self._P_scipy is not None
                else None
            )
            q_eval = _ScipySparseMatmul.apply(self._q_scipy, p_stack)
            A_eval = _ScipySparseMatmul.apply(self._A_scipy, p_stack)
        else:
            # Use torch sparse CSR on GPU (fast there)
            P_eval = (
                (self.P.to(dtype=param_dtype, device=param_device) @ p_stack)
                if self.P is not None
                else None
            )
            q_eval = self.q.to(dtype=param_dtype, device=param_device) @ p_stack  # type: ignore[operator]
            A_eval = self.A.to(dtype=param_dtype, device=param_device) @ p_stack  # type: ignore[operator]

        # Get the solver-specific _CvxpyLayer class
        from cvxpylayers.interfaces import get_torch_cvxpylayer

        _CvxpyLayer = get_torch_cvxpylayer(self.ctx.solver)

        # Give the custom solver the raw parameter values before the solve.
        # This allows CVXPYgen-style solvers that work with CVXPY parameter
        # objects rather than the canonical q/A matrices to capture the values.
        if isinstance(self.ctx.solver, SolverInterface):
            params_numpy = [p.detach().cpu().numpy() for p in params]
            self.ctx.solver.set_params(params_numpy)

        # Determine if gradients are needed (must check here, not inside
        # Function.forward() where torch.is_grad_enabled() is always False)
        needs_grad = torch.is_grad_enabled() and any(
            p.requires_grad for p in params
        )

        # Validate warm start batch compatibility before using cache.
        # moreau.torch always uses batched mode internally (batch_size=1 for unbatched),
        # so compare effective batch sizes directly.
        ws = None
        if warm_start and self._warm_start_cache is not None:
            cached_ws = self._warm_start_cache
            cached_batch = cached_ws.x.shape[0]
            current_batch = batch[0] if batch else 1
            if cached_batch == current_batch:
                ws = cached_ws

        primal, dual, _, solver_data = _CvxpyLayer.apply(  # type: ignore[misc]
            P_eval,
            q_eval,
            A_eval,
            self.ctx,
            solver_args,
            needs_grad,
            ws,
        )

        # Always update warm start cache for Moreau solver (negligible cost)
        if self.ctx.solver == "MOREAU":
            self._warm_start_cache = solver_data._solution.to_warm_start()

        # Recover results and apply GP inverse transform if needed
        return _recover_results(primal, dual, self.ctx, batch)


def scipy_csr_to_torch_csr(
    scipy_csr: scipy.sparse.csr_array | None,
) -> torch.Tensor | None:
    if scipy_csr is None:
        return None
    # Use cast to help type checker understand scipy_csr is not None
    scipy_csr = cast(scipy.sparse.csr_array, scipy_csr)
    # Get the CSR format components
    values = scipy_csr.data
    col_indices = scipy_csr.indices
    row_ptr = scipy_csr.indptr
    num_rows, num_cols = scipy_csr.shape  # type: ignore[misc]

    # Create the torch sparse csr_tensor.
    # Explicitly disable invariant checks for performance — the input
    # comes from scipy CSR, which already guarantees sorted, deduplicated
    # indices.  This also silences the PyTorch warning about implicit state.
    with (
        torch.sparse.check_sparse_tensor_invariants(enable=False),
        warnings.catch_warnings(),
    ):
        warnings.filterwarnings(
            "ignore",
            message="Sparse CSR tensor support is in beta state",
            category=UserWarning,
        )
        torch_csr = torch.sparse_csr_tensor(
            crow_indices=torch.tensor(row_ptr, dtype=torch.int64),
            col_indices=torch.tensor(col_indices, dtype=torch.int64),
            values=torch.tensor(values, dtype=torch.float64),
            size=(num_rows, num_cols),
        )

    return torch_csr
