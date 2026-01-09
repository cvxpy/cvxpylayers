from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, TypeVar

import cvxpy as cp
import cvxpy.constraints
import numpy as np
import scipy.sparse
from cvxpy.reductions.dcp2cone.cone_matrix_stuffing import ParamConeProg

import cvxpylayers.interfaces

if TYPE_CHECKING:
    import torch

T = TypeVar("T")


class SolverData(Protocol):
    """Protocol for data objects returned by solver context."""

    def torch_solve(
        self, solver_args: dict[str, Any] | None = None
    ) -> tuple["torch.Tensor", "torch.Tensor", Any]:
        """Solve the problem using torch backend.

        Returns:
            tuple of (primal, dual, adj_batch) where primal and dual are torch tensors
            and adj_batch is a solver-specific adjoint/backward object.
        """
        ...

    def torch_derivative(
        self, primal: "torch.Tensor", dual: "torch.Tensor", adj_batch: Any
    ) -> tuple["torch.Tensor | None", "torch.Tensor", "torch.Tensor"]:
        """Compute derivatives using torch backend.

        Returns:
            tuple of (dP, dq, dA) gradients as torch tensors (dP can be None).
        """
        ...


class SolverContext(Protocol):
    """Protocol for solver context objects."""

    def torch_to_data(
        self,
        quad_obj_values: "torch.Tensor | None",
        lin_obj_values: "torch.Tensor",
        con_values: "torch.Tensor",
    ) -> SolverData:
        """Convert torch tensors to solver data format."""
        ...


# Default reshape function for numpy/jax arrays
def _default_reshape(array, shape):
    return array.reshape(shape, order="F")


@dataclass
class VariableRecovery:
    primal: slice | None
    dual: slice | None
    shape: tuple[int, ...]
    is_symmetric: bool = False  # True if primal variable is symmetric (requires svec unpacking)
    is_psd_dual: bool = False  # True if this is a PSD constraint dual (requires svec unpacking)

    def recover(self, primal_sol: T, dual_sol: T, reshape_fn=_default_reshape) -> T:
        """Extract and reshape variable from primal or dual solution.

        Args:
            primal_sol: Primal solution array
            dual_sol: Dual solution array
            reshape_fn: Function to reshape array with Fortran order semantics.
                        Defaults to numpy-style reshape with order="F".
        """
        batch = tuple(primal_sol.shape[:-1])
        if self.primal is not None:
            data = primal_sol[..., self.primal]
            if self.is_symmetric:
                # Unpack symmetric primal variable from svec format
                return _unpack_primal_svec(data, self.shape[0], batch)
            return reshape_fn(data, batch + self.shape)  # type: ignore[index]
        if self.dual is not None:
            data = dual_sol[..., self.dual]
            if self.is_psd_dual:
                # Unpack scaled vectorized form (svec) to full symmetric matrix
                return _unpack_svec(data, self.shape[0], batch)
            return reshape_fn(data, batch + self.shape)  # type: ignore[index]
        raise RuntimeError(
            "Invalid VariableRecovery: both primal and dual slices are None. "
            "At least one must be set to recover variable values."
        )


def _unpack_primal_svec(svec: T, n: int, batch: tuple) -> T:
    """Unpack symmetric primal variable from vectorized form.

    cvxpylayers receives symmetric variables in upper triangular row-major order:
    [X[0,0], X[0,1], ..., X[0,n-1], X[1,1], X[1,2], ..., X[n-1,n-1]]

    Args:
        svec: Vectorized form, shape (*batch, n*(n+1)/2)
        n: Matrix dimension
        batch: Batch dimensions

    Returns:
        Full symmetric matrix, shape (*batch, n, n)
    """
    # Build index arrays for vectorized unpacking
    # Upper triangular row-major: (0,0), (0,1), ..., (0,n-1), (1,1), (1,2), ...
    rows, cols = [], []
    for i in range(n):
        for j in range(i, n):
            rows.append(i)
            cols.append(j)

    # Handle torch tensors (maintaining gradient flow)
    try:
        import torch
        is_torch = isinstance(svec, torch.Tensor)
    except ImportError:
        is_torch = False

    if is_torch:
        rows_t = torch.tensor(rows, dtype=torch.long, device=svec.device)
        cols_t = torch.tensor(cols, dtype=torch.long, device=svec.device)

        # Create output tensor
        out_shape = batch + (n, n)
        # Use svec values to build result (need to maintain gradient connection)
        # Create index for assignment: flatten batch dims, then scatter
        batch_size = 1
        for b in batch:
            batch_size *= b
        if batch:
            # Reshape to (batch_size, k) where k = n*(n+1)/2
            svec_flat = svec.reshape(batch_size, -1)
            result = torch.zeros(
                batch_size, n, n, dtype=svec.dtype, device=svec.device
            )
            # Fill upper and lower triangles
            result[:, rows_t, cols_t] = svec_flat
            result[:, cols_t, rows_t] = svec_flat
            result = result.reshape(out_shape)
        else:
            result = torch.zeros(n, n, dtype=svec.dtype, device=svec.device)
            result[rows_t, cols_t] = svec
            result[cols_t, rows_t] = svec
        return result
    else:
        # JAX/numpy path - detect JAX arrays
        try:
            import jax
            import jax.numpy as jnp
            is_jax = isinstance(svec, jax.Array)
        except ImportError:
            is_jax = False

        if is_jax:
            rows_arr = jnp.array(rows)
            cols_arr = jnp.array(cols)
            out_shape = batch + (n, n)
            result = jnp.zeros(out_shape, dtype=svec.dtype)
            result = result.at[..., rows_arr, cols_arr].set(svec)
            result = result.at[..., cols_arr, rows_arr].set(svec)
            return result
        else:
            # Pure numpy path
            rows_np = np.array(rows)
            cols_np = np.array(cols)
            svec_np = np.asarray(svec)
            out_shape = batch + (n, n)
            result = np.zeros(out_shape, dtype=svec_np.dtype)
            result[..., rows_np, cols_np] = svec_np
            result[..., cols_np, rows_np] = svec_np
            return result


def _unpack_svec(svec: T, n: int, batch: tuple) -> T:
    """Unpack scaled vectorized form to full symmetric matrix.

    The svec format stores a symmetric n x n matrix as a vector of length n*(n+1)/2,
    with off-diagonal elements scaled by sqrt(2). This function unpacks to the full matrix.

    Args:
        svec: Scaled vectorized form, shape (*batch, n*(n+1)/2)
        n: Matrix dimension
        batch: Batch dimensions

    Returns:
        Full symmetric matrix, shape (*batch, n, n)
    """
    # Build index arrays for vectorized unpacking
    # svec uses column-major lower triangular ordering: (0,0), (1,0), (1,1), (2,0), ...
    rows, cols, is_diag = [], [], []
    for j in range(n):
        for i in range(j, n):
            rows.append(i)
            cols.append(j)
            is_diag.append(i == j)

    sqrt2 = np.sqrt(2.0)
    scale = np.array([1.0 if d else 1.0 / sqrt2 for d in is_diag])

    # Handle torch tensors (maintaining gradient flow)
    try:
        import torch
        is_torch = isinstance(svec, torch.Tensor)
    except ImportError:
        is_torch = False

    if is_torch:
        rows_t = torch.tensor(rows, dtype=torch.long, device=svec.device)
        cols_t = torch.tensor(cols, dtype=torch.long, device=svec.device)
        scale_t = torch.tensor(scale, dtype=svec.dtype, device=svec.device)

        # Scale off-diagonal elements
        scaled_svec = svec * scale_t

        # Create output tensor
        out_shape = batch + (n, n)
        batch_size = 1
        for b in batch:
            batch_size *= b
        if batch:
            # Reshape to (batch_size, k) where k = n*(n+1)/2
            scaled_flat = scaled_svec.reshape(batch_size, -1)
            result = torch.zeros(
                batch_size, n, n, dtype=svec.dtype, device=svec.device
            )
            # Fill lower and upper triangles
            result[:, rows_t, cols_t] = scaled_flat
            result[:, cols_t, rows_t] = scaled_flat
            result = result.reshape(out_shape)
        else:
            result = torch.zeros(n, n, dtype=svec.dtype, device=svec.device)
            result[rows_t, cols_t] = scaled_svec
            result[cols_t, rows_t] = scaled_svec
        return result
    else:
        # JAX/numpy path - detect JAX arrays
        try:
            import jax
            import jax.numpy as jnp
            is_jax = isinstance(svec, jax.Array)
        except ImportError:
            is_jax = False

        if is_jax:
            rows_arr = jnp.array(rows)
            cols_arr = jnp.array(cols)
            scale_arr = jnp.array(scale)
            scaled_svec = svec * scale_arr
            out_shape = batch + (n, n)
            result = jnp.zeros(out_shape, dtype=svec.dtype)
            result = result.at[..., rows_arr, cols_arr].set(scaled_svec)
            result = result.at[..., cols_arr, rows_arr].set(scaled_svec)
            return result
        else:
            # Pure numpy path
            rows_np = np.array(rows)
            cols_np = np.array(cols)
            svec_np = np.asarray(svec)
            scaled_svec = svec_np * scale
            out_shape = batch + (n, n)
            result = np.zeros(out_shape, dtype=svec_np.dtype)
            result[..., rows_np, cols_np] = scaled_svec
            result[..., cols_np, rows_np] = scaled_svec
            return result


@dataclass
class LayersContext:
    parameters: list[cp.Parameter]
    reduced_P: scipy.sparse.csr_array
    q: scipy.sparse.csr_array | None
    reduced_A: scipy.sparse.csr_array
    cone_dims: dict[str, int | list[int]]
    solver_ctx: SolverContext
    solver: str
    var_recover: list[VariableRecovery]
    user_order_to_col_order: dict[int, int]
    batch_sizes: list[int] | None = (
        None  # Track which params are batched (0=unbatched, N=batch size)
    )
    # GP (Geometric Programming) support
    gp: bool = False
    # Maps original GP parameters to their log-space DCP parameters
    # Used to determine which parameters need log transformation in forward pass
    gp_param_to_log_param: dict[cp.Parameter, cp.Parameter] | None = None

    def validate_params(self, values: list) -> tuple:
        if len(values) != len(self.parameters):
            raise ValueError(
                f"A tensor must be provided for each CVXPY parameter; "
                f"received {len(values)} tensors, expected {len(self.parameters)}",
            )

        # Determine batch size from all parameters
        batch_sizes = []
        for i, (value, param) in enumerate(zip(values, self.parameters, strict=False)):
            # Check if value has the right shape (with or without batch dimension)
            if len(value.shape) == len(param.shape):
                # No batch dimension for this parameter
                if value.shape != param.shape:
                    raise ValueError(
                        f"Invalid parameter shape for parameter {i}. "
                        f"Expected: {param.shape}, Got: {value.shape}",
                    )
                batch_sizes.append(0)
            elif len(value.shape) == len(param.shape) + 1:
                # Has batch dimension
                if value.shape[1:] != param.shape:
                    shape_str = ", ".join(map(str, param.shape))
                    raise ValueError(
                        f"Invalid parameter shape for parameter {i}. "
                        f"Expected batched shape: (batch_size, {shape_str}), "
                        f"Got: {value.shape}",
                    )
                batch_sizes.append(value.shape[0])
            else:
                raise ValueError(
                    f"Invalid parameter dimensionality for parameter {i}. "
                    f"Expected {len(param.shape)} or {len(param.shape) + 1} dimensions, "
                    f"Got: {len(value.shape)} dimensions",
                )

        # Check that all non-zero batch sizes are the same
        nonzero_batch_sizes = [b for b in batch_sizes if b > 0]
        if nonzero_batch_sizes:
            batch_size = nonzero_batch_sizes[0]
            if not all(b == batch_size for b in nonzero_batch_sizes):
                raise ValueError(
                    f"Inconsistent batch sizes. Expected all batched parameters to have "
                    f"the same batch size, but got: {batch_sizes}",
                )
            # Store batch_sizes for use in forward pass
            self.batch_sizes = batch_sizes
            return (batch_size,)
        self.batch_sizes = batch_sizes
        return ()


def _find_parent_constraint(
    var: cp.Variable, problem: cp.Problem
) -> cp.Constraint | None:
    """Find the constraint whose dual_variables contains this variable.

    Args:
        var: A CVXPY Variable that may be a constraint's dual variable
        problem: The CVXPY Problem to search

    Returns:
        The parent constraint if var is a dual variable, None otherwise
    """
    for con in problem.constraints:
        # Compare by ID to avoid triggering CVXPY constraint comparison
        for dv in con.dual_variables:
            if var.id == dv.id:
                return con
    return None


def _build_constr_id_to_slice(param_prob: ParamConeProg) -> dict[int, slice]:
    """Build mapping from constraint ID to slice in dual solution vector.

    The dual solution vector is ordered by cone type:
    Zero (equalities) -> NonNeg (inequalities) -> SOC -> ExpCone -> PSD -> PowCone3D

    Args:
        param_prob: CVXPY's parametrized cone program

    Returns:
        Dictionary mapping constraint ID to slice in dual solution vector
    """
    constr_id_to_slice: dict[int, slice] = {}
    cur_idx = 0

    # Process each cone type in canonical order
    cone_types = [
        cvxpy.constraints.Zero,
        cvxpy.constraints.NonNeg,
        cvxpy.constraints.SOC,
        cvxpy.constraints.ExpCone,
        cvxpy.constraints.PSD,
        cvxpy.constraints.PowCone3D,
    ]

    for cone_type in cone_types:
        for c in param_prob.constr_map.get(cone_type, []):
            # PSD constraints use scaled vectorization (svec) in the dual
            # For an n x n PSD constraint, svec size is n*(n+1)//2
            if cone_type is cvxpy.constraints.PSD:
                n = c.shape[0]  # Matrix dimension
                cone_size = n * (n + 1) // 2
            else:
                cone_size = c.size
            constr_id_to_slice[c.id] = slice(cur_idx, cur_idx + cone_size)
            cur_idx += cone_size

    return constr_id_to_slice


def _validate_problem(
    problem: cp.Problem,
    variables: list[cp.Variable],
    parameters: list[cp.Parameter],
    gp: bool,
) -> None:
    """Validate that the problem is DPP-compliant and inputs are well-formed.

    Args:
        problem: CVXPY problem to validate
        variables: List of CVXPY variables to track (can include constraint dual variables)
        parameters: List of CVXPY parameters
        gp: Whether this is a geometric program (GP)

    Raises:
        ValueError: If problem is not DPP-compliant or inputs are invalid
    """
    # Check if problem follows disciplined parametrized programming (DPP) rules
    if gp:
        if not problem.is_dgp(dpp=True):  # type: ignore[call-arg]
            raise ValueError("Problem must be DPP for geometric programming.")
    else:
        if not problem.is_dcp(dpp=True):  # type: ignore[call-arg]
            raise ValueError("Problem must be DPP.")

    # Validate parameters match problem definition
    if not set(problem.parameters()) == set(parameters):
        raise ValueError("The layer's parameters must exactly match problem.parameters")
    if not isinstance(parameters, list) and not isinstance(parameters, tuple):
        raise ValueError("The layer's parameters must be provided as a list or tuple")
    if not isinstance(variables, list) and not isinstance(variables, tuple):
        raise ValueError("The layer's variables must be provided as a list or tuple")

    # Validate variables: each must be either a primal variable or a constraint dual variable
    primal_vars = set(problem.variables())
    for v in variables:
        if v in primal_vars:
            continue  # Valid primal variable
        parent_con = _find_parent_constraint(v, problem)
        if parent_con is None:
            raise ValueError(
                f"Variable {v} must be a subset of problem.variables or a constraint dual variable"
            )


def _build_user_order_mapping(
    parameters: list[cp.Parameter],
    param_prob: ParamConeProg,
    gp: bool,
    gp_param_to_log_param: dict[cp.Parameter, cp.Parameter] | None,
) -> dict[int, int]:
    """Build mapping from user parameter order to column order.

    CVXPY internally reorders parameters when canonicalizing problems. This
    creates a mapping from the user's parameter order to the internal column
    order used in the canonical form.

    Args:
        parameters: List of CVXPY parameters in user order
        param_prob: CVXPY's parametrized problem object
        gp: Whether this is a geometric program
        gp_param_to_log_param: Mapping from GP params to log-space DCP params

    Returns:
        Dictionary mapping user parameter index to column order index
    """
    # For GP problems, we need to use the log-space DCP parameter IDs
    if gp and gp_param_to_log_param:
        # Map user order index to column using log-space DCP parameters
        user_order_to_col = {
            i: param_prob.param_id_to_col[
                gp_param_to_log_param[p].id if p in gp_param_to_log_param else p.id
            ]
            for i, p in enumerate(parameters)
        }
    else:
        # Standard DCP problem - use original parameters
        user_order_to_col = {
            i: col
            for col, i in sorted(
                [(param_prob.param_id_to_col[p.id], i) for i, p in enumerate(parameters)],
            )
        }

    # Convert column indices to sequential order mapping
    user_order_to_col_order = {}
    for j, i in enumerate(user_order_to_col.keys()):
        user_order_to_col_order[i] = j

    return user_order_to_col_order


def parse_args(
    problem: cp.Problem,
    variables: list[cp.Variable],
    parameters: list[cp.Parameter],
    solver: str | None,
    gp: bool = False,
    verbose: bool = False,
    canon_backend: str | None = None,
    solver_args: dict[str, Any] | None = None,
) -> LayersContext:
    # Validate problem is DPP (disciplined parametrized programming)
    _validate_problem(problem, variables, parameters, gp)

    if solver is None:
        solver = "DIFFCP"

    # Handle GP problems using native CVXPY reduction (cvxpy >= 1.7.4)
    gp_param_to_log_param = None
    if gp:
        # Apply native CVXPY DGP→DCP reduction
        dgp2dcp = cp.reductions.Dgp2Dcp(problem)
        dcp_problem, _ = dgp2dcp.apply(problem)

        # Extract parameter mapping from the reduction
        gp_param_to_log_param = dgp2dcp.canon_methods._parameters

        # Get problem data from the already-transformed DCP problem
        data, _, _ = dcp_problem.get_problem_data(
            solver=solver,
            gp=False,
            verbose=verbose,
            canon_backend=canon_backend,
            solver_opts=solver_args,
        )
    else:
        # Standard DCP path
        data, _, _ = problem.get_problem_data(
            solver=solver,
            gp=False,
            verbose=verbose,
            canon_backend=canon_backend,
            solver_opts=solver_args,
        )

    param_prob = data[cp.settings.PARAM_PROB]  # type: ignore[attr-defined]
    cone_dims = data["dims"]

    # Create solver context
    solver_ctx = cvxpylayers.interfaces.get_solver_ctx(
        solver,
        param_prob,
        cone_dims,
        data,
        solver_args,
    )

    # Build parameter ordering mapping
    user_order_to_col_order = _build_user_order_mapping(
        parameters, param_prob, gp, gp_param_to_log_param
    )

    q = getattr(param_prob, "q", getattr(param_prob, "c", None))

    # Build constraint ID to dual slice mapping for dual variable recovery
    constr_id_to_slice = _build_constr_id_to_slice(param_prob)

    # Build variable recovery info for each requested variable
    # Variables can be either primal (from problem.variables()) or dual (from constraint.dual_variables)
    primal_vars = set(problem.variables())
    var_recover = []
    for v in variables:
        if v in primal_vars:
            # Primal variable: recover from primal solution
            start = param_prob.var_id_to_col[v.id]
            # Check if variable is symmetric (uses svec format in solver)
            # A symmetric variable must be at least 2D with shape (n, n)
            is_sym = (
                hasattr(v, "is_symmetric")
                and v.is_symmetric()
                and len(v.shape) >= 2
            )
            if is_sym:
                # Symmetric variables use svec format: n*(n+1)//2 elements
                n = v.shape[0]
                svec_size = n * (n + 1) // 2
                var_recover.append(
                    VariableRecovery(
                        primal=slice(start, start + svec_size),
                        dual=None,
                        shape=v.shape,
                        is_symmetric=True,
                    )
                )
            else:
                var_recover.append(
                    VariableRecovery(
                        primal=slice(start, start + v.size),
                        dual=None,
                        shape=v.shape,
                    )
                )
        else:
            # Dual variable: find parent constraint and recover from dual solution
            parent_con = _find_parent_constraint(v, problem)
            assert parent_con is not None  # Already validated
            dual_slice = constr_id_to_slice[parent_con.id]
            # Check if this is a PSD constraint (requires svec unpacking)
            is_psd = isinstance(parent_con, cvxpy.constraints.PSD)
            var_recover.append(
                VariableRecovery(
                    primal=None,
                    dual=dual_slice,
                    shape=v.shape,
                    is_psd_dual=is_psd,
                )
            )

    return LayersContext(
        parameters,
        param_prob.reduced_P,
        q,
        param_prob.reduced_A,
        cone_dims,
        solver_ctx,
        solver,
        var_recover=var_recover,
        user_order_to_col_order=user_order_to_col_order,
        gp=gp,
        gp_param_to_log_param=gp_param_to_log_param,
    )
