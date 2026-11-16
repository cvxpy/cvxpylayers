from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import diffcp
import numpy as np
import scipy.sparse as sp
from cvxpy.reductions.canonical_maps import MatrixStructure
from cvxpy.reductions.solvers.conic_solvers.scs_conif import dims_to_solver_dict

try:
    import jax.numpy as jnp
except ImportError:
    jnp = None  # type: ignore[assignment]

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]

try:
    import mlx.core as mx
except ImportError:
    mx = None  # type: ignore[assignment]

if TYPE_CHECKING:
    import cvxpylayers.utils.parse_args as pa

    # Type alias for multi-framework tensor types
    TensorLike = torch.Tensor | jnp.ndarray | np.ndarray | mx.array
else:
    TensorLike = Any


def _detect_batch_size(con_values: TensorLike) -> tuple[int, bool]:
    """Detect batch size and whether input was originally unbatched."""
    ndim = (
        con_values.dim() if hasattr(con_values, "dim")
        else con_values.ndim
    )
    if ndim == 1:
        return 1, True
    else:
        return con_values.shape[1], False


def _build_diffcp_matrices(
    con_values: TensorLike,
    lin_obj_values: TensorLike,
    A_structure: MatrixStructure,
    batch_size: int,
) -> tuple[list[sp.csc_matrix], list[np.ndarray], list[np.ndarray]]:
    """Assemble DIFFCP's ``A``, ``b``, ``c`` from evaluated canonical maps.

    ``con_values`` stacks the stored values of ``A`` (in ``A_structure``)
    above ``b``; ``lin_obj_values`` is ``c``. Both have one column per
    problem in the batch.
    """
    nnz = A_structure.indices.size
    As, bs, cs = [], [], []
    for i in range(batch_size):
        con_vals_i = np.array(con_values[:, i])
        As.append(sp.csc_matrix((con_vals_i[:nnz], A_structure.indices, A_structure.indptr),
                                shape=A_structure.shape))
        bs.append(con_vals_i[nnz:])
        cs.append(np.array(lin_obj_values[:, i]))
    return As, bs, cs


def _compute_gradients(
    adj_batch: Callable,
    dprimal: TensorLike,
    ddual: TensorLike,
    bs: list[np.ndarray],
    batch_size: int,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Compute gradients with respect to ``c`` and ``[A values; b]``."""
    dxs = [np.array(dprimal[i]) for i in range(batch_size)]
    dys = [np.array(ddual[i]) for i in range(batch_size)]
    dss = [np.zeros_like(bs[i]) for i in range(batch_size)]

    dAs, dbs, dcs = adj_batch(dxs, dys, dss)

    # DIFFCP returns dA in A's sparsity structure.
    dA_batch = [np.hstack([dAs[i].data, dbs[i]]) for i in range(batch_size)]
    return list(dcs), dA_batch


class DIFFCP_ctx:
    """DIFFCP inputs from CVXPY's canonical maps (SCS input format)."""

    A_structure: MatrixStructure
    dims: dict

    def __init__(self, A_structure, dims, options=None):
        self.A_structure = A_structure
        self.dims = dims
        self.options = options or {}

    def jax_to_data(self, quad_obj_values,
                    lin_obj_values, con_values) -> "DIFFCP_data":
        if jnp is None:
            raise ImportError(
                "JAX interface requires 'jax' package to be installed. "
                "Install with: pip install jax"
            )

        batch_size, originally_unbatched = _detect_batch_size(con_values)

        if originally_unbatched:
            con_values = jnp.expand_dims(con_values, 1)
            lin_obj_values = jnp.expand_dims(lin_obj_values, 1)

        As, bs, cs = _build_diffcp_matrices(
            con_values, lin_obj_values, self.A_structure, batch_size,
        )

        return DIFFCP_data(
            As=As,
            bs=bs,
            cs=cs,
            cone_dict=dims_to_solver_dict(self.dims),
            batch_size=batch_size,
            originally_unbatched=originally_unbatched,
            options=self.options,
        )

    def mlx_to_data(self, quad_obj_values,
                    lin_obj_values, con_values) -> "DIFFCP_data":

        if mx is None:
            raise ImportError(
                "MLX interface requires 'mlx' package to be installed. "
                "Install with: pip install mlx"
            )

        if isinstance(con_values, np.ndarray):
            con_values_np = con_values
        else:
            con_values_np = np.array(con_values, dtype=np.float32)

        if isinstance(lin_obj_values, np.ndarray):
            lin_obj_values_np = lin_obj_values
        else:
            lin_obj_values_np = np.array(lin_obj_values, dtype=np.float32)

        batch_size, originally_unbatched = _detect_batch_size(con_values_np)

        if originally_unbatched:
            con_values_np = np.expand_dims(con_values_np, 1)
            lin_obj_values_np = np.expand_dims(lin_obj_values_np, 1)

        As, bs, cs = _build_diffcp_matrices(
            con_values_np, lin_obj_values_np, self.A_structure, batch_size,
        )

        return DIFFCP_data(
            As=As,
            bs=bs,
            cs=cs,
            cone_dict=dims_to_solver_dict(self.dims),
            batch_size=batch_size,
            originally_unbatched=originally_unbatched,
            options=self.options,
        )


@dataclass
class DIFFCP_data:
    As: list[sp.csc_matrix]
    bs: list[np.ndarray]
    cs: list[np.ndarray]
    cone_dict: dict[str, int | list[int]]
    batch_size: int
    originally_unbatched: bool
    options: dict

    def jax_solve(self, solver_args=None):
        # Merge layer options with per-call solver_args
        merged_args = {**self.options}
        if solver_args:
            merged_args.update(solver_args)

        xs, ys, _, _, adj_batch = diffcp.solve_and_derivative_batch(
            self.As,
            self.bs,
            self.cs,
            [self.cone_dict] * self.batch_size,
            **merged_args,
        )

        primal = jnp.stack([jnp.array(x) for x in xs])
        dual = jnp.stack([jnp.array(y) for y in ys])

        return primal, dual, adj_batch

    def jax_solve_only(self, solver_args=None):
        # Merge layer options with per-call solver_args
        merged_args = {**self.options}
        if solver_args:
            merged_args.update(solver_args)

        xs, ys, _ = diffcp.solve_only_batch(
            self.As,
            self.bs,
            self.cs,
            [self.cone_dict] * self.batch_size,
            **merged_args,
        )

        primal = jnp.stack([jnp.array(x) for x in xs])
        dual = jnp.stack([jnp.array(y) for y in ys])

        return primal, dual

    def jax_derivative(self, dprimal, ddual, adj_batch):
        dq_batch, dA_batch = _compute_gradients(
            adj_batch, dprimal, ddual, self.bs, self.batch_size
        )

        dq_stacked = jnp.stack([jnp.array(g) for g in dq_batch]).T
        dA_stacked = jnp.stack([jnp.array(g) for g in dA_batch]).T

        if self.originally_unbatched:
            dq_stacked = jnp.squeeze(dq_stacked, 1)
            dA_stacked = jnp.squeeze(dA_stacked, 1)

        return (
            None,
            dq_stacked,
            dA_stacked,
        )

    def mlx_solve(self, solver_args=None):
        if mx is None:
            raise ImportError(
                "MLX interface requires 'mlx' package to be installed. "
                "Install with: pip install mlx"
            )

        # Merge layer options with per-call solver_args
        merged_args = {**self.options}
        if solver_args:
            merged_args.update(solver_args)

        xs, ys, _, _, adj_batch = diffcp.solve_and_derivative_batch(
            self.As,
            self.bs,
            self.cs,
            [self.cone_dict] * self.batch_size,
            **merged_args,
        )
        primal = mx.stack([mx.array(x, dtype=mx.float32) for x in xs])
        dual = mx.stack([mx.array(y, dtype=mx.float32) for y in ys])
        return primal, dual, adj_batch

    def mlx_derivative(self, dprimal, ddual, adj_batch):
        if mx is None:
            raise ImportError(
                "MLX interface requires 'mlx' package to be installed. "
                "Install with: pip install mlx"
            )

        dprimal_np = np.array(dprimal, dtype=np.float32)
        ddual_np = np.array(ddual, dtype=np.float32)

        if dprimal_np.ndim == 1:
            dprimal_np = dprimal_np[np.newaxis, :]
        if ddual_np.ndim == 1:
            ddual_np = ddual_np[np.newaxis, :]

        dq_batch, dA_batch = _compute_gradients(
            adj_batch, dprimal_np, ddual_np, self.bs, self.batch_size
        )

        dq_stacked = mx.stack([mx.array(g, dtype=mx.float32) for g in dq_batch])
        dq_stacked = mx.transpose(dq_stacked)
        dA_stacked = mx.stack([mx.array(g, dtype=mx.float32) for g in dA_batch])
        dA_stacked = mx.transpose(dA_stacked)

        if self.originally_unbatched:
            dq_stacked = mx.squeeze(dq_stacked, 1)
            dA_stacked = mx.squeeze(dA_stacked, 1)

        return (
            None,
            dq_stacked,
            dA_stacked,
        )


if torch is not None:
    class _CvxpyLayer(torch.autograd.Function):
        @staticmethod
        def forward(
            P_eval: torch.Tensor | None,
            q_eval: torch.Tensor,
            A_eval: torch.Tensor,
            cl_ctx: "pa.LayersContext",
            solver_args: dict[str, Any],
            needs_grad: bool = True,
            warm_start: Any = None,
        ) -> tuple[torch.Tensor, torch.Tensor, Any, Any]:
            ctx = cl_ctx.solver_ctx

            batch_size, originally_unbatched = _detect_batch_size(A_eval)

            if originally_unbatched:
                A_eval = A_eval.unsqueeze(1)
                q_eval = q_eval.unsqueeze(1)

            As, bs, cs = _build_diffcp_matrices(A_eval, q_eval, ctx.A_structure, batch_size)

            # Merge layer options with per-call solver_args
            merged_args = {**ctx.options}
            if solver_args:
                merged_args.update(solver_args)

            cone_dicts = [dims_to_solver_dict(ctx.dims)] * batch_size

            # Skip computing the adjoint operator when gradients aren't needed
            if needs_grad:
                xs, ys, _, _, adj_batch = diffcp.solve_and_derivative_batch(
                    As, bs, cs, cone_dicts, **merged_args,
                )
            else:
                xs, ys, _ = diffcp.solve_only_batch(
                    As, bs, cs, cone_dicts, **merged_args,
                )
                adj_batch = None

            primal = torch.stack([torch.from_numpy(x) for x in xs])
            dual = torch.stack([torch.from_numpy(y) for y in ys])

            return primal, dual, adj_batch, (bs, batch_size, originally_unbatched)

        @staticmethod
        def setup_context(ctx: Any, inputs: tuple, outputs: tuple) -> None:
            _, _, adj_batch, backward_data = outputs
            ctx.adj_batch = adj_batch
            ctx.backward_data = backward_data

        @staticmethod
        @torch.autograd.function.once_differentiable
        def backward(
            ctx: Any, dprimal: torch.Tensor, ddual: torch.Tensor, _adj: Any, _data: Any
        ) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor, None, None, None]:
            bs, batch_size, originally_unbatched = ctx.backward_data

            dq_batch, dA_batch = _compute_gradients(
                ctx.adj_batch, dprimal, ddual, bs, batch_size
            )

            dq_stacked = torch.stack([torch.from_numpy(g) for g in dq_batch]).T
            dA_stacked = torch.stack([torch.from_numpy(g) for g in dA_batch]).T

            if originally_unbatched:
                dq_stacked = dq_stacked.squeeze(1)
                dA_stacked = dA_stacked.squeeze(1)

            return None, dq_stacked, dA_stacked, None, None, None, None
