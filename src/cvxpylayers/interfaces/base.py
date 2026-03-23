"""Base class for custom solver interfaces in cvxpylayers.

Users subclass ``SolverInterface``, implement at least one solve method and
one derivative method, and pass the instance as ``solver=`` to ``CvxpyLayer``.

cvxpylayers automatically fills in all missing variants via a **ring of
default implementations**.  Every method's default calls the next in the ring,
and ``@require_one_of`` guarantees at least one link is user-overridden,
breaking the cycle:

Solve ring::

    solve_torch_batch → solve_torch → solve_numpy → solve_numpy_batch
        → solve_jax_batch → solve_jax → solve_mlx → solve_mlx_batch
        → (back to solve_torch_batch)

Each step is one of:

* **batch → single**: loops the single-problem method over the batch dim.
* **single → single**: converts arrays to the next framework.
* **single → batch**: adds a trivial batch dim of 1, calls batch, strips it.
* **batch → batch**: converts arrays to the next framework.

The same ring structure applies to the derivative methods.

Each framework's layer calls its own entry point directly:

* Torch layer  → ``solve_torch_batch`` / ``derivative_torch_batch``
* JAX layer    → ``solve_jax_batch``   / ``derivative_jax_batch``
* MLX layer    → ``solve_mlx_batch``   / ``derivative_mlx_batch``
"""
from __future__ import annotations

from abc import ABC
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# @require_one_of decorator
# ---------------------------------------------------------------------------

def require_one_of(*method_names: str):
    """Class decorator: at least one of *method_names* must be overridden.

    Raises ``TypeError`` at *subclass definition time* if none of the named
    methods appear in the subclass's own MRO entries.  Abstract subclasses
    (those still carrying ``__abstractmethods__``) are exempt.

    Example::

        @require_one_of('method_a', 'method_b')
        class Base(ABC):
            def method_a(self): raise NotImplementedError
            def method_b(self): raise NotImplementedError

        class Good(Base):
            def method_a(self): return 1   # OK

        class Bad(Base):
            pass  # TypeError: Bad must override at least one of: method_a, method_b
    """
    def decorator(cls: type) -> type:
        original_init_subclass = cls.__init_subclass__

        @classmethod  # type: ignore[misc]
        def new_init_subclass(sub_cls: type, **kwargs: Any) -> None:
            # When decorators are stacked, original_init_subclass is a bound
            # classmethod bound to *cls*, not to *sub_cls*.  We must call
            # through __func__ so the correct subclass propagates through the
            # chain instead of the decorator's parent class.
            if hasattr(original_init_subclass, "__func__"):
                original_init_subclass.__func__(sub_cls, **kwargs)
            else:
                original_init_subclass(**kwargs)  # type: ignore[call-arg]
            if getattr(sub_cls, "__abstractmethods__", frozenset()):
                return  # still abstract — defer check to concrete subclass
            for name in method_names:
                if any(
                    name in vars(klass)
                    for klass in sub_cls.__mro__
                    if klass is not cls and klass is not object
                ):
                    return
            raise TypeError(
                f"{sub_cls.__name__} must override at least one of: "
                f"{', '.join(method_names)}"
            )

        cls.__init_subclass__ = new_init_subclass  # type: ignore[method-assign]
        return cls

    return decorator


# ---------------------------------------------------------------------------
# Conversion helpers (all cross-framework paths go through numpy as the hub)
# ---------------------------------------------------------------------------

def _to_numpy_from_torch(t: Any) -> np.ndarray:
    return t.detach().cpu().numpy()


def _to_torch_from_numpy(arr: np.ndarray) -> Any:
    import torch
    return torch.from_numpy(np.asarray(arr))


def _to_jax_from_numpy(arr: np.ndarray) -> Any:
    import jax.numpy as jnp
    return jnp.array(arr)


def _to_numpy_from_jax(arr: Any) -> np.ndarray:
    return np.asarray(arr)


def _to_mlx_from_numpy(arr: np.ndarray) -> Any:
    import mlx.core as mx
    return mx.array(arr)


def _to_numpy_from_mlx(arr: Any) -> np.ndarray:
    return np.asarray(arr)


# ---------------------------------------------------------------------------
# Per-item adjoint splitting helper
# ---------------------------------------------------------------------------

def _split_adj(adj: Any, batch_size: int) -> list[Any]:
    """Return a list of per-item adjoints of length *batch_size*.

    If *adj* is already a list of the right length (produced by looping a
    single-problem method), return it as-is.  Otherwise broadcast the single
    batch-level object to every item.
    """
    if isinstance(adj, list) and len(adj) == batch_size:
        return adj
    return [adj] * batch_size


# ---------------------------------------------------------------------------
# _SOLVE_METHODS / _DERIV_METHODS — all 8 variants for @require_one_of
# ---------------------------------------------------------------------------

_SOLVE_METHODS = (
    "solve_torch_batch",
    "solve_torch",
    "solve_numpy",
    "solve_numpy_batch",
    "solve_jax_batch",
    "solve_jax",
    "solve_mlx",
    "solve_mlx_batch",
)
_DERIV_METHODS = (
    "derivative_torch_batch",
    "derivative_torch",
    "derivative_numpy",
    "derivative_numpy_batch",
    "derivative_jax_batch",
    "derivative_jax",
    "derivative_mlx",
    "derivative_mlx_batch",
)


# ---------------------------------------------------------------------------
# SolverInterface ABC
# ---------------------------------------------------------------------------

@require_one_of(*_SOLVE_METHODS)
@require_one_of(*_DERIV_METHODS)
class SolverInterface(ABC):
    """Base class for plugging a custom solver into ``CvxpyLayer``.

    Subclass this, override **at least one** solve method and **at least one**
    derivative method, then pass an instance as ``solver=`` to
    ``CvxpyLayer``::

        layer = CvxpyLayer(problem, parameters=[A, b], variables=[x],
                           solver=MySolver())

    Each framework's layer calls its own entry point:

    * Torch layer  → ``solve_torch_batch`` / ``derivative_torch_batch``
    * JAX layer    → ``solve_jax_batch``   / ``derivative_jax_batch``
    * MLX layer    → ``solve_mlx_batch``   / ``derivative_mlx_batch``

    All other methods are filled in by default implementations that form a
    ring (see module docstring).  ``@require_one_of`` ensures exactly one link
    in the ring is user-implemented, so the ring terminates without infinite
    recursion.

    **Array shape conventions** (same for every framework variant):

    +---------------------+-----------------------------------+
    | Batched input       | ``(B, n)`` — batch **first**      |
    +---------------------+-----------------------------------+
    | Batched output      | ``(B, n)`` — batch **first**      |
    +---------------------+-----------------------------------+
    | Single input        | ``(n,)`` — 1-D                    |
    +---------------------+-----------------------------------+
    | Single output       | ``(n,)`` — 1-D                    |
    +---------------------+-----------------------------------+

    ``dP`` may be ``None`` for LP-only solvers (no QP gradient).

    **Class attributes** (set as class variables, not in ``__init__``):

    .. code-block:: python

        class MySolver(SolverInterface):
            canon_solver = "CLARABEL"   # default: "SCS"
            supports_quad_obj = True    # default: False

    ``canon_solver``: CVXPY solver name used during problem canonicalization.
    Must match the cone format your solver expects.

    ``supports_quad_obj``: enable ``quad_form_dpp_scope`` for parametric QP
    objectives.

    Examples
    --------
    Numpy, single-problem (simplest)::

        class MySolver(SolverInterface):
            canon_solver = "CLARABEL"

            def solve_numpy(self, P, q, A, dims, solver_args, needs_grad):
                primal, dual = my_c_code(q, A, dims)
                return primal, dual, None

            def derivative_numpy(self, dprimal, ddual, adjoint_data):
                dq, dA = my_c_derivative(dprimal, ddual)
                return None, dq, dA     # dP=None for LP

    Torch, native batched (no conversion overhead)::

        class MySolver(SolverInterface):
            canon_solver = "CLARABEL"
            supports_quad_obj = True

            def solve_torch_batch(self, P, q, A, dims, solver_args, needs_grad):
                # P: (B, nnz_P) or None; q: (B, n); A: (B, m)
                return primal, dual, kkt_state

            def derivative_torch_batch(self, dprimal, ddual, kkt_state):
                # dprimal/ddual: (B, n)
                return dP, dq, dA       # each (B, n), dP may be None
    """

    #: CVXPY solver name for problem canonicalization.
    canon_solver: str = "SCS"

    #: Set True if your solver handles parametric QP objectives.
    supports_quad_obj: bool = False

    # ------------------------------------------------------------------
    # Solve methods — implement at least one
    # ------------------------------------------------------------------
    # Ring order:
    #   torch_batch → torch → numpy → numpy_batch
    #               → jax_batch → jax → mlx → mlx_batch → (torch_batch)
    # ------------------------------------------------------------------

    def solve_torch_batch(
        self,
        P: Any | None,
        q: Any,
        A: Any,
        dims: dict,
        solver_args: dict,
        needs_grad: bool,
    ) -> tuple[Any, Any, Any]:
        """Solve a **batch** of problems; inputs/outputs are torch tensors.

        Args:
            P: ``(B, nnz_P)`` or ``None`` for LP.
            q: ``(B, n_vars + 1)``
            A: ``(B, nnz_A)``
            dims: Cone dimensions dict.
            solver_args: Per-call keyword arguments.
            needs_grad: If ``False`` the adjoint system may be skipped.

        Returns:
            ``(primal, dual, adjoint_data)`` — torch tensors + opaque adjoint.

        Default: loops :meth:`solve_torch` over the batch dimension.
        """
        import torch
        batch = q.shape[0]
        primals, duals, adjs = [], [], []
        for i in range(batch):
            p_i, d_i, a_i = self.solve_torch(
                P[i] if P is not None else None, q[i], A[i],
                dims, solver_args, needs_grad,
            )
            primals.append(p_i)
            duals.append(d_i)
            adjs.append(a_i)
        return torch.stack(primals, dim=0), torch.stack(duals, dim=0), adjs

    def solve_torch(
        self,
        P: Any | None,
        q: Any,
        A: Any,
        dims: dict,
        solver_args: dict,
        needs_grad: bool,
    ) -> tuple[Any, Any, Any]:
        """Solve a **single** problem; inputs/outputs are torch tensors.

        Shapes mirror :meth:`solve_numpy` but using torch tensors.

        Default: converts to numpy, calls :meth:`solve_numpy`, converts back.
        """
        P_np = _to_numpy_from_torch(P) if P is not None else None
        q_np = _to_numpy_from_torch(q)
        A_np = _to_numpy_from_torch(A)
        primal_np, dual_np, adj = self.solve_numpy(P_np, q_np, A_np, dims, solver_args, needs_grad)
        return (
            _to_torch_from_numpy(primal_np).to(dtype=q.dtype, device=q.device),
            _to_torch_from_numpy(dual_np).to(dtype=q.dtype, device=q.device),
            adj,
        )

    def solve_numpy(
        self,
        P: np.ndarray | None,
        q: np.ndarray,
        A: np.ndarray,
        dims: dict,
        solver_args: dict,
        needs_grad: bool,
    ) -> tuple[np.ndarray, np.ndarray, Any]:
        """Solve a **single** problem; inputs/outputs are numpy arrays.

        Args:
            P: ``(nnz_P,)`` or ``None``.
            q: ``(n_vars + 1,)``
            A: ``(nnz_A,)``
            dims: Cone dimensions dict.
            solver_args: Per-call keyword arguments.
            needs_grad: Skip adjoint construction if ``False``.

        Returns:
            ``(primal, dual, adjoint_data)`` — 1-D numpy arrays + opaque adj.

        Default: adds a batch dim of 1, calls :meth:`solve_numpy_batch`,
        strips the batch dim.
        """
        P_b = P[np.newaxis] if P is not None else None
        primal_b, dual_b, adj = self.solve_numpy_batch(
            P_b, q[np.newaxis], A[np.newaxis], dims, solver_args, needs_grad,
        )
        adj_item = adj[0] if isinstance(adj, list) and len(adj) == 1 else adj
        return primal_b[0], dual_b[0], adj_item

    def solve_numpy_batch(
        self,
        P: np.ndarray | None,
        q: np.ndarray,
        A: np.ndarray,
        dims: dict,
        solver_args: dict,
        needs_grad: bool,
    ) -> tuple[np.ndarray, np.ndarray, Any]:
        """Solve a **batch** of problems; inputs/outputs are numpy arrays.

        Args:
            P: ``(B, nnz_P)`` or ``None``.
            q: ``(B, n_vars + 1)``
            A: ``(B, nnz_A)``

        Returns:
            ``(primal, dual, adjoint_data)`` — 2-D numpy arrays + opaque adj.

        Default: converts to JAX, calls :meth:`solve_jax_batch`, converts back.
        """
        P_jax = _to_jax_from_numpy(P) if P is not None else None
        primal_jax, dual_jax, adj = self.solve_jax_batch(
            P_jax, _to_jax_from_numpy(q), _to_jax_from_numpy(A),
            dims, solver_args, needs_grad,
        )
        return _to_numpy_from_jax(primal_jax), _to_numpy_from_jax(dual_jax), adj

    def solve_jax_batch(
        self,
        P: Any | None,
        q: Any,
        A: Any,
        dims: dict,
        solver_args: dict,
        needs_grad: bool,
    ) -> tuple[Any, Any, Any]:
        """Solve a **batch** of problems; inputs/outputs are JAX arrays.

        Default: loops :meth:`solve_jax` over the batch dimension.
        """
        import jax.numpy as jnp
        batch = q.shape[0]
        primals, duals, adjs = [], [], []
        for i in range(batch):
            p_i, d_i, a_i = self.solve_jax(
                P[i] if P is not None else None, q[i], A[i],
                dims, solver_args, needs_grad,
            )
            primals.append(p_i)
            duals.append(d_i)
            adjs.append(a_i)
        return jnp.stack(primals, axis=0), jnp.stack(duals, axis=0), adjs

    def solve_jax(
        self,
        P: Any | None,
        q: Any,
        A: Any,
        dims: dict,
        solver_args: dict,
        needs_grad: bool,
    ) -> tuple[Any, Any, Any]:
        """Solve a **single** problem; inputs/outputs are JAX arrays.

        Default: converts to MLX, calls :meth:`solve_mlx`, converts back.
        """
        P_mlx = _to_mlx_from_numpy(_to_numpy_from_jax(P)) if P is not None else None
        primal_mlx, dual_mlx, adj = self.solve_mlx(
            P_mlx,
            _to_mlx_from_numpy(_to_numpy_from_jax(q)),
            _to_mlx_from_numpy(_to_numpy_from_jax(A)),
            dims, solver_args, needs_grad,
        )
        import jax.numpy as jnp
        return (
            jnp.array(_to_numpy_from_mlx(primal_mlx)),
            jnp.array(_to_numpy_from_mlx(dual_mlx)),
            adj,
        )

    def solve_mlx(
        self,
        P: Any | None,
        q: Any,
        A: Any,
        dims: dict,
        solver_args: dict,
        needs_grad: bool,
    ) -> tuple[Any, Any, Any]:
        """Solve a **single** problem; inputs/outputs are MLX arrays.

        Default: adds a batch dim of 1, calls :meth:`solve_mlx_batch`,
        strips the batch dim.
        """
        import mlx.core as mx
        P_b = mx.expand_dims(P, 0) if P is not None else None
        primal_b, dual_b, adj = self.solve_mlx_batch(
            P_b, mx.expand_dims(q, 0), mx.expand_dims(A, 0),
            dims, solver_args, needs_grad,
        )
        adj_item = adj[0] if isinstance(adj, list) and len(adj) == 1 else adj
        return primal_b[0], dual_b[0], adj_item

    def solve_mlx_batch(
        self,
        P: Any | None,
        q: Any,
        A: Any,
        dims: dict,
        solver_args: dict,
        needs_grad: bool,
    ) -> tuple[Any, Any, Any]:
        """Solve a **batch** of problems; inputs/outputs are MLX arrays.

        Default: converts to torch, calls :meth:`solve_torch_batch`,
        converts back.  Closes the ring.
        """
        import mlx.core as mx
        P_t = _to_torch_from_numpy(_to_numpy_from_mlx(P)) if P is not None else None
        primal_t, dual_t, adj = self.solve_torch_batch(
            P_t,
            _to_torch_from_numpy(_to_numpy_from_mlx(q)),
            _to_torch_from_numpy(_to_numpy_from_mlx(A)),
            dims, solver_args, needs_grad,
        )
        return (
            mx.array(_to_numpy_from_torch(primal_t)),
            mx.array(_to_numpy_from_torch(dual_t)),
            adj,
        )

    # ------------------------------------------------------------------
    # Derivative methods — implement at least one
    # ------------------------------------------------------------------
    # Same ring order as solve:
    #   torch_batch → torch → numpy → numpy_batch
    #               → jax_batch → jax → mlx → mlx_batch → (torch_batch)
    # ------------------------------------------------------------------

    def derivative_torch_batch(
        self,
        dprimal: Any,
        ddual: Any,
        adjoint_data: Any,
    ) -> tuple[Any | None, Any, Any]:
        """Backward pass for a **batch** of problems; tensors are torch.

        Args:
            dprimal: ``(B, n_primal)``
            ddual:   ``(B, n_dual)``
            adjoint_data: from the corresponding solve call.

        Returns:
            ``(dP, dq, dA)`` each ``(B, n)``; ``dP`` may be ``None``.

        Default: loops :meth:`derivative_torch` over the batch dimension.
        """
        import torch
        batch = dprimal.shape[0]
        adj_list = _split_adj(adjoint_data, batch)
        dPs, dqs, dAs = [], [], []
        has_dP = False
        for i in range(batch):
            dP_i, dq_i, dA_i = self.derivative_torch(dprimal[i], ddual[i], adj_list[i])
            if dP_i is not None:
                dPs.append(dP_i)
                has_dP = True
            dqs.append(dq_i)
            dAs.append(dA_i)
        dP = torch.stack(dPs, dim=0) if has_dP else None
        return dP, torch.stack(dqs, dim=0), torch.stack(dAs, dim=0)

    def derivative_torch(
        self,
        dprimal: Any,
        ddual: Any,
        adjoint_data: Any,
    ) -> tuple[Any | None, Any, Any]:
        """Backward pass for a **single** problem; tensors are torch.

        Default: converts to numpy, calls :meth:`derivative_numpy`, converts back.
        """
        dp_np = _to_numpy_from_torch(dprimal)
        dd_np = _to_numpy_from_torch(ddual)
        dP_np, dq_np, dA_np = self.derivative_numpy(dp_np, dd_np, adjoint_data)
        ref = dprimal
        dP = (
            _to_torch_from_numpy(dP_np).to(dtype=ref.dtype, device=ref.device)
            if dP_np is not None
            else None
        )
        return (
            dP,
            _to_torch_from_numpy(dq_np).to(dtype=ref.dtype, device=ref.device),
            _to_torch_from_numpy(dA_np).to(dtype=ref.dtype, device=ref.device),
        )

    def derivative_numpy(
        self,
        dprimal: np.ndarray,
        ddual: np.ndarray,
        adjoint_data: Any,
    ) -> tuple[np.ndarray | None, np.ndarray, np.ndarray]:
        """Backward pass for a **single** problem; arrays are numpy.

        Args:
            dprimal: ``(n_primal,)``
            ddual:   ``(n_dual,)``
            adjoint_data: from the corresponding solve call.

        Returns:
            ``(dP, dq, dA)`` — 1-D numpy arrays; ``dP`` may be ``None``.

        Default: adds a batch dim of 1, calls :meth:`derivative_numpy_batch`,
        strips the batch dim.
        """
        dP_b, dq_b, dA_b = self.derivative_numpy_batch(
            dprimal[np.newaxis], ddual[np.newaxis], [adjoint_data],
        )
        dP = dP_b[0] if dP_b is not None else None
        return dP, dq_b[0], dA_b[0]

    def derivative_numpy_batch(
        self,
        dprimal: np.ndarray,
        ddual: np.ndarray,
        adjoint_data: Any,
    ) -> tuple[np.ndarray | None, np.ndarray, np.ndarray]:
        """Backward pass for a **batch** of problems; arrays are numpy.

        Args:
            dprimal: ``(B, n_primal)``
            ddual:   ``(B, n_dual)``
            adjoint_data: from the corresponding solve call.

        Returns:
            ``(dP, dq, dA)`` each ``(B, n)``; ``dP`` may be ``None``.

        Default: converts to JAX, calls :meth:`derivative_jax_batch`, converts back.
        """
        dP_jax, dq_jax, dA_jax = self.derivative_jax_batch(
            _to_jax_from_numpy(dprimal),
            _to_jax_from_numpy(ddual),
            adjoint_data,
        )
        dP = _to_numpy_from_jax(dP_jax) if dP_jax is not None else None
        return dP, _to_numpy_from_jax(dq_jax), _to_numpy_from_jax(dA_jax)

    def derivative_jax_batch(
        self,
        dprimal: Any,
        ddual: Any,
        adjoint_data: Any,
    ) -> tuple[Any | None, Any, Any]:
        """Backward pass for a **batch** of problems; arrays are JAX arrays.

        Default: loops :meth:`derivative_jax` over the batch dimension.
        """
        import jax.numpy as jnp
        batch = dprimal.shape[0]
        adj_list = _split_adj(adjoint_data, batch)
        dPs, dqs, dAs = [], [], []
        has_dP = False
        for i in range(batch):
            dP_i, dq_i, dA_i = self.derivative_jax(dprimal[i], ddual[i], adj_list[i])
            if dP_i is not None:
                dPs.append(dP_i)
                has_dP = True
            dqs.append(dq_i)
            dAs.append(dA_i)
        dP = jnp.stack(dPs, axis=0) if has_dP else None
        return dP, jnp.stack(dqs, axis=0), jnp.stack(dAs, axis=0)

    def derivative_jax(
        self,
        dprimal: Any,
        ddual: Any,
        adjoint_data: Any,
    ) -> tuple[Any | None, Any, Any]:
        """Backward pass for a **single** problem; arrays are JAX arrays.

        Default: converts to MLX, calls :meth:`derivative_mlx`, converts back.
        """
        dp_mlx = _to_mlx_from_numpy(_to_numpy_from_jax(dprimal))
        dd_mlx = _to_mlx_from_numpy(_to_numpy_from_jax(ddual))
        dP_mlx, dq_mlx, dA_mlx = self.derivative_mlx(dp_mlx, dd_mlx, adjoint_data)
        import jax.numpy as jnp
        dP = jnp.array(_to_numpy_from_mlx(dP_mlx)) if dP_mlx is not None else None
        return dP, jnp.array(_to_numpy_from_mlx(dq_mlx)), jnp.array(_to_numpy_from_mlx(dA_mlx))

    def derivative_mlx(
        self,
        dprimal: Any,
        ddual: Any,
        adjoint_data: Any,
    ) -> tuple[Any | None, Any, Any]:
        """Backward pass for a **single** problem; arrays are MLX arrays.

        Default: adds a batch dim of 1, calls :meth:`derivative_mlx_batch`,
        strips the batch dim.
        """
        import mlx.core as mx
        dP_b, dq_b, dA_b = self.derivative_mlx_batch(
            mx.expand_dims(dprimal, 0),
            mx.expand_dims(ddual, 0),
            [adjoint_data],
        )
        dP = dP_b[0] if dP_b is not None else None
        return dP, dq_b[0], dA_b[0]

    def derivative_mlx_batch(
        self,
        dprimal: Any,
        ddual: Any,
        adjoint_data: Any,
    ) -> tuple[Any | None, Any, Any]:
        """Backward pass for a **batch** of problems; arrays are MLX arrays.

        Default: converts to torch, calls :meth:`derivative_torch_batch`,
        converts back.  Closes the ring.
        """
        import mlx.core as mx
        dp_t = _to_torch_from_numpy(_to_numpy_from_mlx(dprimal))
        dd_t = _to_torch_from_numpy(_to_numpy_from_mlx(ddual))
        dP_t, dq_t, dA_t = self.derivative_torch_batch(dp_t, dd_t, adjoint_data)
        dP = mx.array(_to_numpy_from_torch(dP_t)) if dP_t is not None else None
        return dP, mx.array(_to_numpy_from_torch(dq_t)), mx.array(_to_numpy_from_torch(dA_t))
