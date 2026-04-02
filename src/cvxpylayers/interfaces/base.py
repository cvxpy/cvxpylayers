"""Base class for custom solver interfaces in cvxpylayers.

Users subclass ``SolverInterface``, implement at least one solve method and
one derivative method, and pass the instance as ``solver=`` to ``CvxpyLayer``.

Ring of default implementations
--------------------------------
Every method's *default* body delegates to the **next** method in the ring.
``@require_one_of`` enforces that at least one method is user-overridden.
Because overrides do **not** call ``super()``, they terminate the chain —
there is no infinite recursion.

Solve ring (each arrow = "default delegates to →")::

    solve_torch_batch  ──[batch→single]──►  solve_torch
    solve_torch        ──[single→single]──►  solve_numpy        (torch → numpy)
    solve_numpy        ──[single→batch]──►  solve_numpy_batch
    solve_numpy_batch  ──[batch→batch]───►  solve_jax_batch     (numpy → jax)
    solve_jax_batch    ──[batch→single]──►  solve_jax
    solve_jax          ──[single→single]──►  solve_mlx           (jax → mlx)
    solve_mlx          ──[single→batch]──►  solve_mlx_batch
    solve_mlx_batch    ──[batch→batch]───►  solve_torch_batch   (mlx → torch, closes ring)

Arrow semantics:

* **batch → single**: strips batch dim, loops single-problem method, re-stacks.
* **single → batch**: adds a size-1 batch dim, calls batch method, strips it.
* **single → single** / **batch → batch**: converts arrays to the next framework
  (all cross-framework conversions go through numpy as a hub).

Example — overriding ``solve_numpy``: a call entering at
``solve_torch_batch`` walks the chain until it hits the override::

    solve_torch_batch → solve_torch → ★ solve_numpy   (chain ends here)

The same ring and semantics apply identically to the derivative methods.

Entry points — each framework layer calls its own batch variant directly:

* Torch layer  → ``solve_torch_batch`` / ``derivative_torch_batch``
* JAX layer    → ``solve_jax_batch``   / ``derivative_jax_batch``
* MLX layer    → ``solve_mlx_batch``   / ``derivative_mlx_batch``

Choosing what to implement
--------------------------
* **Simplest** (CPU, no framework dependency): override ``solve_numpy`` +
  ``derivative_numpy``.  All three frameworks will work; batching and
  array conversion are handled automatically.
* **Native batched**: override the ``*_batch`` variant for your framework to
  avoid the per-sample Python loop inserted by the batch→single default.
* **Framework-agnostic batched**: override ``solve_numpy_batch`` +
  ``derivative_numpy_batch``; all frameworks convert to numpy before calling.
"""
from __future__ import annotations

from abc import ABC
from typing import Any

import numpy as np
from cvxpy.reductions.solvers.solver import Solver as _CvxpySolver

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
# Per-item state splitting helper
# ---------------------------------------------------------------------------

def _split_state(state: Any, batch_size: int) -> list[Any]:
    """Return a list of per-item saved states of length *batch_size*.

    If *state* is already a list of the right length (produced by looping a
    single-problem method), return it as-is.  Otherwise broadcast the single
    batch-level object to every item.
    """
    if isinstance(state, list) and len(state) == batch_size:
        return state
    return [state] * batch_size


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
            canon_solver = "CLARABEL"              # string name — default: "SCS"
            # or equivalently, a CVXPY Solver instance:
            canon_solver = cp.reductions.solvers.conic_solvers.clarabel_conif.CLARABEL()
            supports_quad_obj = True               # default: False

    ``canon_solver``: CVXPY solver used during problem canonicalization —
    either a solver name string (e.g. ``"CLARABEL"``) or a
    ``cvxpy.reductions.solvers.solver.Solver`` instance.  Use
    :attr:`canon_solver_name` to get the normalised string in either case.
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
                return primal, dual, self.save_for_backward(primal, dual)

            def derivative_numpy(self, dprimal, ddual, saved_state):
                primal, dual = saved_state
                dq, dA = my_c_derivative(dprimal, ddual, primal, dual)
                return None, dq, dA     # dP=None for LP

    Extra solver state — derivative only needs the factorization, not primal/dual::

        class MySolver(SolverInterface):
            canon_solver = "CLARABEL"

            def save_for_backward(self, primal, dual):
                return self._last_kkt_factor   # only the extra; skip primal/dual

            def solve_numpy(self, P, q, A, dims, solver_args, needs_grad):
                primal, dual, factor = my_factored_solve(q, A, dims)
                self._last_kkt_factor = factor
                return primal, dual, self.save_for_backward(primal, dual)

            def derivative_numpy(self, dprimal, ddual, saved_state):
                factor = saved_state
                dq, dA = my_derivative(dprimal, ddual, factor)
                return None, dq, dA

    Extra solver state — derivative needs primal/dual AND a factorization::

        class MySolver(SolverInterface):
            canon_solver = "CLARABEL"

            def save_for_backward(self, primal, dual):
                return primal, dual, self._last_kkt_factor   # all three needed

            def solve_numpy(self, P, q, A, dims, solver_args, needs_grad):
                primal, dual, factor = my_factored_solve(q, A, dims)
                self._last_kkt_factor = factor
                return primal, dual, self.save_for_backward(primal, dual)

            def derivative_numpy(self, dprimal, ddual, saved_state):
                primal, dual, factor = saved_state
                dq, dA = my_derivative(dprimal, ddual, primal, dual, factor)
                return None, dq, dA

    Torch, native batched (no conversion overhead)::

        class MySolver(SolverInterface):
            canon_solver = "CLARABEL"
            supports_quad_obj = True

            def solve_torch_batch(self, P, q, A, dims, solver_args, needs_grad):
                # P: (B, nnz_P) or None; q: (B, n); A: (B, m)
                return primal, dual, kkt_state

            def derivative_torch_batch(self, dprimal, ddual, saved_state):
                # dprimal/ddual: (B, n)
                return dP, dq, dA       # each (B, n), dP may be None
    """

    #: CVXPY solver for problem canonicalization — a name string or a Solver instance.
    canon_solver: str | _CvxpySolver = "SCS"

    #: Set True if your solver handles parametric QP objectives.
    supports_quad_obj: bool = False

    #: Set True for parameter-space solvers (e.g. CVXPYgen).
    #: ``CvxpyLayer`` will skip canonical-matrix evaluation and call
    #: ``_cpg_solve`` / ``_cpg_solve_and_gradient`` / ``_cpg_gradient`` directly.
    is_parametric: bool = False

    @classmethod
    def from_functions(
        cls,
        solve: Any,
        derivative: Any,
        *,
        save_for_backward: Any | None = None,
        canon_solver: str | _CvxpySolver = "SCS",
        supports_quad_obj: bool = False,
    ) -> "SolverInterface":
        """Create a :class:`SolverInterface` from a pair of plain numpy callables.

        This is the lightest-weight way to plug in a custom solver without
        subclassing.  Both functions operate on **single** (unbatched) numpy
        arrays; the ring of default implementations handles batching and
        cross-framework conversion automatically.

        Args:
            solve: ``(P, q, A, dims, solver_args, needs_grad) -> (primal, dual)``
                **or** ``(primal, dual, saved_state)``.  Arrays are 1-D numpy;
                ``P`` may be ``None`` for LPs.  When only a 2-tuple is returned
                the saved state is built by calling ``save_for_backward(primal,
                dual)`` (see below).
            derivative: ``(dprimal, ddual, saved_state) -> (dP, dq, dA)``
                where arrays are 1-D numpy, ``dP`` may be ``None``.
            save_for_backward: Optional ``(primal, dual) -> saved_state``
                callable.  Called when ``solve`` returns a 2-tuple.  Defaults
                to ``lambda p, d: (p, d)`` — i.e. saves ``(primal, dual)``.
                Pass a custom function to attach extra solver state.
            canon_solver: CVXPY solver name or instance for problem canonicalization.
            supports_quad_obj: Whether the solver handles parametric QP objectives.

        Returns:
            A concrete :class:`SolverInterface` instance wrapping the callables.

        Simple example (2-tuple solve — saved state defaults to ``(primal, dual)``)::

            def my_solve(P, q, A, dims, args, needs_grad):
                primal, dual = fast_conic_solve(q, A, dims)
                return primal, dual                          # no repeated state

            def my_derivative(dprimal, ddual, saved_state):
                primal, dual = saved_state                   # filled automatically
                dq, dA = fast_derivative(dprimal, ddual, primal, dual)
                return None, dq, dA

            layer = CvxpyLayer(problem, parameters=[A, b], variables=[x],
                               solver=SolverInterface.from_functions(my_solve, my_derivative))

        Custom state example (attach a factorization for cheaper derivatives)::

            def my_solve(P, q, A, dims, args, needs_grad):
                primal, dual, factor = fast_conic_solve_with_factor(q, A, dims)
                return primal, dual, (primal, dual, factor)  # explicit 3-tuple

            def my_derivative(dprimal, ddual, saved_state):
                primal, dual, factor = saved_state
                dq, dA = fast_derivative_with_factor(dprimal, ddual, factor)
                return None, dq, dA
        """
        _solve, _derivative = solve, derivative
        _sfb: Any = save_for_backward if save_for_backward is not None else (
            lambda p, d: (p, d)
        )

        def _wrapped_solve(
            self: Any,
            P: Any, q: Any, A: Any,
            dims: Any, args: Any, ng: Any,
        ) -> tuple[Any, Any, Any]:
            result = _solve(P, q, A, dims, args, ng)
            if len(result) == 2:
                primal, dual = result
                return primal, dual, _sfb(primal, dual)
            primal, dual, state = result
            return primal, dual, state

        # Use type() so that @require_one_of sees the overrides at class-creation
        # time (vars() check), avoiding the "must override at least one" error.
        _cls: type[SolverInterface] = type(
            "_FunctionalSolverInterface",
            (cls,),
            {
                "canon_solver": canon_solver,
                "supports_quad_obj": supports_quad_obj,
                "solve_numpy": _wrapped_solve,
                "derivative_numpy": (
                    lambda self, dp, dd, state: _derivative(dp, dd, state)
                ),
            },
        )
        return _cls()

    @classmethod
    def from_parametric_functions(
        cls,
        solve: Any,
        solve_and_gradient: Any = None,
        gradient: Any = None,
    ) -> "SolverInterface":
        """Create a parameter-space :class:`SolverInterface` from CVXPYgen functions.

        This is the recommended way to integrate a CVXPYgen-generated solver.
        The three function arguments map directly to what CVXPYgen exports::

            import cpg_solver  # generated by cvxpygen

            layer = CvxpyLayer(
                problem, parameters=[A, b], variables=[x],
                solver=SolverInterface.from_parametric_functions(
                    solve               = cpg_solver.cpg_solve,
                    solve_and_gradient  = cpg_solver.cpg_solve_and_gradient_info,
                    gradient            = cpg_solver.cpg_gradient,
                ),
            )

        All problem specification (``problem``, ``parameters``, ``variables``)
        is already provided to :class:`~cvxpylayers.torch.CvxpyLayer` — no
        duplicate arguments are needed here.

        :class:`~cvxpylayers.torch.CvxpyLayer` detects :attr:`is_parametric`
        and bypasses the canonical ``q @ p_stack`` / ``A @ p_stack`` evaluation,
        propagating gradients directly through ``param.gradient`` — no
        pseudoinverse required.

        Args:
            solve: ``cpg_solve(problem)`` — runs the compiled solver, sets
                ``var.value``.
            solve_and_gradient: ``cpg_solve_and_gradient_info(problem) ->
                (cpg_var, gp, gd)`` — runs the solver *and* captures
                intermediate data for the backward pass.  Optional; falls back
                to ``solve`` when absent.
            gradient: ``cpg_gradient(problem, grad_primal, grad_dual)`` — sets
                ``param.gradient`` for each CVXPY parameter.  Required for
                backpropagation.

        Returns:
            A concrete :class:`SolverInterface` instance with
            :attr:`is_parametric` ``= True``.
        """
        def _not_impl(self: Any, *a: Any, **kw: Any) -> Any:
            raise NotImplementedError(
                "Parametric SolverInterface must be used through CvxpyLayer."
            )

        _cls: type[SolverInterface] = type(
            "_ParametricSolverInterface",
            (cls,),
            {
                "is_parametric":           True,
                # Satisfy @require_one_of — never reached on the parametric path.
                "solve_numpy":             _not_impl,
                "derivative_numpy":        _not_impl,
                "_cpg_solve":              solve,
                "_cpg_solve_and_gradient": solve_and_gradient,
                "_cpg_gradient":           gradient,
            },
        )
        return _cls()

    @property
    def canon_solver_name(self) -> str:
        """Normalised solver name string.

        Returns ``canon_solver`` as-is when it is already a string, or calls
        ``.name()`` on it when it is a CVXPY ``Solver`` instance.
        """
        raw = self.canon_solver
        return raw.name() if isinstance(raw, _CvxpySolver) else raw

    # ------------------------------------------------------------------
    # Lifecycle hooks (optional overrides — default: no-op)
    # ------------------------------------------------------------------

    def setup(self, ctx: Any) -> None:
        """Called once by ``CvxpyLayer.__init__`` after problem canonicalization.

        Receives the fully-populated :class:`~cvxpylayers.utils.parse_args.LayersContext`
        so the solver can inspect problem structure (cone dimensions, sparse
        canonical matrices, variable recovery info, etc.) and perform any
        one-time setup (e.g. load a compiled C extension, pre-factor matrices).

        Args:
            ctx: The ``LayersContext`` built by
                :func:`~cvxpylayers.utils.parse_args.parse_args`.

        Default: no-op.
        """

    def set_params(self, params: list) -> None:
        """Called by ``CvxpyLayer.forward()`` before each solve.

        Receives the current parameter values as a list of numpy arrays (one
        per CVXPY ``Parameter``, in the same order as passed to
        ``CvxpyLayer``).  The shapes match the CVXPY parameter shapes; batched
        parameters have an extra leading batch dimension.

        Solvers that work with the original CVXPY parameter values (e.g. a
        CVXPYgen-generated solver) should override this to capture them before
        :meth:`solve_numpy` is called.

        Args:
            params: List of ``numpy.ndarray``, one per parameter.

        Default: no-op.
        """

    def save_for_backward(self, primal: np.ndarray, dual: np.ndarray) -> Any:
        """Build the saved state passed to the matching ``derivative_*`` call.

        Returned as the third element of every ``solve_*`` method.  Override
        to attach extra solver data (factorizations, active sets, residuals,
        …) that your derivative implementation needs beyond ``(primal, dual)``::

            def save_for_backward(self, primal, dual):
                return primal, dual, self._last_kkt_factor

            def derivative_numpy(self, dprimal, ddual, saved_state):
                primal, dual, factor = saved_state
                ...

        In :meth:`from_functions`, the solve callable may return either
        ``(primal, dual)`` or ``(primal, dual, saved_state)``.  When only a
        2-tuple is returned the result of ``save_for_backward(primal, dual)``
        is used automatically.

        Args:
            primal: 1-D numpy primal solution from the just-completed solve.
            dual:   1-D numpy dual solution from the just-completed solve.

        Returns:
            Anything needed by ``derivative_*``.  Default: ``(primal, dual)``.
        """
        return primal, dual

    # ------------------------------------------------------------------
    # Solve methods — implement at least one (see module docstring for details)
    # ------------------------------------------------------------------
    # Ring order and step type at each arrow:
    #
    #   torch_batch  -[batch→single]->  torch
    #   torch        -[single→single]-> numpy        (torch  → numpy)
    #   numpy        -[single→batch]->  numpy_batch
    #   numpy_batch  -[batch→batch]->   jax_batch    (numpy  → jax)
    #   jax_batch    -[batch→single]->  jax
    #   jax          -[single→single]-> mlx           (jax   → mlx)
    #   mlx          -[single→batch]->  mlx_batch
    #   mlx_batch    -[batch→batch]->   torch_batch  (mlx → torch, closes ring)
    #
    # The first user override encountered breaks the chain.
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
            needs_grad: If ``False`` the saved state may be omitted.

        Returns:
            ``(primal, dual, saved_state)`` — torch tensors + state for backward.

        Default: loops :meth:`solve_torch` over the batch dimension.
        """
        import torch
        batch = q.shape[0]
        primals, duals, states = [], [], []
        for i in range(batch):
            p_i, d_i, state_i = self.solve_torch(
                P[i] if P is not None else None, q[i], A[i],
                dims, solver_args, needs_grad,
            )
            primals.append(p_i)
            duals.append(d_i)
            states.append(state_i)
        return torch.stack(primals, dim=0), torch.stack(duals, dim=0), states

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

        Default: converts to numpy, calls :meth:`solve_numpy`, converts back.
        """
        P_np = _to_numpy_from_torch(P) if P is not None else None
        q_np = _to_numpy_from_torch(q)
        A_np = _to_numpy_from_torch(A)
        primal_np, dual_np, state = self.solve_numpy(P_np, q_np, A_np, dims, solver_args, needs_grad)
        return (
            _to_torch_from_numpy(primal_np).to(dtype=q.dtype, device=q.device),
            _to_torch_from_numpy(dual_np).to(dtype=q.dtype, device=q.device),
            state,
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
            needs_grad: If ``False``, saved state may be omitted.

        Returns:
            ``(primal, dual, saved_state)`` — 1-D numpy arrays + state for backward.

        Default: adds a batch dim of 1, calls :meth:`solve_numpy_batch`,
        strips the batch dim.
        """
        P_b = P[np.newaxis] if P is not None else None
        primal_b, dual_b, state = self.solve_numpy_batch(
            P_b, q[np.newaxis], A[np.newaxis], dims, solver_args, needs_grad,
        )
        state_item = state[0] if isinstance(state, list) and len(state) == 1 else state
        return primal_b[0], dual_b[0], state_item

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

        ``P``: ``(B, nnz_P)`` or ``None``; ``q``: ``(B, n_vars+1)``; ``A``: ``(B, nnz_A)``.
        Returns ``(primal, dual, saved_state)`` as 2-D arrays.

        Default: converts to JAX, calls :meth:`solve_jax_batch`, converts back.
        """
        P_jax = _to_jax_from_numpy(P) if P is not None else None
        primal_jax, dual_jax, state = self.solve_jax_batch(
            P_jax, _to_jax_from_numpy(q), _to_jax_from_numpy(A),
            dims, solver_args, needs_grad,
        )
        return _to_numpy_from_jax(primal_jax), _to_numpy_from_jax(dual_jax), state

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
        Default: loops :meth:`solve_jax` over the batch dimension."""
        import jax.numpy as jnp
        batch = q.shape[0]
        primals, duals, states = [], [], []
        for i in range(batch):
            p_i, d_i, state_i = self.solve_jax(
                P[i] if P is not None else None, q[i], A[i],
                dims, solver_args, needs_grad,
            )
            primals.append(p_i)
            duals.append(d_i)
            states.append(state_i)
        return jnp.stack(primals, axis=0), jnp.stack(duals, axis=0), states

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
        Default: converts to MLX, calls :meth:`solve_mlx`, converts back."""
        P_mlx = _to_mlx_from_numpy(_to_numpy_from_jax(P)) if P is not None else None
        primal_mlx, dual_mlx, state = self.solve_mlx(
            P_mlx,
            _to_mlx_from_numpy(_to_numpy_from_jax(q)),
            _to_mlx_from_numpy(_to_numpy_from_jax(A)),
            dims, solver_args, needs_grad,
        )
        import jax.numpy as jnp
        return (
            jnp.array(_to_numpy_from_mlx(primal_mlx)),
            jnp.array(_to_numpy_from_mlx(dual_mlx)),
            state,
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
        Default: wraps :meth:`solve_mlx_batch` with a size-1 batch dim."""
        import mlx.core as mx
        P_b = mx.expand_dims(P, 0) if P is not None else None
        primal_b, dual_b, state = self.solve_mlx_batch(
            P_b, mx.expand_dims(q, 0), mx.expand_dims(A, 0),
            dims, solver_args, needs_grad,
        )
        state_item = state[0] if isinstance(state, list) and len(state) == 1 else state
        return primal_b[0], dual_b[0], state_item

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
        Default: converts to torch, calls :meth:`solve_torch_batch`. Closes the ring."""
        import mlx.core as mx
        P_t = _to_torch_from_numpy(_to_numpy_from_mlx(P)) if P is not None else None
        primal_t, dual_t, state = self.solve_torch_batch(
            P_t,
            _to_torch_from_numpy(_to_numpy_from_mlx(q)),
            _to_torch_from_numpy(_to_numpy_from_mlx(A)),
            dims, solver_args, needs_grad,
        )
        return (
            mx.array(_to_numpy_from_torch(primal_t)),
            mx.array(_to_numpy_from_torch(dual_t)),
            state,
        )

    # ------------------------------------------------------------------
    # Derivative methods — implement at least one
    # Same ring order and step types as the solve methods above.
    # ------------------------------------------------------------------

    def derivative_torch_batch(
        self,
        dprimal: Any,
        ddual: Any,
        saved_state: Any,
    ) -> tuple[Any | None, Any, Any]:
        """Backward pass for a **batch** of problems; tensors are torch.

        Args:
            dprimal: ``(B, n_primal)``
            ddual:   ``(B, n_dual)``
            saved_state: value returned by the corresponding ``solve_torch_batch``.

        Returns:
            ``(dP, dq, dA)`` each ``(B, n)``; ``dP`` may be ``None``.

        Default: loops :meth:`derivative_torch` over the batch dimension.
        """
        import torch
        batch = dprimal.shape[0]
        state_list = _split_state(saved_state, batch)
        dPs, dqs, dAs = [], [], []
        has_dP = False
        for i in range(batch):
            dP_i, dq_i, dA_i = self.derivative_torch(dprimal[i], ddual[i], state_list[i])
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
        saved_state: Any,
    ) -> tuple[Any | None, Any, Any]:
        """Backward pass for a **single** problem; tensors are torch.
        Default: converts to numpy, calls :meth:`derivative_numpy`, converts back."""
        dp_np = _to_numpy_from_torch(dprimal)
        dd_np = _to_numpy_from_torch(ddual)
        dP_np, dq_np, dA_np = self.derivative_numpy(dp_np, dd_np, saved_state)
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
        saved_state: Any,
    ) -> tuple[np.ndarray | None, np.ndarray, np.ndarray]:
        """Backward pass for a **single** problem; arrays are numpy.

        Args:
            dprimal: ``(n_primal,)``
            ddual:   ``(n_dual,)``
            saved_state: value returned by the corresponding ``solve_numpy``.

        Returns:
            ``(dP, dq, dA)`` — 1-D numpy arrays; ``dP`` may be ``None``.

        Default: adds a batch dim of 1, calls :meth:`derivative_numpy_batch`,
        strips the batch dim.
        """
        dP_b, dq_b, dA_b = self.derivative_numpy_batch(
            dprimal[np.newaxis], ddual[np.newaxis], [saved_state],
        )
        dP = dP_b[0] if dP_b is not None else None
        return dP, dq_b[0], dA_b[0]

    def derivative_numpy_batch(
        self,
        dprimal: np.ndarray,
        ddual: np.ndarray,
        saved_state: Any,
    ) -> tuple[np.ndarray | None, np.ndarray, np.ndarray]:
        """Backward pass for a **batch** of problems; arrays are numpy.

        ``dprimal``/``ddual``: ``(B, n)``. Returns ``(dP, dq, dA)`` each ``(B, n)``;
        ``dP`` may be ``None``.

        Default: converts to JAX, calls :meth:`derivative_jax_batch`, converts back.
        """
        dP_jax, dq_jax, dA_jax = self.derivative_jax_batch(
            _to_jax_from_numpy(dprimal),
            _to_jax_from_numpy(ddual),
            saved_state,
        )
        dP = _to_numpy_from_jax(dP_jax) if dP_jax is not None else None
        return dP, _to_numpy_from_jax(dq_jax), _to_numpy_from_jax(dA_jax)

    def derivative_jax_batch(
        self,
        dprimal: Any,
        ddual: Any,
        saved_state: Any,
    ) -> tuple[Any | None, Any, Any]:
        """Backward pass for a **batch** of problems; arrays are JAX arrays.
        Default: loops :meth:`derivative_jax` over the batch dimension."""
        import jax.numpy as jnp
        batch = dprimal.shape[0]
        state_list = _split_state(saved_state, batch)
        dPs, dqs, dAs = [], [], []
        has_dP = False
        for i in range(batch):
            dP_i, dq_i, dA_i = self.derivative_jax(dprimal[i], ddual[i], state_list[i])
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
        saved_state: Any,
    ) -> tuple[Any | None, Any, Any]:
        """Backward pass for a **single** problem; arrays are JAX arrays.
        Default: converts to MLX, calls :meth:`derivative_mlx`, converts back."""
        dp_mlx = _to_mlx_from_numpy(_to_numpy_from_jax(dprimal))
        dd_mlx = _to_mlx_from_numpy(_to_numpy_from_jax(ddual))
        dP_mlx, dq_mlx, dA_mlx = self.derivative_mlx(dp_mlx, dd_mlx, saved_state)
        import jax.numpy as jnp
        dP = jnp.array(_to_numpy_from_mlx(dP_mlx)) if dP_mlx is not None else None
        return dP, jnp.array(_to_numpy_from_mlx(dq_mlx)), jnp.array(_to_numpy_from_mlx(dA_mlx))

    def derivative_mlx(
        self,
        dprimal: Any,
        ddual: Any,
        saved_state: Any,
    ) -> tuple[Any | None, Any, Any]:
        """Backward pass for a **single** problem; arrays are MLX arrays.
        Default: wraps :meth:`derivative_mlx_batch` with a size-1 batch dim."""
        import mlx.core as mx
        dP_b, dq_b, dA_b = self.derivative_mlx_batch(
            mx.expand_dims(dprimal, 0),
            mx.expand_dims(ddual, 0),
            [saved_state],
        )
        dP = dP_b[0] if dP_b is not None else None
        return dP, dq_b[0], dA_b[0]

    def derivative_mlx_batch(
        self,
        dprimal: Any,
        ddual: Any,
        saved_state: Any,
    ) -> tuple[Any | None, Any, Any]:
        """Backward pass for a **batch** of problems; arrays are MLX arrays.
        Default: converts to torch, calls :meth:`derivative_torch_batch`. Closes the ring."""
        import mlx.core as mx
        dp_t = _to_torch_from_numpy(_to_numpy_from_mlx(dprimal))
        dd_t = _to_torch_from_numpy(_to_numpy_from_mlx(ddual))
        dP_t, dq_t, dA_t = self.derivative_torch_batch(dp_t, dd_t, saved_state)
        dP = mx.array(_to_numpy_from_torch(dP_t)) if dP_t is not None else None
        return dP, mx.array(_to_numpy_from_torch(dq_t)), mx.array(_to_numpy_from_torch(dA_t))
