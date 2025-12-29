"""TensorFlow implementation of differentiable convex optimization layers."""

from typing import Any, cast

import cvxpy as cp
import numpy as np
import scipy.sparse
import tensorflow as tf

import cvxpylayers.utils.parse_args as pa


def _reshape_fortran(x: tf.Tensor, shape: tuple) -> tf.Tensor:
    """Reshape tensor using Fortran (column-major) order.

    TensorFlow does not support order='F' in reshape, so we implement
    it via transpose-reshape-transpose.

    Args:
        x: Input tensor to reshape
        shape: Target shape

    Returns:
        Reshaped tensor with Fortran ordering
    """
    if len(x.shape) > 0:
        # Reverse dimensions
        x = tf.transpose(x, perm=list(reversed(range(len(x.shape)))))
    # Reshape with reversed shape, then reverse back
    reshaped = tf.reshape(x, list(reversed(shape)))
    if len(shape) > 0:
        reshaped = tf.transpose(reshaped, perm=list(reversed(range(len(shape)))))
    return reshaped


def _apply_gp_log_transform(
    params: tuple[tf.Tensor, ...],
    ctx: pa.LayersContext,
) -> tuple[tf.Tensor, ...]:
    """Apply log transformation to geometric program (GP) parameters.

    Geometric programs are solved in log-space after conversion to DCP.
    This function applies log transformation to the appropriate parameters.

    Args:
        params: Tuple of parameter tensors in original GP space
        ctx: Layer context containing GP parameter mapping info

    Returns:
        Tuple of transformed parameters (log-space for GP params, unchanged otherwise)
    """
    if not ctx.gp or not ctx.gp_param_to_log_param:
        return params

    params_transformed = []
    for i, param in enumerate(params):
        cvxpy_param = ctx.parameters[i]
        if cvxpy_param in ctx.gp_param_to_log_param:
            # This parameter needs log transformation for GP
            params_transformed.append(tf.math.log(param))
        else:
            params_transformed.append(param)
    return tuple(params_transformed)


def _flatten_and_batch_params(
    params: tuple[tf.Tensor, ...],
    ctx: pa.LayersContext,
    batch: tuple,
) -> tf.Tensor:
    """Flatten and batch parameters into a single stacked tensor.

    Converts a tuple of parameter tensors (potentially with mixed batched/unbatched)
    into a single concatenated tensor suitable for matrix multiplication with the
    parametrized problem matrices.

    Args:
        params: Tuple of parameter tensors
        ctx: Layer context with batch info and ordering
        batch: Batch dimensions tuple (empty if unbatched)

    Returns:
        Concatenated parameter tensor with shape (num_params, batch_size) or (num_params,)
    """
    flattened_params: list[tf.Tensor | None] = [None] * (len(params) + 1)

    for i, param in enumerate(params):
        # Check if this parameter is batched or needs broadcasting
        if ctx.batch_sizes[i] == 0 and batch:
            # Unbatched parameter - expand to match batch size
            param_expanded = tf.expand_dims(param, 0)
            param_expanded = tf.broadcast_to(param_expanded, batch + tuple(param.shape))
            flattened_params[ctx.user_order_to_col_order[i]] = _reshape_fortran(
                param_expanded,
                batch + (-1,),
            )
        else:
            # Already batched or no batch dimension needed
            flattened_params[ctx.user_order_to_col_order[i]] = _reshape_fortran(
                param,
                batch + (-1,),
            )

    # Add constant 1.0 column for offset terms in canonical form
    flattened_params[-1] = tf.ones(batch + (1,), dtype=params[0].dtype)
    assert all(p is not None for p in flattened_params), "All parameters must be assigned"

    p_stack = tf.concat(cast(list[tf.Tensor], flattened_params), axis=-1)
    # When batched, p_stack is (batch_size, num_params) but we need (num_params, batch_size)
    if batch:
        p_stack = tf.transpose(p_stack)
    return p_stack


def _recover_results(
    primal: tf.Tensor,
    dual: tf.Tensor,
    ctx: pa.LayersContext,
    batch: tuple,
) -> tuple[tf.Tensor, ...]:
    """Recover variable values from primal/dual solutions.

    Extracts the requested variables from the solver's primal and dual
    solutions, applies inverse GP transformation if needed, and removes
    batch dimension for unbatched inputs.

    Args:
        primal: Primal solution from solver
        dual: Dual solution from solver
        ctx: Layer context with variable recovery info
        batch: Batch dimensions tuple (empty if unbatched)

    Returns:
        Tuple of recovered variable tensors
    """

    def tf_reshape_fn(x, shape):
        """Reshape using Fortran order for TensorFlow."""
        return _reshape_fortran(x, shape)

    # Extract each variable using its slice and reshape
    results = tuple(var.recover(primal, dual, reshape_fn=tf_reshape_fn) for var in ctx.var_recover)

    # Apply exp transformation to recover from log-space for GP
    if ctx.gp:
        results = tuple(tf.exp(r) for r in results)

    # Squeeze batch dimension for unbatched inputs
    if not batch:
        results = tuple(tf.squeeze(r, axis=0) for r in results)

    return results


def scipy_csr_to_tf_sparse(
    scipy_csr: scipy.sparse.csr_array | None,
) -> tf.SparseTensor | None:
    """Convert scipy CSR sparse matrix to TensorFlow SparseTensor.

    Args:
        scipy_csr: Scipy CSR sparse array or None

    Returns:
        TensorFlow SparseTensor or None if input was None
    """
    if scipy_csr is None:
        return None
    scipy_csr = cast(scipy.sparse.csr_array, scipy_csr)

    # Convert to COO format for TensorFlow
    coo = scipy_csr.tocoo()
    indices = np.column_stack((coo.row, coo.col)).astype(np.int64)
    values = coo.data.astype(np.float64)
    shape = coo.shape

    sparse_tensor = tf.SparseTensor(
        indices=indices,
        values=values,
        dense_shape=shape,
    )

    # Reorder to canonical form (required by some TF sparse ops)
    return tf.sparse.reorder(sparse_tensor)


class CvxpyLayer:
    """A differentiable convex optimization layer for TensorFlow.

    This layer wraps a parametrized CVXPY problem, solving it in the forward pass
    and computing gradients via implicit differentiation. Compatible with
    ``tf.GradientTape``.

    Example:
        >>> import cvxpy as cp
        >>> import tensorflow as tf
        >>> from cvxpylayers.tensorflow import CvxpyLayer
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
        >>> # Solve and compute gradients
        >>> A_tf = tf.Variable(tf.random.normal((3, 2), dtype=tf.float64))
        >>> b_tf = tf.Variable(tf.random.normal((3,), dtype=tf.float64))
        >>> with tf.GradientTape() as tape:
        ...     (solution,) = layer(A_tf, b_tf)
        ...     loss = tf.reduce_sum(solution)
        >>> grads = tape.gradient(loss, [A_tf, b_tf])
    """

    def __init__(
        self,
        problem: cp.Problem,
        parameters: list[cp.Parameter],
        variables: list[cp.Variable],
        solver: str | None = None,
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
                at runtime. Order must match the order of tensors passed to __call__().
            variables: List of CVXPY Variables whose optimal values will be returned
                by __call__(). Order determines the order of returned tensors.
            solver: CVXPY solver to use (e.g., ``cp.CLARABEL``, ``cp.SCS``).
                If None, uses the default diffcp solver.
            gp: If True, problem is a geometric program. Parameters will be
                log-transformed before solving.
            verbose: If True, print solver output.
            canon_backend: Backend for canonicalization. Options are 'diffcp',
                'cuclarabel', or None (auto-select).
            solver_args: Default keyword arguments passed to the solver.
                Can be overridden per-call in __call__().

        Raises:
            AssertionError: If problem is not DPP-compliant.
            ValueError: If parameters or variables are not part of the problem.
        """
        if solver_args is None:
            solver_args = {}
        self.ctx = pa.parse_args(
            problem,
            variables,
            parameters,
            solver,
            gp=gp,
            verbose=verbose,
            canon_backend=canon_backend,
            solver_args=solver_args,
        )
        if self.ctx.reduced_P.reduced_mat is not None:  # type: ignore[attr-defined]
            self.P = scipy_csr_to_tf_sparse(self.ctx.reduced_P.reduced_mat)  # type: ignore[attr-defined]
        else:
            self.P = None
        self.q: tf.SparseTensor = scipy_csr_to_tf_sparse(self.ctx.q.tocsr())  # type: ignore[assignment]
        self.A: tf.SparseTensor = scipy_csr_to_tf_sparse(self.ctx.reduced_A.reduced_mat)  # type: ignore[attr-defined,assignment]

    def __call__(
        self, *params: tf.Tensor, solver_args: dict[str, Any] | None = None
    ) -> tuple[tf.Tensor, ...]:
        """Solve the optimization problem and return optimal variable values.

        Args:
            *params: Tensor values for each CVXPY Parameter, in the same order
                as the ``parameters`` argument to __init__(). Each tensor shape must
                match the corresponding Parameter shape, optionally with a batch
                dimension prepended. Batched and unbatched parameters can be mixed;
                unbatched parameters are broadcast.
            solver_args: Keyword arguments passed to the solver, overriding any
                defaults set in __init__().

        Returns:
            Tuple of tensors containing optimal values for each CVXPY Variable
            specified in the ``variables`` argument to __init__(). If inputs are
            batched, outputs will have matching batch dimensions.

        Raises:
            RuntimeError: If the solver fails to find a solution.

        Example:
            >>> # Single problem
            >>> (x_opt,) = layer(A_tensor, b_tensor)
            >>>
            >>> # Batched: solve 10 problems in parallel
            >>> A_batch = tf.random.normal((10, 3, 2), dtype=tf.float64)
            >>> b_batch = tf.random.normal((10, 3), dtype=tf.float64)
            >>> (x_batch,) = layer(A_batch, b_batch)  # x_batch.shape = (10, 2)
        """
        if solver_args is None:
            solver_args = {}
        batch = self.ctx.validate_params(list(params))

        # Apply log transformation to GP parameters
        params = _apply_gp_log_transform(params, self.ctx)

        # Flatten and batch parameters
        p_stack = _flatten_and_batch_params(params, self.ctx, batch)

        # Get the dtype from input parameters
        param_dtype = params[0].dtype

        # TensorFlow sparse_dense_matmul requires 2D tensors, so expand if unbatched
        p_stack_2d = p_stack if batch else tf.expand_dims(p_stack, axis=1)

        # Evaluate parametrized matrices using sparse-dense matrix multiplication
        if self.P is not None:
            P_sparse = tf.cast(self.P, param_dtype)
            P_eval = tf.sparse.sparse_dense_matmul(P_sparse, p_stack_2d)
        else:
            P_eval = None

        q_sparse = tf.cast(self.q, param_dtype)
        q_eval = tf.sparse.sparse_dense_matmul(q_sparse, p_stack_2d)

        A_sparse = tf.cast(self.A, param_dtype)
        A_eval = tf.sparse.sparse_dense_matmul(A_sparse, p_stack_2d)

        # Store data and adjoint in closure for backward pass
        # This avoids TensorFlow trying to trace through DIFFCP's Python-based solver
        data_container: dict[str, Any] = {}

        @tf.custom_gradient
        def solve_problem(q_eval_inner, A_eval_inner):
            # Forward pass: solve the optimization problem
            data = self.ctx.solver_ctx.tf_to_data(P_eval, q_eval_inner, A_eval_inner)  # type: ignore[attr-defined]
            primal, dual, adj_batch = data.tf_solve(solver_args)  # type: ignore[attr-defined]
            # Store for backward pass (outside TF tracing)
            data_container["data"] = data
            data_container["adj_batch"] = adj_batch

            def grad(dprimal, ddual):
                # Backward pass using adjoint method
                data = data_container["data"]
                adj_batch = data_container["adj_batch"]
                _, dq, dA = data.tf_derivative(dprimal, ddual, adj_batch)
                return dq, dA

            return (primal, dual), grad

        primal, dual = solve_problem(q_eval, A_eval)

        # Recover results and apply GP inverse transform if needed
        return _recover_results(primal, dual, self.ctx, batch)
