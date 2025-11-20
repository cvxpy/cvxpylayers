"""
CVXPy Layer for Apple MLX Framework
A differentiable convex optimization layer that integrates CVXPY with MLX's
automatic diff system
"""


import cvxpy as cp  # type: ignore
import numpy as np  # type: ignore
from cvxpy.reductions.solvers.conic_solvers.scs_conif import \
    dims_to_solver_dict  # type: ignore

from cvxpylayers.utils import (BackwardContext, ForwardContext, backward_numpy,
                               forward_numpy)

try:
    import mlx.core as mx  # type: ignore

except ImportError:
    raise ImportError("Unable to import mlx. Please install MLX framework.")


def to_numpy(x):
    """Convert MLX array to numpy array"""
    return np.array(x, dtype=np.float64)


def to_mlx(x, dtype=mx.float64):
    """Convert numpy array to MLX array"""
    return mx.array(x, dtype=dtype)


def _validate_params(params, param_order):
    """Validate parameter shapes and extract batch information

    Args:
        params: Sequence of MLX arrays representing CVXPY parameters
        param_order: List of CVXPY Parameter objects in expected order

    Returns:
        tuple: (dtype, batch_sizes, batch, batch_size)
    """
    batch_sizes = []
    dtype = params[0].dtype

    for i, (p, q) in enumerate(zip(params, param_order)):
        # Check dtype consistency
        if p.dtype != dtype:
            raise ValueError(f"Parameter {i} has dtype {p.dtype}"
                             "but expected {dtype}")

        # Check and extract batch dimension
        if p.ndim == q.ndim:
            batch_size = 0  # No batch dimension
        elif p.ndim == q.ndim + 1:
            batch_size = p.shape[0]
            if batch_size == 0:
                raise ValueError(
                    f"Batch dimension for parameter {i} is zero but should be"
                    "non-zero"
                )
        else:
            raise ValueError(
                f"Parameter {i} has {p.ndim} dimensions,"
                "expected {q.ndim} or {q.ndim + 1}"
            )

        batch_sizes.append(batch_size)

        # Validate parameter shape (excluding batch dimension)
        p_shape = p.shape if batch_size == 0 else p.shape[1:]
        if not np.all(np.array(p_shape) == np.array(param_order[i].shape)):
            raise ValueError(
                f"Parameter {i} has shape {p_shape},"
                "expected {param_order[i].shape}"
            )

    batch_sizes = np.array(batch_sizes)
    batch = np.any(batch_sizes > 0)

    if batch:
        nonzero_batch_sizes = batch_sizes[batch_sizes > 0]
        batch_size = nonzero_batch_sizes[0]
        if np.any(nonzero_batch_sizes != batch_size):
            raise ValueError(f"Inconsistent batch sizes: {batch_sizes}")
    else:
        batch_size = 1

    return dtype, batch_sizes, batch, batch_size


def _cvxpy_layer_fn(
    params,
    forward_numpy,
    backward_numpy,
    param_order,
    param_ids,
    variables,
    var_dict,
    compiler,
    cone_dims,
    gp,
    dgp2dcp,
    solver_args,
):
    """Core CVXPY layer function with
    custom Vector Jacobian Product (vjp)

    This function creates a differentiable optimization layer using MLX's
    custom_function decorator to define custom forward and backward passes.
    """

    # Storage for passing info from forward to backward pass
    _info_storage = {}

    @mx.custom_function
    def cvxpy_solve(*params):
        """Forward pass: solve the optimization problem"""
        # Validate inputs and extract batch information
        dtype, batch_sizes, batch, batch_size = _validate_params(
                                                params, param_order)

        old_params_to_new_params = None
        if gp:
            old_params_to_new_params = dgp2dcp.canon_methods._parameters

        # Convert MLX arrays to numpy for solver
        params_numpy = [to_numpy(p) for p in params]

        context = ForwardContext(
            gp=gp,
            solve_and_derivative=True,  # Always compute derivatives for MLX
            batch=batch,  # type: ignore
            batch_size=batch_size,
            batch_sizes=batch_sizes,  # type: ignore
            compiler=compiler,
            param_ids=param_ids,
            param_order=param_order,
            old_params_to_new_params=old_params_to_new_params,  # type: ignore
            cone_dims=cone_dims,
            solver_args=solver_args,
            variables=variables,
            var_dict=var_dict,
        )

        # Actual Solver
        sol, info = forward_numpy(params_numpy, context)

        # Store metadata for backward pass (MLX custom_function
        # can only return arrays)
        _info_storage["info"] = info
        _info_storage["dtype"] = dtype
        _info_storage["batch_sizes"] = batch_sizes
        _info_storage["batch"] = batch
        _info_storage["batch_size"] = batch_size
        _info_storage["old_params_to_new_params"] = old_params_to_new_params

        # Convert solution back to MLX arrays
        sol_mlx = [to_mlx(s, dtype) for s in sol]

        # Return as tuple of arrays (required by MLX)
        return tuple(sol_mlx)

    # The vector-Jacobian product for mlx custom function. similar to
    # jax.vjp[1] for backprop
    @cvxpy_solve.vjp
    def cvxpy_solve_vjp(primals, cotangents, outputs):
        """Backward pass: compute parameter gradients"""
        # Retrieve stored metadata from forward pass
        info = _info_storage["info"]
        dtype = _info_storage["dtype"]
        batch_sizes = _info_storage["batch_sizes"]
        batch = _info_storage["batch"]
        batch_size = _info_storage["batch_size"]
        old_params_to_new_params = _info_storage["old_params_to_new_params"]

        # Convert cotangents (gradients w.r.t. outputs) to proper format
        if isinstance(cotangents, tuple):
            dvars = list(cotangents)
        elif isinstance(cotangents, list):
            dvars = cotangents
        else:
            dvars = [cotangents]

        # Convert gradients to numpy for cvxpylayers backward pass
        dvars_numpy = [to_numpy(dvar) for dvar in dvars]

        # Create backward context for cvxpylayers
        backward_context = BackwardContext(
            info=info,
            gp=gp,
            batch=batch,
            batch_size=batch_size,
            batch_sizes=batch_sizes,
            variables=variables,
            compiler=compiler,
            param_ids=param_ids,
            param_order=param_order if gp else None,  # type: ignore
            params=primals if gp else None,  # type: ignore
            old_params_to_new_params=old_params_to_new_params,
            sol=info["sol"] if gp else None,  # type: ignore
        )

        # Compute parameter gradients
        grad_numpy, _ = backward_numpy(dvars_numpy, backward_context)

        # Convert gradients back to MLX arrays
        grad_mlx = [to_mlx(g, dtype) for g in grad_numpy]

        return tuple(grad_mlx)

    # Execute the custom function
    result = cvxpy_solve(*params)

    # Return appropriate format based on number of variables
    if len(variables) == 1:
        return result[0]  # type: ignore

    else:
        return list(result)  # type: ignore


class CvxpyLayer:
    """A differentiable convex optimization layer for MLX

    A CvxpyLayer solves a parametrized convex optimization problem given by a
    CVXPY problem. It solves the problem in its forward pass, and computes
    the derivative of problem's solution map with respect to the parameters in
    its backward pass using MLX's automatic differentiation.

    The CVXPY problem must be a disciplined parametrized program (DPP).
    """

    def __init__(self, problem, parameters, variables, gp=False,
                 custom_method=None):
        """Construct a CvxpyLayer

        Args:
            problem: The CVXPY problem; must be DPP.
            parameters: A list of CVXPY Parameters in the problem; the order
                       of the Parameters determines the order in which the
                       parameter values must be supplied in the forward pass.
                       Must include every parameter involved in problem.
            variables: A list of CVXPY Variables in the problem; the order of
                       values returned from the forward pass.
            gp: Whether to parse the problem using DGP (True or False).
            custom_method: A tuple of two custom methods for the forward and
                         backward pass. If None, uses default cvxpylayers
                         methods.

        Raises:
            ValueError: If the problem is not DPP, parameters don't match, or
                       variables are not a subset of problem variables.
        """
        if custom_method is None:
            self._forward_numpy, self._backward_numpy = forward_numpy, \
                                                        backward_numpy
        else:
            self._forward_numpy, self._backward_numpy = custom_method

        self.gp = gp

        # Validate problem type
        if self.gp:
            if not problem.is_dgp(dpp=True):
                raise ValueError("Problem must be DGP")
        else:
            if not problem.is_dcp(dpp=True):
                raise ValueError("Problem must be DCP")

        # Validate parameters and variables
        if not set(problem.parameters()) == set(parameters):
            raise ValueError(
                "The layer's parameters must exactly match "
                "problem.parameters"
            )
        if not set(variables).issubset(set(problem.variables())):
            raise ValueError(
                "Argument variables must be a subset of problem.variables"
            )
        if not isinstance(parameters, (list, tuple)):
            raise ValueError(
                "The layer's parameters must be provided as a list or tuple"
            )
        if not isinstance(variables, (list, tuple)):
            raise ValueError(
                "The layer's variables must be provided as a list or tuple"
            )

        self.param_order = parameters
        self.variables = variables
        self.var_dict = {v.id for v in self.variables}
        self.dgp2dcp = None

        # Compile the problem
        if self.gp:
            # Geometric programming requires initial parameter values
            for param in parameters:
                if param.value is None:
                    raise ValueError(
                        "An initial value for each parameter is required"
                        "when gp=True."
                    )
            data, solving_chain, _ = problem.get_problem_data(
                solver=cp.SCS, gp=True, solver_opts={"use_quad_obj": False}
            )
            self.compiler = data[cp.settings.PARAM_PROB]
            self.dgp2dcp = solving_chain.get(cp.reductions.Dgp2Dcp)
            self.param_ids = [p.id for p in self.compiler.parameters]
        else:
            # Standard convex programming
            data, _, _ = problem.get_problem_data(
                solver=cp.SCS, solver_opts={"use_quad_obj": False}
            )
            self.compiler = data[cp.settings.PARAM_PROB]
            self.param_ids = [p.id for p in self.param_order]

        self.cone_dims = dims_to_solver_dict(data["dims"])

    def __call__(self, *params, solver_args={}):
        """Solve problem (or batch of problems) corresponding to `params`

        Args:
            *params: Sequence of MLX arrays; the n-th array specifies
                    the value for the n-th CVXPY Parameter. These arrays
                    can be batched: if an array has one more dimension than
                    expected, the first dimension is interpreted as the
                    batch size.
            solver_args: Dict of optional arguments to send to `diffcp`.

        Returns:
            List of optimal variable values, one for each CVXPY Variable
            supplied to the constructor. If there's only one variable,
            returns the single array directly.

        Raises:
            ValueError: If the number of input parameters doesn't match
                       the number of CVXPY parameters.
        """
        if len(params) != len(self.param_ids):
            raise ValueError(
                f"Expected {len(self.param_ids)} parameters, got {len(params)}"
            )

        return _cvxpy_layer_fn(
            params=params,
            forward_numpy=self._forward_numpy,
            backward_numpy=self._backward_numpy,
            param_order=self.param_order,
            param_ids=self.param_ids,
            variables=self.variables,
            var_dict=self.var_dict,
            compiler=self.compiler,
            cone_dims=self.cone_dims,
            gp=self.gp,
            dgp2dcp=self.dgp2dcp,
            solver_args=solver_args,
        )

    def forward(self, *params, **kwargs):
        """Alias for __call__ ,can be safely removed. just pytorch style
        programming """
        return self.__call__(*params, **kwargs)


def make_cvxpy_layer(problem, parameters, variables,
                     gp=False, custom_method=None):
    """Factory function to create a CvxpyLayer

    This is a convenience function that can be used
    in functional-style MLX code.

    Args:
        problem: The CVXPY problem; must be DPP.
        parameters: A list of CVXPY Parameters in the problem.
        variables: A list of CVXPY Variables in the problem.
        gp: Whether to parse the problem using DGP (True or False).
        custom_method: A tuple of two custom methods for forward
        and backward pass.

    Returns:
        A function that can be called with MLX arrays as arguments.
    """
    layer = CvxpyLayer(problem, parameters, variables, gp, custom_method)
    return layer
