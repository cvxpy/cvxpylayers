import cvxpy as cp
import numpy as np
from cvxpy.reductions.solvers.conic_solvers.scs_conif import dims_to_solver_dict

from cvxpylayers.utils import (
    BackwardContext,
    ForwardContext,
    backward_numpy,
    forward_numpy,
)

try:
    import torch
except ImportError:
    raise ImportError("Unable to import torch. Please install at "
                      "https://pytorch.org.")

torch_major_version = int(torch.__version__.split('.')[0])
if torch_major_version < 1:
    raise ImportError("cvxpylayers requires PyTorch >= 1.0; please "
                      "upgrade your installation of PyTorch, which is "
                      "version %s." % torch.__version__)


class CvxpyLayer(torch.nn.Module):
    """A differentiable convex optimization layer

    A CvxpyLayer solves a parametrized convex optimization problem given by a
    CVXPY problem. It solves the problem in its forward pass, and it computes
    the derivative of problem's solution map with respect to the parameters in
    its backward pass. The CVPXY problem must be a disciplined parametrized
    program.

    Example usage:
        ```
        import cvxpy as cp
        import torch
        from cvxpylayers.torch import CvxpyLayer

        n, m = 2, 3
        x = cp.Variable(n)
        A = cp.Parameter((m, n))
        b = cp.Parameter(m)
        constraints = [x >= 0]
        objective = cp.Minimize(0.5 * cp.pnorm(A @ x - b, p=1))
        problem = cp.Problem(objective, constraints)
        assert problem.is_dpp()

        cvxpylayer = CvxpyLayer(problem, parameters=[A, b], variables=[x])
        A_tch = torch.randn(m, n, requires_grad=True)
        b_tch = torch.randn(m, requires_grad=True)

        # solve the problem
        solution, = cvxpylayer(A_tch, b_tch)

        # compute the gradient of the sum of the solution with respect to A, b
        solution.sum().backward()
        ```
    """

    def __init__(self, problem, parameters, variables, gp=False, custom_method=None):
        """Construct a CvxpyLayer

        Args:
          problem: The CVXPY problem; must be DPP.
          parameters: A list of CVXPY Parameters in the problem; the order
                      of the Parameters determines the order in which parameter
                      values must be supplied in the forward pass. Must include
                      every parameter involved in problem.
          variables: A list of CVXPY Variables in the problem; the order of the
                     Variables determines the order of the optimal variable
                     values returned from the forward pass.
          gp: Whether to parse the problem using DGP (True or False).
          custom_method: A tuple of two custom methods for the forward and
                      backward pass.
        """
        super(CvxpyLayer, self).__init__()
        
        if custom_method is None:
            self._forward_numpy, self._backward_numpy = forward_numpy, backward_numpy
        else:
            self._forward_numpy, self._backward_numpy = custom_method

        self.gp = gp
        if self.gp:
            if not problem.is_dgp(dpp=True):
                raise ValueError('Problem must be DPP.')
        else:
            if not problem.is_dcp(dpp=True):
                raise ValueError('Problem must be DPP.')

        if not set(problem.parameters()) == set(parameters):
            raise ValueError("The layer's parameters must exactly match "
                             "problem.parameters")
        if not set(variables).issubset(set(problem.variables())):
            raise ValueError("Argument variables must be a subset of "
                             "problem.variables")
        if not isinstance(parameters, list) and \
           not isinstance(parameters, tuple):
            raise ValueError("The layer's parameters must be provided as "
                             "a list or tuple")
        if not isinstance(variables, list) and \
           not isinstance(variables, tuple):
            raise ValueError("The layer's variables must be provided as "
                             "a list or tuple")

        self.param_order = parameters
        self.variables = variables
        self.var_dict = {v.id for v in self.variables}

        # Construct compiler
        self.dgp2dcp = None

        if self.gp:
            for param in parameters:
                if param.value is None:
                    raise ValueError("An initial value for each parameter is "
                                     "required when gp=True.")
            data, solving_chain, _ = problem.get_problem_data(
                solver=cp.SCS, gp=True, solver_opts={'use_quad_obj': False})
            self.compiler = data[cp.settings.PARAM_PROB]
            self.dgp2dcp = solving_chain.get(cp.reductions.Dgp2Dcp)
            self.param_ids = [p.id for p in self.compiler.parameters]
        else:
            data, _, _ = problem.get_problem_data(
                solver=cp.SCS, solver_opts={'use_quad_obj': False})
            self.compiler = data[cp.settings.PARAM_PROB]
            self.param_ids = [p.id for p in self.param_order]
        self.cone_dims = dims_to_solver_dict(data["dims"])

    def forward(self, *params, solver_args={}):
        """Solve problem (or a batch of problems) corresponding to `params`

        Args:
          params: a sequence of torch Tensors; the n-th Tensor specifies
                  the value for the n-th CVXPY Parameter. These Tensors
                  can be batched: if a Tensor has 3 dimensions, then its
                  first dimension is interpreted as the batch size. These
                  Tensors must all have the same dtype and device.
          solver_args: a dict of optional arguments, to send to `diffcp`. Keys
                       should be the names of keyword arguments.

        Returns:
          a list of optimal variable values, one for each CVXPY Variable
          supplied to the constructor.
        """
        if len(params) != len(self.param_ids):
            raise ValueError('A tensor must be provided for each CVXPY '
                             'parameter; received %d tensors, expected %d' % (
                                 len(params), len(self.param_ids)))
        info = {}
        f = _CvxpyLayerFn(
            _forward_numpy=self._forward_numpy,
            _backward_numpy=self._backward_numpy,
            param_order=self.param_order,
            param_ids=self.param_ids,
            variables=self.variables,
            var_dict=self.var_dict,
            compiler=self.compiler,
            cone_dims=self.cone_dims,
            gp=self.gp,
            dgp2dcp=self.dgp2dcp,
            solver_args=solver_args,
            info=info,
        )
        sol = f(*params)
        self.info = info
        return sol


def to_numpy(x):
    # convert torch tensor to numpy array
    return x.cpu().detach().double().numpy()


def to_torch(x, dtype, device):
    # convert numpy array or tensor to torch tensor
    if isinstance(x, torch.Tensor):
        return x.type(dtype).to(device)
    else:
        return torch.from_numpy(x).type(dtype).to(device)


def _CvxpyLayerFn(
        _forward_numpy,
        _backward_numpy,
        param_order,
        param_ids,
        variables,
        var_dict,
        compiler,
        cone_dims,
        gp,
        dgp2dcp,
        solver_args,
        info):
    class _CvxpyLayerFnFn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, *params):
            # infer dtype, device, and whether or not params are batched
            ctx.dtype = params[0].dtype
            ctx.device = params[0].device

            ctx.batch_sizes = []
            for i, (p, q) in enumerate(zip(params, param_order)):
                # check dtype, device of params
                if p.dtype != ctx.dtype:
                    raise ValueError(
                        "Two or more parameters have different dtypes. "
                        "Expected parameter %d to have dtype %s but "
                        "got dtype %s." %
                        (i, str(ctx.dtype), str(p.dtype))
                    )
                if p.device != ctx.device:
                    raise ValueError(
                        "Two or more parameters are on different devices. "
                        "Expected parameter %d to be on device %s "
                        "but got device %s." %
                        (i, str(ctx.device), str(p.device))
                    )

                # check and extract the batch size for the parameter
                # 0 means there is no batch dimension for this parameter
                # and we assume the batch dimension is non-zero
                if p.ndimension() == q.ndim:
                    batch_size = 0
                elif p.ndimension() == q.ndim + 1:
                    batch_size = p.size(0)
                    if batch_size == 0:
                        raise ValueError(
                            "The batch dimension for parameter {} is zero "
                            "but should be non-zero.".format(i))
                else:
                    raise ValueError(
                        "Invalid parameter size passed in. Expected "
                        "parameter {} to have have {} or {} dimensions "
                        "but got {} dimensions".format(
                            i, q.ndim, q.ndim + 1, p.ndimension()))

                ctx.batch_sizes.append(batch_size)

                # validate the parameter shape
                p_shape = p.shape if batch_size == 0 else p.shape[1:]
                if not np.all(p_shape == param_order[i].shape):
                    raise ValueError(
                        "Inconsistent parameter shapes passed in. "
                        "Expected parameter {} to have non-batched shape of "
                        "{} but got {}.".format(
                                i,
                                q.shape,
                                p.shape))

            ctx.batch_sizes = np.array(ctx.batch_sizes)
            ctx.batch = np.any(ctx.batch_sizes > 0)

            if ctx.batch:
                nonzero_batch_sizes = ctx.batch_sizes[ctx.batch_sizes > 0]
                ctx.batch_size = nonzero_batch_sizes[0]
                if np.any(nonzero_batch_sizes != ctx.batch_size):
                    raise ValueError(
                        "Inconsistent batch sizes passed in. Expected "
                        "parameters to have no batch size or all the same "
                        "batch size but got sizes: {}.".format(
                            ctx.batch_sizes))
            else:
                ctx.batch_size = 1

            if gp:
                ctx.params = params
                ctx.old_params_to_new_params = (
                    dgp2dcp.canon_methods._parameters
                )
            
            # convert to numpy arrays
            params_numpy = [to_numpy(p) for p in params]
            
            context = ForwardContext(
                gp=gp,
                solve_and_derivative=any(p.requires_grad for p in params),
                batch=ctx.batch,
                batch_size=ctx.batch_size,
                batch_sizes=ctx.batch_sizes,
                compiler=compiler,
                param_ids=param_ids,
                param_order=param_order,
                old_params_to_new_params=ctx.old_params_to_new_params if gp else None,
                cone_dims=cone_dims,
                solver_args=solver_args,
                variables=variables,
                var_dict=var_dict,
            )
            
            sol, info_forward = _forward_numpy(params_numpy, context)
            
            # convert to torch tensors and incorporate info_forward
            sol = [to_torch(s, ctx.dtype, ctx.device) for s in sol]
            info.update(info_forward)

            return tuple(sol)

        @staticmethod
        def backward(ctx, *dvars):
            
            # convert to numpy arrays
            dvars_numpy = [to_numpy(dvar) for dvar in dvars]
                
            context = BackwardContext(
                info=info,
                gp=gp,
                batch=ctx.batch,
                batch_size=ctx.batch_size,
                batch_sizes=ctx.batch_sizes,
                variables=variables,
                compiler=compiler,
                param_ids=param_ids,
                param_order=param_order if gp else None,
                params=ctx.params if gp else None,
                old_params_to_new_params=ctx.old_params_to_new_params if gp else None,
                sol=info['sol'] if gp else None,
            )

            grad_numpy, info_backward = _backward_numpy(dvars_numpy, context)
            
            # convert to torch tensors and incorporate info_backward
            grad = [to_torch(g, ctx.dtype, ctx.device) for g in grad_numpy]
            info.update(info_backward)

            return tuple(grad)

    return _CvxpyLayerFnFn.apply
