import torch
import cvxpylayers.utils.parse_args as pa


class GpuCvxpyLayer(torch.nn.Module):
    def __init__(
        self, problem, parameters, variables, solver=None, gp=False, solver_args={}
    ):
        super().__init__()
        assert gp is False
        self.ctx = pa.parse_args(problem, variables, parameters, solver, solver_args)
        if self.ctx.reduced_P.reduced_mat is not None:
            self.P = torch.nn.Buffer(
                scipy_csr_to_torch_csr(self.ctx.reduced_P.reduced_mat)
            )
        else:
            self.P = None
        self.q = torch.nn.Buffer(scipy_csr_to_torch_csr(self.ctx.q.tocsr()))
        self.A = torch.nn.Buffer(scipy_csr_to_torch_csr(self.ctx.reduced_A.reduced_mat))

    def forward(self, *params, solver_args={}):
        batch = self.ctx.validate_params(params)
        flattened_params = (len(params) + 1) * [None]
        for i, param in enumerate(params):
            flattened_params[self.ctx.user_order_to_col_order[i]] = reshape_fortran(
                param, batch + (-1,)
            )
        flattened_params[-1] = torch.ones(
            batch + (1,), dtype=params[0].dtype, device=params[0].device
        )
        p_stack = torch.cat(flattened_params, -1)
        P_eval = self.P @ p_stack if self.P is not None else None
        q_eval = self.q @ p_stack
        A_eval = self.A @ p_stack
        primal, dual, _, _ = _CvxpyLayer.apply(P_eval, q_eval, A_eval, self.ctx)
        return tuple(var.recover(primal, dual) for var in self.ctx.var_recover)


class _CvxpyLayer(torch.autograd.Function):
    @staticmethod
    def forward(P_eval, q_eval, A_eval, cl_ctx):
        data = cl_ctx.solver_ctx.torch_to_data(P_eval, q_eval, A_eval)
        return *data.torch_solve(), data

    @staticmethod
    def setup_context(ctx, inputs, outputs):
        primal, dual, backwards, data = outputs
        ctx.backwards = backwards
        ctx.data = data

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, primal, dual, backwards, data):
        (
            dP,
            dq,
            dA,
        ) = ctx.data.torch_derivative(primal, dual, ctx.backwards)
        return dP, dq, dA, None


def reshape_fortran(x, shape):
    if len(x.shape) > 0:
        x = x.permute(*reversed(range(len(x.shape))))
    return x.reshape(*reversed(shape)).permute(*reversed(range(len(shape))))


def scipy_csr_to_torch_csr(scipy_csr):
    if scipy_csr is None:
        return None
    # Get the CSR format components
    values = scipy_csr.data
    col_indices = scipy_csr.indices
    row_ptr = scipy_csr.indptr
    num_rows, num_cols = scipy_csr.shape

    # Create the torch sparse csr_tensor
    torch_csr = torch.sparse_csr_tensor(
        crow_indices=torch.tensor(row_ptr, dtype=torch.int64),
        col_indices=torch.tensor(col_indices, dtype=torch.int64),
        values=torch.tensor(values, dtype=torch.float64),
        size=(num_rows, num_cols),
    )

    return torch_csr


CvxpyLayer = GpuCvxpyLayer
