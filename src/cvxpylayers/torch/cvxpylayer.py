import torch
import cvxpylayers.utils.parse_args as pa


class GpuCvxpyLayer(torch.nn.Module):
    def __init__(self, problem, parameters, variables, solver=None, gp=False, solver_args={}):
        super().__init__()
        assert gp is False
        self.ctx = pa.parse_args(problem, variables, parameters, solver, solver_args)
        self.P = torch.nn.Buffer(scipy_csr_to_torch_csr(self.ctx.reduced_P.reduced_mat))
        self.q = torch.nn.Buffer(scipy_csr_to_torch_csr(self.ctx.q))
        self.A = torch.nn.Buffer(scipy_csr_to_torch_csr(self.ctx.reduced_A.reduced_mat))


    def forward(self, *params):
        batch = self.ctx.validate_params(params)
        flattened_params = (len(params)+1) * [None]
        for i, param in enumerate(params):
            p = torch.Tensor()
            p.set_(
                    param.untyped_storage(),
                    param.storage_offset(),
                    param.size(),
                    param.stride()[:len(batch)] + tuple(reversed(param.stride()[len(batch):])))
            flattened_params[self.ctx.user_order_to_col_order[i]] = p.reshape(batch + (-1,))
        flattened_params[-1] = torch.ones(batch + (1,), dtype=params[0].dtype, device=params[0].device)
        p_stack = torch.cat(flattened_params, -1)
        P_eval = self.P @ p_stack
        q_eval = self.q @ p_stack
        A_eval = self.A @ p_stack
        primal, dual, _ = _CvxpyLayer.apply(P_eval, q_eval, A_eval, self.ctx)
        return tuple(var.recover(primal, dual) for var in self.ctx.var_recover)

class _CvxpyLayer(torch.autograd.Function):
    @staticmethod
    def forward(P_eval, q_eval, A_eval, cl_ctx):
        data = cl_ctx.solver_ctx.torch_to_data(P_eval, q_eval, A_eval)
        return data.torch_solve()

    @staticmethod
    def setup_context(ctx, inputs, outputs):
        P_eval, q_eval, A_eval, cl_ctx = inputs
        primal, dual, backwards = outputs
        ctx.save_for_backward(P_eval, q_eval, A_eval)
        ctx.backwards = backwards

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, primal, dual):
        return ctx.data.torch_derivative(primal, dual, ctx.backwards)

def scipy_csr_to_torch_csr(scipy_csr):
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
        size=(num_rows, num_cols)
    )

    return torch_csr

CvxpyLayer = GpuCvxpyLayer
