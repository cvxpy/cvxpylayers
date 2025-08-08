import torch
import cvxpylayers.utils.parse_args as pa


class GpuCvxpyLayer(torch.nn.Module):
    def __init__(self, problem, parameters, variables, solver=None, kwargs={}):
        super().__init__()
        self.ctx = pa.parse_args(problem, variables, parameters, solver, kwargs)
        self.P = torch.nn.Buffer(scipy_csr_to_torch_csr(ctx.reduced_P))
        self.q = torch.nn.Buffer(scipy_csr_to_torch_csr(ctx.q))
        self.A = torch.nn.Buffer(scipy_csr_to_torch_csr(ctx.reduced_A))


    def forward(self, *params):
        batch = ctx.validate_params(params)
        flattened_params = len(params) * [None]
        for i, param in enumerate(params):
            p = torch.Tensor()
            p.set_(
                    param.untyped_storage(),
                    param.storage_offset(),
                    param.size(),
                    param.stride()[:len(batch)] + tuple(reversed(param.stride()[len(batch):])))
            p.reshape(batch + (-1,))
            flattened_params[ctx.user_order_col_order[i]] = p
        p_stack = torch.cat(flattened_params, -1)
        P_eval = self.P @ p_stack
        q_eval = self.q @ p_stack
        A_eval = self.A @ p_stack
        primal, dual = _CvxpyLayer(P_eval, q_eval, A_eval, ctx)
        return tuple(var(primal, dual) for var in ctx.var_recover)

class _CvxpyLayer(torch.autograd.Function):
    @staticmethod
    def forward(P_eval, q_eval, A_eval, cl_ctx):
        data = cl_ctx.solver.torch_to_data(P_eval, q_eval, A_eval)
        return data.solve()
        

    @staticmethod
    def setup_context(ctx, inputs, outputs):
        P_eval, q_eval, A_eval, cl_ctx = inputs
        primal, dual, data = outputs
        ctx.save_for_backward(P_eval, q_eval, A_eval)
        ctx.data = data

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, primal, dual):
        return ctx.data.torch_derivative(primal, dual)

def scipy_csr_to_torch_csr(scipy_csr):
    # Get the CSR format components
    values = scipy_csr.data
    row_indices = scipy_csr.indices
    row_ptr = scipy_csr.indptr
    num_rows, num_cols = scipy_csr.shape

    # Create the torch sparse csr_tensor
    torch_csr = torch.sparse_csr_tensor(
        row_ptr=torch.tensor(row_ptr, dtype=torch.int64),
        col=torch.tensor(row_indices, dtype=torch.int64),
        value=torch.tensor(values, dtype=torch.float64),
        size=(num_rows, num_cols)
    )

    return torch_csr

CvxpyLayer = GpuCvxpyLayer
