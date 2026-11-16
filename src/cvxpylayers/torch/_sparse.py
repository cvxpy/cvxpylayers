"""
Copyright, the CVXPY authors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import numpy as np
import scipy.sparse
import torch


@torch.library.custom_op("cvxpylayers::csr_mm", mutates_args=())
def _csr_mm(
    crow: torch.Tensor,
    col: torch.Tensor,
    values: torch.Tensor,
    x: torch.Tensor,
    rows: int,
    cols: int,
    transpose: bool,
) -> torch.Tensor:
    if x.device.type == "cpu":
        matrix = scipy.sparse.csr_array(
            (values.numpy(), col.numpy(), crow.numpy()), shape=(rows, cols)
        )
        if transpose:
            matrix = matrix.T
        return torch.from_numpy(np.asarray(matrix @ x.numpy()))
    matrix = torch.sparse_csr_tensor(crow, col, values, size=(rows, cols))
    if transpose:
        matrix = matrix.transpose(0, 1)
    if x.ndim == 1:
        return torch.sparse.mm(matrix, x.unsqueeze(1)).squeeze(1)
    return torch.sparse.mm(matrix, x)


@_csr_mm.register_fake
def _csr_mm_fake(crow, col, values, x, rows, cols, transpose):
    return x.new_empty((cols if transpose else rows, *x.shape[1:]))


def _setup_context(ctx, inputs, output):
    crow, col, values, _x, rows, cols, transpose = inputs
    ctx.save_for_backward(crow, col, values)
    ctx.rows, ctx.cols, ctx.transpose = rows, cols, transpose


def _backward(ctx, grad):
    crow, col, values = ctx.saved_tensors
    dx = _csr_mm(crow, col, values, grad, ctx.rows, ctx.cols, not ctx.transpose)
    return None, None, None, dx, None, None, None


_csr_mm.register_autograd(_backward, setup_context=_setup_context)


class _CsrMatrix(torch.nn.Module):
    """Constant sparse map with dense buffers that Dynamo can represent."""

    def __init__(self, matrix):
        super().__init__()
        self.rows, self.cols = matrix.shape
        self.register_buffer("crow", torch.as_tensor(matrix.indptr, dtype=torch.int64))
        self.register_buffer("col", torch.as_tensor(matrix.indices, dtype=torch.int64))
        self.register_buffer("values", torch.as_tensor(matrix.data, dtype=torch.float64))

    def forward(self, x):
        return _csr_mm(
            self.crow.to(device=x.device),
            self.col.to(device=x.device),
            self.values.to(dtype=x.dtype, device=x.device),
            x,
            self.rows,
            self.cols,
            False,
        )
