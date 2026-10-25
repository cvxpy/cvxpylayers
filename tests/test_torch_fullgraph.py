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

import cvxpy as cp
import numpy as np
import pytest

torch = pytest.importorskip("torch")
moreau = pytest.importorskip("moreau")
from torch._dynamo.testing import CompileCounterWithBackend

from cvxpylayers.torch import CvxpyLayer
from cvxpylayers.torch._sparse import _csr_mm


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    device = request.param
    if not moreau.device_available(device) or (device == "cuda" and not torch.cuda.is_available()):
        pytest.skip(f"Moreau {device} required")
    return device


def _tensor(value, device, batch_shape=(), dtype=torch.float64):
    t = torch.tensor(value, dtype=dtype, device=device)
    if batch_shape:
        t = t.expand(*batch_shape, *t.shape).clone()
    return t.requires_grad_()


@pytest.mark.parametrize("batch_shape", [(), (1,), (3,)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_fullgraph_layer_parameters_and_duals(batch_shape, dtype, device):
    x = cp.Variable(2)
    d = cp.Parameter(2, nonneg=True)
    a = cp.Parameter((1, 2))
    q = cp.Parameter(2)
    b = cp.Parameter(1)
    con = a @ x == b
    problem = cp.Problem(cp.Minimize(0.5 * d @ cp.square(x) + q @ x), [con])
    layer = CvxpyLayer(problem, [d, a, q, b], [x, con.dual_variables[0]], solver="MOREAU")
    torch._dynamo.reset()
    backend = CompileCounterWithBackend("inductor")

    def model(d, a, q, b):
        primal, dual = layer(d, a, 1.2 * q, b)
        return primal.square().sum() + 0.3 * dual.sum()

    compiled = torch.compile(model, fullgraph=True, backend=backend)
    pending, all_inputs, references = [], [], []
    for scale in (1.0, 1.4):
        # Mix batched and shared parameters, including changes to P and A.
        inputs = (
            _tensor([2.0 * scale, 3.0], device, dtype=dtype),
            _tensor([[1.0, scale]], device, batch_shape, dtype),
            _tensor([0.2, -0.3], device, batch_shape, dtype),
            _tensor([1.0], device, dtype=dtype),
        )
        actual = compiled(*inputs)
        expected = model(*inputs)
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
        pending.append(actual)
        all_inputs.extend(inputs)
        references.extend(torch.autograd.grad(expected, inputs))
    grads = torch.autograd.grad(sum(pending), all_inputs)
    for actual, expected in zip(grads, references):
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
    assert backend.frame_count == 1
    torch._dynamo.reset()


@pytest.mark.parametrize("problem_kind", ["direct", "lp", "psd"])
def test_fullgraph_cones_and_warm_start(problem_kind, device):
    if problem_kind == "psd":
        x = cp.Variable((2, 2), symmetric=True)
        p = cp.Parameter((2, 2))
        con = x >> 0
        objective = 0.5 * cp.sum_squares(x) + cp.sum(cp.multiply(p, x))
        values = [[-2.0, 0.3], [0.3, 1.0]]
    else:
        x, p = cp.Variable(2), cp.Parameter(2)
        con = x >= 0
        objective = p @ x
        if problem_kind == "direct":
            objective += 0.5 * cp.sum_squares(x)
            values = [-2.0, 1.0]
        else:
            values = [2.0, 1.0]
    layer = CvxpyLayer(
        cp.Problem(cp.Minimize(objective), [con]), [p], [x, con.dual_variables[0]], solver="MOREAU"
    )
    torch._dynamo.reset()
    compiled = torch.compile(layer, fullgraph=True)
    for scale in (1.0, 1.2, 0.8):
        p = _tensor(np.asarray(values) * scale, device)
        actual = compiled(p, warm_start=True)
        assert not layer._warm_start_cache.x.requires_grad
        assert not layer._warm_start_cache.z_x.requires_grad
        expected = layer(p)
        for a, e in zip(actual, expected):
            torch.testing.assert_close(a, e, atol=1e-5, rtol=1e-5)
        (actual_grad,) = torch.autograd.grad(sum(t.square().sum() for t in actual), p)
        (expected_grad,) = torch.autograd.grad(sum(t.square().sum() for t in expected), p)
        torch.testing.assert_close(actual_grad, expected_grad, atol=1e-4, rtol=1e-4)
    torch._dynamo.reset()


def test_sparse_operator_contract(device):
    crow = torch.tensor([0, 1, 3], device=device)
    col = torch.tensor([0, 0, 1], device=device)
    values = torch.tensor([1.0, 2.0, 3.0], device=device, dtype=torch.float64)
    for shape in ((2,), (2, 3)):
        x = torch.randn(shape, device=device, dtype=torch.float64, requires_grad=True)
        torch.library.opcheck(_csr_mm, (crow, col, values, x, 2, 2, False))
        assert torch.autograd.gradcheck(lambda x: _csr_mm(crow, col, values, x, 2, 2, False), x)


def test_fullgraph_gp_options_and_inference(device):
    x, p = cp.Variable(2, pos=True), cp.Parameter(2, pos=True)
    layer = CvxpyLayer(
        cp.Problem(cp.Minimize(cp.sum(x)), [x >= p]), [p], [x], gp=True, solver="MOREAU"
    )
    torch._dynamo.reset()
    compiled = torch.compile(layer, fullgraph=True)
    value = _tensor([2.0, 3.0], device)
    options = {"max_iter": 80, "ipm_settings": {"tol_gap_abs": 1e-9}}
    (result,) = compiled(value, solver_args=options)
    torch.testing.assert_close(result, value, atol=1e-5, rtol=1e-5)
    (grad,) = torch.autograd.grad(result.sum(), value)
    torch.testing.assert_close(grad, torch.ones_like(value), atol=1e-5, rtol=1e-5)
    with torch.no_grad():
        (result,) = compiled(value, solver_args=options)
    assert not result.requires_grad
    torch.testing.assert_close(result, value, atol=1e-5, rtol=1e-5)
    torch._dynamo.reset()
