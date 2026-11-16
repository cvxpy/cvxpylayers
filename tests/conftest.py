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

import logging

import pytest


@pytest.fixture
def compile_without_fallback(caplog):
    """Require actual Inductor compilation without hidden backend failures."""
    torch = pytest.importorskip("torch")
    testing = pytest.importorskip("torch._dynamo.testing")
    torch._dynamo.reset()
    backend = testing.CompileCounterWithBackend("inductor")
    requested = False

    def compile_forward(forward):
        nonlocal requested
        requested = True
        return torch.compile(forward, backend=backend, fullgraph=False)

    yield compile_forward
    torch._dynamo.reset()
    if not requested:
        return  # CUDA may have been skipped before compilation was requested.
    assert backend.op_count > 0
    failures = [
        record.getMessage()
        for record in caplog.get_records("call")
        if record.name.startswith("torch.")
        and (
            record.levelno >= logging.ERROR
            or "Backend compiler exception" in record.getMessage()
        )
    ]
    assert not failures, "\n".join(failures)
