# CVXPYlayers

CVXPYlayers is a Python library for constructing differentiable convex optimization layers in PyTorch and JAX.
CVXPYlayers 1.0 supports keeping the data on the GPU with the CuClarabel backend.

This library accompanies our [NeurIPS 2019 paper](https://web.stanford.edu/~boyd/papers/pdf/diff_cvxpy.pdf)
on differentiable convex optimization layers.
For an informal introduction to convex optimization layers, see our
[blog post](https://locuslab.github.io/2019-10-28-cvxpylayers/).

Our package uses [CVXPY](https://github.com/cvxgrp/cvxpy) for specifying
parametrized convex optimization problems.

- [Installation](#installation)
- [Usage](#usage)
- [Examples](#examples)
- [Contributing](#contributing)
- [Projects using cvxpylayers](#projects-using-cvxpylayers)
- [License](#contributing)
- [Citing](#citing)

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install
cvxpylayers.

```bash
pip install cvxpylayers
```

Our package includes convex optimization layers for
PyTorch, JAX, and TensorFlow 2.0;
the layers are functionally equivalent. You will need to install
[PyTorch](https://pytorch.org),
[JAX](https://jax.dev), or
[TensorFlow](https://www.tensorflow.org)
separately, which can be done by following the instructions on their websites.

Additionally, to use the fully GPU-accelerated pathway, install:

- [Julia](https://julialang.org/)
- Install [CuClarabel](https://github.com/oxfordcontrol/Clarabel.jl/tree/CuClarabel/)
- Install [juliacall](https://juliapy.github.io/PythonCall.jl/stable/juliacall/)
- Install [cupy](https://cupy.dev/)
- Install [diffqcp](https://github.com/cvxgrp/diffqcp)
- Install [lineax](https://github.com/patrick-kidger/lineax) from main. (*e.g.*, `uv add "lineax @ git+https://github.com/patrick-kidger/lineax.git"`)
