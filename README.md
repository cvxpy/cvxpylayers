# cvxpylayers

![cvxpylayers logo](cvxpylayers_logo.png)

[![Build Status](https://github.com/cvxpy/cvxpylayers/actions/workflows/build.yml/badge.svg)](https://github.com/cvxpy/cvxpylayers/actions/workflows/build.yml)
[![Documentation](https://img.shields.io/badge/docs-online-brightgreen?logo=read-the-docs&style=flat)](https://www.cvxgrp.org/cvxpylayers/)
[![Coverage Status](https://coveralls.io/repos/github/cvxpy/cvxpylayers/badge.svg?branch=master)](https://coveralls.io/github/cvxpy/cvxpylayers?branch=master)

## About

cvxpylayers is a Python library for constructing differentiable convex
optimization layers in PyTorch, JAX, and TensorFlow using CVXPY.
A convex optimization layer solves a parametrized convex optimization problem
in the forward pass to produce a solution.
It computes the derivative of the solution with respect to
the parameters in the backward pass.

This library accompanies our [NeurIPS 2019 paper](https://web.stanford.edu/~boyd/papers/pdf/diff_cvxpy.pdf)
on differentiable convex optimization layers.
For an informal introduction to convex optimization layers, see our
[blog post](https://locuslab.github.io/2019-10-28-cvxpylayers/). Our package uses 
[CVXPY](https://github.com/cvxgrp/cvxpy) for specifying parametrized convex optimization 
problems.

For more information, see the [documentation](https://www.cvxgrp.org/cvxpylayers/).

## Installation

cvxpylayers is available on [pypi](https://pypi.org/project/cvxpylayers/) and can be 
installed with:

```bash
pip install cvxpylayers
```

cvxpylayers has the following dependencies:
* Python 3
* [NumPy](https://pypi.org/project/numpy/)
* [CVXPY](https://github.com/cvxgrp/cvxpy) >= 1.1.a4
* [PyTorch](https://pytorch.org) >= 1.0, [JAX](https://github.com/google/jax) >= 0.2.12
* [diffcp](https://github.com/cvxgrp/diffcp) >= 1.0.13

To make use of differentiable convex optimization layers in PyTorch or JAX, you will 
need to install the corresponding dependencies. You can do this manually
for [PyTorch](https://pytorch.org) and [JAX](https://github.com/google/jax) or use
the commands given below.


### PyTorch

Install manuall from [PyTorch](https://pytorch.org) or run:

```bash
pip install cvxpylayers[torch]  
```

### JAX

Install manually from [JAX](https://github.com/google/jax) or run:

```bash
pip install cvxpylayers[jax] 
```

## Usage

Below are usage examples of our PyTorch and JAX layers. Note that the parametrized 
convex optimization problems must be constructed in CVXPY, and must be
[DPP](https://www.cvxpy.org/tutorial/advanced/index.html#disciplined-parametrized-programming)
compliant. For more information on DPP, see the [CVXPY documentation](https://www.cvxpy.org/tutorial/dpp/index.html).

### PyTorch

```python
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

Note: `CvxpyLayer` cannot be traced with `torch.jit`.

### JAX
```python
import cvxpy as cp
import jax
from cvxpylayers.jax import CvxpyLayer

n, m = 2, 3
x = cp.Variable(n)
A = cp.Parameter((m, n))
b = cp.Parameter(m)
constraints = [x >= 0]
objective = cp.Minimize(0.5 * cp.pnorm(A @ x - b, p=1))
problem = cp.Problem(objective, constraints)
assert problem.is_dpp()

cvxpylayer = CvxpyLayer(problem, parameters=[A, b], variables=[x])
key = jax.random.PRNGKey(0)
key, k1, k2 = jax.random.split(key, 3)
A_jax = jax.random.normal(k1, shape=(m, n))
b_jax = jax.random.normal(k2, shape=(m,))

solution, = cvxpylayer(A_jax, b_jax)

# compute the gradient of the summed solution with respect to A, b
dcvxpylayer = jax.grad(lambda A, b: sum(cvxpylayer(A, b)[0]), argnums=[0, 1])
gradA, gradb = dcvxpylayer(A_jax, b_jax)
```

Note: `CvxpyLayer` cannot be traced with the JAX `jit` or `vmap` operations.