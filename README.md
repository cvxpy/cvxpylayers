# CVXPYlayers

CVXPYlayers is a Python library for constructing differentiable convex
optimization layers in PyTorch, JAX, and MLX using CVXPY.
A convex optimization layer solves a parametrized convex optimization problem
in the forward pass to produce a solution.
It computes the derivative of the solution with respect to
the parameters in the backward pass.

**CVXPYlayers 1.0** supports keeping the data on the GPU with the CuClarabel backend.

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
- [Projects using CVXPYlayers](#projects-using-cvxpylayers)
- [License](#license)
- [Citing](#citing)

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install
cvxpylayers.

```bash
pip install cvxpylayers
```

Our package includes convex optimization layers for
PyTorch, JAX, and MLX;
the layers are functionally equivalent. You will need to install
[PyTorch](https://pytorch.org),
[JAX](https://github.com/google/jax), or
[MLX](https://github.com/ml-explore/mlx)
separately, which can be done by following the instructions on their websites.

CVXPYlayers has the following dependencies:
* Python >= 3.11
* [NumPy](https://pypi.org/project/numpy/) >= 1.22.4
* [CVXPY](https://github.com/cvxgrp/cvxpy) >= 1.7.4
* [diffcp](https://github.com/cvxgrp/diffcp) >= 1.1.0

Additionally, install one of the following frameworks:
* [PyTorch](https://pytorch.org) >= 2.0
* [JAX](https://github.com/google/jax) >= 0.4.0
* [MLX](https://github.com/ml-explore/mlx)

### GPU-accelerated pathway (CVXPYlayers 1.0)

To use the fully GPU-accelerated pathway, install:

- [Julia](https://julialang.org/)
- [CuClarabel](https://github.com/oxfordcontrol/Clarabel.jl/tree/CuClarabel/)
- [juliacall](https://juliapy.github.io/PythonCall.jl/stable/juliacall/)
- [cupy](https://cupy.dev/)
- [diffqcp](https://github.com/cvxgrp/diffqcp)
- [lineax](https://github.com/patrick-kidger/lineax) from main (*e.g.*, `uv add "lineax @ git+https://github.com/patrick-kidger/lineax.git"`)

## Usage
Below are usage examples of our PyTorch and JAX layers.
Note that the parametrized convex optimization problems must be constructed
in CVXPY, using
[DPP](https://www.cvxpy.org/tutorial/advanced/index.html#disciplined-parametrized-programming).

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

layer = CvxpyLayer(problem, parameters=[A, b], variables=[x])
A_tch = torch.randn(m, n, requires_grad=True)
b_tch = torch.randn(m, requires_grad=True)

# solve the problem
(solution,) = layer(A_tch, b_tch)

# compute the gradient of the sum of the solution with respect to A, b
solution.sum().backward()
```

#### PyTorch on GPU with CuClarabel

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

device = torch.device("cuda")
layer = CvxpyLayer(problem, parameters=[A, b], variables=[x], solver=cp.CUCLARABEL).to(device)
A_tch = torch.randn(m, n, requires_grad=True, device=device)
b_tch = torch.randn(m, requires_grad=True, device=device)

# solve the problem
(solution,) = layer(A_tch, b_tch)

# compute the gradient of the sum of the solution with respect to A, b
solution.sum().backward()
```

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

layer = CvxpyLayer(problem, parameters=[A, b], variables=[x])
key = jax.random.PRNGKey(0)
key, k1, k2 = jax.random.split(key, 3)
A_jax = jax.random.normal(k1, shape=(m, n))
b_jax = jax.random.normal(k2, shape=(m,))

(solution,) = layer(A_jax, b_jax)

# compute the gradient of the summed solution with respect to A, b
dlayer = jax.grad(lambda A, b: sum(layer(A, b)[0]), argnums=[0, 1])
gradA, gradb = dlayer(A_jax, b_jax)
```

### Dual variables
CVXPYlayers can return constraint dual variables (Lagrange multipliers) alongside the primal solution:

```python
import cvxpy as cp
import torch
from cvxpylayers.torch import CvxpyLayer

x = cp.Variable(2)
c = cp.Parameter(2)
b = cp.Parameter()

eq_con = cp.sum(x) == b
prob = cp.Problem(cp.Minimize(c @ x), [eq_con, x >= 0])

# Request both primal and dual variables
layer = CvxpyLayer(prob, parameters=[c, b], variables=[x, eq_con.dual_variables[0]])

c_tch = torch.tensor([1.0, 2.0], requires_grad=True)
b_tch = torch.tensor(1.0, requires_grad=True)

x_star, eq_dual = layer(c_tch, b_tch)
```

### Log-log convex programs
CVXPYlayers can also differentiate through log-log convex programs (LLCPs), which generalize geometric programs. Use the keyword argument `gp=True` when constructing a `CvxpyLayer` for an LLCP. Below is a simple usage example

```python
import cvxpy as cp
import torch
from cvxpylayers.torch import CvxpyLayer

x = cp.Variable(pos=True)
y = cp.Variable(pos=True)
z = cp.Variable(pos=True)

a = cp.Parameter(pos=True, value=2.)
b = cp.Parameter(pos=True, value=1.)
c = cp.Parameter(value=0.5)

objective_fn = 1/(x*y*z)
objective = cp.Minimize(objective_fn)
constraints = [a*(x*y + x*z + y*z) <= b, x >= y**c]
problem = cp.Problem(objective, constraints)
assert problem.is_dgp(dpp=True)

layer = CvxpyLayer(problem, parameters=[a, b, c],
                   variables=[x, y, z], gp=True)
a_tch = torch.tensor(a.value, requires_grad=True)
b_tch = torch.tensor(b.value, requires_grad=True)
c_tch = torch.tensor(c.value, requires_grad=True)

x_star, y_star, z_star = layer(a_tch, b_tch, c_tch)
sum_of_solution = x_star + y_star + z_star
sum_of_solution.backward()
```

## Solvers

CVXPYlayers supports multiple solvers including [Clarabel](https://github.com/oxfordcontrol/Clarabel.rs)
and [SCS](https://github.com/cvxgrp/scs).
For GPU acceleration, use [CuClarabel](https://github.com/oxfordcontrol/Clarabel.jl/tree/CuClarabel/).

### Passing arguments to the solvers
One can pass arguments to solvers by adding the argument as a key-value pair
in the `solver_args` argument.
For example, to increase the tolerance of SCS to `1e-8` one would write:
```
layer(*parameters, solver_args={"eps": 1e-8})
```
If SCS is not converging, we highly recommend using the following arguments to `SCS`:
```
solver_args={"eps": 1e-8, "max_iters": 10000, "acceleration_lookback": 0}
```

## Examples
Our [examples](examples) subdirectory contains simple applications of convex optimization
layers.

## Contributing
Pull requests are welcome. For major changes, please open an issue first to
discuss what you would like to change.

Please make sure to update tests as appropriate.

### Running tests

CVXPYlayers uses the `pytest` framework for running tests.
To install `pytest`, run:
```bash
pip install pytest
```

Execute the tests from the main directory of this repository with:
```bash
pytest tests/
```

## Projects using CVXPYlayers
Below is a list of projects using CVXPYlayers. If you have used CVXPYlayers in a project, you're welcome to make a PR to add it to this list.
* [Learning Convex Optimization Control Policies](https://web.stanford.edu/~boyd/papers/learning_cocps.html)
* [Learning Convex Optimization Models](https://web.stanford.edu/~boyd/papers/learning_copt_models.html)
* [DeepDow](https://github.com/jankrepl/deepdow) - Portfolio optimization with deep learning
* [NeuroMANCER](https://github.com/pnnl/neuromancer) - PNNL's PyTorch library for constrained optimization, physics-informed system identification, and model predictive control

## License
CVXPYlayers carries an Apache 2.0 license.

## Citing
If you use CVXPYlayers for research, please cite our accompanying [NeurIPS paper](https://web.stanford.edu/~boyd/papers/pdf/diff_cvxpy.pdf):

```
@inproceedings{cvxpylayers2019,
  author={Agrawal, A. and Amos, B. and Barratt, S. and Boyd, S. and Diamond, S. and Kolter, Z.},
  title={Differentiable Convex Optimization Layers},
  booktitle={Advances in Neural Information Processing Systems},
  year={2019},
}
```

If you use CVXPYlayers to differentiate through a log-log convex program, please cite the accompanying [paper](https://web.stanford.edu/~boyd/papers/diff_dgp.html):

```
@article{agrawal2020differentiating,
  title={Differentiating through log-log convex programs},
  author={Agrawal, Akshay and Boyd, Stephen},
  journal={arXiv},
  archivePrefix={arXiv},
  eprint={2004.12553},
  primaryClass={math.OC},
  year={2020},
}
```
