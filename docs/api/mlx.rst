MLX API
=======

.. module:: cvxpylayers.mlx

The MLX layer is optimized for Apple Silicon (M1/M2/M3) using the MLX framework.

CvxpyLayer
----------

.. autoclass:: cvxpylayers.mlx.CvxpyLayer
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__, __call__

Usage Example
-------------

.. code-block:: python

   import cvxpy as cp
   import mlx.core as mx
   from cvxpylayers.mlx import CvxpyLayer

   # Define problem
   n, m = 2, 3
   x = cp.Variable(n)
   A = cp.Parameter((m, n))
   b = cp.Parameter(m)
   problem = cp.Problem(
       cp.Minimize(cp.sum_squares(A @ x - b)),
       [x >= 0]
   )

   # Create layer
   layer = CvxpyLayer(problem, parameters=[A, b], variables=[x])

   # Solve
   A_mx = mx.random.normal((m, n))
   b_mx = mx.random.normal((m,))

   (x_sol,) = layer(A_mx, b_mx)

Computing Gradients
-------------------

Use ``mx.grad`` to compute gradients:

.. code-block:: python

   def loss_fn(A, b):
       (x,) = layer(A, b)
       return mx.sum(x)

   # Gradient with respect to A and b
   grad_fn = mx.grad(loss_fn, argnums=[0, 1])
   dA, db = grad_fn(A_mx, b_mx)

   # Evaluate gradients
   mx.eval(dA, db)

Value and Gradient
------------------

Compute both value and gradient efficiently:

.. code-block:: python

   def loss_fn(A, b):
       (x,) = layer(A, b)
       return mx.sum(x)

   value_and_grad_fn = mx.value_and_grad(loss_fn, argnums=[0, 1])
   loss_val, (dA, db) = value_and_grad_fn(A_mx, b_mx)

Moreau CPU Backend
------------------

Select Moreau explicitly to use its CPU interior-point solver and implicit
derivatives. Install the ``mlx`` and ``moreau`` extras (on Linux, also install
``mlx[cpu]``):

.. code-block:: bash

   pip install "cvxpylayers[mlx,moreau]"

.. code-block:: python

   import cvxpy as cp
   import mlx.core as mx
   from cvxpylayers.mlx import CvxpyLayer

   x = cp.Variable(2)
   q = cp.Parameter(2)
   problem = cp.Problem(
       cp.Minimize(0.5 * cp.sum_squares(x) + q @ x),
       [x >= 0],
   )
   layer = CvxpyLayer(problem, parameters=[q], variables=[x], solver=cp.MOREAU)

   # This stream also keeps the surrounding MLX operations on CPU.
   with mx.stream(mx.cpu):
       q_mx = mx.array([-2.0, 3.0])
       x_sol = layer(q_mx)[0]  # [2.0, 0.0]
       dq = mx.grad(lambda q: mx.sum(layer(q)[0]))(q_mx)  # [-1.0, 0.0]
       mx.eval(x_sol, dq)

Moreau's solve and backward pass always run on CPU, including when MLX's default
device is a GPU. This interface does not accelerate Moreau with Metal or CUDA.
It transfers canonical problem data to NumPy in double precision and returns MLX
arrays in the input dtype. PyTorch and JAX are not required.

The backend supports quadratic objectives, batched and broadcast parameters,
primal and dual outputs, and ``mx.grad`` / ``mx.value_and_grad``. Each forward
call retains its own solver state until its backward pass. It uses Moreau's IPM
algorithm; explicit GPU devices and the active-set algorithm are rejected.
Solver settings can be provided at construction or overridden per call, for example
``solver_args={"max_iter": 100}``.

MLX Device Selection
--------------------

MLX operations surrounding the solver follow MLX's selected stream. On Apple
Silicon they can use unified memory, while the optimization solve runs on CPU:

.. code-block:: python

   A_mx = mx.random.normal((1000, 500))
   b_mx = mx.random.normal((1000,))

   (x_sol,) = layer(A_mx, b_mx)
   mx.eval(x_sol)  # Force evaluation

Notes
-----

- MLX uses lazy evaluation; call ``mx.eval()`` to force computation
- The MLX layer supports batched execution like PyTorch and JAX
- Moreau's MLX backend uses CPU solves and first-order reverse-mode derivatives
