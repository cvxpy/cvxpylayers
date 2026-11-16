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

MLX defaults to Moreau’s CPU solver.
Select ``solver=cp.DIFFCP`` to use DIFFCP. Install the ``mlx`` extra (on Linux,
also install ``mlx[cpu]``):

.. code-block:: bash

   pip install "cvxpylayers[mlx]"

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
   layer = CvxpyLayer(problem, parameters=[q], variables=[x])

   with mx.stream(mx.cpu):
       q_mx = mx.array([-2.0, 3.0])
       x_sol = layer(q_mx)[0]  # [2.0, 0.0]
       dq = mx.grad(lambda q: mx.sum(layer(q)[0]))(q_mx)  # [-1.0, 0.0]
       mx.eval(x_sol, dq)

Pass Moreau settings through ``solver_args`` at construction or call time.
