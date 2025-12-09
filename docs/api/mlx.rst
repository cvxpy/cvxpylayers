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

Apple Silicon Optimization
--------------------------

MLX automatically uses the unified memory architecture of Apple Silicon,
allowing efficient computation without explicit device management:

.. code-block:: python

   # No need to move tensors to GPU - MLX handles this automatically
   A_mx = mx.random.normal((1000, 500))
   b_mx = mx.random.normal((1000,))

   (x_sol,) = layer(A_mx, b_mx)
   mx.eval(x_sol)  # Force evaluation

Notes
-----

- MLX uses lazy evaluation; call ``mx.eval()`` to force computation
- The MLX layer supports batched execution like PyTorch and JAX
- For best performance on Apple Silicon, prefer MLX over PyTorch CPU
