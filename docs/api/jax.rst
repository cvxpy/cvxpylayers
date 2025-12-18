JAX API
=======

.. module:: cvxpylayers.jax

The JAX layer is a callable class compatible with JAX transformations like ``jax.grad`` and ``jax.vmap``. Support for ``jax.jit`` is coming soon.

CvxpyLayer
----------

.. autoclass:: cvxpylayers.jax.CvxpyLayer
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__, __call__

Usage Example
-------------

.. code-block:: python

   import cvxpy as cp
   import jax
   import jax.numpy as jnp
   from cvxpylayers.jax import CvxpyLayer

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
   key = jax.random.PRNGKey(0)
   key, k1, k2 = jax.random.split(key, 3)
   A_jax = jax.random.normal(k1, shape=(m, n))
   b_jax = jax.random.normal(k2, shape=(m,))

   (x_sol,) = layer(A_jax, b_jax)

Computing Gradients
-------------------

Use ``jax.grad`` to compute gradients:

.. code-block:: python

   def loss_fn(A, b):
       (x,) = layer(A, b)
       return jnp.sum(x)

   # Gradient with respect to A and b
   grad_fn = jax.grad(loss_fn, argnums=[0, 1])
   dA, db = grad_fn(A_jax, b_jax)

JIT Compilation
---------------

.. note::

   Support for ``jax.jit`` is coming soon.

Vectorization with vmap
-----------------------

Use ``jax.vmap`` for batched execution:

.. code-block:: python

   # Batched solve
   batch_size = 10
   A_batch = jax.random.normal(key, shape=(batch_size, m, n))

   @jax.vmap
   def solve_single(A):
       (x,) = layer(A, b_jax)
       return x

   x_batch = solve_single(A_batch)  # Shape: (10, n)

   # Or use built-in batching
   (x_batch,) = layer(A_batch, b_jax)
