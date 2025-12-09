API Reference
=============

CVXPYlayers provides the ``CvxpyLayer`` class for each supported framework.
The API is consistent across frameworks, with minor differences for framework integration.

.. toctree::
   :maxdepth: 2

   torch
   jax
   mlx

Quick Reference
---------------

All frameworks share the same constructor signature:

.. code-block:: python

   CvxpyLayer(
       problem,           # cvxpy.Problem
       parameters,        # list of cvxpy.Parameter
       variables,         # list of cvxpy.Variable
       solver=None,       # optional solver
       gp=False,          # geometric program mode
       verbose=False,     # solver verbosity
       solver_args=None,  # default solver arguments
   )

Framework Differences
---------------------

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Framework
     - Base Class
     - Call Method
   * - PyTorch
     - ``torch.nn.Module``
     - ``forward(*params)``
   * - JAX
     - Callable class
     - ``__call__(*params)``
   * - MLX
     - Callable class
     - ``__call__(*params)``
