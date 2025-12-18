PyTorch API
===========

.. module:: cvxpylayers.torch

The PyTorch layer extends ``torch.nn.Module`` for seamless integration with PyTorch models.

CvxpyLayer
----------

.. autoclass:: cvxpylayers.torch.CvxpyLayer
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__, __call__

Usage Example
-------------

.. code-block:: python

   import cvxpy as cp
   import torch
   from cvxpylayers.torch import CvxpyLayer

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

   # Solve with gradients
   A_t = torch.randn(m, n, requires_grad=True)
   b_t = torch.randn(m, requires_grad=True)
   (x_sol,) = layer(A_t, b_t)

   # Backpropagate
   x_sol.sum().backward()

GPU Usage
---------

For GPU acceleration with CuClarabel:

.. code-block:: python

   import cvxpy as cp
   import torch
   from cvxpylayers.torch import CvxpyLayer

   device = torch.device("cuda")

   layer = CvxpyLayer(
       problem,
       parameters=[A, b],
       variables=[x],
       solver=cp.CUCLARABEL
   ).to(device)

   A_gpu = torch.randn(m, n, device=device, requires_grad=True)
   b_gpu = torch.randn(m, device=device, requires_grad=True)
   (x_sol,) = layer(A_gpu, b_gpu)

Integration with nn.Module
--------------------------

.. code-block:: python

   import torch.nn as nn

   class OptNet(nn.Module):
       def __init__(self):
           super().__init__()
           self.fc = nn.Linear(10, 6)

           x = cp.Variable(2)
           A = cp.Parameter((3, 2))
           b = cp.Parameter(3)
           problem = cp.Problem(cp.Minimize(cp.sum_squares(A @ x - b)))

           self.cvx = CvxpyLayer(problem, parameters=[A, b], variables=[x])

       def forward(self, features, b):
           A = self.fc(features).view(-1, 3, 2)
           (x,) = self.cvx(A, b)
           return x
