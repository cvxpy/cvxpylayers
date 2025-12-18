# PyTorch Quick Start

A minimal example demonstrating CVXPYlayers with PyTorch.

## Code

```{literalinclude} ../../examples/torch/torch_example.py
:language: python
:linenos:
```

## Explanation

This example:

1. **Defines a parametrized QP**: A simple quadratic program with a linear constraint
2. **Creates a CvxpyLayer**: Wraps the CVXPY problem for differentiable optimization
3. **Solves with gradients**: Computes the solution and backpropagates through it

## Running

```bash
cd examples/torch
python torch_example.py
```

## Expected Output

```
Solution: tensor([...], grad_fn=<...>)
Gradient w.r.t. A: tensor([...])
```
