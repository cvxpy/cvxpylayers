# Hello World

A minimal example showing how to use a differentiable convex optimization layer in PyTorch.

## What it does

Solves `min ||Ax - b||^2` as a `CvxpyLayer`, computes gradients via backprop, and verifies they match `torch.linalg.lstsq`.

## Run

```bash
pip install -r requirements.txt
python hello_world.py
```
