# Hello World

A minimal example showing how to use a differentiable convex optimization layer in JAX.

## What it does

Solves `min ||Ax - b||^2` as a `CvxpyLayer`, computes gradients via `jax.grad`, and verifies they match `jnp.linalg.lstsq`.

## Run

```bash
pip install -r requirements.txt
python hello_world.py
```
