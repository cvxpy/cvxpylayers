# Optimal Transport

Entropy-regularized optimal transport as a differentiable layer.

## Problem

Given two discrete distributions with support points `x` and `y` and probability vectors `a` and `b`, solve:

```
min  trace(C'P) - eps * H(P)
s.t. P @ 1 = a,  P' @ 1 = b,  P >= 0
```

where `C_ij = (x_i - y_j)^2` is the cost matrix and `H(P)` is the entropy.

After computing the transport plan, we use `jax.grad` to find how to change the support points, probabilities, or regularization to increase a specific coupling.

## Run

```bash
pip install -r requirements.txt
python optimal_transport.py
```

## Reference

A. Agrawal, B. Amos, S. Barratt, S. Boyd, S. Diamond, J. Z. Kolter. [Differentiable Convex Optimization Layers](https://proceedings.neurips.cc/paper/2019/file/9ce3c52fc54362e22053399d3181c638-Paper.pdf). *NeurIPS*, 2019.
