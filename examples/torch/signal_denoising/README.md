# Signal Denoising

Learn regularization parameters for a total-variation denoising layer.

## Problem

Given a noisy signal `x`, the denoising layer solves:

```
min  ||theta @ (x - y)||^2 + lambda * ||diff(y)||^2
```

where `y` is the denoised output. Both the weighting matrix `theta` (n x n) and regularization strength `lambda` are learned end-to-end via Adam to minimize MSE on noisy cosine signals.

## Run

```bash
pip install -r requirements.txt
python signal_denoising.py
```

## Reference

A. Agrawal, S. Barratt, S. Boyd. [Learning Convex Optimization Models](https://ieeexplore.ieee.org/abstract/document/9459585). *IEEE/CAA Journal of Automatica Sinica*, 2021.
