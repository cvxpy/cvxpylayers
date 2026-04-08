# Batching

Solve a batch of least-squares problems in one call using `CvxpyLayer`.

The first dimension of each parameter tensor is the batch dimension. The layer solves all problems in the batch and returns a batched solution tensor with gradients.

## Run

```bash
pip install -r requirements.txt
python batching.py
```
