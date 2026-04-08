# LQR

Learn the optimal value function for a Linear Quadratic Regulator (LQR) by differentiating through a control policy layer.

## Problem

Given a discrete-time linear system `x_{t+1} = Ax_t + Bu_t + w_t`, we define a policy as a `CvxpyLayer` that solves:

```
u* = argmin  u'Ru + ||P_sqrt @ (Ax + Bu)||^2
```

We learn `P_sqrt` via SGD on the total trajectory cost, so the policy converges to the optimal LQR controller.

## Run

```bash
pip install -r requirements.txt
python lqr.py
```

## Reference

A. Agrawal, S. Barratt, S. Boyd, B. Stellato. [Learning Convex Optimization Control Policies](https://proceedings.mlr.press/v120/agrawal20a.html). *L4DC*, 2020.
