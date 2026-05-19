# Examples

Explore CVXPYlayers through interactive notebooks and scripts.

## Quick Start

::::{grid} 1 1 3 3
:gutter: 3

:::{grid-item-card} **PyTorch**
:link: https://github.com/cvxpy/cvxpylayers/blob/master/examples/torch/torch_example.py
:link-type: url
:class-card: sd-card-pytorch

Most popular for deep learning. Full `torch.nn.Module` integration with autograd support.
:::

:::{grid-item-card} **JAX**
:link: https://github.com/cvxpy/cvxpylayers/blob/master/examples/jax/jax_example.py
:link-type: url
:class-card: sd-card-jax

Functional style with `jax.grad` and `jax.vmap`. Support for `jax.jit` coming soon.
:::

:::{grid-item-card} **MLX**
:link: https://github.com/cvxpy/cvxpylayers/blob/master/examples/mlx/mlx_example.py
:link-type: url
:class-card: sd-card-mlx

Optimized for Apple Silicon. Unified memory for M1/M2/M3 chips.
:::

::::

---

## Tutorials

Step-by-step introductions to CVXPYlayers:

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} PyTorch Tutorial
:link: https://colab.research.google.com/github/cvxpy/cvxpylayers/blob/master/examples/torch/tutorial.ipynb
:link-type: url

Complete walkthrough of defining problems, creating layers, and training.
:::

:::{grid-item-card} JAX Tutorial
:link: https://colab.research.google.com/github/cvxpy/cvxpylayers/blob/master/examples/jax/tutorial.ipynb
:link-type: url

Learn to use CVXPYlayers with JAX transformations.
:::

::::

---

## Control Systems

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Linear Quadratic Regulator (LQR)
:link: https://colab.research.google.com/github/cvxpy/cvxpylayers/blob/master/examples/torch/lqr.ipynb
:link-type: url

Learn optimal value function parameters for LQR control.
:::

:::{grid-item-card} Constrained LQR
:link: https://colab.research.google.com/github/cvxpy/cvxpylayers/blob/master/examples/torch/constrained_lqr.ipynb
:link-type: url

LQR with control input bounds and state constraints.
:::

:::{grid-item-card} Vehicle Path Planning
:link: https://colab.research.google.com/github/cvxpy/cvxpylayers/blob/master/examples/torch/vehicle.ipynb
:link-type: url

Autonomous vehicle trajectory optimization.
:::

:::{grid-item-card} Model Predictive Control
:link: https://colab.research.google.com/github/cvxpy/cvxpylayers/blob/master/examples/torch/constrained_mpc.ipynb
:link-type: url

MPC with learned cost-to-go function.
:::

:::{grid-item-card} Approximate Dynamic Programming
:link: https://colab.research.google.com/github/cvxpy/cvxpylayers/blob/master/examples/torch/convex_approximate_dynamic_programming.ipynb
:link-type: url

Convex approximations for dynamic programming problems.
:::

::::

---

## Finance & Portfolio Optimization

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Markowitz Portfolio
:link: https://colab.research.google.com/github/cvxpy/cvxpylayers/blob/master/examples/torch/markowitz_tuning.ipynb
:link-type: url

Classic mean-variance optimization with dynamic rebalancing.
:::

:::{grid-item-card} Portfolio with VIX
:link: https://colab.research.google.com/github/cvxpy/cvxpylayers/blob/master/examples/torch/Portfolio%20optimization%20with%20vix.ipynb
:link-type: url

Volatility-aware portfolio optimization using VIX index.
:::

::::

---

## Machine Learning

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Monotonic Regression
:link: https://colab.research.google.com/github/cvxpy/cvxpylayers/blob/master/examples/torch/monotonic_output_regression.ipynb
:link-type: url

Learning monotonic input-output relationships.
:::

:::{grid-item-card} Signal Denoising
:link: https://colab.research.google.com/github/cvxpy/cvxpylayers/blob/master/examples/torch/signal_denoising.ipynb
:link-type: url

Signal/image denoising with learned parameters.
:::

:::{grid-item-card} ReLU Layers
:link: https://colab.research.google.com/github/cvxpy/cvxpylayers/blob/master/examples/torch/ReLU%20Layers.ipynb
:link-type: url

Optimization layers with ReLU activations.
:::

:::{grid-item-card} Data Poisoning Attack
:link: https://colab.research.google.com/github/cvxpy/cvxpylayers/blob/master/examples/torch/data_poisoning_attack.ipynb
:link-type: url

Adversarial attacks on machine learning models.
:::

::::

---

## Resource Allocation

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Resource Allocation
:link: https://colab.research.google.com/github/cvxpy/cvxpylayers/blob/master/examples/torch/resource_allocation.ipynb
:link-type: url

Water and resource distribution optimization.
:::

:::{grid-item-card} Supply Chain
:link: https://colab.research.google.com/github/cvxpy/cvxpylayers/blob/master/examples/torch/supply_chain.ipynb
:link-type: url

Supply chain network flow optimization.
:::

::::

---

## Engineering

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Stiffness Constants
:link: https://colab.research.google.com/github/cvxpy/cvxpylayers/blob/master/examples/torch/optimizing_stiffness_constants.ipynb
:link-type: url

Optimizing mechanical stiffness parameters.
:::

:::{grid-item-card} GP Circuit Sizing
:link: https://colab.research.google.com/github/cvxpy/cvxpylayers/blob/master/examples/torch/gp_circuit_sizing.ipynb
:link-type: url

Geometric program for transistor sizing with `gp=True`. Sensitivity analysis of delay vs. area/power budgets.
:::

::::

---

## Optimization Techniques

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Batched Portfolio with Duals
:link: https://colab.research.google.com/github/cvxpy/cvxpylayers/blob/master/examples/torch/batched_portfolio.ipynb
:link-type: url

Batched solving, parameter broadcasting, and dual variable extraction for portfolio optimization.
:::

:::{grid-item-card} Robust Optimization
:link: https://colab.research.google.com/github/cvxpy/cvxpylayers/blob/master/examples/torch/robust_optimization.ipynb
:link-type: url

Learn the right amount of robustness from data. Worst-case design under uncertainty.
:::

:::{grid-item-card} Predict-then-Optimize
:link: https://colab.research.google.com/github/cvxpy/cvxpylayers/blob/master/examples/torch/predict_then_optimize.ipynb
:link-type: url

Neural net + CvxpyLayer end-to-end pipeline. Decision-focused learning vs. two-stage training.
:::

:::{grid-item-card} Convex Regression
:link: https://colab.research.google.com/github/cvxpy/cvxpylayers/blob/master/examples/torch/convex_regression.ipynb
:link-type: url

Fit a convex piecewise-linear function (max-of-affines) to data via SGD.
:::

::::

---

## Running Locally

Clone and run any example:

```bash
git clone https://github.com/cvxpy/cvxpylayers.git
cd cvxpylayers
pip install -e ".[torch]"
pip install matplotlib jupyter

# Run a notebook
jupyter notebook examples/torch/lqr.ipynb
```

## Related Papers

These examples accompany published research:

- **Learning Convex Optimization Control Policies**: `lqr`, `constrained_lqr`, `vehicle`, `supply_chain`, `markowitz_tuning`
- **Learning Convex Optimization Models**: `monotonic_output_regression`, `signal_denoising`, `constrained_mpc`
- **Differentiable Convex Optimization Layers** (NeurIPS 2019): `data_poisoning_attack`
