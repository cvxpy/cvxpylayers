# CVXPYlayers Examples

This directory contains examples demonstrating how to use cvxpylayers across different frameworks and application domains.

## Getting Started

Start with the basic examples for your framework of choice:

| Framework | Quick Start | Tutorial |
|-----------|-------------|----------|
| PyTorch | [`torch/torch_example.py`](torch/torch_example.py) | [`torch/tutorial.ipynb`](torch/tutorial.ipynb) |
| JAX | [`jax/jax_example.py`](jax/jax_example.py) | [`jax/tutorial.ipynb`](jax/tutorial.ipynb) |
| MLX | [`mlx/mlx_example.py`](mlx/mlx_example.py) | - |

## Examples by Domain

### Control Systems

| Example | Description |
|---------|-------------|
| [`torch/lqr.ipynb`](torch/lqr.ipynb) | Linear Quadratic Regulator - learn optimal value function parameters |
| [`torch/constrained_lqr.ipynb`](torch/constrained_lqr.ipynb) | LQR with control input bounds and state constraints |
| [`torch/vehicle.ipynb`](torch/vehicle.ipynb) | Autonomous vehicle path planning and trajectory optimization |
| [`torch/constrained_mpc.ipynb`](torch/constrained_mpc.ipynb) | Model Predictive Control with learned cost-to-go |
| [`torch/convex_approximate_dynamic_programming.ipynb`](torch/convex_approximate_dynamic_programming.ipynb) | Dynamic programming with convex approximation |

### Finance & Portfolio Optimization

| Example | Description |
|---------|-------------|
| [`torch/markowitz_tuning.ipynb`](torch/markowitz_tuning.ipynb) | Portfolio optimization with dynamic rebalancing |
| [`torch/Portfolio optimization with vix.ipynb`](torch/Portfolio%20optimization%20with%20vix.ipynb) | Portfolio optimization incorporating VIX volatility |

### Machine Learning

| Example | Description |
|---------|-------------|
| [`torch/ReLU Layers.ipynb`](torch/ReLU%20Layers.ipynb) | Replace ReLU activations with differentiable optimization layers |
| [`torch/monotonic_output_regression.ipynb`](torch/monotonic_output_regression.ipynb) | Learning monotonic input-output relationships |
| [`torch/signal_denoising.ipynb`](torch/signal_denoising.ipynb) | Signal/image denoising with learned parameters |
| [`torch/data_poisoning_attack.ipynb`](torch/data_poisoning_attack.ipynb) | Adversarial data poisoning attack on logistic regression |

### Resource Allocation

| Example | Description |
|---------|-------------|
| [`torch/resource_allocation.ipynb`](torch/resource_allocation.ipynb) | Water/resource distribution optimization |
| [`torch/supply_chain.ipynb`](torch/supply_chain.ipynb) | Supply chain network flow optimization |

### Physics & Engineering

| Example | Description |
|---------|-------------|
| [`torch/optimizing_stiffness_constants.ipynb`](torch/optimizing_stiffness_constants.ipynb) | Optimal design - spring stiffness coefficients |

## Related Papers

Several examples accompany published research:

- **Learning Convex Optimization Control Policies** (COCP): `lqr.ipynb`, `constrained_lqr.ipynb`, `vehicle.ipynb`, `supply_chain.ipynb`, `markowitz_tuning.ipynb`
- **Learning Convex Optimization Models**: `monotonic_output_regression.ipynb`, `signal_denoising.ipynb`, `constrained_mpc.ipynb`
- **Differentiable Convex Optimization Layers** (NeurIPS 2019): `data_poisoning_attack.ipynb`

## Dependencies

Most examples require:
- `cvxpylayers` with the appropriate framework (`pip install cvxpylayers[torch]`, `[jax]`, or `[mlx]`)
- `matplotlib` for visualization
- `numpy`, `scipy` for numerical operations

Some notebooks have additional dependencies (e.g., `networkx`, `seaborn`, `PIL`).
