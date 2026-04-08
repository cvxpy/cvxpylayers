# JAX Examples

Curated examples of [cvxpylayers](https://github.com/cvxpy/cvxpylayers) with JAX.

Each example is self-contained in its own folder with:
- `README.md` -- description and usage
- An example script (`.py`)
- `requirements.txt` -- dependencies

## Examples

| Example | Description |
|---------|-------------|
| [hello_world](hello_world/) | Differentiate through a least-squares problem |
| [batching](batching/) | Solve a batch of least-squares problems in one call |
| [lqr](lqr/) | Learn an LQR controller by differentiating through trajectory rollouts |
| [optimal_transport](optimal_transport/) | Sensitivity analysis of entropy-regularized optimal transport |
| [signal_denoising](signal_denoising/) | Learn regularization parameters for total-variation denoising |

## References

- A. Agrawal, B. Amos, S. Barratt, S. Boyd, S. Diamond, J. Z. Kolter. [Differentiable Convex Optimization Layers](https://proceedings.neurips.cc/paper/2019/file/9ce3c52fc54362e22053399d3181c638-Paper.pdf). *NeurIPS*, 2019.
- A. Agrawal, S. Barratt, S. Boyd, B. Stellato. [Learning Convex Optimization Control Policies](https://proceedings.mlr.press/v120/agrawal20a.html). *L4DC*, 2020.
- A. Agrawal, S. Barratt, S. Boyd. [Learning Convex Optimization Models](https://ieeexplore.ieee.org/abstract/document/9459585). *IEEE/CAA Journal of Automatica Sinica*, 2021.
- S. Barratt, S. Boyd. [Least Squares Auto-Tuning](https://www.tandfonline.com/doi/abs/10.1080/0305215X.2020.1754406). *Engineering Optimization*, 2020.
- S. Barratt, G. Angeris, S. Boyd. [Automatic Repair of Convex Optimization Problems](https://link.springer.com/article/10.1007/s11081-020-09508-9). *Optimization and Engineering*, 2020.
