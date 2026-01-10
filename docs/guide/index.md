# User Guide

This guide covers the core concepts and features of CVXPYlayers.

## Contents

```{toctree}
:maxdepth: 2

basic-usage
batching
dual-variables
geometric-programs
solvers
```

## Overview

CVXPYlayers converts CVXPY problems into differentiable layers. The key concepts are:

1. **DPP Compliance**: Problems must follow Disciplined Parametrized Programming rules
2. **Parameters vs Variables**: Parameters are inputs; variables are outputs
3. **Implicit Differentiation**: Gradients computed via the KKT conditions
4. **Cone Program Representation**: Problems are canonicalized to standard form

## Data Flow

```
Parameters (torch/jax/mlx tensors)
    |
    v
+-------------------+
| Validate & Batch  |  Check shapes, broadcast
+-------------------+
    |
    v
+-------------------+
| Canonicalize      |  CVXPY -> cone program
+-------------------+
    |
    v
+-------------------+
| Solve             |  diffcp/CuClarabel/etc.
+-------------------+
    |
    v
+-------------------+
| Extract Variables |  Map solution back
+-------------------+
    |
    v
Variables (with gradients attached)
```
