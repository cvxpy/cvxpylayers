
import numpy as np
import diffcp
import time
from dataclasses import dataclass
from typing import Any


@dataclass
class ForwardContext:
    gp: bool
    solve_and_derivative: bool
    batch: bool
    batch_size: int
    batch_sizes: list
    compiler: callable
    param_ids: list
    param_order: list
    old_params_to_new_params: dict
    cone_dims: list
    solver_args: dict
    variables: list
    var_dict: dict


@dataclass
class BackwardContext:
    info: dict
    gp: bool
    batch: bool
    batch_size: int
    batch_sizes: list
    variables: list
    compiler: Any
    param_ids: list
    param_order: list
    params: list
    old_params_to_new_params: dict
    sol: list
    
    
def forward_numpy(params_numpy, context):
    """Forward pass in numpy."""
    
    info = {}
    
    if context.gp:
        param_map = {}
        # construct a list of params for the DCP problem
        for param, value in zip(context.param_order, params_numpy):
            if param in context.old_params_to_new_params:
                new_id = context.old_params_to_new_params[param].id
                param_map[new_id] = np.log(value)
            else:
                new_id = param.id
                param_map[new_id] = value
        params_numpy = [param_map[pid] for pid in context.param_ids]
    
    # canonicalize problem
    start = time.time()
    As, bs, cs, cone_dicts, shapes = [], [], [], [], []
    for i in range(context.batch_size):
        params_numpy_i = [
            p if sz == 0 else p[i]
            for p, sz in zip(params_numpy, context.batch_sizes)]
        c, _, neg_A, b = context.compiler.apply_parameters(
            dict(zip(context.param_ids, params_numpy_i)),
            keep_zeros=True)
        A = -neg_A  # cvxpy canonicalizes -A
        As.append(A)
        bs.append(b)
        cs.append(c)
        cone_dicts.append(context.cone_dims)
        shapes.append(A.shape)
    info['canon_time'] = time.time() - start
    info['shapes'] = shapes

    # compute solution and derivative function
    start = time.time()
    try:
        if context.solve_and_derivative:
            xs, _, _, _, DT_batch = diffcp.solve_and_derivative_batch(
                As, bs, cs, cone_dicts, **context.solver_args)
            info['DT_batch'] = DT_batch
        else:
            xs, _, _ = diffcp.solve_only_batch(
                As, bs, cs, cone_dicts, **context.solver_args)
    except diffcp.SolverError as e:
        print(
            "Please consider re-formulating your problem so that "
            "it is always solvable or increasing the number of "
            "solver iterations.")
        raise e
    info['solve_time'] = time.time() - start

    # extract solutions and append along batch dimension
    start = time.time()
    sol = [[] for i in range(len(context.variables))]
    for i in range(context.batch_size):
        sltn_dict = context.compiler.split_solution(
            xs[i], active_vars=context.var_dict)
        for j, v in enumerate(context.variables):
            sol[j].append(np.expand_dims(sltn_dict[v.id], axis=0))
    sol = [np.concatenate(s, axis=0) for s in sol]

    if not context.batch:
        sol = [np.squeeze(s, axis=0) for s in sol]

    if context.gp:
        sol = [np.exp(s) for s in sol]
        info['sol'] = sol
            
    return sol, info


def backward_numpy(dvars_numpy, context):
    """Backward pass in numpy."""
        
    info = {}
    
    if context.gp:
        # derivative of exponential recovery transformation
        dvars_numpy = [dvar*s for dvar, s in zip(dvars_numpy, context.sol)]
    
    if not context.batch:
        dvars_numpy = [np.expand_dims(dvar, 0) for dvar in dvars_numpy]

    # differentiate from cvxpy variables to cone problem data
    dxs, dys, dss = [], [], []
    for i in range(context.batch_size):
        del_vars = {}
        for v, dv in zip(context.variables, [dv[i] for dv in dvars_numpy]):
            del_vars[v.id] = dv
        dxs.append(context.compiler.split_adjoint(del_vars))
        dys.append(np.zeros(context.info['shapes'][i][0]))
        dss.append(np.zeros(context.info['shapes'][i][0]))

    dAs, dbs, dcs = context.info['DT_batch'](dxs, dys, dss)

    # differentiate from cone problem data to cvxpy parameters
    start = time.time()
    grad = [[] for _ in range(len(context.param_ids))]
    for i in range(context.batch_size):
        del_param_dict = context.compiler.apply_param_jac(
            dcs[i], -dAs[i], dbs[i])
        for j, pid in enumerate(context.param_ids):
            grad[j].append(np.expand_dims(del_param_dict[pid], 0))
    grad = [np.concatenate(g, axis=0) for g in grad]

    if context.gp:
        # differentiate through the log transformation of params
        dcp_grad = grad
        grad = []
        dparams = {pid: g for pid, g in zip(context.param_ids, dcp_grad)}
        for param, value in zip(context.param_order, context.params):
            g = 0.0 if param.id not in dparams else dparams[param.id]
            if param in context.old_params_to_new_params:
                dcp_param_id = context.old_params_to_new_params[param].id
                # new_param.value == log(param), apply chain rule
                g += (1.0 / value) * dparams[dcp_param_id]
            grad.append(g)
    info['dcanon_time'] = time.time() - start

    if not context.batch:
        grad = [g.squeeze(0) for g in grad]
    else:
        for i, sz in enumerate(context.batch_sizes):
            if sz == 0:
                grad[i] = grad[i].sum(axis=0)
    
    return grad, info

def extract_linops_as_csr(prob, solver, kwargs):
    data, *_ = prob.get_problem_data(solver, **kwargs)
    param_prog = data['param_prog']
    return param_prog.reduced_P.reduced_mat, param_prog.c, param_prog.reduced_A.reduced_mat, data
