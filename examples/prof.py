#!/usr/bin/env python3

import argparse
import sys

import numpy as np
import numpy.random as npr

import itertools
import time

import torch

from qpth.qp import QPFunction
from cvxpylayers.torch.cvxpylayer import CvxpyLayer

from scipy.linalg import sqrtm
from scipy import sparse
import cvxpy as cp
import pandas as pd

from IPython.core import ultratb
sys.excepthook = ultratb.FormattedTB(
    mode='Verbose',
    color_scheme='Linux', call_pdb=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nTrials', type=int, default=10)
    args = parser.parse_args()
    npr.seed(0)
    prof(args)


def prof(args):
    trials = []
    for nz, nbatch, cuda in itertools.product(
            [128], [128], [True, False]):
        print('--- {} vars/cons, batch size: {}, cuda: {} ---'.format(
            nz, nbatch, cuda))
        for i in range(args.nTrials):
            print('  + Trial {}'.format(i))
            t = prof_dense_qp(i, nz, nbatch, 'dense', cuda)
            trials += t
            print(t)

    for nz, nbatch, cuda in itertools.product(
            [1024], [32], [False]):
        print('--- {} vars/cons, batch size: {}, cuda: {} ---'.format(
            nz, nbatch, cuda))
        for i in range(args.nTrials):
            print('  + Trial {}'.format(i))
            t = prof_sparse_qp(i, nz, nbatch, None, cuda)
            trials += t
            print(t)

    df = pd.DataFrame(trials)
    df.to_csv('results.csv', index=False)


def prof_sparse_qp(trial, nz, nbatch, cons, cuda=True):
    trials = []

    npr.seed(trial)

    A = sparse.random(nz, nz, density=.01) + \
        sparse.eye(nz)
    A_rows, A_cols = A.nonzero()

    G = sparse.random(nz, nz, density=.01) + \
        sparse.eye(nz)
    G_rows, G_cols = G.nonzero()
    Q = sparse.eye(nz)

    xs = npr.randn(nbatch, nz)
    p = npr.randn(nbatch, nz)
    b = np.array([A @ xs[i] for i in range(nbatch)])
    h = np.array([G @ xs[i] for i in range(nbatch)])

    def convert(A):
        A = [A.todense() for _ in range(nbatch)]
        return torch.from_numpy(np.array(A)).double().requires_grad_()

    Q_tch, A_tch, G_tch = [convert(mat) for mat in [Q, A, G]]

    p_tch, b_tch, h_tch = [
        torch.from_numpy(x).double().requires_grad_()
        for x in [p, b, h]
    ]

    if cuda:
        p_tch, Q_tch, G_tch, h_tch, A_tch, b_tch = [
            x.cuda() for x in [p_tch, Q_tch, G_tch, h_tch, A_tch, b_tch]]

    torch.cuda.synchronize()
    torch.cuda.synchronize()
    start = time.time()
    x = QPFunction(verbose=False, eps=1e-8, notImprovedLim=5,
                   maxIter=1000)(Q_tch, p_tch, G_tch, h_tch, A_tch, b_tch)
    torch.cuda.synchronize()
    t = time.time() - start
    trials.append({
        'trial': trial,
        'nz': nz,
        'nbatch': nbatch,
        'cuda': cuda,
        'mode': 'qpth',
        'direction': 'forward',
        'time': t,
        'qp': 'sparse'
    })

    y = x.sum()
    start = time.time()
    y.backward()
    t = time.time() - start
    trials.append({
        'trial': trial,
        'nz': nz,
        'nbatch': nbatch,
        'cuda': cuda,
        'mode': 'qpth',
        'direction': 'backward',
        'time': t,
        'qp': 'sparse'
    })

    _p = cp.Parameter((nz, 1))
    _b = cp.Parameter((nz, 1))
    _h = cp.Parameter((nz, 1))

    _z = cp.Variable((nz, 1))

    obj = cp.Minimize(0.5 * cp.sum_squares(_z) + _p.T @ _z)
    cons = [G @ _z <= _h,
            A @ _z == _b]
    prob = cp.Problem(obj, cons)

    p_tch, b_tch, h_tch = [torch.from_numpy(x).unsqueeze(-1).requires_grad_()
                           for x in [p, b, h]]

    solver_args = {
        'mode': 'lsqr',
        'verbose': False,
        'max_iters': 1000,
        'eps': 1e-6,
        'use_indirect': False,
        'gpu': False,
        'n_jobs_forward': -1,
        'n_jobs_backward': -1
    }
    solve = CvxpyLayer(prob, [_p, _b, _h], [_z])

    start = time.time()
    z, = solve(p_tch, b_tch, h_tch, solver_args=solver_args)
    t = time.time() - start
    trials.append({
        'trial': trial,
        'nz': nz,
        'nbatch': nbatch,
        'cuda': cuda,
        'mode': 'cvxpylayers',
        'direction': 'forward',
        'time': t,
        'qp': 'sparse',
        'canon_time': solve.info.get("canon_time")
    })

    y = z.sum()
    start = time.time()
    y.backward()
    t = time.time() - start
    trials.append({
        'trial': trial,
        'nz': nz,
        'nbatch': nbatch,
        'cuda': cuda,
        'mode': 'cvxpylayers',
        'direction': 'backward',
        'time': t,
        'qp': 'sparse',
        'dcanon_time': solve.info.get("dcanon_time")
    })
    return trials


def prof_dense_qp(trial, nz, nbatch, cons, cuda=True):
    trials = []

    npr.seed(trial)

    L = npr.rand(nbatch, nz, nz)
    Q = np.matmul(L, L.transpose((0, 2, 1))) + 1e-3 * np.eye(nz, nz)
    p = npr.randn(nbatch, nz)

    if cons == 'dense':
        nineq = nz
        G = npr.randn(nbatch, nineq, nz)
        z0 = npr.randn(nbatch, nz)
        s0 = npr.rand(nbatch, nineq)
        h = np.matmul(G, np.expand_dims(z0, axis=(2))).squeeze(2) + s0
    elif cons == 'box':
        nineq = 2 * nz
        G = np.concatenate((-np.eye(nz), np.eye(nz)))
        G = np.stack([G] * nbatch)
        h = np.ones((nbatch, 2 * nz))
    else:
        raise NotImplementedError

    p_tch, Q_tch, G_tch, h_tch = [
        torch.from_numpy(x).double().requires_grad_()
        for x in [p, Q, G, h]
    ]
    if cuda:
        p_tch, Q_tch, G_tch, h_tch = [x.cuda()
                                      for x in [p_tch, Q_tch, G_tch, h_tch]]

    e = torch.Tensor()

    torch.cuda.synchronize()
    torch.cuda.synchronize()
    start = time.time()
    x = QPFunction(verbose=False, eps=1e-8, notImprovedLim=5,
                   maxIter=1000)(Q_tch, p_tch, G_tch, h_tch, e, e)
    torch.cuda.synchronize()
    t = time.time() - start
    trials.append({
        'trial': trial,
        'nz': nz,
        'nbatch': nbatch,
        'cuda': cuda,
        'mode': 'qpth',
        'direction': 'forward',
        'time': t,
        'qp': 'dense'
    })

    y = x.sum()
    start = time.time()
    y.backward()
    t = time.time() - start
    trials.append({
        'trial': trial,
        'nz': nz,
        'nbatch': nbatch,
        'cuda': cuda,
        'mode': 'qpth',
        'direction': 'backward',
        'time': t,
        'qp': 'dense'
    })

    _Q_sqrt = cp.Parameter((nz, nz))
    _p = cp.Parameter((nz, 1))
    _G = cp.Parameter((nineq, nz))
    _h = cp.Parameter((nineq, 1))
    _z = cp.Variable((nz, 1))
    obj = cp.Minimize(0.5 * cp.sum_squares(_Q_sqrt @ _z) + _p.T @ _z)
    cons = [_G @ _z <= _h]
    prob = cp.Problem(obj, cons)

    Q_sqrt = np.array([sqrtm(q) for q in Q])
    Q_sqrt_tch, p_tch, G_tch, h_tch = [
        torch.from_numpy(x).double().requires_grad_()
        for x in [Q_sqrt, p, G, h]]

    solver_args = {
        'mode': 'dense',
        'verbose': False,
        'max_iters': 1000,
        'eps': 1e-6,
        'use_indirect': False,
        'gpu': False,
        'n_jobs_forward': 12,
        'n_jobs_backward': 12
    }
    solve = CvxpyLayer(prob, [_Q_sqrt, _p, _G, _h], [_z])

    start = time.time()
    z, = solve(
        Q_sqrt_tch, p_tch.unsqueeze(-1), G_tch, h_tch.unsqueeze(-1),
        solver_args=solver_args
    )
    t = time.time() - start
    trials.append({
        'trial': trial,
        'nz': nz,
        'nbatch': nbatch,
        'cuda': cuda,
        'mode': 'cvxpylayers',
        'direction': 'forward',
        'time': t,
        'qp': 'dense',
        'canon_time': solve.info.get("canon_time")
    })

    y = z.sum()
    start = time.time()
    y.backward()
    t = time.time() - start
    trials.append({
        'trial': trial,
        'nz': nz,
        'nbatch': nbatch,
        'cuda': cuda,
        'mode': 'cvxpylayers',
        'direction': 'backward',
        'time': t,
        'qp': 'dense',
        'dcanon_time': solve.info.get("dcanon_time")
    })

    return trials


if __name__ == '__main__':
    main()
