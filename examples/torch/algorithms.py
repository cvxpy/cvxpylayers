import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

torch.set_default_dtype(torch.double)


def fit(loss, params, X, Y, Xval, Yval, opt, opt_kwargs={"lr":1e-3}, batch_size=128, epochs=100, verbose=False, callback=None):
    """

    Arguments:
        loss: given x and y in batched form, evaluates loss.
        params: list of parameters to optimize.
        X: input data, torch tensor.
        Y: output data, torch tensor.
        Xval: input validation data, torch tensor.
        Yval: output validation data, torch tensor.
    """

    train_dset = TensorDataset(X, Y)
    train_loader = DataLoader(train_dset, batch_size=batch_size, shuffle=True)
    opt = opt(params, **opt_kwargs)

    train_losses = []
    val_losses = []
    for epoch in range(epochs):
        with torch.no_grad():
            val_losses.append(loss(Xval, Yval).item())
        if verbose:
            print("%03d | %3.5f" % (epoch + 1, val_losses[-1]))
        batch = 1
        train_losses.append([])
        for Xbatch, Ybatch in train_loader:
            opt.zero_grad()
            l = loss(Xbatch, Ybatch)
            l.backward()
            opt.step()
            train_losses[-1].append(l.item())
            if verbose:
                print("batch %03d / %03d | %3.5f" %
                      (batch, len(train_loader), np.mean(train_losses[-1])))
            batch += 1
            if callback is not None:
                callback()
    return val_losses, train_losses
