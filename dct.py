import scipy
import torch
import numpy as np


def dct_and_idct(x):
    x = x.numpy()
    x_dct = scipy.fftpack.dct(x, axis=1, type=2)
    # Remove 80%
    thershold = np.quantile(np.abs(x_dct), 0.75)
    x_dct[np.abs(x_dct) < thershold] = 0
    # Remove 95%
    x_dct = torch.tensor(x_dct).to(torch.float8_e5m2)
    x_dct = x_dct.to(torch.float32).numpy()
    x_inv = scipy.fftpack.idct(x_dct, axis=1, type=2) / x_dct.shape[1] / 2
    res = torch.tensor(x_inv)
    return res


# t = torch.load("k_0.pth")
# d = dct_and_idct(t)
# for i in range(10):
    # print(d[0][i], t[0][i])