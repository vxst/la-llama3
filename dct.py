import scipy
import torch
import numpy as np


def k_compress_uncompress(x):
    x = x.numpy()
    x_dct = scipy.fftpack.dct(x, axis=1, type=2)
    # Remove 75%
    thershold = np.quantile(np.abs(x_dct), 0.75)
    x_dct[np.abs(x_dct) < thershold] = 0
    # to 2 bit!
    x_dct = torch.tensor(x_dct).to(torch.float8_e5m2)
    x_dct = x_dct.to(torch.float32).numpy()
    x_inv = scipy.fftpack.idct(x_dct, axis=1, type=2) / x_dct.shape[1] / 2
    res = torch.tensor(x_inv)
    return res

def v_compress_uncompress(x):
    x = x.numpy()
    x_dct_x = scipy.fftpack.dct(x, axis=1, type=4)
    x_dct_s = np.concatenate([x_dct_x[:, :2048], x_dct_x[:, 3072:]], axis=1)
    # Remove 50%
    thershold = np.quantile(np.abs(x_dct_s), 0.5)
    x_dct_s[np.abs(x_dct_s) < thershold] = 0
    # to 4 bit!
    x_dct_s = torch.tensor(x_dct_s).to(torch.float8_e5m2)
    x_dct_s = x_dct_s.to(torch.float32).numpy()
    x_dct = np.zeros_like(x)
    x_dct[:, :2048] = x_dct_s[:, :2048]
    x_dct[:, 3072:] = x_dct_s[:, 2048:]
    x_inv = scipy.fftpack.idct(x_dct, axis=1, type=4) / x_dct.shape[1] / 2
    res = torch.tensor(x_inv)
    return res

# t = torch.load("k_0.pth")
# d = dct_and_idct(t)
# for i in range(10):
    # print(d[0][i], t[0][i])
