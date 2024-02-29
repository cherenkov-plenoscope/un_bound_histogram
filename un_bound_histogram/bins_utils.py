import numpy as np


def argmax(bins):
    if len(bins) == 0:
        raise RuntimeError("No values in bins yet.")
    m = 0
    max_key = -1
    for key in bins:
        c = bins[key]
        if c > m:
            max_key = key
            m = c
    return max_key


def range(bins):
    return (min(bins.keys()), max(bins.keys()))


def sum(bins):
    total = 0
    for key in bins:
        total += bins[key]
    return total


def range2d(bins):
    kk = np.array(list(bins.keys()))
    xk = kk[:, 0]
    yk = kk[:, 1]
    x_range = (min(xk), max(xk))
    y_range = (min(yk), max(yk))
    return (x_range, y_range)
