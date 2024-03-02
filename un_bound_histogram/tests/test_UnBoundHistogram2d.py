import un_bound_histogram
import numpy as np
import pytest


def test_no_assignment():
    ubh = un_bound_histogram.UnBoundHistogram2d(
        x_bin_width=0.1,
        y_bin_width=0.1,
    )
    assert ubh.sum() == 0.0

    dd = ubh.to_dict()
    assert dd["x_bin_width"] == ubh.x_bin_width
    assert dd["y_bin_width"] == ubh.y_bin_width

    assert "bins" in dd


def test_x_y_to_w():
    prng = np.random.Generator(np.random.PCG64(12))

    for i in range(100):
        NUM = prng.integers(low=1, high=100)

        x = prng.integers(
            low=np.iinfo(np.int32).min,
            high=np.iinfo(np.int32).max,
            size=NUM,
        )
        y = prng.integers(
            low=np.iinfo(np.int32).min,
            high=np.iinfo(np.int32).max,
            size=NUM,
        )

        w = un_bound_histogram._x_y_to_w(x=x, y=y)
        xb, yb = un_bound_histogram._w_to_x_y(w=w)

        np.testing.assert_array_equal(x, xb)
        np.testing.assert_array_equal(y, yb)


def test_normal():
    prng = np.random.Generator(np.random.PCG64(9))
    SIZE = 100000
    XLOC = 3.0
    YLOC = -4.5

    ubh = un_bound_histogram.UnBoundHistogram2d(
        x_bin_width=0.1,
        y_bin_width=0.1,
    )

    ubh.assign(
        x=prng.normal(loc=XLOC, scale=1.0, size=SIZE),
        y=prng.normal(loc=YLOC, scale=1.0, size=SIZE),
    )

    xb_max, yb_max = ubh.argmax()
    x_max = xb_max * ubh.x_bin_width
    y_max = yb_max * ubh.y_bin_width

    assert XLOC - 0.5 < x_max < XLOC + 0.5
    assert YLOC - 0.5 < y_max < YLOC + 0.5

    x_range, y_range = ubh.range()

    assert x_range[0] <= xb_max <= x_range[1]
    assert y_range[0] <= yb_max <= y_range[1]

    assert ubh.sum() == SIZE


def test_to_array():
    prng = np.random.Generator(np.random.PCG64(9))
    SIZE = 100000
    XLOC = 3.0
    YLOC = -4.5

    ubh = un_bound_histogram.UnBoundHistogram2d(
        x_bin_width=0.1,
        y_bin_width=0.1,
    )

    ubh.assign(
        x=prng.normal(loc=XLOC, scale=1.0, size=SIZE),
        y=prng.normal(loc=YLOC, scale=1.0, size=SIZE),
    )

    xbins, ybins, counts = ubh.to_array()

    assert len(xbins) == len(ybins)
    assert len(xbins) == len(counts)
    assert len(xbins) == len(ubh.counts)
    assert SIZE == np.sum(counts)
    for i in range(len(xbins)):
        xb = xbins[i]
        yb = ybins[i]
        c = counts[i]
        assert (xb, yb) in ubh.counts
        assert ubh.counts[(xb, yb)] == c
