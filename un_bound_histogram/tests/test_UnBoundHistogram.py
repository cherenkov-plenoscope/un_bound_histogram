import un_bound_histogram
import numpy as np
import pytest


def test_no_assignment():
    ubh = un_bound_histogram.UnBoundHistogram(bin_width=0.1)
    assert ubh.sum() == 0.0

    with pytest.raises(RuntimeError):
        ubh.quantile(q=0.5)

    with pytest.raises(RuntimeError):
        ubh.percentile(p=0.5)

    with pytest.raises(RuntimeError):
        ubh.modus()

    dd = ubh.to_dict()
    assert "bin_width" in dd
    assert dd["bin_width"] == ubh.bin_width

    assert "bins" in dd


def test_normal():
    prng = np.random.Generator(np.random.PCG64(9))
    SIZE = 100000
    LOC = 3.0

    ubh = un_bound_histogram.UnBoundHistogram(bin_width=0.1)

    ubh.assign(x=prng.normal(loc=LOC, scale=1.0, size=SIZE))

    assert ubh.sum() == SIZE
    assert LOC - 0.05 < ubh.quantile(q=0.5) < LOC + 0.05
    assert len(ubh.counts) > 50


def test_assignment():
    ubh = un_bound_histogram.UnBoundHistogram(bin_width=1.0)

    ubh.assign(x=0)
    assert ubh.counts[0] == 1

    ubh.assign(x=0.001)
    assert ubh.counts[0] == 2

    ubh.assign(x=0.5)
    assert ubh.counts[0] == 3

    ubh.assign(x=0.999)
    assert ubh.counts[0] == 4

    ubh.assign(x=1.0)
    assert ubh.counts[0] == 4
    assert ubh.counts[1] == 1


def test_single_bin():
    ubh = un_bound_histogram.UnBoundHistogram(bin_width=1.0)

    ubh.assign(x=0.5 * np.ones(100))
    assert ubh.sum() == 100
    assert ubh.counts[0] == 100
    assert ubh.modus() == 0.5
    assert ubh.quantile(q=0.5) == 0.5


def test_gap():
    ubh = un_bound_histogram.UnBoundHistogram(bin_width=1.0)

    ubh.assign(x=0.5 * np.ones(100))
    ubh.assign(x=2.5 * np.ones(100))
    assert ubh.sum() == 200
    assert ubh.counts[0] == 100
    assert ubh.counts[2] == 100

    assert ubh.modus() == 0.5

    np.testing.assert_almost_equal(ubh.quantile(q=0.5), 1.0)


def test_to_array():
    prng = np.random.Generator(np.random.PCG64(9))
    SIZE = 10000
    LOC = 3.0

    ubh = un_bound_histogram.UnBoundHistogram(bin_width=0.1)
    ubh.assign(x=prng.normal(loc=LOC, scale=1.0, size=SIZE))

    bins, counts = ubh.to_array()

    assert len(bins) == len(counts)
    assert len(bins) == len(ubh.counts)
    assert SIZE == np.sum(counts)
    for i in range(len(bins)):
        b = bins[i]
        c = counts[i]
        assert b in ubh.counts
        assert ubh.counts[b] == c
