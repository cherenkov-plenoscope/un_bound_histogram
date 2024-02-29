from .version import __version__
from . import bins_utils
import numpy as np


class UnBoundHistogram:
    """
    A histogram with a fix bin_width that is sparse and grows dynamically.
    Bins with zero counts are not allocated.
    """

    def __init__(self, bin_width):
        assert bin_width > 0
        self.bin_width = bin_width
        self.bins = {}

    def assign(self, x):
        """
        Assign value x to the bins of the histogram.
        The value of the bin will be raised by one.

        Parameters
        ----------
        x : numpy.array of float or int
            Value.
        """
        xb = x / self.bin_width
        xb = np.floor(xb).astype(int)
        unique, counts = np.unique(xb, return_counts=True)
        bins = dict(zip(unique, counts))
        for key in bins:
            if key in self.bins:
                self.bins[key] += bins[key]
            else:
                self.bins[key] = bins[key]

    def sum(self):
        """
        Returns
        -------
        sum : int
            The sum of all content in all bins.
        """
        return bins_utils.sum(bins=self.bins)

    def argmax(self):
        """
        Returns
        -------
        argmax : int
            The key of the bin with the largest content.
        """
        return bins_utils.argmax(bins=self.bins)

    def range(self):
        """
        Returns
        -------
        range : (int, int)
            The key with the lowest value and the key with the highest value.
        """
        return bins_utils.range(bins=self.bins)

    def modus(self):
        if len(self.bins) == 0:
            raise RuntimeError("No values in bins yet.")

        modus_key = self.argmax()
        return (modus_key + 0.5) * self.bin_width

    def quantile(self, q):
        if len(self.bins) == 0:
            raise RuntimeError("No values in bins yet.")
        assert 0 <= q <= 1.0
        total = self.sum()
        target = total * q
        sorted_keys = sorted(self.bins.keys())
        part = 0
        for key in sorted_keys:
            if part + self.bins[key] < target:
                part += self.bins[key]
            else:
                break
        missing = target - part
        assert missing <= self.bins[key]
        bin_frac = missing / self.bins[key]
        bin_center = key
        bin_quantile = bin_center + bin_frac
        quantile = bin_quantile * self.bin_width
        return quantile

    def percentile(self, p):
        return self.quantile(q=p * 1e-2)

    def to_dict(self):
        return {"bin_width": self.bin_width, "bins": self.bins}

    def __repr__(self):
        out = "{:s}(bin_width={:f})".format(
            self.__class__.__name__, self.bin_width
        )
        return out


class UnBoundHistogram2d:
    """
    A 2d histogram with a fix bin_width that is sparse and grows dynamically.
    Bins with zero counts are not allocated.
    """

    def __init__(self, x_bin_width, y_bin_width):
        """
        Parameters
        ----------
        x_bin_width : float
            Width of bins on x axis.
        y_bin_width : float
            Width of bins on y axis.
        """
        assert x_bin_width > 0
        assert y_bin_width > 0
        self.x_bin_width = x_bin_width
        self.y_bin_width = y_bin_width
        self.bins = {}

    def assign(self, x, y):
        """
        Assign value pair (x, y) to the bins of the histogram.
        The value of the bin will be raised by one.

        Parameters
        ----------
        x : numpy.array of float or int
            X component of value.
        y : numpy.array of float or int
            Y component of value.
        """
        xb = x / self.x_bin_width
        yb = y / self.y_bin_width

        xb = np.floor(xb).astype(np.int32)
        yb = np.floor(yb).astype(np.int32)

        wb = _x_y_to_w(x=xb, y=yb)
        wunique, wcounts = np.unique(wb, return_counts=True)
        xunique, yunique = _w_to_x_y(w=wunique)
        for i in range(len(wunique)):
            xkey = xunique[i]
            ykey = yunique[i]
            xykey = (xkey, ykey)
            if xykey in self.bins:
                self.bins[xykey] += wcounts[i]
            else:
                self.bins[xykey] = wcounts[i]

    def sum(self):
        """
        Returns
        -------
        sum : int
            The sum of all content in all bins.
        """
        return bins_utils.sum(bins=self.bins)

    def argmax(self):
        """
        Returns
        -------
        argmax : int
            The key of the bin with the largest content.
        """
        return bins_utils.argmax(bins=self.bins)

    def range(self):
        return bins_utils.range2d(bins=self.bins)

    def to_dict(self):
        return {
            "x_bin_width": self.x_bin_width,
            "y_bin_width": self.x_bin_width,
            "bins": self.bins,
        }

    def __repr__(self):
        out = "{:s}(bin_width={:f})".format(
            self.__class__.__name__, self.bin_width
        )
        return out


def _x_y_to_w(x, y):
    assert np.all(np.iinfo(np.int32).min <= x)
    assert np.all(x <= np.iinfo(np.int32).max)
    assert np.all(np.iinfo(np.int32).min <= y)
    assert np.all(y <= np.iinfo(np.int32).max)

    bb = np.c_[x, y].astype(np.int32).tobytes()
    return np.frombuffer(bb, dtype=np.uint64)


def _w_to_x_y(w):
    bb = w.tobytes()
    buff = np.frombuffer(bb, dtype=np.int32)
    xy = buff.reshape((buff.shape[0] // 2, 2))
    return xy[:, 0], xy[:, 1]
