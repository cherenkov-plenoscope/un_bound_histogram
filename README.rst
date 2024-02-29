##################
Un Bound Histogram
##################
|TestStatus| |PyPiStatus| |BlackStyle| |BlackPackStyle| |MITLicenseBadge|

This ``UnBoundhHistogram`` has bins with a fixed width. It is sparse and thus
does not allocate memory for bins with zero content. It's range is almost
un-bound (integer limits). Bins are allocated and populated as needed during
assignment.
Making a histogram in an almost un bound range is usefule when one does not
know the range of the data in advance and when streaming thrhough the data is
costly. ``UnBoundhHistogram`` was created to histogram vast streams of data
generated in costly simulations for particle physics.
Buzz word bingo: big data.

*******
Install
*******

.. code-block:: bash

    pip install un_bound_histogram


*****
Usage
*****
.. code-block:: python

    import un_bound_histogram
    import numpy

    prng = numpy.random.Generator(numpy.random.PCG64(1337))

    h = un_bound_histogram.UnBoundHistogram(bin_width=0.1)

    h.assign(x=prng.normal(loc=5.0, scale=2.0, size=1000000))

    # assign multiple times to grow the histogram.
    h.assign(x=prng.normal(loc=-3.0, scale=1.0, size=1000000))
    h.assign(x=prng.normal(loc=1.0, scale=0.5, size=1000000))

    assert 0.9 < h.percentile(50) < 1.1
    assert h.sum() == 3 * 1000000


The ``UnBoundHistogram`` has a few statistical estimators built in, such as
``modus()`` and ``quantile()/percentile()``.

There is also a two dimensional implementation ``UnBoundHistogram2d``. See
tests for examples.

.. code-block:: python

    import un_bound_histogram
    import numpy as np

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


.. |TestStatus| image:: https://github.com/cherenkov-plenoscope/un_bound_histogram/actions/workflows/test.yml/badge.svg?branch=main
    :target: https://github.com/cherenkov-plenoscope/un_bound_histogram/actions/workflows/test.yml

.. |PyPiStatus| image:: https://img.shields.io/pypi/v/un_bound_histogram
    :target: https://pypi.org/project/un_bound_histogram

.. |BlackStyle| image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black

.. |BlackPackStyle| image:: https://img.shields.io/badge/pack%20style-black-000000.svg
    :target: https://github.com/cherenkov-plenoscope/black_pack

.. |MITLicenseBadge| image:: https://img.shields.io/badge/License-MIT-yellow.svg
    :target: https://opensource.org/licenses/MIT

