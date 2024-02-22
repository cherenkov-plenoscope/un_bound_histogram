##################
Un Bound Histogram
##################
|TestStatus| |PyPiStatus| |BlackStyle| |BlackPackStyle| |MITLicenseBadge|

This ``UnBoundhHistogram`` has bins with a fixed width. But if needed, it
offers an almost un-bound amount of ``2**64`` bins.
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

