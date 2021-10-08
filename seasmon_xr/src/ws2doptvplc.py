from math import log, sqrt

import numba
from numba import guvectorize, float64, int16
import numpy

from ._helper import lazycompile
from .ws2d import ws2d
from .ws2doptvp import _ws2doptvp
from .autocorr import autocorr_1d


@lazycompile(
    guvectorize(
        [(int16[:], float64, float64, float64, int16[:], float64[:])],
        "(n),(),(),() -> (n),()",
        nopython=True,
    )
)
def ws2doptvplc(y, nodata, p, lc, out, lopt):
    """Whittaker filter V-curve optimization of S, asymmetric weights and
    srange determined by autocorrelation

    Args:
        y (numpy.array): raw data array (1d, expected in float64)
        nodata (double, int): nodata value
        p (float): Envelope value for asymmetric weights
        lc (float): lag1 autocorrelation"""

    m = y.shape[0]
    w = numpy.zeros(y.shape, dtype=float64)

    n = 0
    for ii in range(m):
        if y[ii] == nodata:
            w[ii] = 0
        else:
            n += 1
            w[ii] = 1

    if n > 1:
        if lc > 0.5:
            llas = numpy.arange(-2, 1.2, 0.2, dtype=float64)
        elif lc <= 0.5:
            llas = numpy.arange(0, 3.2, 0.2, dtype=float64)
        else:
            llas = numpy.arange(-1, 1.2, 0.2, dtype=float64)

        m1 = m - 1
        m2 = m - 2
        nl = len(llas)
        nl1 = nl - 1
        i = 0
        k = 0
        j = 0
        p1 = 1 - p

        fits = numpy.zeros(nl)
        pens = numpy.zeros(nl)
        z = numpy.zeros(m)
        znew = numpy.zeros(m)
        diff1 = numpy.zeros(m1)
        lamids = numpy.zeros(nl1)
        v = numpy.zeros(nl1)
        wa = numpy.zeros(m)
        ww = numpy.zeros(m)

        # Compute v-curve
        for lix in range(nl):
            l = pow(10, llas[lix])

            for i in range(10):
                for j in range(m):
                    y_tmp = y[j]
                    z_tmp = z[j]
                    if y_tmp > z_tmp:
                        wa[j] = p
                    else:
                        wa[j] = p1
                    ww[j] = w[j] * wa[j]

                znew[:] = ws2d(y, l, ww)
                z_tmp = 0.0
                j = 0
                for j in range(m):
                    z_tmp += abs(znew[j] - z[j])

                if z_tmp == 0.0:
                    break

                z[0:m] = znew[0:m]

            for i in range(m):
                w_tmp = w[i]
                y_tmp = y[i]
                z_tmp = z[i]
                fits[lix] += pow(w_tmp * (y_tmp - z_tmp), 2)
            fits[lix] = log(fits[lix])

            for i in range(m1):
                z_tmp = z[i]
                z2 = z[i + 1]
                diff1[i] = z2 - z_tmp
            for i in range(m2):
                z_tmp = diff1[i]
                z2 = diff1[i + 1]
                pens[lix] += pow(z2 - z_tmp, 2)
            pens[lix] = log(pens[lix])

        # Construct v-curve
        llastep = llas[1] - llas[0]

        for i in range(nl1):
            l1 = llas[i]
            l2 = llas[i + 1]
            fit1 = fits[i]
            fit2 = fits[i + 1]
            pen1 = pens[i]
            pen2 = pens[i + 1]
            v[i] = sqrt(pow(fit2 - fit1, 2) + pow(pen2 - pen1, 2)) / (log(10) * llastep)
            lamids[i] = (l1 + l2) / 2

        vmin = v[k]
        for i in range(1, nl1):
            if v[i] < vmin:
                vmin = v[i]
                k = i

        lopt[0] = pow(10, lamids[k])

        z[:] = 0.0

        for i in range(10):
            for j in range(m):
                y_tmp = y[j]
                z_tmp = z[j]

                if y_tmp > z_tmp:
                    wa[j] = p
                else:
                    wa[j] = p1
                ww[j] = w[j] * wa[j]

            znew[0:m] = ws2d(y, lopt[0], ww)
            z_tmp = 0.0
            j = 0
            for j in range(m):
                z_tmp += abs(znew[j] - z[j])

            if z_tmp == 0.0:
                break

            z[0:m] = znew[0:m]

        z = ws2d(y, lopt[0], ww)
        numpy.round_(z, 0, out)

    else:
        out[:] = y[:]
        lopt[0] = 0.0


@lazycompile(numba.jit(nopython=True, parallel=True, nogil=True))
def ws2doptvplc_tyx(tyx, p, nodata):
    """Whittaker filter V-curve optimization of S, asymmetric weights and
    srange determined by autocorrelation

    Args:
        tyx (numpy.array): raw data array (int16 usually, T,Y,X axis order)
        nodata (double, int): nodata value
        p (float): Envelope value for asymmetric weights

    Returns:
       zz  smoothed version of the input data (zz.shape == tyx.shape)
       lopts optimization parameters (lopts.shape == zz.shape[1:])
    """

    nt, nr, nc = tyx.shape
    zz = numpy.zeros((nt, nr, nc), dtype=tyx.dtype)
    lopts = numpy.zeros((nr, nc), dtype="float64")

    _llas = (
        numpy.arange(-2, 1.2, 0.2, dtype=float64),  # lc > 0.5
        numpy.arange(0, 3.2, 0.2, dtype=float64),
    )  # lc <= 0.5

    for rr in numba.prange(nr):
        # needs to be here so that each thread gets it's own version
        xx = numpy.zeros(nt, dtype=float64)
        ww = numpy.zeros(nt, dtype=float64)

        for cc in range(nc):
            # Copy values into local array and convert to float64
            #
            # For now `nodata` values remain unchanged
            #
            # other option is to replace `nodata` with last observed valid value or with 0.0 if
            # none observed yet. Replaced values only impact autocorr, have no
            # impact on smoothing due to them having 0 weight.
            #
            #   w[i] = (x[i] != nodata).astype("float64")
            ngood = 0
            # last_valid_value = float64(0)
            for i in range(nt):
                v = tyx[i, rr, cc]
                if v != nodata:
                    xx[i] = v
                    ww[i] = 1
                    # last_valid_value = v
                    ngood += 1
                else:
                    # xx[i] = last_valid_value
                    xx[i] = v  # keeping original behaviour for now
                    ww[i] = 0

            if ngood > 1:
                lc = autocorr_1d(xx)
                llas = _llas[0] if lc > 0.5 else _llas[1]
                _xx, _lopts = _ws2doptvp(xx, ww, numba.float64(p), llas)
                numpy.round_(_xx, 0, zz[:, rr, cc])
                lopts[rr, cc] = _lopts

    return zz, lopts
