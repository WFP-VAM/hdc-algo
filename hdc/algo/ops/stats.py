"""Numba accelerated statistical funtions."""
from math import erf, log, sqrt

import numba
import numba.core.types as nt
import numba_scipy  # pylint: disable=unused-import
import numpy as np
import scipy.special as sc

from ._helper import lazycompile


@numba.njit
def brentq(xa, xb, s):
    """
    Root finding optimization using Brent's method.

    adapted from:

    https://github.com/scipy/scipy/blob/f2ef65dc7f00672496d7de6154744fee55ef95e9/scipy/optimize/Zeros/brentq.c#L37
    Copyright (c) 2001-2002 Enthought, Inc.  2003-2019, SciPy Developers.
    All rights reserved.
    """
    xpre = xa
    xcur = xb
    xblk = 0.0
    fblk = 0.0
    spre = 0.0
    scur = 0.0
    maxiter = 100
    xtol = 2e-12
    rtol = 8.881784197001252e-16

    func = lambda a: log(a) - sc.digamma(a) - s

    fpre = func(xpre)
    fcur = func(xcur)

    if (fpre * fcur) > 0:
        return 0.0

    if fpre == 0:
        return xpre

    if fcur == 0:
        return xcur

    iterations = 0

    for _ in range(maxiter):
        iterations += 1

        if (fpre * fcur) < 0:
            xblk = xpre
            fblk = fpre
            spre = scur = xcur - xpre

        if abs(fblk) < abs(fcur):
            xpre, xcur = xcur, xblk
            xblk = xpre

            fpre, fcur = fcur, fblk
            fblk = fpre

        delta = (xtol + rtol * abs(xcur)) / 2
        sbis = (xblk - xcur) / 2

        if fcur == 0 or (abs(sbis) < delta):
            return xcur

        if (abs(spre) > delta) and (abs(fcur) < abs(fpre)):
            if xpre == xblk:
                # interpolate
                stry = -fcur * (xcur - xpre) / (fcur - fpre)
            else:
                # extrapolate
                dpre = (fpre - fcur) / (xpre - xcur)
                dblk = (fblk - fcur) / (xblk - xcur)
                stry = (
                    -fcur * (fblk * dblk - fpre * dpre) / (dblk * dpre * (fblk - fpre))
                )

            if (2 * abs(stry)) < min(abs(spre), 3 * abs(sbis) - delta):
                # good short step
                spre = scur
                scur = stry
            else:
                # bisect
                spre = sbis
                scur = sbis

        else:
            # bisect
            spre = sbis
            scur = sbis

        xpre = xcur
        fpre = fcur

        if abs(scur) > delta:
            xcur += scur
        else:
            xcur += delta if sbis > 0 else -delta

        fcur = func(xcur)

    return xcur


@numba.njit
def gammafit(x):
    """
    Calculate gamma distribution parameters for timeseries.

    Adapted from:
    https://github.com/scipy/scipy/blob/f2ef65dc7f00672496d7de6154744fee55ef95e9/scipy/stats/_continuous_distns.py#L2554
    Copyright (c) 2001-2002 Enthought, Inc.  2003-2019, SciPy Developers.
    All rights reserved.
    """
    n = 0
    xts = 0
    logs = 0

    for xx in x:
        if xx > 0:
            xts += xx
            logs += log(xx)
            n += 1

    if n == 0:
        return (0, 0)

    xtsbar = xts / n
    s = log(xtsbar) - (logs / n)

    if s == 0:
        return (0, 0)

    a_est = (3 - s + sqrt((s - 3) ** 2 + 24 * s)) / (12 * s)
    xa = a_est * (1 - 0.4)
    xb = a_est * (1 + 0.4)
    a = brentq(xa, xb, s)
    if a == 0:
        return (0, 0)

    b = xtsbar / a

    return (a, b)


@numba.njit
def spifun(x, a=None, b=None, cal_start=None, cal_stop=None):
    """Calculate SPI with gamma distribution for 3d array."""
    y = np.full_like(x, -9999)
    r, c, t = x.shape

    if cal_start is None:
        cal_start = 0

    if cal_stop is None:
        cal_stop = t

    cal_ix = np.arange(cal_start, cal_stop)

    for ri in range(r):
        for ci in range(c):
            xt = x[ri, ci, :]
            valid_ix = []

            for tix in range(t):
                if xt[tix] > 0:
                    valid_ix.append(tix)

            n_valid = len(valid_ix)

            p_zero = 1 - (n_valid / t)

            if p_zero > 0.9:
                y[ri, ci, :] = -9999
                continue

            if a is None or b is None:
                alpha, beta = gammafit(xt[cal_ix])
            else:
                alpha, beta = (a, b)

            if alpha == 0 or beta == 0:
                y[ri, ci, :] = -9999
                continue

            spi = np.full(t, p_zero, dtype=nt.float64)  # type: ignore

            for tix in valid_ix:
                spi[tix] = p_zero + (
                    (1 - p_zero)
                    * sc.gammainc(alpha, xt[tix] / beta)  # pylint: disable=no-member
                )

            for tix in range(t):
                spi[tix] = sc.ndtri(spi[tix]) * 1000

            np.round_(spi, 0, spi)

            y[ri, ci, :] = spi[:]

    return y


@numba.njit
def mk_score(x):
    """Calculate MK score (S) and Kendall's Tau

    https://vsp.pnnl.gov/help/vsample/design_trend_mann_kendall.htm

    Hussain et al., (2019). pyMannKendall: a python package for non parametric Mann Kendall family of trend tests..
    Journal of Open Source Software, 4(39), 1556, https://doi.org/10.21105/joss.01556
    """
    n = len(x)
    _s1 = 0
    _s2 = 0

    for k in range(0, n - 1):
        for kk in range(k + 1, n):
            if x[kk] > x[k]:
                _s1 += 1
            if x[kk] < x[k]:
                _s2 += 1

    s = _s1 - _s2
    tau = s / (0.5 * n * (n - 1))
    return s, tau


@numba.njit
def mk_variance_s(x):
    """Mann-Kendall's variance S

    https://vsp.pnnl.gov/help/vsample/design_trend_mann_kendall.htm

    Hussain et al., (2019). pyMannKendall: a python package for non parametric Mann Kendall family of trend tests..
    Journal of Open Source Software, 4(39), 1556, https://doi.org/10.21105/joss.01556
    """
    xu = np.unique(x)
    n = len(x)

    if len(xu) == n:
        return (n * (n - 1) * (2 * n + 5)) / 18

    tp = 0
    for i in range(len(xu)):
        _tp = 0
        for ii in range(n):
            if xu[i] == x[ii]:
                _tp += 1

        tp += _tp * (_tp - 1) * (2 * _tp + 5)

    return ((n * (n - 1) * (2 * n + 5)) - tp) / 18


@numba.njit
def mk_z_score(s, vs):
    """Calculate Mann-Kendall's Z (MKZ)

    https://vsp.pnnl.gov/help/vsample/design_trend_mann_kendall.htm

    Hussain et al., (2019). pyMannKendall: a python package for non parametric Mann Kendall family of trend tests..
    Journal of Open Source Software, 4(39), 1556, https://doi.org/10.21105/joss.01556
    """
    if s > 0:
        return (s - 1) / sqrt(vs)

    if s < 0:
        return (s + 1) / sqrt(vs)

    return 0


@numba.njit
def mk_p_value(z, alpha=0.05):
    """Calculate Mann-Kendall's p value and significance

    https://vsp.pnnl.gov/help/vsample/design_trend_mann_kendall.htm

    Hussain et al., (2019). pyMannKendall: a python package for non parametric Mann Kendall family of trend tests..
    Journal of Open Source Software, 4(39), 1556, https://doi.org/10.21105/joss.01556
    """
    p = 2 * (1 - (0.5 * (1.0 + erf(abs(z) * sqrt(0.5)))))
    h = int(abs(z) > sc.ndtri(1 - alpha / 2))

    return p, h


@numba.njit
def mk_sens_slope(x):
    """Calculate Sen's slope and intercept for Mann-Kendall test

    Hussain et al., (2019). pyMannKendall: a python package for non parametric Mann Kendall family of trend tests..
    Journal of Open Source Software, 4(39), 1556, https://doi.org/10.21105/joss.01556
    """
    ix = 0
    n = x.size
    nd = int(n * (n - 1) / 2)
    d = np.ones(nd)
    for i in range(n - 1):
        for j in range(i + 1, n):
            d[ix] = (x[j] - x[i]) / (j - i)
            ix += 1

    slope = np.median(d)
    intercept = np.median(x) - (n - 1) / 2 * slope

    return slope, intercept


@numba.njit
def mann_kendall_trend_yxt(x):
    """Calculate Mann-Kendall trend over y, x, t array

    This function calculates MK tend for each pixel over a
    3-d y, x, t array and returns an array of shape (y, x, 4)
    containing Kendall's Tau, P value, Sen's slope and a trend indicator
    in the last array dimension.
    The trend indicator can be -1, 0 or +1 and signifies a significand decresing trend,
    no (significant) trend or a significant upward trend, respectively.
    """
    ys, xs, ts = x.shape

    r = np.zeros(shape=(ys, xs, 4), dtype="float32")

    for yix in range(ys):
        for xix in range(xs):
            pass
            s, tau = mk_score(x[yix, xix, 0:ts])
            sv = mk_variance_s(x[yix, xix, 0:ts])
            z = mk_z_score(s, sv)
            p, h = mk_p_value(z)
            slope, _ = mk_sens_slope(x[yix, xix, 0:ts])

            if not h:
                trend = 0
            else:
                if z > 0:
                    trend = 1
                if z < 0:
                    trend = -1

            r[yix, xix, 0] = tau
            r[yix, xix, 1] = p
            r[yix, xix, 2] = slope
            r[yix, xix, 3] = trend

    return r


@numba.njit
def mann_kendall_trend_1d(x):
    """Caluclate Mann-Kendall trend metric on 1-d array

    This function calculates MK tend metrics (Kendall's Tau, P value, Sen's slope and a trend indicator).
    The trend indicator can be -1, 0 or +1 and signifies a significand decresing trend,
    no (significant) trend or a significant upward trend, respectively.
    """
    s, tau = mk_score(x)
    sv = mk_variance_s(x)
    z = mk_z_score(s, sv)
    p, h = mk_p_value(z)
    slope, _ = mk_sens_slope(x)

    if not h:
        return tau, p, slope, 0

    if z > 0:
        trend = 1
    elif z < 0:
        trend = -1
    else:
        trend = 0

    return tau, p, slope, trend


@lazycompile(
    numba.guvectorize(
        [
            "(int16[:], float64, float32[:], float32[:], float32[:], int8[:])",
            "(float32[:], float64, float32[:], float32[:], float32[:], int8[:])",
        ],
        "(n),() -> (),(),(),()",
        nopython=True,
    )
)
def _mann_kendall_trend_gu_nd(x, nodata, tau, p, slope, trend):
    """Guvectorize wrapper for mann_kendall_trend_1d with nodata"""
    if (x != nodata).any():
        tau[0], p[0], slope[0], trend[0] = mann_kendall_trend_1d(x)

    else:
        tau[0] = nodata
        p[0] = nodata
        slope[0] = nodata
        trend[0] = 0


@lazycompile(
    numba.guvectorize(
        [
            "(int16[:], float32[:], float32[:], float32[:], int8[:])",
            "(float32[:], float32[:], float32[:], float32[:], int8[:])",
        ],
        "(n) -> (),(),(),()",
        nopython=True,
    )
)
def _mann_kendall_trend_gu(x, tau, p, slope, trend):
    """Guvectorize wrapper for mann_kendall_trend_1d"""
    tau[0], p[0], slope[0], trend[0] = mann_kendall_trend_1d(x)
