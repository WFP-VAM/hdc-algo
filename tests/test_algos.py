"""Tests for pixel alogrithms"""
# pylint: disable=no-name-in-module,redefined-outer-name,no-value-for-parameter
# pyright: reportGeneralTypeIssues=false
import numpy as np
import pytest

from hdc.algo.ops import (
    autocorr,
    autocorr_1d,
    autocorr_tyx,
    lroo,
    ws2dgu,
    ws2doptv,
    ws2doptvp,
    ws2doptvplc,
    ws2dwcv,
    ws2dwcvp,
    ws2dpgu,
)
from hdc.algo.ops.stats import (
    brentq,
    gammafit,
    gammastd,
    gammastd_yxt,
    gammastd_grp,
    mean_grp,
    mk_score,
    mk_p_value,
    mk_z_score,
    mk_variance_s,
    mk_sens_slope,
    mann_kendall_trend_1d,
    rolling_sum,
)


@pytest.fixture
def ts():
    """Testdata"""
    np.random.seed(42)
    x = np.random.gamma(1, size=10)
    return x


def test_lroo(ts):
    x_lroo = lroo(np.array((ts > 0.9) * 1, dtype="uint8"))
    assert x_lroo == 3


def pearson_reference(X, Y):
    return ((X - X.mean()) * (Y - Y.mean())).mean() / (X.std() * Y.std())


def autocorr_1d_reference(x, nodata=None):
    if nodata is not None:
        _x = x.astype("float64")
        _x[x == nodata] = np.nan
        x = _x

    X = x[:-1]
    Y = x[1:]
    if np.isnan(x).any():
        X, Y = X.copy(), Y.copy()
        X[np.isnan(X)] = np.nanmean(X)
        Y[np.isnan(Y)] = np.nanmean(Y)

    return pearson_reference(X, Y)


def test_autocorr(ts):
    ac = autocorr(ts.reshape(1, 1, -1))
    np.testing.assert_almost_equal(ac, 0.00398337)

    np.testing.assert_almost_equal(autocorr_1d.py_func(ts), 0.00398337)
    np.testing.assert_almost_equal(autocorr_1d_reference(ts), 0.00398337)
    np.testing.assert_almost_equal(autocorr_tyx(ts.reshape(-1, 1, 1)), 0.00398337)


def test_autocorr_nodata(ts_ndvi):
    nodata, ts = ts_ndvi
    rr = autocorr_1d.py_func(ts, nodata)
    rr_ref = autocorr_1d_reference(ts, nodata)
    assert rr == pytest.approx(rr_ref, rel=1e-3)


def test_brentq():
    x = brentq.py_func(
        xa=0.6446262296476516, xb=1.5041278691778537, s=0.5278852360624721
    )
    assert x == pytest.approx(1.083449238500003)


def test_gammafit(ts):
    parameters = gammafit.py_func(ts)
    assert parameters == pytest.approx((1.083449238500003, 0.9478709674697126))


def test_gammastd(ts):
    xspi = gammastd_yxt.py_func(ts.reshape(1, 1, -1))
    assert xspi.shape == (1, 1, 10)
    np.testing.assert_array_equal(
        xspi[0, 0, :],
        [-382.0, 1654.0, 588.0, 207.0, -1097.0, -1098.0, -1677.0, 1094.0, 213.0, 514.0],
    )


def test_gammastd(ts):
    xspi = gammastd.py_func(ts, 0, 10)

    np.testing.assert_array_equal(
        (xspi * 1000).round(),
        [-382.0, 1654.0, 588.0, 207.0, -1097.0, -1098.0, -1677.0, 1094.0, 213.0, 514.0],
    )


def test_gammastd_nofit(ts):
    xspi = gammastd.py_func(ts, 0, len(ts), a=1, b=2)
    xspi
    np.testing.assert_array_equal(
        (xspi * 1000).round(),
        [
            -809.0,
            765.0,
            -44.0,
            -341.0,
            -1396.0,
            -1396.0,
            -1889.0,
            343.0,
            -336.0,
            -101.0,
        ],
    )


def test_gammastd_selfit(ts):
    xspi = gammastd_yxt.py_func(ts.reshape(1, 1, -1), cal_start=0, cal_stop=3)
    assert xspi.shape == (1, 1, 10)
    np.testing.assert_array_equal(
        xspi[0, 0, :],
        [
            -1211.0,
            1236.0,
            -32.0,
            -492.0,
            -2099.0,
            -2099.0,
            -2833.0,
            572.0,
            -484.0,
            -120.0,
        ],
    )


def test_gammastd_selfit_2(ts):
    cal_start = 2
    cal_stop = 8
    a, b = gammafit.py_func(ts[cal_start:cal_stop])
    xspi_ref = gammastd.py_func(ts, cal_start=cal_start, cal_stop=cal_stop, a=a, b=b)
    xspi = gammastd.py_func(ts, cal_start=cal_start, cal_stop=cal_stop)
    np.testing.assert_equal(xspi, xspi_ref)


def test_gammastd_grp(ts):
    tts = np.repeat(ts, 5).astype("float32")
    grps = np.tile(np.arange(5), 10).astype("int16")
    xspi = gammastd_grp(tts, grps, np.unique(grps).size, 0, 10)

    res = [
        -0.38238713,
        1.6544422,
        0.5879236,
        0.20665395,
        -1.0974495,
        -1.0975538,
        -1.6773673,
        1.093621,
        0.21322519,
        0.5144766,
    ]

    np.testing.assert_almost_equal(xspi, (np.repeat(res, 5) * 1000).round())


def test_mean_grp(ts):
    tts = np.repeat(ts, 5).astype("float32")
    grps = np.tile(np.arange(5), 10).astype("int16")
    res = mean_grp(tts, grps, np.unique(grps).size, 0)
    assert (res == 1.02697).all()


def test_rolling_sum():
    res = rolling_sum(np.arange(10).astype("float32"), 3)
    np.testing.assert_almost_equal(
        res,
        np.array(
            [0.0, 0.0, 3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0], dtype="float32"
        ),
    )


def test_ws2dgu(ts):
    _ts = ts * 10
    z = ws2dgu(_ts, 10, 0)
    np.testing.assert_array_equal(z, [15, 14, 12, 9, 8, 7, 7, 9, 10, 12])


def test_ws2dpgu(ts):
    _ts = ts * 10
    z = ws2dpgu(_ts, 10, 0, 0.9)
    np.testing.assert_array_equal(z, [26, 24, 22, 20, 18, 17, 16, 15, 15, 14])


def test_ws2doptv(ts):
    _ts = ts * 10
    z, l = ws2doptv(_ts, 0, np.arange(-2, 2))
    np.testing.assert_array_equal(z, [10, 21, 16, 9, 3, 2, 5, 13, 12, 12])
    assert l == pytest.approx(0.31622776601683794)


def test_ws2doptvp(ts):
    _ts = ts * 10
    z, l = ws2doptvp(_ts, 0, 0.9, np.arange(-2, 2))
    np.testing.assert_array_equal(z, [13, 28, 19, 9, 3, 2, 7, 19, 15, 12])
    assert l == pytest.approx(0.03162277660168379)


def test_ws2doptvplc(ts):
    _ts = (ts * 10).astype("int16")
    z, l = ws2doptvplc(_ts, 0, 0.9, 0.9)
    np.testing.assert_array_equal(z, [12, 28, 19, 9, 3, 4, 13, 19, 14, 12])
    assert l == pytest.approx(0.03162277660168379)


def test_ws2dwcv(ts):
    _ts = ts * 10

    z, l = ws2dwcv(_ts, 0, np.arange(-2, 2), False)
    np.testing.assert_array_equal(z, [15, 14, 12, 9, 8, 7, 7, 9, 10, 12])
    assert l == 10.0

    z, l = ws2dwcv(_ts, 0, np.arange(-2, 2), True)
    np.testing.assert_array_equal(z, [16, 15, 12, 10, 8, 7, 8, 9, 11, 12])
    assert l == 10.0


def test_ws2dwcvp(ts):
    _ts = ts * 10
    z, l = ws2dwcvp(_ts, 0, 0.8, np.arange(-2, 2), False)
    np.testing.assert_array_equal(z, [22, 20, 18, 16, 14, 13, 13, 13, 13, 13])
    assert l == 10.0

    _ts = ts * 10
    z, l = ws2dwcvp(_ts, 0, 0.8, np.arange(-2, 2), True)
    np.testing.assert_array_equal(z, [23, 21, 19, 16, 15, 13, 13, 13, 13, 13])
    assert l == 10.0


def test_mk_score(ts):
    assert mk_score.py_func(ts) == pytest.approx((-5, -0.1111111))


def test_mk_variance_s(ts):
    assert mk_variance_s.py_func(ts) == pytest.approx(125)


def test_mk_z_score(ts):
    s, _ = mk_score.py_func(ts)
    sv = mk_variance_s.py_func(ts)
    assert mk_z_score.py_func(s, sv) == pytest.approx(-0.35777087639996635)


def test_mk_p_value(ts):
    s, _ = mk_score.py_func(ts)
    sv = mk_variance_s.py_func(ts)
    z = mk_z_score.py_func(s, sv)

    assert mk_p_value(z) == pytest.approx((0.7205147871362552, 0))


def test_mk_sens_slope(ts):
    assert mk_sens_slope.py_func(ts) == pytest.approx(
        (-0.054893050926832804, 1.1630310828723567)
    )


def test_mann_kendall_trend_1d(ts):
    assert mann_kendall_trend_1d.py_func(ts) == pytest.approx(
        (-0.1111111111111111, 0.7205147871362552, -0.054893050926832804, 0)
    )


def test_mann_kendall_trend_1d_na(ts):
    ts[-1] = np.nan
    *_, slope, _t = mann_kendall_trend_1d.py_func(ts)
    assert slope == pytest.approx(-0.06725773844053026)


def test_mann_kendall_trend_1d_trend(ts):
    *_, trend = mann_kendall_trend_1d.py_func(ts + np.arange(ts.size))
    assert trend == 1

    *_, trend = mann_kendall_trend_1d.py_func(ts - np.arange(ts.size))
    assert trend == -1
