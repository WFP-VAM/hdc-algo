"""Tests for xarray accessors"""

# pylint: disable=redefined-outer-name,unused-import,no-member,no-name-in-module,missing-function-docstring
# pyright: reportGeneralTypeIssues=false

import dask.array as da
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from hdc.algo import ops
from hdc.algo.accessors import MissingTimeError
from hdc.algo.ops.ws2d import ws2d


def to_da(xx):
    return xr.DataArray(
        data=da.from_array(xx.data),
        dims=xx.dims,
        coords=xx.coords,
        attrs=xx.attrs,
    )


@pytest.fixture
def darr():
    np.random.seed(42)
    x = xr.DataArray(
        np.random.randint(1, 100, (5, 2, 2)),
        dims=("time", "y", "x"),
        name="band",
        attrs={"nodata": -9999},
    )
    x["time"] = pd.date_range(start="2000-01-01", periods=5, freq="10D")

    return x


@pytest.fixture
def res_spi():
    return np.array(
        [
            [[217, 397, 644, -1947, 847], [1401, -979, 979, -920, -508]],
            [[-1673, 943, 1061, 116, -424], [988, 1178, 103, -1139, -1139]],
        ]
    )


@pytest.fixture
def zones():
    x = xr.DataArray([[0, 1], [0, 1]], dims=("y", "x"), name="band", attrs={"nodata": -1})

    return x


def test_period_years_dekad(darr):
    np.testing.assert_array_equal(darr.time.dekad.year, darr.time.dt.year)


def test_period_months_dekad(darr):
    np.testing.assert_array_equal(darr.time.dekad.month, darr.time.dt.month)


def test_period_idx_dekad(darr):
    assert isinstance(darr.time.dekad.idx, xr.DataArray)
    np.testing.assert_array_equal(darr.time.dekad.idx, [1, 2, 3, 3, 1])


def test_period_yidx_dekad(darr):
    assert isinstance(darr.time.dekad.yidx, xr.DataArray)
    np.testing.assert_array_equal(darr.time.dekad.yidx, [1, 2, 3, 3, 4])


def test_period_yidx_linspace_dekad(darr):
    assert isinstance(darr.time.dekad.linspace, xr.DataArray)
    np.testing.assert_array_equal(darr.time.dekad.linspace, [0, 1, 2, 2, 3])


def test_period_ndays_dekad(darr):
    assert isinstance(darr.time.dekad.ndays, xr.DataArray)
    np.testing.assert_equal(darr.time.dekad.ndays.values, np.array([10, 10, 11, 11, 10]))


def test_period_start_date_dekad(darr):
    assert isinstance(darr.time.dekad.start_date, xr.DataArray)
    np.testing.assert_equal(
        darr.time.dekad.start_date.values,
        np.array(
            [
                "2000-01-01T00:00:00.000000000",
                "2000-01-11T00:00:00.000000000",
                "2000-01-21T00:00:00.000000000",
                "2000-01-21T00:00:00.000000000",
                "2000-02-01T00:00:00.000000000",
            ],
            dtype="datetime64[ns]",
        ),
    )


def test_period_end_date_dekad(darr):
    assert isinstance(darr.time.dekad.end_date, xr.DataArray)
    np.testing.assert_equal(
        darr.time.dekad.end_date.values,
        np.array(
            [
                "2000-01-10T23:59:59.999999000",
                "2000-01-20T23:59:59.999999000",
                "2000-01-31T23:59:59.999999000",
                "2000-01-31T23:59:59.999999000",
                "2000-02-10T23:59:59.999999000",
            ],
            dtype="datetime64[ns]",
        ),
    )


def test_period_raw_dekad(darr):
    assert isinstance(darr.time.dekad.raw, xr.DataArray)
    np.testing.assert_array_equal(darr.time.dekad.raw, [72000, 72001, 72002, 72002, 72003])


def test_period_labels_dekad(darr):
    assert isinstance(darr.time.dekad.label, xr.DataArray)
    np.testing.assert_array_equal(
        darr.time.dekad.label,
        ["200001d1", "200001d2", "200001d3", "200001d3", "200002d1"],
    )


def test_period_labels_dekad_single(darr):
    np.testing.assert_array_equal(darr.isel(time=0).time.dekad.label, ["200001d1"])


def test_period_exception(darr):
    with pytest.raises(TypeError):
        _ = darr.x.dekad


def test_season_label(darr, season_ranges):
    assert isinstance(darr.time.season.label(season_ranges), xr.DataArray)
    np.testing.assert_array_equal(
        darr.time.season.label(season_ranges), [200001, 200001, 200001, 200001, 200002]
    )


def test_season_idx(darr, season_ranges):
    assert isinstance(darr.time.season.idx(season_ranges), xr.DataArray)
    np.testing.assert_array_equal(darr.time.season.idx(season_ranges), [1, 1, 1, 1, 2])


def test_season_ndays(darr, season_ranges):
    assert isinstance(darr.time.season.ndays(season_ranges), xr.DataArray)
    np.testing.assert_equal(darr.time.season.ndays(season_ranges).values, np.array([31, 31, 31, 31, 29]))


def test_season_start_date(darr, season_ranges):
    assert isinstance(darr.time.season.start_date(season_ranges), xr.DataArray)
    np.testing.assert_equal(
        darr.time.season.start_date(season_ranges).values,
        np.array(
            ["2000-01-01T00:00:00.000000000"] * 4 + ["2000-02-01T00:00:00.000000000"],
            dtype="datetime64[ns]",
        ),
    )


def test_season_end_date(darr, season_ranges):
    assert isinstance(darr.time.season.end_date(season_ranges), xr.DataArray)
    np.testing.assert_equal(
        darr.time.season.end_date(season_ranges).values,
        np.array(
            ["2000-01-31T23:59:59.999999000"] * 4 + ["2000-02-29T23:59:59.999999000"],
            dtype="datetime64[ns]",
        ),
    )


def test_season_exception(darr, season_ranges):
    with pytest.raises(TypeError):
        _ = darr.x.season.label(season_ranges)


def test_algo_lroo(darr):
    _res = np.array([[3, 0], [4, 2]], dtype="uint8")

    darr_lroo = ((darr > 30) * 1).astype("uint8").hdc.algo.lroo()
    assert isinstance(darr_lroo, xr.DataArray)
    np.testing.assert_array_equal(darr_lroo, _res)


def test_algo_croo(darr):
    _res = np.array([[1, 0], [4, 0]], dtype="uint8")

    darr_croo = ((darr > 30) * 1).hdc.algo.croo()
    assert isinstance(darr_croo, xr.DataArray)
    np.testing.assert_array_equal(darr_croo, _res)


def test_anom_ratio(darr):
    _res = np.array([[85.0, 443.0], [18.0, 83.0]])

    np.testing.assert_array_equal(darr.isel(time=0).hdc.anom.ratio(darr.isel(time=1)).round(), _res)


def test_anom_diff(darr):
    _res = np.array([[-9, 72], [-68, -15]])

    np.testing.assert_array_equal(darr.isel(time=0).hdc.anom.diff(darr.isel(time=1)), _res)


def test_algo_autocorr(darr):
    _res = np.array([[-0.7395127, -0.69477093], [-0.1768683, 0.6412078]], dtype="float32")
    darr_autocorr = darr.hdc.algo.autocorr()
    assert isinstance(darr_autocorr, xr.DataArray)
    np.testing.assert_almost_equal(darr_autocorr, _res)


def test_algo_autocorr_da(darr):
    darr_da = to_da(darr)
    _ = darr_da.hdc.algo.autocorr()

    darr_da_t = darr_da.transpose(..., "time")
    _ = darr_da_t.hdc.algo.autocorr()


def test_algo_spi(darr, res_spi):
    _res = darr.hdc.algo.spi()
    assert isinstance(_res, xr.DataArray)
    np.testing.assert_array_equal(_res, res_spi)


def test_algo_spi_grouped(darr, res_spi):
    _res = darr.astype("float32").hdc.algo.spi(groups=np.array([0] * 5, dtype="int16"))
    assert isinstance(_res, xr.DataArray)
    np.testing.assert_array_equal(_res, res_spi)


def test_algo_spi_grouped_2(darr, res_spi):
    _res = darr.astype("float32").hdc.algo.spi(groups=["D1", "D1", "D1", "D1", "D1"])
    assert isinstance(_res, xr.DataArray)
    np.testing.assert_array_equal(_res, res_spi)


def test_algo_spi_transp(darr, res_spi):
    _darr = darr.transpose(..., "time")
    _res = _darr.hdc.algo.spi()
    assert isinstance(_res, xr.DataArray)
    np.testing.assert_array_equal(_res, res_spi)


def test_algo_spi_attrs_default(darr):
    _res = darr.hdc.algo.spi()
    assert _res.attrs["spi_calibration_begin"] == str(darr.time.to_index()[0])
    assert _res.attrs["spi_calibration_end"] == str(darr.time.to_index()[-1])


def test_algo_spi_attrs_start(darr):
    _res = darr.hdc.algo.spi(calibration_begin="2000-01-02")
    assert _res.attrs["spi_calibration_begin"] == "2000-01-11 00:00:00"


def test_algo_spi_attrs_stop(darr):
    _res = darr.hdc.algo.spi(calibration_end="2000-02-09")
    assert _res.attrs["spi_calibration_end"] == "2000-01-31 00:00:00"


def test_algo_spi_decoupled_1(darr, res_spi):
    _res = darr.hdc.algo.spi(calibration_begin="2000-01-01", calibration_end="2000-02-10")

    assert isinstance(_res, xr.DataArray)
    np.testing.assert_array_equal(_res, res_spi)

    assert _res.attrs["spi_calibration_begin"] == "2000-01-01 00:00:00"
    assert _res.attrs["spi_calibration_end"] == "2000-02-10 00:00:00"


def test_algo_spi_decoupled_2(darr):
    res_spi = np.array(
        [
            [[406, 583, 826, -1704, 1028], [1182, -1019, 792, -964, -581]],
            [[-1628, 770, 878, 13, -480], [795, 1001, -173, -1550, -1550]],
        ]
    )

    _res = darr.hdc.algo.spi(calibration_begin="2000-01-01", calibration_end="2000-01-31")

    assert isinstance(_res, xr.DataArray)
    np.testing.assert_array_equal(_res, res_spi)

    assert _res.attrs["spi_calibration_begin"] == "2000-01-01 00:00:00"
    assert _res.attrs["spi_calibration_end"] == "2000-01-31 00:00:00"


def test_algo_spi_decoupled_3(darr):
    res_spi = np.array(
        [
            [[240, 401, 622, -1706, 804], [2213, -790, 1675, -717, -201]],
            [[-3454, 852, 1044, -504, -1390], [1230, 1427, 320, -925, -925]],
        ]
    )

    _res = darr.hdc.algo.spi(calibration_begin="2000-01-11")

    assert isinstance(_res, xr.DataArray)
    np.testing.assert_array_equal(_res, res_spi)

    assert _res.attrs["spi_calibration_begin"] == "2000-01-11 00:00:00"
    assert _res.attrs["spi_calibration_end"] == str(darr.time.to_index()[-1])


def test_algo_spi_nodata(darr):
    _ = darr.attrs.pop("nodata")
    with pytest.raises(ValueError):
        _ = darr.hdc.algo.spi()
    _ = darr.hdc.algo.spi(nodata=-9999)


def test_algo_spi_decoupled_err_1(darr):
    with pytest.raises(ValueError):
        _res = darr.hdc.algo.spi(
            calibration_begin="2000-03-01",
        )


def test_algo_spi_decoupled_err_2(darr):
    with pytest.raises(ValueError):
        _res = darr.hdc.algo.spi(
            calibration_end="1999-01-01",
        )


def test_algo_spi_decoupled_err_3(darr):
    with pytest.raises(ValueError):
        _res = darr.hdc.algo.spi(
            calibration_begin="2000-01-01",
            calibration_end="2000-01-01",
        )


def test_algo_spi_decoupled_err_4(darr):
    with pytest.raises(ValueError):
        _res = darr.hdc.algo.spi(
            calibration_begin="2000-02-01",
            calibration_end="2000-01-01",
        )


def test_algo_spi_exception(darr):
    with pytest.raises(MissingTimeError):
        _ = darr.isel(time=0).hdc.algo.spi()


def test_algo_croo_exception(darr):
    with pytest.raises(MissingTimeError):
        _ = darr.isel(time=0).hdc.algo.croo()


def test_algo_lroo_exception(darr):
    with pytest.raises(MissingTimeError):
        _ = darr.isel(time=0).hdc.algo.lroo()


def test_agg_sum_1(darr):
    n = 0
    ii = -1
    for _x in darr.hdc.iteragg.sum(1):
        np.testing.assert_array_equal(_x.squeeze(), darr[ii, :])
        assert _x.attrs["agg_n"] == 1
        assert _x.time == darr.time[ii]
        ii -= 1
        n += 1


def test_agg_sum_1_lim_begin(darr):
    n = 0
    begin = "2000-01-21"
    ii = darr.time.to_index().get_loc(begin)
    for _x in darr.hdc.iteragg.sum(1, begin=begin):
        np.testing.assert_array_equal(_x.squeeze(), darr[ii, :])
        assert _x.attrs["agg_n"] == 1
        assert _x.time == darr.time[ii]
        ii -= 1
        n += 1
    assert n == 3


def test_agg_sum_1_lim_end(darr):
    n = 0
    end = "2000-01-21"
    ii = -1
    _last_x = None
    for _x in darr.hdc.iteragg.sum(1, end=end):
        np.testing.assert_array_equal(_x.squeeze(), darr[ii, :])
        assert _x.attrs["agg_n"] == 1
        assert _x.time == darr.time[ii]
        ii -= 1
        n += 1
        _last_x = _x
    assert n == 3
    assert _last_x is not None
    assert _last_x.time.dt.strftime("%Y-%m-%d").values == end


def test_agg_sum_1_lim_begin_end(darr):
    n = 0
    begin = "2000-01-31"
    end = "2000-01-11"
    ii = darr.time.to_index().get_loc(begin)
    _last_x = None
    for _x in darr.hdc.iteragg.sum(1, begin=begin, end=end):
        np.testing.assert_array_equal(_x.squeeze(), darr[ii, :])
        assert _x.attrs["agg_n"] == 1
        assert _x.time == darr.time[ii]
        ii -= 1
        n += 1
        _last_x = _x

    assert n == 3
    assert _last_x is not None
    assert _last_x.time.dt.strftime("%Y-%m-%d").values == end


def test_agg_sum_3(darr):
    n = 0
    for _x in darr.hdc.iteragg.sum(n=3):
        np.testing.assert_array_equal(
            _x.squeeze(),
            darr.sel(time=slice(_x.attrs["agg_start"], _x.attrs["agg_stop"])).sum("time"),
        )
        assert _x.attrs["agg_n"] == 3
        assert str(_x.time.to_index()[0]) == _x.attrs["agg_stop"]
        n += 1
    assert n == 3


def test_agg_mean_1(darr):
    n = 0
    ii = -1
    for _x in darr.hdc.iteragg.mean(1):
        np.testing.assert_array_equal(_x.squeeze(), darr[ii, :])
        assert _x.attrs["agg_n"] == 1
        assert _x.time == darr.time[ii]
        ii -= 1
        n += 1
    assert n == 5


def test_agg_mean_1_lim_begin(darr):
    n = 0
    begin = "2000-01-21"
    ii = darr.time.to_index().get_loc(begin)
    for _x in darr.hdc.iteragg.mean(1, begin=begin):
        np.testing.assert_array_equal(_x.squeeze(), darr[ii, :])
        assert _x.attrs["agg_n"] == 1
        assert _x.time == darr.time[ii]
        ii -= 1
        n += 1
    assert n == 3


def test_agg_mean_1_lim_end(darr):
    n = 0
    end = "2000-01-21"
    ii = -1
    _last_x = None
    for _x in darr.hdc.iteragg.mean(1, end=end):
        np.testing.assert_array_equal(_x.squeeze(), darr[ii, :])
        assert _x.attrs["agg_n"] == 1
        assert _x.time == darr.time[ii]
        ii -= 1
        n += 1
        _last_x = _x
    assert n == 3
    assert _last_x is not None
    assert _last_x.time.dt.strftime("%Y-%m-%d").values == end


def test_agg_mean_1_lim_begin_end(darr):
    n = 0
    begin = "2000-01-31"
    end = "2000-01-11"
    ii = darr.time.to_index().get_loc(begin)
    _last_x = None
    for _x in darr.hdc.iteragg.mean(1, begin=begin, end=end):
        np.testing.assert_array_equal(_x.squeeze(), darr[ii, :])
        assert _x.attrs["agg_n"] == 1
        assert _x.time == darr.time[ii]
        ii -= 1
        n += 1
        _last_x = _x
    assert n == 3
    assert _last_x is not None
    assert _last_x.time.dt.strftime("%Y-%m-%d").values == end


def test_agg_mean_3(darr):
    n = 0
    for _x in darr.hdc.iteragg.mean(3):
        np.testing.assert_array_equal(
            _x.squeeze(),
            darr.sel(time=slice(_x.attrs["agg_start"], _x.attrs["agg_stop"])).mean("time"),
        )
        assert _x.attrs["agg_n"] == 3
        assert str(_x.time.to_index()[0]) == _x.attrs["agg_stop"]
        n += 1
    assert n == 3


def test_agg_full_3(darr):
    n = 0
    for _x in darr.hdc.iteragg.full(3):
        np.testing.assert_array_equal(
            _x.squeeze(),
            darr.sel(time=slice(_x.attrs["agg_start"], _x.attrs["agg_stop"])),
        )
        assert _x.time.size == 3
        assert _x.attrs["agg_n"] == 3
        assert str(_x.time.to_index()[0]) == _x.attrs["agg_start"]
        assert str(_x.time.to_index()[-1]) == _x.attrs["agg_stop"]
        n += 1
    assert n == 3


def test_whit_whits_s(darr):
    _res = np.array(
        [
            [[54, 54, 55, 56, 60], [74, 60, 48, 35, 24]],
            [[49, 56, 59, 58, 56], [82, 60, 37, 14, -7]],
        ],
        dtype="int16",
    )
    _darr = darr.hdc.whit.whits(nodata=0, s=10)
    np.testing.assert_array_equal(_darr, _res)

    y = darr[:, 0, 0].astype("float64").data
    w = ((y != 0) * 1).astype("float64")
    z = np.rint(ws2d(y, 10.0, w))

    np.testing.assert_array_equal(_darr[0, 0, :].data, z)


def test_whit_whits_sg(darr):
    sg = np.full((2, 2), -0.5)
    _res = np.array(
        [
            [[55, 59, 55, 38, 72], [82, 49, 50, 34, 27]],
            [[26, 71, 81, 61, 39], [80, 69, 33, 8, -3]],
        ],
        dtype="int16",
    )
    _darr = darr.hdc.whit.whits(nodata=0, sg=sg)
    np.testing.assert_array_equal(_darr, _res)

    y = darr[:, 0, 0].astype("float64").data
    w = ((y != 0) * 1).astype("float64")
    l = 10**-0.5
    z = np.rint(ws2d(y, l, w))

    np.testing.assert_array_equal(_darr[0, 0, :].data, z)


def test_whit_whits_sg_zeros(darr):
    sg = np.full((2, 2), -np.inf)
    _res = darr.transpose(..., "time")
    _darr = darr.hdc.whit.whits(nodata=0, sg=sg)
    np.testing.assert_array_equal(_darr, _res)


def test_whit_whits_sg_p(darr):
    sg = np.full((2, 2), -0.5)
    _res = np.array(
        [
            [[56, 64, 70, 73, 85], [91, 77, 67, 49, 30]],
            [[60, 79, 83, 72, 55], [100, 80, 51, 23, 0]],
        ],
        dtype="int16",
    )
    _darr = darr.hdc.whit.whits(nodata=0, sg=sg, p=0.90)
    np.testing.assert_array_equal(_darr, _res)

    y = darr[:, 0, 0].astype("float64").data
    l = 10**-0.5
    z = np.rint(ops.ws2dpgu(y, l, 0, 0.90))

    np.testing.assert_array_equal(_darr[0, 0, :].data, z)


def test_whit_whits_sg_p_zeros(darr):
    sg = np.full((2, 2), -np.inf)
    _res = darr.transpose(..., "time")
    _darr = darr.hdc.whit.whits(nodata=0, sg=sg, p=0.90)
    np.testing.assert_array_equal(_darr, _res)


def test_whit_whitsvc(darr):
    srange = np.arange(-2, 2)
    _res = np.array(
        [
            [[55, 59, 55, 38, 72], [82, 49, 50, 34, 27]],
            [[26, 71, 81, 61, 39], [80, 69, 33, 8, -3]],
        ],
        dtype="int16",
    )
    _darr = darr.hdc.whit.whitsvc(nodata=0, srange=srange)
    assert isinstance(_darr, xr.Dataset)
    assert "band" in _darr
    assert "sgrid" in _darr

    np.testing.assert_array_equal(_darr.band, _res)
    np.testing.assert_array_equal(_darr.sgrid, np.full((2, 2), -0.5))

    y = darr[:, 0, 0].astype("float64").data

    z, l = ops.ws2doptv(y, 0.0, srange)

    np.testing.assert_array_equal(_darr.band[0, 0, :].data, np.rint(z))
    assert np.log10(l) == -0.5


def test_whit_whitsvc_unnamed(darr):
    srange = np.arange(-2, 2)
    _res = np.array(
        [
            [[55, 59, 55, 38, 72], [82, 49, 50, 34, 27]],
            [[26, 71, 81, 61, 39], [80, 69, 33, 8, -3]],
        ],
        dtype="int16",
    )

    darr.name = None
    _darr = darr.hdc.whit.whitsvc(nodata=0, srange=srange)
    assert isinstance(_darr, xr.Dataset)
    assert "band" in _darr
    assert "sgrid" in _darr

    np.testing.assert_array_equal(_darr.band, _res)
    np.testing.assert_array_equal(_darr.sgrid, np.full((2, 2), -0.5))

    y = darr[:, 0, 0].astype("float64").data

    z, l = ops.ws2doptv(y, 0.0, srange)

    np.testing.assert_array_equal(_darr.band[0, 0, :].data, np.rint(z))
    assert np.log10(l) == -0.5


def test_whit_whitsvcp(darr):
    srange = np.arange(-2, 2)
    _res = np.array(
        [
            [[54, 66, 71, 49, 86], [92, 60, 70, 43, 30]],
            [[60, 79, 83, 72, 55], [85, 84, 40, 10, 1]],
        ],
        dtype="int16",
    )
    _darr = darr.hdc.whit.whitsvc(nodata=0, srange=srange, p=0.90)
    assert isinstance(_darr, xr.Dataset)
    assert "band" in _darr
    assert "sgrid" in _darr

    np.testing.assert_array_equal(_darr.band, _res)
    np.testing.assert_array_equal(_darr.sgrid, np.array([[-1.5, -1.5], [-0.5, -1.5]]))

    y = darr[:, 0, 0].astype("float64").data

    z, l = ops.ws2doptvp(y, 0.0, 0.90, srange)

    np.testing.assert_array_equal(_darr.band[0, 0, :].data, np.rint(z))
    assert np.log10(l) == -1.5


def test_whit_whitsvcp_unnamed(darr):
    srange = np.arange(-2, 2)
    _res = np.array(
        [
            [[54, 66, 71, 49, 86], [92, 60, 70, 43, 30]],
            [[60, 79, 83, 72, 55], [85, 84, 40, 10, 1]],
        ],
        dtype="int16",
    )

    darr.name = None
    _darr = darr.hdc.whit.whitsvc(nodata=0, srange=srange, p=0.90)
    assert isinstance(_darr, xr.Dataset)
    assert "band" in _darr
    assert "sgrid" in _darr

    np.testing.assert_array_equal(_darr.band, _res)
    np.testing.assert_array_equal(_darr.sgrid, np.array([[-1.5, -1.5], [-0.5, -1.5]]))

    y = darr[:, 0, 0].astype("float64").data

    z, l = ops.ws2doptvp(y, 0.0, 0.90, srange)

    np.testing.assert_array_equal(_darr.band[0, 0, :].data, np.rint(z))
    assert np.log10(l) == -1.5


def test_whit_whitswcv(darr):
    srange = np.arange(-2, 2)
    _res = (
        np.array(
            [
                [[53, 63, 73, 83, 93], [79, 64, 50, 37, 24]],
                [[79, 83, 88, 69, 38], [73, 86, 47, 4, 2]],
            ],
            dtype="int16",
        ),
        np.array([[1.0, 1.0], [-2.0, -2.0]]),
    )

    _darr = darr.hdc.whit.whitswcv(nodata=0, srange=srange)

    assert isinstance(_darr, xr.Dataset)
    assert "band" in _darr
    assert "sgrid" in _darr

    np.testing.assert_array_equal(_darr.band, _res[0])
    np.testing.assert_array_equal(_darr.sgrid, _res[1])

    y = darr[:, 0, 0].astype("float64").data

    z, l = ops.ws2dwcv(y, 0.0, srange, True)

    np.testing.assert_array_equal(_darr.band[0, 0, :].data, np.rint(z))
    assert np.log10(l) == 1.0


def test_whit_whitswcv_unnamed(darr):
    srange = np.arange(-2, 2)
    _res = (
        np.array(
            [
                [[53, 63, 73, 83, 93], [79, 64, 50, 37, 24]],
                [[79, 83, 88, 69, 38], [73, 86, 47, 4, 2]],
            ],
            dtype="int16",
        ),
        np.array([[1.0, 1.0], [-2.0, -2.0]]),
    )

    darr.name = None
    _darr = darr.hdc.whit.whitswcv(nodata=0, srange=srange)
    assert isinstance(_darr, xr.Dataset)
    assert "band" in _darr
    assert "sgrid" in _darr

    np.testing.assert_array_equal(_darr.band, _res[0])
    np.testing.assert_array_equal(_darr.sgrid, _res[1])

    y = darr[:, 0, 0].astype("float64").data

    z, l = ops.ws2dwcv(y, 0.0, srange, True)

    np.testing.assert_array_equal(_darr.band[0, 0, :].data, np.rint(z))
    assert np.log10(l) == 1.0


def test_whit_whitswcvp(darr):
    srange = np.arange(-2, 2)
    _res = (
        np.array(
            [
                [[53, 64, 74, 85, 95], [91, 76, 61, 46, 31]],
                [[81, 84, 88, 69, 39], [76, 86, 50, 9, 2]],
            ],
            dtype="int16",
        ),
        np.array([[1.0, 1.0], [-2.0, -2.0]]),
    )

    _darr = darr.hdc.whit.whitswcv(nodata=0, srange=srange, p=0.80)

    assert isinstance(_darr, xr.Dataset)
    assert "band" in _darr
    assert "sgrid" in _darr

    np.testing.assert_array_equal(_darr.band, _res[0])
    np.testing.assert_array_equal(_darr.sgrid, _res[1])

    y = darr[:, 0, 0].astype("float64").data

    z, l = ops.ws2dwcvp(y, 0.0, 0.80, srange, True)

    np.testing.assert_array_equal(_darr.band[0, 0, :].data, np.rint(z))
    assert np.log10(l) == 1.0


def test_whit_whitswcvp_unnamed(darr):
    srange = np.arange(-2, 2)
    _res = (
        np.array(
            [
                [[53, 64, 74, 85, 95], [91, 76, 61, 46, 31]],
                [[81, 84, 88, 69, 39], [76, 86, 50, 9, 2]],
            ],
            dtype="int16",
        ),
        np.array([[1.0, 1.0], [-2.0, -2.0]]),
    )

    darr.name = None
    _darr = darr.hdc.whit.whitswcv(nodata=0, srange=srange, p=0.80)
    assert isinstance(_darr, xr.Dataset)
    assert "band" in _darr
    assert "sgrid" in _darr

    np.testing.assert_array_equal(_darr.band, _res[0])
    np.testing.assert_array_equal(_darr.sgrid, _res[1])

    y = darr[:, 0, 0].astype("float64").data

    z, l = ops.ws2dwcvp(y, 0.0, 0.80, srange, True)

    np.testing.assert_array_equal(_darr.band[0, 0, :].data, np.rint(z))
    assert np.log10(l) == 1.0


def test_whit_whitint(darr):
    labels_daily = np.array(
        [
            2000011,
            2000011,
            2000011,
            2000011,
            2000011,
            2000011,
            2000011,
            2000011,
            2000011,
            2000011,
            2000012,
            2000012,
            2000012,
            2000012,
            2000012,
            2000012,
            2000012,
            2000012,
            2000012,
            2000012,
            2000013,
            2000013,
            2000013,
            2000013,
            2000013,
            2000013,
            2000013,
            2000013,
            2000013,
            2000013,
            2000013,
            2000021,
            2000021,
            2000021,
            2000021,
            2000021,
            2000021,
            2000021,
            2000021,
            2000021,
            2000021,
            2000022,
            2000022,
            2000022,
            2000022,
            2000022,
            2000022,
            2000022,
            2000022,
            2000022,
            2000022,
        ],
        dtype="int32",
    )

    template = np.array(
        [
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
    )

    temp = template.copy()
    res = darr.astype("int16").hdc.whit.whitint(labels_daily, temp)
    assert "newtime" in res.dims
    assert res.newtime.size == 5

    xx = darr[:, 0, 0].values

    w = template.copy()
    temp = template.copy()

    for ix, ii in enumerate(np.where(template != 0)[0]):
        template[ii] = xx[ix]
    template[-1] = xx[-1]

    z = ws2d(template, 0.00001, w)

    _df = pd.DataFrame({"data": z, "labels": labels_daily})
    xx_int = _df.groupby("labels").mean().round().astype("int16")

    np.testing.assert_array_equal(res[0, 0, :], xx_int.data.values)


def test_whit_whits_exception(darr):
    with pytest.raises(MissingTimeError):
        _ = darr.isel(time=0).hdc.whit.whits(nodata=-3000)


def test_whit_whitsvc_exception(darr):
    with pytest.raises(MissingTimeError):
        _ = darr.isel(time=0).hdc.whit.whitsvc(nodata=-3000)


def test_whit_whitswcv_exception(darr):
    with pytest.raises(MissingTimeError):
        _ = darr.isel(time=0).hdc.whit.whitswcv(nodata=-3000)


def test_whit_whitint_exception(darr):
    with pytest.raises(MissingTimeError):
        _ = darr.isel(time=0).hdc.whit.whitint(None, None)


def test_zonal_mean(darr, zones):
    res = np.array(
        np.array(
            [
                [[33.5, 2.0], [82.5, 2.0]],
                [[72.0, 2.0], [54.0, 2.0]],
                [[81.5, 2.0], [49.5, 2.0]],
                [[28.0, 2.0], [12.0, 2.0]],
                [[63.0, 2.0], [16.0, 2.0]],
            ]
        ),
        dtype="float32",
    )

    z_ids = np.unique(zones.data)
    x = darr.hdc.zonal.mean(zones, z_ids)
    assert x.shape == (5, 2, 2)
    np.testing.assert_equal(x.coords["zones"].data, [0, 1])
    assert list(x.coords["stat"].values) == ["mean", "valid"]
    np.testing.assert_almost_equal(x, res)
    assert x.dtype == res.dtype
    assert x.nodata == darr.nodata


def test_zonal_mean_nodata(darr, zones):
    res = np.array(
        [
            [[15.0, 1.0], [82.5, 2.0]],
            [[72.0, 2.0], [54.0, 2.0]],
            [[81.5, 2.0], [49.5, 2.0]],
            [[28.0, 2.0], [12.0, 2.0]],
            [[63.0, 2.0], [16.0, 2.0]],
        ],
        dtype="float32",
    )

    darr[0, 0, 0] = darr.nodata

    z_ids = np.unique(zones.data)
    x = darr.hdc.zonal.mean(zones, z_ids)
    np.testing.assert_almost_equal(x, res)
    assert x.dtype == res.dtype
    assert x.nodata == darr.nodata


def test_zonal_mean_nodata_nan(darr, zones):
    darr[[0, -1], :] = darr.nodata

    z_ids = np.unique(zones.data)
    x = darr.hdc.zonal.mean(zones, z_ids)
    assert np.isnan(x.data[[0, -1], :, 0]).all()
    assert np.all(x.data[[0, -1], :, 1] == 0)
    assert x.nodata == darr.nodata


def test_zonal_mean_nodata_nan_float(darr, zones):
    res = np.array(
        [
            [[15.0, 1.0], [82.5, 2.0]],
            [[72.0, 2.0], [54.0, 2.0]],
            [[81.5, 2.0], [49.5, 2.0]],
            [[28.0, 2.0], [12.0, 2.0]],
            [[63.0, 2.0], [16.0, 2.0]],
        ],
        dtype="float64",
    )

    darr = darr.astype("float64")
    darr[0, 0, 0] = "nan"

    z_ids = np.unique(zones.data)
    x = darr.hdc.zonal.mean(zones, z_ids, dtype="float64")
    np.testing.assert_almost_equal(x, res)
    assert x.dtype == res.dtype
    assert x.nodata == darr.nodata


def test_zonal_zone_nodata_nan(darr, zones):
    res = np.array(
        [
            [["nan", 0.0], [82.5, 2.0]],
            [["nan", 0.0], [54.0, 2.0]],
            [["nan", 0.0], [49.5, 2.0]],
            [["nan", 0.0], [12.0, 2.0]],
            [["nan", 0.0], [16.0, 2.0]],
        ],
        dtype="float64",
    )

    zones.attrs = {"nodata": 0}
    z_ids = np.unique(zones.data)
    x = darr.hdc.zonal.mean(zones, z_ids, dim_name="foo", dtype="float64")
    np.testing.assert_almost_equal(x, res)
    assert x.dtype == res.dtype
    assert x.nodata == darr.nodata


def test_zonal_dimname(darr, zones):
    z_ids = np.unique(zones.data)
    x = darr.hdc.zonal.mean(zones, z_ids, dim_name="foo", dtype="float64")
    assert x.dims == ("time", "foo", "stat")


def test_zonal_nodata_exc(darr, zones):
    z_ids = np.unique(zones.data)
    del darr.attrs["nodata"]
    with pytest.raises(ValueError):
        _ = darr.hdc.zonal.mean(zones, z_ids)


def test_zonal_zone_nodata_exc(darr, zones):
    z_ids = np.unique(zones.data)
    del zones.attrs["nodata"]
    with pytest.raises(ValueError):
        _ = darr.hdc.zonal.mean(zones, z_ids)


def test_zonal_type_exc(darr, zones):
    z_ids = np.unique(zones.data)
    with pytest.raises(ValueError):
        _ = darr.hdc.zonal.mean(zones.data, z_ids)


def test_rolling_sum(darr):
    res = darr.hdc.rolling.sum(window_size=3).transpose("time", ...)
    xr.testing.assert_equal(
        res,
        darr.rolling(time=3).sum().dropna("time"),
    )
    np.testing.assert_equal(darr[-3:].sum("time").data, res[-1].data)


def test_rolling_sum_nodata(darr):
    darr[:, 0, 0] = -9999
    xr.testing.assert_equal(
        darr.hdc.rolling.sum(3, nodata=-9999).transpose("time", ...),
        darr.rolling(time=3).sum().where(darr != -9999, -9999).dropna("time"),
    )


def test_mean_grp(darr):
    grps = np.array([0, 0, 1, 1, 2], dtype="int16")
    tst = darr.hdc.algo.mean_grp(grps)[..., [0, 2, 4]]
    cntrl = darr.groupby(xr.DataArray(grps, dims=("time",))).mean().transpose(..., "group")

    np.testing.assert_allclose(tst.data, cntrl.data)


def test_mean_grp_exc(darr):
    grps = np.array([0, 0, 1, 1, 2], dtype="int16")
    _darr = darr.copy()
    _ = _darr.attrs.pop("nodata")
    with pytest.raises(ValueError):
        _darr.hdc.algo.mean_grp(grps)

    with pytest.raises(MissingTimeError):
        darr.rename(time="foo").hdc.algo.mean_grp(grps)


def test_theta(darr):
    """Test theta anomaly calculation."""
    result = darr.hdc.anom.theta()
    # Dimensions may be reordered by apply_ufunc, but should contain same dims
    assert set(result.dims) == set(darr.dims)
    assert result.dtype in ["float32", "float64"]
    # Check that result is finite (logit transformation)
    assert np.isfinite(result).all()


def test_theta_nodata(darr):
    """Test theta with explicit nodata."""
    darr[:, 0, 0] = -9999
    result = darr.hdc.anom.theta(nodata=-9999)
    assert set(result.dims) == set(darr.dims)
    assert result.dtype == "float32"


def test_theta_missing_dimension(darr):
    """Test theta raises error for missing dimension."""
    with pytest.raises(ValueError, match="Dimension foo not found"):
        darr.hdc.anom.theta(dim_name="foo")


def test_logit(darr):
    """Test logit transformation."""
    # Use values strictly between 0 and 1 for logit (avoid 0 and 1 which give inf)
    test_data = (darr - darr.min()) / (darr.max() - darr.min() + 1e-6)
    test_data = test_data * 0.98 + 0.01  # Scale to (0.01, 0.99)
    result = test_data.hdc.algo.logit()
    assert set(result.dims) == set(test_data.dims)
    # Logit of values in (0, 1) should be finite
    assert np.isfinite(result).all()


def test_invlogit(darr):
    """Test inverse logit transformation."""
    # Use any values (invlogit accepts any real number)
    result = darr.hdc.algo.invlogit()
    assert set(result.dims) == set(darr.dims)
    # Invlogit should always be between 0 and 1
    assert (result >= 0).all()
    assert (result <= 1).all()


def test_logit_invlogit_roundtrip(darr):
    """Test that logit and invlogit are inverse operations."""
    # Use values strictly between 0 and 1 for logit
    test_data = (darr - darr.min()) / (darr.max() - darr.min() + 1e-6)
    test_data = test_data * 0.98 + 0.01  # Scale to (0.01, 0.99)
    logit_result = test_data.hdc.algo.logit()
    invlogit_result = logit_result.hdc.algo.invlogit()
    # Should recover original values approximately (transpose to match dimension order)
    np.testing.assert_allclose(
        test_data.transpose(*invlogit_result.dims).values,
        invlogit_result.values,
        rtol=1e-5,
    )
