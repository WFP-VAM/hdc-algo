"""Xarray Accesor classes."""

from typing import Iterable, List, Optional, Tuple, Union
from warnings import warn

import dask.array as da
import numpy as np
import xarray
from dask import is_dask_collection
from dask.base import tokenize

from . import ops
from .dekad import Dekad
from .season import Season
from .utils import get_calibration_indices, to_linspace

__all__ = [
    "Anomalies",
    "Dekad",
    "Season",
    "IterativeAggregation",
    "PixelAlgorithms",
    "WhittakerSmoother",
]


class MissingTimeError(Exception):
    """Exception for missing time dimension when required."""


class AccessorBase:
    """Base class for accessors."""

    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def _check_for_timedim(self):
        if "time" not in self._obj.dims:
            return False
        return True


class AccessorTimeBase(AccessorBase):
    """Base class for accessors with required time dimension."""

    def __init__(self, xarray_obj):
        """Construct with DataArray|Dataset."""
        if not np.issubdtype(xarray_obj, np.datetime64):
            raise TypeError(
                "'This accessor is only available for "
                "DataArray with datetime64 dtype"
            )

        if not hasattr(xarray_obj, "time"):
            raise ValueError("Data array is missing 'time' accessor!")

        if "time" not in xarray_obj.dims:
            xarray_obj = xarray_obj.expand_dims("time")
        self._obj = xarray_obj

        super().__init__(xarray_obj)

    @property
    def year(self):
        return self._obj.time.dt.year

    @property
    def month(self):
        return self._obj.time.dt.month

    @property
    def day(self):
        return self._obj.time.dt.day


class Period(AccessorTimeBase):
    # pylint: disable=no-member,undefined-variable
    """
    Baseclass to extend time dimension with period functionality.

    Adds functionality for working with periods, such as dekads and pentads
    """

    @property
    def _tseries(self):
        return self._obj.time.to_series()

    @property
    def idx(self):
        return self._tseries.apply(lambda x: self._period_cls(x).idx).to_xarray()

    @property
    def midx(self):
        warn(
            """Please use `.idx` property (renaming for consistency).
            The `.midx` property will be removed in a future release.""",
            DeprecationWarning,
            2,
        )
        return self.idx

    @property
    def yidx(self):
        return self._tseries.apply(lambda x: self._period_cls(x).yidx).to_xarray()

    @property
    def ndays(self):
        return self._tseries.apply(lambda x: self._period_cls(x).ndays).to_xarray()

    @property
    def label(self):
        return self._tseries.apply(lambda x: str(self._period_cls(x))).to_xarray()

    @property
    def start_date(self):
        return self._tseries.apply(lambda x: self._period_cls(x).start_date).to_xarray()

    @property
    def end_date(self):
        return self._tseries.apply(lambda x: self._period_cls(x).end_date).to_xarray()

    @property
    def raw(self):
        return self._tseries.apply(lambda x: self._period_cls(x).raw).to_xarray()

    @property
    def linspace(self):
        return self.yidx - 1


@xarray.register_dataset_accessor("dekad")
@xarray.register_dataarray_accessor("dekad")
class DekadPeriod(Period):
    """Accessor class for dekad period."""

    _period_cls = Dekad


@xarray.register_dataset_accessor("season")
@xarray.register_dataarray_accessor("season")
class SeasonPeriod(AccessorTimeBase):
    """Accessor class for handling seasonal indexing of an xarray object."""

    @property
    def _tseries(self):
        return self._obj.time.to_series()

    def label(
        self, season_ranges: List[Tuple[int, int]]
    ) -> Union[xarray.DataArray, xarray.Dataset]:
        """
        Assigns a seasonal label (e.g., '2021-01') to each time step in the xarray object.
        """
        if "time" not in self._obj.coords:
            raise ValueError("The xarray object must have a 'time' coordinate.")

        return self._tseries.apply(
            lambda date: Season(date, season_ranges).id
        ).to_xarray()

    def idx(
        self, season_range: Optional[List[Tuple[int, int]]] = None
    ) -> Union[xarray.DataArray, xarray.Dataset]:
        """Returns the index of the season within the year."""
        return self._tseries.apply(
            lambda x: Season(x, season_range).season_index(Dekad(x).yidx)
        ).to_xarray()

    def ndays(
        self, season_range: Optional[List[Tuple[int, int]]]
    ) -> Union[xarray.DataArray, xarray.Dataset]:
        """Returns the number of days in each season."""
        return self._tseries.apply(lambda x: Season(x, season_range).ndays).to_xarray()

    def start_date(
        self, season_range: Optional[List[Tuple[int, int]]]
    ) -> Union[xarray.DataArray, xarray.Dataset]:
        """Returns the start date of each season."""
        return self._tseries.apply(
            lambda x: Season(x, season_range).start_date
        ).to_xarray()

    def end_date(
        self, season_range: Optional[List[Tuple[int, int]]]
    ) -> Union[xarray.DataArray, xarray.Dataset]:
        """Returns the end date of each season."""
        return self._tseries.apply(
            lambda x: Season(x, season_range).end_date
        ).to_xarray()


class IterativeAggregation(AccessorBase):
    """Class to aggregate multiple coordinate slices."""

    def sum(
        self,
        n: Optional[int] = None,
        dim: str = "time",
        begin: Optional[Union[str, int, float]] = None,
        end: Optional[Union[str, int, float]] = None,
        method: Optional[str] = None,
    ):
        """Generate sum-aggregations over dim for periods n."""
        yield from self._iteragg(np.nansum, n, dim, begin, end, method)

    def mean(
        self,
        n: Optional[int] = None,
        dim: str = "time",
        begin: Optional[Union[str, int, float]] = None,
        end: Optional[Union[str, int, float]] = None,
        method: Optional[str] = None,
    ):
        """Generate mean-aggregations over dim for slices of n."""
        yield from self._iteragg(np.nanmean, n, dim, begin, end, method)

    def full(
        self,
        n: Optional[int] = None,
        dim: str = "time",
        begin: Optional[Union[str, int, float]] = None,
        end: Optional[Union[str, int, float]] = None,
        method: Optional[str] = None,
    ):
        """Generate mean-aggregations over dim for slices of n."""
        yield from self._iteragg(None, n, dim, begin, end, method)

    def _iteragg(self, func, n, dim, begin, end, method):
        if dim not in self._obj.dims:
            raise ValueError(f"Dimension {dim} doesn't exist in xarray object!")

        _index = self._obj[dim].to_index()

        if n is None:
            n = self._obj[dim].size
        assert n != 0, "n must be non-zero"

        if begin is not None:
            try:
                (begin_ix,) = _index.get_indexer([begin], method=method) + 1
            except KeyError:
                raise ValueError(
                    f"Value {begin} for 'begin' not found in index for dim {dim}"
                ) from None
        else:
            begin_ix = self._obj.sizes[dim]

        if end is not None:
            try:
                (end_ix,) = _index.get_indexer([end], method=method)
            except KeyError:
                raise ValueError(
                    f"Value {end} for 'end' not found in index for dim {dim}"
                ) from None
        else:
            end_ix = 0

        for ii in range(begin_ix, 0, -1):
            jj = ii - n
            if ii <= end_ix:
                break
            if jj >= 0 and (ii - jj) == n:
                region = {dim: slice(jj, ii)}
                _obj = self._obj[region].assign_attrs(
                    {
                        "agg_start": str(_index[jj]),
                        "agg_stop": str(_index[ii - 1]),
                        "agg_n": _index[jj:ii].size,
                    }
                )

                if func is not None:
                    _obj = _obj.reduce(func, dim, keep_attrs=True)
                    if dim == "time":
                        _obj = _obj.expand_dims(time=[self._obj.time[ii - 1].values])

                yield _obj


class Anomalies(AccessorBase):
    """Class to calculate anomalies from reference."""

    def ratio(self, reference, offset=0):
        """Calculate anomaly as ratio."""
        return (self._obj + offset) / (reference + offset) * 100

    def diff(self, reference, offset=0):
        """Calculate anomaly as difference."""
        return (self._obj + offset) - (reference + offset)


class WhittakerSmoother(AccessorBase):
    """Class for applying different version of the Whittaker smoother."""

    def whits(
        self,
        nodata: Union[int, float],
        sg: Optional[xarray.DataArray] = None,
        s: Optional[float] = None,
        p: Optional[float] = None,
    ) -> xarray.Dataset:
        """
        Apply whittaker with fixed S.

        Fixed S can be either provided as constant or
        as sgrid with a constant per pixel

        Args:
            nodata: nodata value
            sg: sgrid,
            s: S value
            p: Envelope value for asymmetric weights

        Returns:
            ds_out: xarray.Dataset with smoothed data
        """
        if not self._check_for_timedim():
            raise MissingTimeError("Whittaker filter requires a time dimension!")
        if sg is None and s is None:
            raise ValueError("Need S or sgrid")

        lmda = 10**sg if sg is not None else s

        if p is not None:
            xout = xarray.apply_ufunc(
                ops.ws2dpgu,
                self._obj,
                lmda,
                nodata,
                p,
                input_core_dims=[["time"], [], [], []],
                output_core_dims=[["time"]],
                dask="parallelized",
                keep_attrs=True,
            )

        else:
            xout = xarray.apply_ufunc(
                ops.ws2dgu,
                self._obj,
                lmda,
                nodata,
                input_core_dims=[["time"], [], []],
                output_core_dims=[["time"]],
                dask="parallelized",
                keep_attrs=True,
            )

        return xout

    def whitsvc(
        self,
        nodata: Union[int, float],
        lc: Optional[xarray.DataArray] = None,
        srange: Optional[np.ndarray] = None,
        p: Optional[float] = None,
    ) -> xarray.Dataset:
        """
        Apply whittaker with V-curve optimization of S.

        Args:
            nodata: nodata value
            lc: lag1 autocorrelation DataArray,
            srange: values of S for V-curve optimization (mandatory if no autocorrelation raster)
            p: Envelope value for asymmetric weights

        Returns:
            ds_out: xarray.Dataset with smoothed data and sgrid
        """
        if not self._check_for_timedim():
            raise MissingTimeError("Whittaker filter requires a time dimension!")

        if lc is not None:
            if p is None:
                raise ValueError(
                    "If lc is set, a p value needs to be specified as well."
                )

            ds_out, sgrid = xarray.apply_ufunc(
                ops.ws2doptvplc,
                self._obj,
                nodata,
                p,
                lc,
                input_core_dims=[["time"], [], [], []],
                output_core_dims=[["time"], []],
                dask="parallelized",
                keep_attrs=True,
            )

        else:
            if srange is None:
                raise ValueError("Need either lagcorr or srange!")

            if p:
                ds_out, sgrid = xarray.apply_ufunc(
                    ops.ws2doptvp,
                    self._obj,
                    nodata,
                    p,
                    srange,
                    input_core_dims=[["time"], [], [], ["dim0"]],
                    output_core_dims=[["time"], []],
                    dask="parallelized",
                    keep_attrs=True,
                )

            else:
                ds_out, sgrid = xarray.apply_ufunc(
                    ops.ws2doptv,
                    self._obj,
                    nodata,
                    srange,
                    input_core_dims=[["time"], [], ["dim0"]],
                    output_core_dims=[["time"], []],
                    dask="parallelized",
                    keep_attrs=True,
                )

        ds_out = ds_out.to_dataset(name=(ds_out.name or "band"))
        ds_out["sgrid"] = np.log10(sgrid).astype("float32")

        return ds_out

    def whitswcv(
        self,
        nodata: Union[int, float],
        srange: Optional[np.ndarray] = None,
        p: Optional[float] = None,
        robust: bool = True,
    ) -> xarray.Dataset:
        """
        Apply whittaker with Generalized Cross Validation optimization of S.

        Args:
            nodata: nodata value
            srange: values of S for GCV optimization
            p: Envelope value for asymmetric weights
            robust (boolean): performs a robust fitting by computing robust weights if True

        Returns:
            ds_out: xarray.Dataset with smoothed data and sgrid
        """
        if not self._check_for_timedim():
            raise MissingTimeError("Whittaker filter requires a time dimension!")

        if p:
            if srange is None:
                srange = np.arange(-1.8, 4.2, 0.2, dtype=np.float64)

            ds_out, sgrid = xarray.apply_ufunc(
                ops.ws2dwcvp,
                self._obj,
                nodata,
                p,
                srange,
                robust,
                input_core_dims=[["time"], [], [], ["dim0"], []],
                output_core_dims=[["time"], []],
                dask="parallelized",
                keep_attrs=True,
            )

        else:
            if srange is None:
                srange = np.arange(-1.8, 4.2, 0.2)

            ds_out, sgrid = xarray.apply_ufunc(
                ops.ws2dwcv,
                self._obj,
                nodata,
                srange,
                robust,
                input_core_dims=[["time"], [], ["dim0"], []],
                output_core_dims=[["time"], []],
                dask="parallelized",
                keep_attrs=True,
            )

        ds_out = ds_out.to_dataset(name=(ds_out.name or "band"))
        ds_out["sgrid"] = np.log10(sgrid).astype("float32")

        return ds_out

    def whitint(self, labels_daily: np.ndarray, template: np.ndarray):
        """Compute temporal interpolation using the Whittaker filter."""
        if not self._check_for_timedim():
            raise MissingTimeError(
                "Whittaker temporal interpolation requires a time dimension!"
            )

        if self._obj.dtype != "int16":
            raise NotImplementedError(
                "Temporal interpolation works currently only with int16 input!"
            )

        template_out = np.zeros(np.unique(labels_daily).size, dtype="u1")

        ds_out = xarray.apply_ufunc(
            ops.tinterpolate,
            self._obj,
            template,
            labels_daily,
            template_out,
            input_core_dims=[["time"], ["dim0"], ["dim1"], ["dim2"]],
            output_core_dims=[["newtime"]],
            dask_gufunc_kwargs={"output_sizes": {"newtime": template_out.size}},
            output_dtypes=["int16"],
            dask="parallelized",
            keep_attrs=True,
        )

        return ds_out


class PixelAlgorithms(AccessorBase):
    """Set of algorithms to be applied to pixel timeseries."""

    def spi(
        self,
        calibration_begin: Optional[str] = None,
        calibration_end: Optional[str] = None,
        nodata: Optional[Union[float, int]] = None,
        groups: Optional[Iterable[Union[int, float, str]]] = None,
        dtype="int16",
    ):
        """Calculate the SPI along the time dimension.

        Calculates the Standardized Precipitation Index along the time dimension.
        Optionally, a calibration begin and / or end date can be provided which
        determine the part of the timeseries used to fit the gamma distribution.

        `groups` can be supplied as list of group labels.
        If `groups` is supplied, the SPI will be computed for each individual group.
        This is intended to be used when SPI should be calculated for specific timesteps.
        """
        if not self._check_for_timedim():
            raise MissingTimeError("SPI requires a time dimension!")

        if nodata is None:
            if (nodata := self._obj.attrs.get("nodata")) is None:
                raise ValueError(
                    "Need nodata attribute defined, or nodata argument provided."
                )

        # pylint: disable=import-outside-toplevel
        from .ops.stats import gammastd_grp, gammastd_yxt

        tix = self._obj.get_index("time")

        if calibration_begin is None:
            calibration_begin = tix[0]

        if calibration_end is None:
            calibration_end = tix[-1]

        if calibration_begin > tix[-1:]:
            raise ValueError("Calibration begin cannot be greater than last timestamp!")

        if calibration_end < tix[:1]:
            raise ValueError("Calibration end cannot be smaller than first timestamp!")

        if groups is None:
            calstart_ix, calstop_ix = get_calibration_indices(
                tix, (calibration_begin, calibration_end)
            )

            if calstart_ix >= calstop_ix:
                raise ValueError("calibration_begin < calibration_end!")

            if abs(calstop_ix - calstart_ix) <= 1:
                raise ValueError(
                    "Timeseries too short for calculating SPI. Please adjust calibration period!"
                )

            res = xarray.apply_ufunc(
                gammastd_yxt,
                self._obj,
                kwargs={
                    "nodata": nodata,
                    "cal_start": calstart_ix,
                    "cal_stop": calstop_ix,
                },
                input_core_dims=[["time"]],
                output_core_dims=[["time"]],
                keep_attrs=True,
                dask="parallelized",
                dask_gufunc_kwargs={"meta": self._obj.data.astype(dtype)},
            )

        else:

            groups, keys = to_linspace(np.array(groups, dtype="str"))

            if len(groups) != len(self._obj.time):
                raise ValueError("Need array of groups same length as time dimension!")

            groups = groups.astype("int16")
            num_groups = len(keys)

            cal_indices = get_calibration_indices(
                tix, (calibration_begin, calibration_end), groups, num_groups
            )
            # assert for mypy
            assert isinstance(cal_indices, np.ndarray)

            if np.any(cal_indices[:, 0] >= cal_indices[:, 1]):
                raise ValueError("calibration_begin < calibration_end!")

            if np.any(np.diff(cal_indices, axis=1) <= 1):
                raise ValueError(
                    "Timeseries too short for calculating SPI. Please adjust calibration period!"
                )

            res = xarray.apply_ufunc(
                gammastd_grp,
                self._obj,
                groups,
                num_groups,
                nodata,
                cal_indices,
                input_core_dims=[["time"], ["grps"], [], [], ["start", "stop"]],
                output_core_dims=[["time"]],
                keep_attrs=True,
                dask="parallelized",
                dask_gufunc_kwargs={"meta": self._obj.data.astype(dtype)},
            )

        res.attrs.update(
            {
                "spi_calibration_begin": str(tix[tix >= calibration_begin][0]),
                "spi_calibration_end": str(tix[tix <= calibration_end][-1]),
            }
        )

        return res

    def croo(self):
        """Compute current run of ones along time dimension."""
        if not self._check_for_timedim():
            raise MissingTimeError("CROO requires a time dimension!")

        xsort = self._obj.sortby("time", ascending=False)
        xtemp = xsort.where(xsort == 1).cumsum("time", skipna=False)
        xtemp = xtemp.where(~xtemp.isnull(), 0).argmax("time")
        x_crbt = xtemp + xsort.isel(time=0)

        return x_crbt

    def lroo(self):
        """Longest run of ones along time dimension."""
        if not self._check_for_timedim():
            raise MissingTimeError("LROO requires a time dimension!")

        return xarray.apply_ufunc(
            ops.lroo,
            self._obj,
            input_core_dims=[["time"]],
            dask="parallelized",
            output_dtypes=["uint8"],
            keep_attrs=True,
        )

    def autocorr(self):
        """
        Calculate the autocorrelation along time.

        Returns:
            xarray.DataArray with lag1 autocorrelation
        """
        xx = self._obj
        if (nodata := xx.attrs.get("nodata", None)) is None:
            warn("Calculating autocorr without nodata value defined!")
        if xx.dims[0] == "time":
            # I don't know how to tell xarray's map_blocks about
            # changing dtype and losing first dimension, so use
            # dask version directly
            if is_dask_collection(xx):
                # merge all time slices if not already
                if len(xx.chunks[0]) != 1:
                    xx = xx.chunk({"time": -1})

                data = da.map_blocks(
                    ops.autocorr_tyx, xx.data, nodata, dtype="float32", drop_axis=0
                )
            else:
                data = ops.autocorr_tyx(xx.data, nodata)

            coords = {k: c for k, c in xx.coords.items() if k != "time"}
            return xarray.DataArray(data=data, dims=xx.dims[1:], coords=coords)

        return xarray.apply_ufunc(
            ops.autocorr,
            xx,
            nodata,
            input_core_dims=[["time"], []],
            dask="parallelized",
            output_dtypes=["float32"],
        )

    def mktrend(self):
        """Calculate the Mann-Kendall trend along the time dimension."""
        # pylint: disable=import-outside-toplevel
        from .ops.stats import _mann_kendall_trend_gu, _mann_kendall_trend_gu_nd

        nodata = self._obj.attrs.get("nodata", None)
        if nodata is None:
            warn("Calculating trend without nodata value defined!")

            x = xarray.apply_ufunc(
                _mann_kendall_trend_gu,
                self._obj,
                input_core_dims=[["time"]],
                output_core_dims=[[], [], [], []],
                output_dtypes=["float32", "float32", "float32", "int8"],
                dask="parallelized",
                keep_attrs=True,
            )
        else:
            x = xarray.apply_ufunc(
                _mann_kendall_trend_gu_nd,
                self._obj,
                nodata,
                input_core_dims=[["time"], []],
                output_core_dims=[[], [], [], []],
                output_dtypes=["float32", "float32", "float32", "int8"],
                dask="parallelized",
                keep_attrs=True,
            )

        x = xarray.merge(
            [
                xx.to_dataset(name=n)
                for xx, n in zip(x, ["tau", "pvalue", "slope", "trend"])
            ]
        )

        x.trend.attrs["nodata"] = -2
        return x

    def mean_grp(
        self,
        groups: Iterable[int],
        nodata: Optional[Union[int, float]] = None,
    ):
        """Calculate mean over groups along time dimension.

        This calculates a simple average over groups along the time
        dimension. The groups are identified by an int16 array, and
        the **need to be in ascending order from 0 to n-1**!

        The function will return an array of original size with averages
        at the respective positions.
        """
        if not self._check_for_timedim():
            raise MissingTimeError("Grouped mean requires a time dimension!")

        # pylint: disable=import-outside-toplevel
        from .ops.stats import mean_grp

        if nodata is None:
            nodata = self._obj.attrs.get("nodata", None)

        if nodata is None:
            raise ValueError("Need to define nodata value!")

        groups = (
            np.array(groups, dtype="int16")
            if not isinstance(groups, np.ndarray)
            else groups
        )

        if groups.size != self._obj.time.size:
            raise ValueError("Need array of groups same length as time dimension!")

        num_groups = np.unique(groups).size

        return xarray.apply_ufunc(
            mean_grp,
            self._obj,
            groups,
            num_groups,
            nodata,
            input_core_dims=[["time"], ["grps"], [], []],
            output_core_dims=[["time"]],
            keep_attrs=True,
            dask="parallelized",
            dask_gufunc_kwargs={"meta": self._obj.data},
        )


class RollingWindowAlgos(AccessorBase):
    """Class to calculate rolling window algos on dimenson."""

    def sum(
        self,
        window_size: int,
        dtype: str = "float32",
        dimension: str = "time",
        nodata: Optional[Union[int, float]] = None,
    ):
        # pylint: disable=import-outside-toplevel
        from .ops.stats import rolling_sum

        if nodata is None:
            if (nodata := self._obj.attrs.get("nodata")) is None:
                raise ValueError(
                    "Need nodata attribute defined, or nodata argument provided."
                )

        xx = xarray.apply_ufunc(
            rolling_sum,
            self._obj,
            window_size,
            nodata,
            input_core_dims=[[dimension], [], []],
            output_core_dims=[[dimension]],
            keep_attrs=True,
            dask="parallelized",
            dask_gufunc_kwargs={"meta": self._obj.astype(dtype).data},
        )
        xx = xx[..., window_size - 1 :]
        return xx


class ZonalStatistics(AccessorBase):
    """Class to claculate zonal statistics."""

    def mean(
        self,
        zones: xarray.DataArray,
        zone_ids: Union[List, np.ndarray],
        dtype: str = "float32",
        dim_name: str = "zones",
        name: Optional[str] = None,
    ) -> xarray.DataArray:
        """Calculate the zonal mean.

        The mean for each pixel is calculated for each zone (group) in
        `zones`.

        The zones in the `zones` raster have to be numbered in a linear fashion,
        starting with 0 for the first zone and num_zones for the last zone.

        Args:
            zones: zonal pixels (Y,X)
            zone_ids: list or array with zone IDs (from 0 to (n-1))
            dtype: datatype
            dim_name: name for new output dimension
            name: name for output dataarray
        """
        from .ops.zonal import do_mean  # pylint: disable=import-outside-toplevel

        xx = self._obj

        if isinstance(xx, xarray.Dataset):
            raise NotImplementedError("zonal needs dataarray as input")

        if "nodata" not in xx.attrs:
            raise ValueError("Input xarray DataArray needs nodata attribute")

        if not isinstance(zones, xarray.DataArray):
            raise ValueError("Zones need to be xarray.DataArray!")

        if "nodata" not in zones.attrs:
            raise ValueError("Zones xarray DataArray needs nodata attribute")

        # set null values to nodata value
        xx = xx.where(xx.notnull(), xx.nodata)
        attrs = xx.attrs
        num_zones = len(zone_ids)
        dims = (xx.dims[0], dim_name, "stat")
        coords = {
            dims[0]: xx.coords[dims[0]],
            dim_name: zone_ids,
            "stat": ["mean", "valid"],
        }

        # convert str datatype to type
        dtype = np.dtype(dtype).type

        if is_dask_collection(xx):
            dask_name = name
            if isinstance(dask_name, str):
                dask_name = f"{name}-{tokenize(xx.data, zones.data, dtype)}"

            chunks = [xx.data.chunks[0], (num_zones,), (2,)]

            data = da.map_blocks(
                do_mean,
                xx.data,
                zones.data,
                num_zones,
                xx.nodata,
                zones.nodata,
                drop_axis=[1, 2],
                new_axis=[1, 2],
                chunks=chunks,
                out_dtype=dtype,
                name=dask_name,
            )
        else:
            data = do_mean(
                xx.data,
                zones.data,
                num_zones,
                xx.nodata,
                zones.nodata,
                out_dtype=dtype,
            )

        return xarray.DataArray(
            data=data, dims=dims, coords=coords, attrs=attrs, name=name
        )


@xarray.register_dataset_accessor("hdc")
@xarray.register_dataarray_accessor("hdc")
class HDC:
    """xarray accessor for HDC xarray tools."""

    def __init__(self, xarray_obj):
        self.algo = PixelAlgorithms(xarray_obj)
        self.anom = Anomalies(xarray_obj)
        self.iteragg = IterativeAggregation(xarray_obj)
        self.rolling = RollingWindowAlgos(xarray_obj)
        self.whit = WhittakerSmoother(xarray_obj)
        self.zonal = ZonalStatistics(xarray_obj)
