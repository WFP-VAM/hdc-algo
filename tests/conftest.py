from pathlib import Path

import dask.array as da
import numpy as np
import pytest
import xarray as xr


@pytest.fixture(scope="session")
def data_dir() -> Path:
    return Path(__file__).parent / "data"


@pytest.fixture(scope="session")
def ts_ndvi(data_dir):
    yield (-3000, np.loadtxt(data_dir / "ts-ndvi.txt", dtype="int16"))


def to_da(xx):
    return xr.DataArray(
        data=da.from_array(xx.data),
        dims=xx.dims,
        coords=xx.coords,
        attrs=xx.attrs,
    )
