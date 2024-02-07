"""Numba accelerated zonal statistics."""

from numba import njit
import numpy as np

from ._helper import lazycompile


@lazycompile(njit)
def do_mean(pixels, z_pixels, num_zones, nodata, z_nodata, out_dtype=np.float32):
    """Calculate the zonal mean.

    The mean for each pixel of `pixels` is calculated for each zone in
    `z_pixels` and returned with the number of valid pixels

    The zones in `z_pixels` have to be numbered in a linear fashion,
    starting with 0 for the first zone and num_zones for the last zone.

    The shape of the returned array is (T, num_zones, 2).

    Args:
        pixels: input value pixels (T,Y,X)
        z_pixels: zonal pixels (Y,X)
        nodata: nodata value in values
        num_zones: number of zones in z_pixels
        out_dtype: datatype
    """
    t, nr, nc = pixels.shape
    result = np.zeros((t, num_zones, 2), dtype=out_dtype)

    # 0 mean
    # 1 valids

    for tix in range(t):
        for rw in range(nr):
            for cl in range(nc):
                pix = pixels[tix, rw, cl]
                z_idx = z_pixels[rw, cl]
                if (pix != nodata) and (z_idx != z_nodata):
                    result[tix, z_idx, 0] += pix
                    result[tix, z_idx, 1] += 1

        for idx in range(result.shape[1]):
            if result[tix, idx, 1] > 0:
                result[tix, idx, 0] = result[tix, idx, 0] / result[tix, idx, 1]
            else:
                result[tix, idx, 0] = np.nan

    return result
