def _init_extension():
    """Register SciPy functions with Numba.

    This entry_point is called by Numba when it initializes.
    """
    # pylint: disable=unused-import,import-outside-toplevel
    from . import sparse, special
