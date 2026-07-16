#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
© M. Sc. Florian Quintes, 2021-2022.

@contact: florian.quintes@pc.uni.freiburg.de

@author: Florian Quintes
"""
from pynvml import (
    nvmlInit,
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetMemoryInfo,
    nvmlShutdown,
)
import math


def get_chunks(needed: int, max_use: float = 0.8, device: int = 0) -> int:
    """
    Get the number of chunks.

    Parameters
    ----------
    needed : int
        Maximum needed number of bytes for the computation.
    max_use : float, optional
        Fraction of the available memory which will be used. (0., 1.]. The
        default is 0.8.
    device : int, optional
        Device number. Only needed for multiple GPU setup. The default is 0.

    Returns
    -------
    int
        Number of chunks.

    """
    if max_use <= 0.0 or max_use > 1.0:
        raise ValueError("max_use needs to be in (0., 1.].")

    free_mem = _get_available_memory(device) * max_use
    n_chunks = math.ceil(needed / free_mem)
    return n_chunks


def get_chunksize(n_chunks: int, arr_size: int) -> int:
    """
    Get the size of one chunk.

    Parameters
    ----------
    n_chunks : int
        Number of chunks.
    arr_size : int
        Size of the dimension at which the array will be sliced for chunked
        computation.

    Returns
    -------
    int
        Chunksize used for slicing.

    """
    return math.floor(arr_size / n_chunks)


def _get_available_memory(device: int = 0) -> int:
    """
    Get the available free GPU memory in bytes.

    Parameters
    ----------
    device : int, optional
        Device number. Only needed for multiple GPU setup. The default is 0.

    Returns
    -------
    int
        Number of  free bytes.

    """
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(device)
    info = nvmlDeviceGetMemoryInfo(handle)
    nvmlShutdown()
    return info.free
