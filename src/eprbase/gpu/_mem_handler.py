#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Memory management utilities for GPU-accelerated EPR simulations.

This module provides functions to determine optimal chunk sizes for memory-efficient
computation on GPU devices, using NVIDIA Management Library (NVML) for memory queries.
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
    Calculate the number of chunks needed for memory-efficient computation.

    Parameters
    ----------
    needed : int
        Total memory requirement in bytes for the computation.
    max_use : float, optional
        Fraction of available GPU memory to use (0.0, 1.0]. Default is 0.8.
    device : int, optional
        GPU device index. Default is 0.

    Returns
    -------
    int
        Number of chunks required to stay within memory limits.

    Raises
    ------
    ValueError
        If max_use is not in the range (0.0, 1.0].
    """
    if max_use <= 0.0 or max_use > 1.0:
        raise ValueError("max_use needs to be in (0., 1.].")

    free_mem = _get_available_memory(device) * max_use
    n_chunks = math.ceil(needed / free_mem)
    return n_chunks


def get_chunksize(n_chunks: int, arr_size: int) -> int:
    """
    Calculate the size of each computation chunk.

    Parameters
    ----------
    n_chunks : int
        Total number of chunks.
    arr_size : int
        Size of the dimension to be split for chunked computation.

    Returns
    -------
    int
        Size of each chunk for memory-efficient processing.
    """
    return math.floor(arr_size / n_chunks)


def _get_available_memory(device: int = 0) -> int:
    """
    Get the available free GPU memory in bytes.

    Parameters
    ----------
    device : int, optional
        GPU device index. Default is 0.

    Returns
    -------
    int
        Number of free bytes available on the specified GPU.
    """
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(device)
    info = nvmlDeviceGetMemoryInfo(handle)
    nvmlShutdown()
    return info.free
