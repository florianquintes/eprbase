#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
© M. Sc. Florian Quintes, 2021-2022.

@contact: florian.quintes@pc.uni.freiburg.de

@author: Florian Quintes
"""
try:
    import cupy as cp
except ImportError as error:
    raise ImportError(
        "GPU-Unterstützung fehlt. Installiere sie mit: "
        "uv sync --extra gpu"
    ) from error

try:
    device_count = cp.cuda.runtime.getDeviceCount()
except cp.cuda.runtime.CUDARuntimeError as error:
    raise RuntimeError(
        "Keine funktionsfähige CUDA-GPU gefunden."
    ) from error

if device_count == 0:
    raise RuntimeError("Keine CUDA-GPU verfügbar.")
