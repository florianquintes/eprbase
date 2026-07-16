#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
© M. Sc. Florian Quintes, 2021-2022.

@contact: florian.quintes@pc.uni.freiburg.de

@author: Florian Quintes
"""
import cupy as cp
from cupyx.scipy.interpolate import RBFInterpolator, PchipInterpolator

# TODO: cupy.CubiCSpline verwenden, wenn die aktuelle Version veröffentlicht ist
# Mit der neuen Cupy-Version werden viele Interpolatoren kommen

CUPY_FLOAT = cp.float32


class Interpolator:

    def __init__(self, theta: cp.array, phi: cp.array, data: tuple):
        self._theta_or = theta
        self._phi_or = phi
        self._Dooh = cp.allclose(phi, cp.zeros(phi.shape, dtype=CUPY_FLOAT))
        self._xyz = self._get_xyz(self._theta_or, self._phi_or)
        (
            self._res_fields,
            self._intensities,
            self._widths,
            self._transitions,
        ) = data
        self._transitions = self._transitions[0]

        if self._Dooh:
            self._init_intensity_interpolator_dooh()
            self._init_position_interpolator_dooh()
            self._init_width_interpolator_dooh()
        else:
            self._init_intensity_interpolator()
            self._init_position_interpolator()
            self._init_width_interpolator()

    def get_intensities(self, theta, phi) -> cp.array:
        if self._Dooh:
            return self._intens_interp(theta)
        else:
            xyz = self._get_xyz(theta, phi)
            return self._intens_interp(xyz).astype(CUPY_FLOAT)

    def get_positions(self, theta, phi) -> cp.array:
        if self._Dooh:
            return self._pos_interp(theta)
        else:
            xyz = self._get_xyz(theta, phi)
            return self._pos_interp(xyz).astype(CUPY_FLOAT)

    def get_widths(self, theta, phi) -> cp.array:
        if self._Dooh:
            return self._widths_interp(theta)
        else:
            xyz = self._get_xyz(theta, phi)
            return self._widths_interp(xyz).astype(CUPY_FLOAT)

    def get_transitions(self, grid_points: int) -> cp.array:
        return cp.repeat(
            self._transitions[cp.newaxis, :, :], grid_points, axis=0
        )

    # TODO: neighbors wieder verwenden, sobald neue Version veröffentlicht ist
    def _init_intensity_interpolator(self):
        intensities = self._intensities  # TODO
        self._intens_interp = RBFInterpolator(
            self._xyz,
            intensities,
            smoothing=0,
            kernel="linear",  # , neighbors=6
        )

    def _init_position_interpolator(self):
        fields = self._res_fields  # TODO
        self._pos_interp = RBFInterpolator(
            self._xyz,
            fields,
            # neighbors=18,
            smoothing=0,
            kernel="thin_plate_spline",
        )

    def _init_width_interpolator(self):
        widths = self._widths  # TODO
        self._widths_interp = RBFInterpolator(
            self._xyz,
            widths,
            # neighbors=18,
            smoothing=0,
            kernel="thin_plate_spline",
        )

    # TODO: CubicSpline verwenden, sobald veröffentlicht
    def _init_intensity_interpolator_dooh(self):
        intensities = self._intensities  # TODO
        self._intens_interp = PchipInterpolator(self._theta_or, intensities)

    def _init_position_interpolator_dooh(self):
        fields = self._res_fields  # TODO
        self._pos_interp = PchipInterpolator(self._theta_or, fields)

    def _init_width_interpolator_dooh(self):
        widths = self._widths  # TODO
        self._widths_interp = PchipInterpolator(self._theta_or, widths)

    def _get_xyz(self, theta, phi):
        x = cp.sin(theta) * cp.cos(phi)
        y = cp.sin(theta) * cp.sin(phi)
        z = cp.cos(theta)
        return cp.array([x, y, z], dtype=CUPY_FLOAT).T
