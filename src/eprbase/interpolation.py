#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
© M. Sc. Florian Quintes, 2021-2022.

@contact: florian.quintes@pc.uni.freiburg.de

@author: Florian Quintes
"""
import numpy as np
from scipy.interpolate import RBFInterpolator, CubicSpline


class Interpolator:

    def __init__(self, theta: np.array, phi: np.array, data: tuple):
        self._theta_or = theta
        self._phi_or = phi
        self._Dooh = np.allclose(phi, np.zeros(phi.shape))
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

    def get_intensities(self, theta, phi) -> np.array:
        if self._Dooh:
            return self._intens_interp(theta)
        else:
            xyz = self._get_xyz(theta, phi)
            return self._intens_interp(xyz)

    def get_positions(self, theta, phi) -> np.array:
        if self._Dooh:
            return self._pos_interp(theta)
        else:
            xyz = self._get_xyz(theta, phi)
            return self._pos_interp(xyz)

    def get_widths(self, theta, phi) -> np.array:
        if self._Dooh:
            return self._widths_interp(theta)
        else:
            xyz = self._get_xyz(theta, phi)
            return self._widths_interp(xyz)

    def get_transitions(self, grid_points: int) -> np.array:
        return np.repeat(
            self._transitions[np.newaxis, :, :], grid_points, axis=0
        )

    def _init_intensity_interpolator(self):
        intensities = self._intensities  # TODO
        self._intens_interp = RBFInterpolator(
            self._xyz, intensities, neighbors=6, smoothing=0, kernel="linear"
        )

    def _init_position_interpolator(self):
        fields = self._res_fields  # TODO
        self._pos_interp = RBFInterpolator(
            self._xyz,
            fields,
            neighbors=18,
            smoothing=0,
            kernel="thin_plate_spline",
        )

    def _init_width_interpolator(self):
        widths = self._widths  # TODO
        self._widths_interp = RBFInterpolator(
            self._xyz,
            widths,
            neighbors=18,
            smoothing=0,
            kernel="thin_plate_spline",
        )

    def _init_intensity_interpolator_dooh(self):
        intensities = self._intensities  # TODO
        self._intens_interp = CubicSpline(self._theta_or, intensities)

    def _init_position_interpolator_dooh(self):
        fields = self._res_fields  # TODO
        self._pos_interp = CubicSpline(self._theta_or, fields)

    def _init_width_interpolator_dooh(self):
        widths = self._widths  # TODO
        self._widths_interp = CubicSpline(self._theta_or, widths)

    def _get_xyz(self, theta, phi):
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        return np.array([x, y, z]).T
