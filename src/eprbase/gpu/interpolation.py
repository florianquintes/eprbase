#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interpolation utilities for EPR simulation data on GPU.

This module provides the :class:`Interpolator` class to perform 3D interpolation
of EPR simulation results (intensities, field positions, linewidths) on spherical
grids with different symmetries (Dooh and general cases) using CuPy for GPU acceleration.

© M. Sc. Florian Quintes, 2026.
@contact: florian.quintes@pc.uni.freiburg.de
@author: Florian Quintes
"""

import cupy as cp
from cupyx.scipy.interpolate import RBFInterpolator, CubicSpline

CUPY_FLOAT = cp.float32


class Interpolator:
    """
    Spherical data interpolator for EPR simulations on GPU.

    Handles interpolation of EPR simulation results on spherical grids with
    different symmetries (Dooh and general cases) using CuPy arrays. Supports
    interpolation of intensities, field positions, linewidths, and transition
    matrices with GPU acceleration.

    Parameters
    ----------
    theta : cp.array
        Original theta angles in radians.
    phi : cp.array
        Original phi angles in radians.
    data : tuple
        Tuple containing (field_positions, intensities, linewidths, transitions).

    Attributes
    ----------
    _Dooh : bool
        Whether the data has Dooh symmetry.
    _xyz : cp.array
        Cartesian coordinates of original data points.
    _res_fields : cp.array
        Resonance field positions.
    _intensities : cp.array
        Signal intensities.
    _widths : cp.array
        Linewidths.
    _transitions : cp.array
        Transition matrices.
    """

    def __init__(self, theta: cp.array, phi: cp.array, data: tuple):
        """
        Initialize the interpolator with simulation data on GPU.

        Parameters
        ----------
        theta : cp.array
            Original theta angles in radians.
        phi : cp.array
            Original phi angles in radians.
        data : tuple
            Tuple containing (field_positions, intensities, linewidths, transitions).
        """
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
        """
        Interpolate signal intensities for given angles on GPU.

        Parameters
        ----------
        theta : cp.array
            Theta angles in radians.
        phi : cp.array
            Phi angles in radians.

        Returns
        -------
        cp.array
            Interpolated intensities.
        """
        if self._Dooh:
            return self._intens_interp(theta)
        else:
            xyz = self._get_xyz(theta, phi)
            return self._intens_interp(xyz).astype(CUPY_FLOAT)

    def get_positions(self, theta, phi) -> cp.array:
        """
        Interpolate resonance field positions for given angles on GPU.

        Parameters
        ----------
        theta : cp.array
            Theta angles in radians.
        phi : cp.array
            Phi angles in radians.

        Returns
        -------
        cp.array
            Interpolated field positions.
        """
        if self._Dooh:
            return self._pos_interp(theta)
        else:
            xyz = self._get_xyz(theta, phi)
            return self._pos_interp(xyz).astype(CUPY_FLOAT)

    def get_widths(self, theta, phi) -> cp.array:
        """
        Interpolate linewidths for given angles on GPU.

        Parameters
        ----------
        theta : cp.array
            Theta angles in radians.
        phi : cp.array
            Phi angles in radians.

        Returns
        -------
        cp.array
            Interpolated linewidths.
        """
        if self._Dooh:
            return self._widths_interp(theta)
        else:
            xyz = self._get_xyz(theta, phi)
            return self._widths_interp(xyz).astype(CUPY_FLOAT)

    def get_transitions(self, grid_points: int) -> cp.array:
        """
        Get transition matrices for given grid points on GPU.

        Parameters
        ----------
        grid_points : int
            Number of grid points.

        Returns
        -------
        cp.array
            Transition matrices repeated for each grid point.
        """
        return cp.repeat(self._transitions[cp.newaxis, :, :], grid_points, axis=0)

    def _init_intensity_interpolator(self) -> None:
        """
        Initialize intensity interpolator for general cases on GPU.

        Uses radial basis function interpolation with linear kernel.
        """
        intensities = self._intensities  # TODO
        self._intens_interp = RBFInterpolator(
            self._xyz, intensities, smoothing=0, kernel="linear", neighbors=6
        )

    def _init_position_interpolator(self) -> None:
        """
        Initialize field position interpolator for general cases on GPU.

        Uses radial basis function interpolation with thin-plate spline kernel.
        """
        fields = self._res_fields  # TODO
        self._pos_interp = RBFInterpolator(
            self._xyz,
            fields,
            neighbors=18,
            smoothing=0,
            kernel="thin_plate_spline",
        )

    def _init_width_interpolator(self) -> None:
        """
        Initialize linewidth interpolator for general cases on GPU.

        Uses radial basis function interpolation with thin-plate spline kernel.
        """
        widths = self._widths  # TODO
        self._widths_interp = RBFInterpolator(
            self._xyz,
            widths,
            neighbors=18,
            smoothing=0,
            kernel="thin_plate_spline",
        )

    def _init_intensity_interpolator_dooh(self) -> None:
        """
        Initialize cubic spline interpolator for intensity data with Dooh symmetry.

        Uses cubic spline interpolation along the theta axis for Dooh-symmetric data.
        """
        intensities = self._intensities  # TODO
        self._intens_interp = CubicSpline(self._theta_or, intensities)

    def _init_position_interpolator_dooh(self) -> None:
        """
        Initialize cubic spline interpolator for field position data with Dooh symmetry.

        Uses cubic spline interpolation along the theta axis for Dooh-symmetric data.
        """
        fields = self._res_fields  # TODO
        self._pos_interp = CubicSpline(self._theta_or, fields)

    def _init_width_interpolator_dooh(self) -> None:
        """
        Initialize cubic spline interpolator for linewidth data with Dooh symmetry.

        Uses cubic spline interpolation along the theta axis for Dooh-symmetric data.
        """
        widths = self._widths  # TODO
        self._widths_interp = CubicSpline(self._theta_or, widths)

    def _get_xyz(self, theta, phi) -> cp.array:
        """
        Convert spherical coordinates to Cartesian coordinates on GPU.

        Parameters
        ----------
        theta : cp.array
            Theta angles in radians.
        phi : cp.array
            Phi angles in radians.

        Returns
        -------
        cp.array
            Cartesian coordinates (x, y, z).
        """
        x = cp.sin(theta) * cp.cos(phi)
        y = cp.sin(theta) * cp.sin(phi)
        z = cp.cos(theta)
        return cp.array([x, y, z], dtype=CUPY_FLOAT).T
