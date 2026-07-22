#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interpolation utilities for EPR simulation data.

This module provides the :class:`Interpolator` class to perform 3D interpolation
of EPR simulation results (intensities, field positions, linewidths) on spherical
grids with different symmetries (Dooh and general cases).

© M. Sc. Florian Quintes, 2026.
@contact: florian.quintes@pc.uni.freiburg.de
@author: Florian Quintes
"""

import numpy as np
from scipy.interpolate import RBFInterpolator, CubicSpline


class Interpolator:
    """
    Spherical data interpolator for EPR simulations.

    Handles interpolation of EPR simulation results on spherical grids with
    different symmetries (Dooh and general cases). Supports interpolation
    of intensities, field positions, linewidths, and transition matrices.

    Parameters
    ----------
    theta : np.array
        Original theta angles in radians.
    phi : np.array
        Original phi angles in radians.
    data : tuple
        Tuple containing (field_positions, intensities, linewidths, transitions).

    Attributes
    ----------
    _Dooh : bool
        Whether the data has Dooh symmetry.
    _xyz : np.array
        Cartesian coordinates of original data points.
    _res_fields : np.array
        Resonance field positions.
    _intensities : np.array
        Signal intensities.
    _widths : np.array
        Linewidths.
    _transitions : np.array
        Transition matrices.
    """

    def __init__(self, theta: np.array, phi: np.array, data: tuple):
        """
        Initialize the interpolator with simulation data.

        Parameters
        ----------
        theta : np.array
            Original theta angles in radians.
        phi : np.array
            Original phi angles in radians.
        data : tuple
            Tuple containing (field_positions, intensities, linewidths, transitions).
        """
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
        """
        Interpolate signal intensities for given angles.

        Parameters
        ----------
        theta : np.array
            Theta angles in radians.
        phi : np.array
            Phi angles in radians.

        Returns
        -------
        np.array
            Interpolated intensities.
        """
        if self._Dooh:
            return self._intens_interp(theta)
        else:
            xyz = self._get_xyz(theta, phi)
            return self._intens_interp(xyz)

    def get_positions(self, theta, phi) -> np.array:
        """
        Interpolate resonance field positions for given angles.

        Parameters
        ----------
        theta : np.array
            Theta angles in radians.
        phi : np.array
            Phi angles in radians.

        Returns
        -------
        np.array
            Interpolated field positions.
        """
        if self._Dooh:
            return self._pos_interp(theta)
        else:
            xyz = self._get_xyz(theta, phi)
            return self._pos_interp(xyz)

    def get_widths(self, theta, phi) -> np.array:
        """
        Interpolate linewidths for given angles.

        Parameters
        ----------
        theta : np.array
            Theta angles in radians.
        phi : np.array
            Phi angles in radians.

        Returns
        -------
        np.array
            Interpolated linewidths.
        """
        if self._Dooh:
            return self._widths_interp(theta)
        else:
            xyz = self._get_xyz(theta, phi)
            return self._widths_interp(xyz)

    def get_transitions(self, grid_points: int) -> np.array:
        """
        Get transition matrices for given grid points.

        Parameters
        ----------
        grid_points : int
            Number of grid points.

        Returns
        -------
        np.array
            Transition matrices repeated for each grid point.
        """
        return np.repeat(self._transitions[np.newaxis, :, :], grid_points, axis=0)

    def _init_intensity_interpolator(self):
        """
        Initialize general 3D intensity interpolator.

        Uses radial basis function interpolation with linear kernel.
        """
        intensities = self._intensities  # TODO
        self._intens_interp = RBFInterpolator(
            self._xyz, intensities, neighbors=6, smoothing=0, kernel="linear"
        )

    def _init_position_interpolator(self):
        """
        Initialize general 3D field position interpolator.

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

    def _init_width_interpolator(self):
        """
        Initialize general 3D linewidth interpolator.

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

    def _init_intensity_interpolator_dooh(self):
        """
        Initialize intensity interpolator for Dooh symmetry.

        Uses cubic spline interpolation along the theta axis.
        """
        intensities = self._intensities  # TODO
        self._intens_interp = CubicSpline(self._theta_or, intensities)

    def _init_position_interpolator_dooh(self):
        """
        Initialize field position interpolator for Dooh symmetry.

        Uses cubic spline interpolation along the theta axis.
        """
        fields = self._res_fields  # TODO
        self._pos_interp = CubicSpline(self._theta_or, fields)

    def _init_width_interpolator_dooh(self):
        """
        Initialize linewidth interpolator for Dooh symmetry.

        Uses cubic spline interpolation along the theta axis.
        """
        widths = self._widths  # TODO
        self._widths_interp = CubicSpline(self._theta_or, widths)

    def _get_xyz(self, theta, phi):
        """
        Convert spherical coordinates to Cartesian coordinates.

        Parameters
        ----------
        theta : np.array
            Theta angles in radians.
        phi : np.array
            Phi angles in radians.

        Returns
        -------
        np.array
            Cartesian coordinates (x, y, z).
        """
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        return np.array([x, y, z]).T
