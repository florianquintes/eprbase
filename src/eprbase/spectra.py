#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EPR spectra construction utilities.

This module provides the :class:`Spectra` class to construct EPR spectra
from transition data using either summation or projection methods. Supports
both Gaussian lineshapes and convolution-based spectral construction.

© M. Sc. Florian Quintes, 2026.
@contact: florian.quintes@pc.uni.freiburg.de
@author: Florian Quintes
"""

import numpy as np
import numexpr as ne
from scipy.special import erf


class Spectra:
    """
    EPR spectra constructor.

    Handles the construction of EPR spectra from transition data using
    either summation or projection methods. Supports both Gaussian lineshapes
    and convolution-based spectral construction.

    Parameters
    ----------
    res_fields : list of np.array
        Resonance fields for each transition (shape: (M, N))
    intensities : list of np.array
        Peak intensities for each transition (shape: (M, N))
    widths : list of np.array
        Linewidths for each transition (shape: (M, N))
    transitions : list of np.array
        Energy level indices for each transition (shape: (M, N, 2))
    weights : np.array, optional
        Orientation weights from grid.get_areas() (shape: (M,))
    triangles : np.array, optional
        Delaunay triangle indices from grid.get_triangle_idx() (shape: (P, 4))
    """

    def __init__(
        self,
        res_fields: list,
        intensities: list,
        widths: list,
        transitions: list,
        weights: np.array = None,
        triangles: np.array = None,
    ) -> None:
        """
        Initialize the Spectra object.

        Parameters
        ----------
        res_fields : list of np.array
            Resonance fields for each transition (shape: (M, N))
        intensities : list of np.array
            Peak intensities for each transition (shape: (M, N))
        widths : list of np.array
            Linewidths for each transition (shape: (M, N))
        transitions : list of np.array
            Energy level indices for each transition (shape: (M, N, 2))
        weights : np.array, optional
            Orientation weights from grid.get_areas() (shape: (M,))
        triangles : np.array, optional
            Delaunay triangle indices from grid.get_triangle_idx() (shape: (P, 4))
        """
        self._res_fields = res_fields
        self._intensities = intensities
        self._widths = widths
        self._transitions = transitions
        self._weights = weights
        self._triangles = triangles

    def by_summation(self, field: np.array) -> np.array:
        """
        Construct spectra by simple summation of Gaussian peaks.

        Parameters
        ----------
        field : np.array
            Magnetic field axis for the spectra (shape: (M,))

        Returns
        -------
        np.array
            EPR spectra (shape: (M,))
        """
        repeats = [len(arr) for arr in self._res_fields]
        weights = np.repeat(self._weights, repeats).astype(np.float32)[:, np.newaxis]
        center = np.concatenate(self._res_fields).astype(np.float32)
        intensity = np.concatenate(self._intensities).astype(np.float32)
        sigma = np.concatenate(self._widths).astype(np.float32)
        gauss = self._get_gaussian(field, center, intensity, sigma)
        spectra = np.sum(gauss * weights, axis=0)
        return spectra

    def by_projection(self, field: np.array) -> np.array:
        """
        Construct spectra by projection onto Delaunay triangles.

        Parameters
        ----------
        field : np.array
            Magnetic field axis for the spectra (shape: (M,))

        Returns
        -------
        np.array
            EPR spectra (shape: (M,))
        """
        self._sort_by_transition()
        res_fields, intens, widths = self._get_points_for_projection()
        areas = self._triangles[:, 3]
        spectras = self._get_triangle_spec(field, res_fields, intens, areas, widths)
        spectra = np.einsum("abc -> c", spectras)
        # sig = widths.mean() ** 2 / np.log(2)
        # gaussian = np.exp(-((field - field.mean()) ** 2 / sig))
        # spectra = np.convolve(spectra, gaussian, "same")

        # import matplotlib.pyplot as plt

        # plt.figure()
        # plt.plot(field, gaussian)
        # plt.show()
        return spectra

    def _sort_by_transition(self) -> None:
        """
        Sort transition data by energy level indices.

        Sorts intensities, resonance fields, and linewidths by transition indices
        to prepare for projection method. Maintains original grid point ordering.

        Notes
        -----
        After sorting, data is transposed to shape (grid_points, transitions).
        """
        sorting = np.lexsort(
            (
                np.array(self._transitions)[:, :, 1],
                np.array(self._transitions)[:, :, 0],
            )
        )
        shp = np.array(self._intensities).shape
        self._sorted_intensities = np.empty(shp)
        self._sorted_fields = np.empty(shp)
        self._sorted_widths = np.empty(shp)

        for i in range(shp[0]):
            self._sorted_intensities[i] = np.array(self._intensities)[i][sorting[i]]
            self._sorted_fields[i] = np.array(self._res_fields)[i][sorting[i]]
            self._sorted_widths[i] = np.array(self._widths)[i][sorting[i]]

        self._sorted_intensities = self._sorted_intensities.T
        self._sorted_fields = self._sorted_fields.T
        self._sorted_widths = self._sorted_widths.T

    def _get_points_for_projection(self) -> list[np.array, np.array, np.array]:
        """
        Prepare transition data for projection method.

        Returns
        -------
        fields : np.array
            Resonance fields for each triangle edge (shape: (nTransitions, nTriangles, 3))
        intens : np.array
            Intensities for each triangle edge (shape: (nTransitions, nTriangles, 3))
        widths : np.array
            Linewidths for each triangle edge (shape: (nTransitions, nTriangles, 3))
        """
        idx = np.int32(self._triangles[:, 0:3])
        intens = self._sorted_intensities[:, idx]
        fields = self._sorted_fields[:, idx]
        widths = self._sorted_widths[:, idx]
        return fields, intens, widths

    def _get_triangle_spec(
        self,
        field: np.array,
        res_fields: np.array,
        intensities: np.array,
        areas: np.array,
        widths: np.array,
    ) -> np.array:
        """
        Construct spectra for individual Delaunay triangles.

        Parameters
        ----------
        field : np.array
            Magnetic field axis (shape: (M,))
        res_fields : np.array
            Resonance fields for triangle edges (shape: (nTransitions, nTriangles, 3))
        intensities : np.array
            Intensities for triangle edges (shape: (nTransitions, nTriangles, 3))
        areas : np.array
            Triangle areas (shape: (nTriangles,))
        widths : np.array
            Linewidths for triangle edges (shape: (nTransitions, nTriangles, 3))

        Returns
        -------
        np.array
            Subspectra for each transition and triangle (shape: (nTransitions, nTriangles, M))
        """
        heigth = areas * intensities.sum(axis=2) / 3
        x = np.sort(res_fields)
        y = np.zeros((*heigth.shape, 3))
        y[:, :, 1] = heigth
        triangles = np.empty((*res_fields.shape[:2], *field.shape))
        print(heigth.min(), heigth.max())
        print(widths.shape)
        print(triangles.shape)
        # sigma = widths.mean(axis=2)
        n = 0
        for i in range(triangles.shape[0]):
            for j in range(triangles.shape[1]):
                if np.all(np.isclose(x[i, j], x[i, j][0])):
                    pos = np.abs(field - x[i, j][0]).argmin()
                    triangles[i, j] = np.zeros(field.size)
                    triangles[i, j][pos] = heigth[i, j]
                else:
                    triangles[i, j] = np.interp(field, x[i, j], y[i, j])

                # sig = sigma[i, j] ** 2 / np.log(2)
                # gaussian = (
                #     1
                #     / sigma[i, j]
                #     * np.exp(-((field - field.mean()) ** 2 / sig))
                # )
                # triangles[i, j] = np.convolve(
                #     triangles[i, j], gaussian, "same"
                # )
                # spread = x[i, j, 2] - x[i, j, 0]
                # if spread > 0:
                #     lambda_b = sigma[i, j] / spread
                #     alpha = 1
                #     c = 0.154
                #     c_1 = 1.57246
                #     c_2 = 18.6348
                #     sig_1 = sigma[i, j] * (1 + alpha * c / (lambda_b**2))
                #     sig_2 = sigma[i, j] * (
                #         1
                #         + alpha
                #         / np.sqrt(c_1 * lambda_b**2 + c_2 * lambda_b**2)
                #     )
                #     sig = sig_2
                # else:
                #     sig = sigma[i, j]
                # triangles[i, j] = self._get_single_gaussian(
                #     field, x[i, j].mean(), y[i, j, 1], sig
                # )
                # if x[i, j, 0] < x[i, j, 1] and x[i, j, 1] < x[i, j, 2]:
                #     sig = sigma[i, j] * 1.7
                #     triangles[i, j] = heigth[i, j] * elementary_spec(
                #         field, x[i, j], sig
                #     )
                #     # triangles[i, j] = np.zeros(field.size)
                # else:
                #     sig = sigma[i, j]
                #     triangles[i, j] = self._get_single_gaussian(
                #         field, x[i, j].mean(), y[i, j, 1], sig
                #     )
                #     n += 1
        print(n)

        return triangles

    def _get_single_gaussian(self, field, center, intensity, sigma):
        """
        Generate Gaussian lineshape for a single peak.

        Parameters
        ----------
        field : np.array
            Magnetic field axis (shape: (M,))
        center : float
            Peak center position
        intensity : float
            Peak intensity
        sigma : float
            Linewidth (FWHM)

        Returns
        -------
        np.array
            Gaussian lineshape (shape: (M,))
        """
        sigma = sigma**2 / np.log(2)
        b = field - center
        gaussian = intensity * np.exp(-(b**2) / sigma)
        return gaussian

    def _get_gaussian(
        self,
        field: np.array,
        center: np.array,
        intensity: np.array,
        sigma: np.array,
    ) -> np.array:
        """
        Generate Gaussian lineshapes for multiple peaks.

        Parameters
        ----------
        field : np.array
            Magnetic field axis (shape: (M,))
        center : np.array
            Peak center positions (shape: (N,))
        intensity : np.array
            Peak intensities (shape: (N,))
        sigma : np.array
            Linewidths (FWHM) (shape: (N,))

        Returns
        -------
        np.array
            Gaussian lineshapes (shape: (N, M))
        """
        sigma = sigma.astype(np.float32)
        intensity = intensity.astype(np.float32)
        field = field.astype(np.float32)
        center = center.astype(np.float32)
        sigma = (sigma**2 / np.log(2))[:, np.newaxis]

        b = field[np.newaxis, :] - center[:, np.newaxis]

        gaussians_1 = intensity[:, np.newaxis] * ne.evaluate(
            "exp(-(field**2) / sigma)", local_dict={"field": b, "sigma": sigma}
        )
        return gaussians_1


def conv_function(x, gamma):
    """
    Convolution function for elementary spectra construction.

    Parameters
    ----------
    x : np.array
        Field values
    gamma : float
        Linewidth parameter

    Returns
    -------
    np.array
        Convolution result
    """
    x_g = x / gamma
    F_x = np.exp(-2 * (x_g**2)) / (np.sqrt(2 * np.pi)) + x_g * (
        1 + erf(np.sqrt(2) * x_g)
    )
    return F_x


def elementary_spec(field, y, gamma):
    """
    Construct elementary spectrum for triangular region.

    Parameters
    ----------
    field : np.array
        Magnetic field axis
    y : np.array
        Triangle edge positions
    gamma : float
        Linewidth parameter

    Returns
    -------
    np.array
        Elementary spectrum
    """
    F_y1 = conv_function(field - y[0], gamma)
    F_y2 = conv_function(field - y[1], gamma)
    F_y2_ = conv_function(y[1] - field, gamma)
    F_y3 = conv_function(y[2] - field, gamma)
    S_y = (
        gamma
        / (y[2] - y[0])
        * ((F_y1 - F_y2) / (y[1] - y[0]) + (F_y3 - F_y2_) / (y[2] - y[1]) - 2 / gamma)
    )
    return S_y
