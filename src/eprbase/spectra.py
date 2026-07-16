#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
© M. Sc. Florian Quintes, 2021-2022.

@contact: florian.quintes@pc.uni.freiburg.de

@author: Florian Quintes
"""
import numpy as np
import numexpr as ne


class Spectra:
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
        res_fields : list, (M, N)
            Resonance field of each transition.
        intensities : list, (M, N)
            Peak intensity of each transition.
        widths : list, (M, N)
            Linewidth of each peak.
        transitions : list, (M, N, 2)
            Energy level indices for each transition.
        weights : np.array, (M,)
            Weigth for each orientation used for the summation. Provided by
            epr_grid.Grid().get_areas().
        triangles : np.array, (P, 4)
            Indices for each Delaunay triangle with its areas used for
            projection. Provided by epr_grid.Grid().get_triangle_idx()

        """
        self._res_fields = res_fields
        self._intensities = intensities
        self._widths = widths
        self._transitions = transitions
        self._weights = weights
        self._triangles = triangles

    def by_summation(self, field: np.array) -> np.array:
        """
        Construct the spectra by simple summation of each Gaussian.

        Parameters
        ----------
        field : np.array, (M,)
            Magnetic field axis for the spectra.

        Returns
        -------
        spectra : np.array, (M,)
            EPR spectra.

        """
        repeats = [len(arr) for arr in self._res_fields]
        weights = np.repeat(self._weights, repeats).astype(np.float32)[
            :, np.newaxis
        ]
        center = np.concatenate(self._res_fields).astype(np.float32)
        intensity = np.concatenate(self._intensities).astype(np.float32)
        sigma = np.concatenate(self._widths).astype(np.float32)
        gauss = self._get_gaussian(field, center, intensity, sigma)
        spectra = np.sum(gauss * weights, axis=0)
        return spectra

    def by_projection(self, field: np.array) -> np.array:
        """
        Construct the spectra by projection.

        Parameters
        ----------
        field : np.array, (M,)
            Magnetic field axis for the spectra.

        Returns
        -------
        spectra : np.array, (M,)
            EPR spectra.

        """
        self._sort_by_transition()
        res_fields, intens, widths = self._get_points_for_projection()
        areas = self._triangles[:, 3]
        spectras = self._get_triangle_spec(
            field, res_fields, intens, areas, widths
        )
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
        Sort intensity, resonance field and linewidths for each transition.

        axis=0 : transitions, axis=1 : grid points.
        Hence `self._sorted_intensities[0, 1]` will return the intensity of the
        first transition at the second grid point.
        `self._sorted_intensities[0]` returns all intensities for the first
        transition. The order of the grid points is not affected by the
        sorting.

        Returns
        -------
        None.

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
            self._sorted_intensities[i] = np.array(self._intensities)[i][
                sorting[i]
            ]
            self._sorted_fields[i] = np.array(self._res_fields)[i][sorting[i]]
            self._sorted_widths[i] = np.array(self._widths)[i][sorting[i]]

        self._sorted_intensities = self._sorted_intensities.T
        self._sorted_fields = self._sorted_fields.T
        self._sorted_widths = self._sorted_widths.T

    def _get_points_for_projection(self) -> [np.array, np.array, np.array]:
        """
        Get the intensities, resonance fields and linewidths for each triangle.

        Returns
        -------
        fields : np.array, (nTransitions, nTriangles, 3)
            Resonance fields.
        intens : np.array, (nTransitions, nTriangles, 3)
            Intensities.
        widths : np.array, (nTransitions, nTriangles, 3)
            Linewidths.
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
        Construct the triangle spectrum for one Delaunay triangle.

        Parameters
        ----------
        field : np.array, (M,)
            Magnetic field axis for the spectra.
        res_fields : np.array, (nTransitions, nTriangles, 3)
            Resonance field for each edge of the Delaunay triangle.
        intensities : np.array, (nTransitions, nTriangles, 3)
            Intensity for each edge of the Delaunay triangle.
        areas : np.array, (nTriangles)
            Areas of the Delaunay triangles.

        Returns
        -------
        triangles : np.array
            All subspectra for each transition and each Delaunay triangle.

        """
        heigth = areas * intensities.sum(axis=2) / 3
        x = np.sort(res_fields)
        y = np.zeros((*heigth.shape, 3))
        y[:, :, 1] = heigth
        triangles = np.empty((*res_fields.shape[:2], *field.shape))
        print(heigth.min(), heigth.max())
        print(widths.shape)
        print(triangles.shape)
        sigma = widths.mean(axis=2)
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
        Get the gaussian lineshape along the given field for multiple ones.

        Parameters
        ----------
        field : np.array, (M,)
            Magnetic field axis.
        center : np.array, (N,)
            Center point of each gaussian.
        intensity : np.array, (N,)
            Peak intensity of each gaussian.
        sigma : np.array, (N,)
            FWHM of each gaussian.

        Returns
        -------
        gaussians : np.array, (N, M)
            Gaussian lineshapes.
        """
        sigma = sigma.astype(np.float32)
        intensity = intensity.astype(np.float32)
        field = field.astype(np.float32)
        center = center.astype(np.float32)
        sigma = (sigma**2 / np.log(2))[:, np.newaxis]

        b = field[np.newaxis, :] - center[:, np.newaxis]

        gaussians_1 = intensity[:, np.newaxis] * ne.evaluate(
            "exp(-(b**2) / sigma)"
        )
        return gaussians_1


from scipy.special import erf


def conv_function(x, gamma):
    x_g = x / gamma
    F_x = np.exp(-2 * (x_g**2)) / (np.sqrt(2 * np.pi)) + x_g * (
        1 + erf(np.sqrt(2) * x_g)
    )
    return F_x


def elementary_spec(field, y, gamma):
    F_y1 = conv_function(field - y[0], gamma)
    F_y2 = conv_function(field - y[1], gamma)
    F_y2_ = conv_function(y[1] - field, gamma)
    F_y3 = conv_function(y[2] - field, gamma)
    S_y = (
        gamma
        / (y[2] - y[0])
        * (
            (F_y1 - F_y2) / (y[1] - y[0])
            + (F_y3 - F_y2_) / (y[2] - y[1])
            - 2 / gamma
        )
    )
    return S_y
