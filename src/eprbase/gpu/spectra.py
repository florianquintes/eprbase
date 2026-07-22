#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EPR spectra construction utilities for GPU-accelerated simulations.

This module provides the :class:`Spectra` class to construct EPR spectra
from transition data using either summation or projection methods on GPU.
Uses CuPy for GPU acceleration and memory-efficient chunked computation.

© M. Sc. Florian Quintes, 2026.
@contact: florian.quintes@pc.uni.freiburg.de
@author: Florian Quintes
"""

import cupy as cp
from tqdm import trange
from eprbase.gpu._mem_handler import get_chunks, get_chunksize

CUPY_FLOAT = cp.float32
CUPY_CMPLX = cp.complex64


class Spectra:
    """
    EPR spectra constructor for GPU-accelerated simulations.

    Handles the construction of EPR spectra from transition data using
    either summation or projection methods on GPU. Supports memory-efficient
    chunked computation for large datasets.

    Parameters
    ----------
    res_fields : list of cp.array
        Resonance fields for each transition (shape: (M, N))
    intensities : list of cp.array
        Peak intensities for each transition (shape: (M, N))
    widths : list of cp.array
        Linewidths for each transition (shape: (M, N))
    transitions : list of cp.array
        Energy level indices for each transition (shape: (M, N, 2))
    weights : cp.array, optional
        Orientation weights from grid.get_areas() (shape: (M,))
    triangles : cp.array, optional
        Delaunay triangle indices from grid.get_triangle_idx() (shape: (P, 4))
    """

    def __init__(
        self,
        res_fields: list,
        intensities: list,
        widths: list,
        transitions: list,
        weights: cp.array = None,
        triangles: cp.array = None,
    ) -> None:
        """
        Initialize the Spectra object for GPU-accelerated EPR spectra construction.

        Parameters
        ----------
        res_fields : list of cp.array
            Resonance fields for each transition (shape: (M, N))
        intensities : list of cp.array
            Peak intensities for each transition (shape: (M, N))
        widths : list of cp.array
            Linewidths for each transition (shape: (M, N))
        transitions : list of cp.array
            Energy level indices for each transition (shape: (M, N, 2))
        weights : cp.array, optional
            Orientation weights from grid.get_areas() (shape: (M,))
        triangles : cp.array, optional
            Delaunay triangle indices from grid.get_triangle_idx() (shape: (P, 4))
        """
        self._res_fields = res_fields
        self._intensities = intensities
        self._widths = widths
        self._transitions = transitions
        self._weights = weights
        self._triangles = triangles

    def by_summation(self, field: cp.array) -> cp.array:
        """
        Construct spectra by simple summation of Gaussian peaks on GPU.

        Parameters
        ----------
        field : cp.array
            Magnetic field axis for the spectra (shape: (M,))

        Returns
        -------
        cp.array
            EPR spectra (shape: (M,))
        """
        repeats = [len(arr) for arr in self._res_fields]
        weights = cp.repeat(self._weights, repeats).astype(CUPY_FLOAT)[:, cp.newaxis]
        weights = cp.squeeze(weights)
        intensity = cp.concatenate(self._intensities)
        wintensity = intensity * weights
        del weights, intensity
        center = cp.concatenate(self._res_fields)
        sigma = cp.concatenate(self._widths)
        spectra = self._get_gaussian_sum(field, center, wintensity, sigma)
        return spectra

    def by_projection(self, field: cp.array) -> cp.array:
        """
        Construct spectra by projection onto Delaunay triangles on GPU.

        Parameters
        ----------
        field : cp.array
            Magnetic field axis for the spectra (shape: (M,))

        Returns
        -------
        cp.array
            EPR spectra (shape: (M,))
        """
        self._sort_by_transition()
        res_fields, intens, widths = self._get_points_for_projection()
        areas = self._triangles[:, 3]
        spectras = self._get_triangle_spec(field, res_fields, intens, areas, widths)
        spectra = cp.einsum("abc -> c", spectras)
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
        Sort transition data by energy level indices on GPU.

        Sorts intensities, resonance fields, and linewidths by transition indices
        to prepare for projection method. Maintains original grid point ordering.
        """
        self._transitions = cp.array(self._transitions)
        sorting = cp.lexsort(
            cp.array(
                [
                    self._transitions[:, :, 1].flatten(),
                    self._transitions[:, :, 0].flatten(),
                ],
                dtype=CUPY_FLOAT,
            )
        )

        shp = cp.array(self._intensities, dtype=CUPY_FLOAT).shape[::-1]
        self._sorted_intensities = (
            cp.array(self._intensities, dtype=CUPY_FLOAT)
            .flatten()[sorting]
            .reshape(shp)
        )
        self._sorted_fields = (
            cp.array(self._res_fields, dtype=CUPY_FLOAT).flatten()[sorting].reshape(shp)
        )
        self._sorted_widths = (
            cp.array(self._widths, dtype=CUPY_FLOAT).flatten()[sorting].reshape(shp)
        )

    def _get_points_for_projection(self) -> list[cp.array, cp.array, cp.array]:
        """
        Prepare transition data for projection method on GPU.

        Returns
        -------
        fields : cp.array
            Resonance fields for each triangle edge (shape: (nTransitions, nTriangles, 3))
        intens : cp.array
            Intensities for each triangle edge (shape: (nTransitions, nTriangles, 3))
        widths : cp.array
            Linewidths for each triangle edge (shape: (nTransitions, nTriangles, 3))
        """
        idx = self._triangles[:, 0:3].astype(cp.uint32)
        intens = self._sorted_intensities[:, idx]
        fields = self._sorted_fields[:, idx]
        widths = self._sorted_widths[:, idx]
        return fields, intens, widths

    def _get_triangle_spec(
        self,
        field: cp.array,
        res_fields: cp.array,
        intensities: cp.array,
        areas: cp.array,
        widths: cp.array,
    ) -> cp.array:
        """
        Construct spectra for individual Delaunay triangles on GPU.

        Parameters
        ----------
        field : cp.array
            Magnetic field axis (shape: (M,))
        res_fields : cp.array
            Resonance fields for triangle edges (shape: (nTransitions, nTriangles, 3))
        intensities : cp.array
            Intensities for triangle edges (shape: (nTransitions, nTriangles, 3))
        areas : cp.array
            Triangle areas (shape: (nTriangles,))
        widths : cp.array
            Linewidths for triangle edges (shape: (nTransitions, nTriangles, 3))

        Returns
        -------
        cp.array
            Subspectra for each transition and triangle (shape: (nTransitions, nTriangles, M))
        """
        heigth = areas * intensities.sum(axis=2) / 3
        x = cp.sort(res_fields)
        y = cp.zeros((*heigth.shape, 3), dtype=CUPY_FLOAT)
        y[:, :, 1] = heigth
        triangles = cp.empty((*res_fields.shape[:2], *field.shape), dtype=CUPY_FLOAT)
        print(heigth.min(), heigth.max())
        print(widths.shape)
        print(triangles.shape)
        # sigma = widths.mean(axis=2)
        n = 0
        for i in range(triangles.shape[0]):
            for j in range(triangles.shape[1]):
                if cp.all(cp.isclose(x[i, j], x[i, j][0])):
                    pos = cp.abs(field - x[i, j][0]).argmin()
                    triangles[i, j] = cp.zeros(field.size, dtype=CUPY_FLOAT)
                    triangles[i, j][pos] = heigth[i, j]
                else:
                    triangles[i, j] = cp.interp(field, x[i, j], y[i, j]).astype(
                        CUPY_FLOAT
                    )

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

    def _get_gaussian_sum(
        self,
        field: cp.array,
        center: cp.array,
        wintensity: cp.array,
        sigma: cp.array,
    ) -> cp.array:
        """
        Calculate all Gaussians and return their sum on GPU.

        Parameters
        ----------
        field : cp.array
            Magnetic field axis (shape: (M,))
        center : cp.array
            Centers of the Gaussians (=resonance fields)
        wintensity : cp.array
            Weighted intensity for each Gaussian
        sigma : cp.array
            Gaussian linewidth for each Gaussian

        Returns
        -------
        cp.array
            Sum of all Gaussians (shape: (M,))
        """
        sigma = (sigma**2 / cp.log(2, dtype=CUPY_FLOAT))[:, cp.newaxis]

        chunks = get_chunks(field.size * center.size * 4 * 4)
        chunk_size = get_chunksize(chunks, field.size)
        gaussians = cp.empty(field.shape, dtype=CUPY_FLOAT)
        for i in trange(chunks):
            lb = i * chunk_size
            if i < chunks - 1:
                rb = (i + 1) * chunk_size
            else:
                rb = field.size
            b = field[cp.newaxis, lb:rb] - center[:, cp.newaxis]
            exp = -(b**2) / sigma
            del b
            gaussians[lb:rb] = (wintensity[:, cp.newaxis] * cp.exp(exp)).sum(axis=0)
            del exp

        return gaussians


# from scipy.special import erf


# def conv_function(x, gamma):
#     x_g = x / gamma
#     F_x = np.exp(-2 * (x_g**2)) / (np.sqrt(2 * np.pi)) + x_g * (
#         1 + erf(np.sqrt(2) * x_g)
#     )
#     return F_x


# def elementary_spec(field, y, gamma):
#     F_y1 = conv_function(field - y[0], gamma)
#     F_y2 = conv_function(field - y[1], gamma)
#     F_y2_ = conv_function(y[1] - field, gamma)
#     F_y3 = conv_function(y[2] - field, gamma)
#     S_y = (
#         gamma
#         / (y[2] - y[0])
#         * (
#             (F_y1 - F_y2) / (y[1] - y[0])
#             + (F_y3 - F_y2_) / (y[2] - y[1])
#             - 2 / gamma
#         )
#     )
#     return S_y
