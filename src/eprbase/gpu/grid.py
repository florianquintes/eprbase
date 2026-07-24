#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Spherical grid generation utilities for EPR simulations on GPU.

This module provides the :class:`Grid` class to generate SOPHE-like spherical
grids under different point-group symmetries and to compute integration weights,
Voronoi areas, and spherical triangle indices using CuPy for GPU acceleration.

© M. Sc. Florian Quintes, 2026.
@contact: florian.quintes@pc.uni.freiburg.de
@author: Florian Quintes
"""

import numpy as np
import cupy as cp
from scipy.spatial import SphericalVoronoi, geometric_slerp
import matplotlib.pyplot as plt


class Grid:
    """
    Spherical integration grid for EPR simulations on GPU.

    The grid is generated in spherical coordinates and can be returned either in
    spherical or Cartesian representation. Symmetry-dependent reductions are
    supported through point groups, with corresponding weight factors for
    integration on the sphere.

    Parameters
    ----------
    grid : str, default="SOPHE"
        Grid family to generate. Currently only ``"SOPHE"`` is supported.
    point_group : str, default="Ci"
        Point-group symmetry used to construct the reduced grid.
    knots : int, default=15
        Resolution parameter controlling the number of grid knots.
    """

    def __init__(self, grid: str = "SOPHE", point_group="Ci", knots: int = 15):
        """
        Initialize a new spherical grid object.

        Parameters
        ----------
        grid : str, default="SOPHE"
            Grid family name.
        point_group : str, default="Ci"
            Point-group symmetry identifier.
        knots : int, default=15
            Grid resolution parameter.
        """
        self._symmetry = point_group
        self._points = int(knots)

        if grid.lower() == "sophe":
            self._get_SOPHE_grid()
        else:
            raise ValueError("{} is not a valid grid!".format(grid))

        self._cartesian = False
        self._sv = None

    def get_grid(self, point_group: str = "Ci", cartesian: bool = False) -> cp.array:
        """
        Get the grid for a given point group.

        Parameters
        ----------
        point_group : str, optional
            Symmetry identifier. ``"C1"`` returns the full-sphere grid.
        cartesian : bool, optional
            If ``True``, coordinates are returned in Cartesian form ``(x, y, z)``.
            Otherwise spherical coordinates ``(r, theta, phi)`` are returned.

        Returns
        -------
        cp.array
            Grid coordinates with shape ``(N, 3)``.
        """
        if self._symmetry != point_group:
            self._symmetry = point_group
            self._get_SOPHE_grid()
            self._sv = None
            self._cartesian = False

        if cartesian and not self._cartesian:
            self._grid = spherical_to_cartesian(self._grid[:, 1], self._grid[:, 2])
            self._border_points = spherical_to_cartesian(
                self._border_points[:, 1], self._border_points[:, 2]
            )
            self._cartesian = True

        return self._grid

    def get_areas(self) -> cp.array:
        """
        Get integration areas for all current grid points.

        Returns
        -------
        cp.array
            Area weights with shape ``(N,)``. For ``"O3"`` and ``"Dooh"``,
            precomputed weights are returned directly; otherwise Voronoi-cell
            areas multiplied by symmetry weight factors are returned.
        """
        if self._symmetry in ["O3", "Dooh"]:
            return self._weight_factors
        else:
            areas = self._get_Voronoi_area() * self._weight_factors
        return areas

    def show(self, voronoi=False):
        """
        Visualize the current grid and optionally its Voronoi tessellation.

        Parameters
        ----------
        voronoi : bool, default=False
            If ``True`` and a Voronoi tessellation is available, Voronoi vertices
            and geodesic edges are plotted in addition to grid points.
        """
        if voronoi and self._sv is not None:
            self._sv.sort_vertices_of_regions()
            if not self._cartesian:
                grid = spherical_to_cartesian(self._grid[:, 1], self._grid[:, 2])
            else:
                grid = self._grid

            t_vals = cp.linspace(0, 1, 2000)
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")

            # plot the unit sphere for reference (optional)
            u = np.linspace(0, 2 * np.pi, 100)
            v = np.linspace(0, np.pi, 100)
            x = np.outer(np.cos(u), np.sin(v))
            y = np.outer(np.sin(u), np.sin(v))
            z = np.outer(np.ones(np.size(u)), np.cos(v))
            ax.plot_surface(x, y, z, color="y", alpha=0.1)

            # Plot all points with color indication for their weight
            g_1 = grid[self._weight_factors == 1]
            g_2 = grid[self._weight_factors == 0.5]
            g_3 = grid[self._weight_factors == 0.25]
            g_4 = grid[self._weight_factors == 1 / 6]
            g_5 = grid[self._weight_factors == 1 / 3]

            ax.scatter(g_1[:, 0].get(), g_1[:, 1].get(), g_1[:, 2].get(), color="b")
            ax.scatter(g_2[:, 0].get(), g_2[:, 1].get(), g_2[:, 2].get(), color="k")
            ax.scatter(g_3[:, 0].get(), g_3[:, 1].get(), g_3[:, 2].get(), color="y")
            ax.scatter(g_4[:, 0].get(), g_4[:, 1].get(), g_4[:, 2].get(), color="r")
            ax.scatter(g_5[:, 0].get(), g_5[:, 1].get(), g_5[:, 2].get(), color="m")

            # plot Voronoi vertices
            ax.scatter(
                self._sv.vertices[:, 0],
                self._sv.vertices[:, 1],
                self._sv.vertices[:, 2],
                c="g",
            )

            # indicate Voronoi regions (as Euclidean polygons)
            for region in self._sv.regions:
                n = len(region)
                for i in range(n):
                    start = self._sv.vertices[region][i]
                    end = self._sv.vertices[region][(i + 1) % n]
                    result = geometric_slerp(start, end, t_vals.get())
                    ax.plot(result[..., 0], result[..., 1], result[..., 2], c="k")

            ax.azim = 10
            ax.elev = 40
            _ = ax.set_xticks([])
            _ = ax.set_yticks([])
            _ = ax.set_zticks([])
            fig.set_size_inches(4, 4)

        else:
            fig = plt.figure(figsize=(12, 12))
            ax = fig.add_subplot(projection="3d")
            ax.scatter(
                self._grid[:, 0].get(),
                self._grid[:, 1].get(),
                self._grid[:, 2].get(),
            )

        plt.show()

    def _get_SOPHE_grid(self) -> cp.array:
        r"""
        Generate a SOPHE grid for the configured symmetry.

        The base construction in one octant follows

        .. math::

            \theta_{k, l}&=\frac{k}{M}\cdot\frac{\pi}{2}\;\;\;\;0\leq k\leq M\\
            \phi_{k, l} & = \frac{l}{k}\cdot\frac{\pi}{2}\;\;\;\; 0\leq l\leq k

        Returns
        -------
        cp.array
            Spherical coordinates of the grid with shape ``(N, 3)``.
        """
        phi_max, octants, border = self._get_grid_params()
        M = self._points

        # 03
        if octants == -1:
            theta = cp.zeros(1)
            phi = cp.zeros(1)
            r = cp.ones(1)
            weights = cp.ones(1)
            self._grid = cp.array([r, theta, phi]).T
            self._weight_factors = weights
            self._border_points = None
            return

        # Dooh
        elif octants == 0:
            dtheta = np.pi / (2 * (M - 1))
            theta = cp.linspace(0, M - 1, M) * dtheta
            phi = cp.zeros(M)
            r = cp.ones(M)
            weights = 4 * np.pi * cp.sin(dtheta / 2) * cp.sin(theta)
            weights[0] = 2 * np.pi * (1 - cp.cos(dtheta / 2))
            weights[-1] = 2 * np.pi * cp.sin(dtheta / 2)
            self._grid = cp.array([r, theta, phi]).T
            self._weight_factors = weights
            self._border_points = None
            return

        else:
            # Pole
            theta = [0]
            phi = [0]
            weight_fac = [phi_max / (2 * np.pi)]

        # Coordinates without Voronoi border points
        if octants != 4 and octants != 8:
            for k in range(1, M + 1):
                for column in range(0, octants * k + border):
                    t = (k / M) * (np.pi / 2)
                    p = (column / (octants * k)) * phi_max

                    if border:
                        if column == 0 or column == octants * k:
                            weight = 0.5
                        else:
                            weight = 1.0
                    else:
                        weight = 1.0

                    if k == M:
                        weight *= 0.5

                    theta.append(t)
                    phi.append(p)
                    weight_fac.append(weight)

        else:
            for k in range(1, M + 1):
                for column in range(0, 4 * k):
                    t = (k / M) * (np.pi / 2)
                    p = (column / (4 * k)) * phi_max

                    if k == M and octants == 4:
                        weight = 0.5
                    else:
                        weight = 1.0

                    theta.append(t)
                    phi.append(p)
                    weight_fac.append(weight)

        theta = cp.array(theta)
        phi = cp.array(phi)
        self._weight_factors = cp.array(weight_fac)

        if octants == 8:
            theta_2 = np.pi - theta[theta != np.pi / 2]
            phi_2 = phi[theta != np.pi / 2]
            theta = cp.hstack((theta, theta_2))
            phi = cp.hstack((phi, phi_2))
            self._weight_factors = cp.ones(theta.size)

        r = cp.ones(theta.size)
        self._grid = cp.array([r, theta, phi]).T

        # Add border points for Voronoi area calculation
        theta = []
        phi = []
        if octants != 4 and octants != 8:
            for k in range(1, M + 2):
                if k == 1:
                    n_points = octants
                    max_points = int(((2 * np.pi) / phi_max) * n_points)
                    lb = -(max_points - n_points - 1 - border)
                else:
                    lb = -1
                for column in range(lb, octants * k + 1 + border):
                    if k != M + 1 and column > -1 and column < octants * k + border:
                        continue
                    elif k != M + 1:
                        t = (k / M) * (np.pi / 2)
                        p = (column / (octants * k)) * phi_max
                    else:
                        if column > (k - 2) * octants + border:
                            continue
                        t = (k / M) * (np.pi / 2)
                        p = (column / (octants * (M - 1))) * phi_max

                    theta.append(t)
                    phi.append(p)

        elif octants == 4:
            k = M - 1
            for column in range(0, 4 * k):
                t = np.pi - (k / M) * (np.pi / 2)
                p = (column / (octants * k)) * phi_max

                theta.append(t)
                phi.append(p)

        theta = cp.array(theta)
        phi = cp.array(phi)
        r = cp.ones(theta.size)
        self._border_points = cp.array([r, theta, phi]).T

    def _get_EasySpin_grid(self, M: int) -> cp.array:
        r"""
        Generate an EasySpin-style one-octant grid.

        .. math::

            \theta_A &= \frac{k}{M}\cdot\frac{\pi}{2} \\
            \phi_A &= \frac{l}{k}\cdot\frac{\pi}{2} \\
            \theta_B &= \frac{M-l}{M}\cdot\frac{\pi}{2} \\
            \phi_B &= \frac{k-l}{M-l}\cdot\frac{\pi}{2} \\
            \theta_C &= \frac{M-k+l}{M}\cdot\frac{\pi}{2} \\
            \phi_C &= \frac{M-k}{M-k+l}\cdot\frac{\pi}{2}

        .. math::

            \begin{pmatrix}x_i\\y_i\\z_i\end{pmatrix} &= \begin{pmatrix}
            \sin(\theta_i)\cos(\phi_i)\\\sin(\theta_i)\sin(\phi_i)\\
                \cos(\theta_i)\end{pmatrix}\\
            \begin{pmatrix}x\\y\\z\end{pmatrix} &= \frac{1}{3}\left(
            \begin{pmatrix}x_A\\y_A\\z_A\end{pmatrix}+
            \begin{pmatrix}y_B\\z_B\\x_B\end{pmatrix}+
            \begin{pmatrix}z_C\\x_C\\y_C\end{pmatrix}\right)


        Returns points in spherical coordinates after averaging three mapped
        Cartesian constructions.

        Parameters
        ----------
        M : int
            Resolution parameter. Returns ``N = (M+1)(M+2)/2`` knots.

        Returns
        -------
        cp.array
            Spherical coordinates with shape ``(N, 3)``
            as ``(radius, elevation, azimuth)``.
        """
        k, column = cp.tril_indices(M + 1)
        theta_a = (k / M) * (np.pi / 2)
        phi_a = (column / k) * (np.pi / 2)
        phi_a = cp.nan_to_num(phi_a, nan=np.pi / 2)

        theta_b = ((M - column) / M) * (np.pi / 2)
        phi_b = ((k - column) / (M - column)) * (np.pi / 2)
        phi_b = cp.nan_to_num(phi_b, nan=np.pi / 2)

        theta_c = ((M - k + column) / M) * (np.pi / 2)
        phi_c = ((M - k) / (M - k + column)) * (np.pi / 2)
        phi_c = cp.nan_to_num(phi_c, nan=np.pi / 2)

        x_a = cp.sin(theta_a) * cp.cos(phi_a)
        y_a = cp.sin(theta_a) * cp.sin(phi_a)
        z_a = cp.cos(theta_a)

        x_b = cp.sin(theta_b) * cp.cos(phi_b)
        y_b = cp.sin(theta_b) * cp.sin(phi_b)
        z_b = cp.cos(theta_b)

        x_c = cp.sin(theta_c) * cp.cos(phi_c)
        y_c = cp.sin(theta_c) * cp.sin(phi_c)
        z_c = cp.cos(theta_c)

        x = (x_a + y_b + z_c) / 3
        y = (y_a + z_b + x_c) / 3
        z = (z_a + x_b + y_c) / 3

        points = cartesian_to_spherical(x, y, z)
        return points

    def _get_Voronoi_area(self) -> cp.array:
        """
        Compute Voronoi-cell areas for the current spherical grid.

        Returns
        -------
        cp.array
            Voronoi-cell areas for primary grid points with shape ``(N,)``.
        """
        cartesian = self._cartesian
        coordinates = cp.vstack((self._grid, self._border_points))

        # Transform to spherical and back to obtain a consistent radius
        if cartesian:
            coordinates = cartesian_to_spherical(
                coordinates[:, 0], coordinates[:, 1], coordinates[:, 2]
            )
            coordinates = spherical_to_cartesian(coordinates[:, 1], coordinates[:, 2])

        # Transform to cartesian
        coordinates = spherical_to_cartesian(coordinates[:, 1], coordinates[:, 2])

        self._sv = SphericalVoronoi(coordinates.get(), radius=1)
        areas = cp.array(self._sv.calculate_areas())

        return areas[: self._grid[:, 0].size]

    def _get_grid_params(self) -> list[float, int, bool]:
        """
        Get symmetry-dependent grid construction parameters.

        Returns
        -------
        list
            A list ``[phi_max, octants, border]`` where ``phi_max`` is the azimuth
            range in radians, ``octants`` controls symmetry expansion, and
            ``border`` indicates whether border handling is enabled.
        """
        point_group = [
            "C1",
            "Ci",
            "C2h",
            "S6",
            "C4h",
            "C6h",
            "D2h",
            "Th",
            "D3d",
            "D4h",
            "Oh",
            "D6h",
            "Dooh",
            "O3",
        ]
        pg_idx_dic = dict(zip(point_group, np.linspace(0, 13, 14, dtype=np.int8)))

        # phi in 1/4 *pi
        phi = [8, 8, 4, 8 / 3, 2, 4 / 3, 2, 2, 4 / 3, 1, 1, 2 / 3, 0, 0]
        bounds = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
        octants = [8, 4, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 0, -1]

        idx = pg_idx_dic[self._symmetry]

        return phi[idx] * np.pi / 4, octants[idx], bool(bounds[idx])

    def get_triangle_idx(self) -> cp.array:
        # TODO: Ordentlich machen und optimieren!
        # TODO: Zwingend für Cupy optimieren! Zu viele arrays werden erstellt.
        """
        Get triangle indices for spherical Delaunay triangulation.

        Returns
        -------
        cp.array
            Triangle table with shape ``(T, 4)``:
            ``(idx1, idx2, idx3, area)``, where indices refer to grid points and
            ``area`` is the spherical triangle area.
        """
        phi_max, octants, border = self._get_grid_params()
        M = self._points

        # u, v are parameterisation variables
        u = self._grid[:, 1]
        v = self._grid[:, 2]

        xyz = spherical_to_cartesian(u, v)

        if octants == 1:
            uptris = []
            downtris = []
            for n in range(M):
                if n == 0:
                    uprow = cp.array([0])
                    downrow = cp.linspace(1, 1 + border, 1 + border, dtype=cp.int32)
                    start_2 = 1
                    stop_2 = 1 + border
                    uptris.append(
                        cp.array(
                            [
                                cp.array(0),
                                downrow[0],
                                downrow[1 % (1 + border)],
                            ]
                        )
                    )
                    continue
                else:
                    start = 1 * start_2
                    stop = 1 * stop_2
                    start_2 = stop + 1
                    stop_2 = start_2 + (n + border)
                    uprow = cp.linspace(start, stop, stop - start + 1, dtype=cp.int32)
                    downrow = cp.linspace(
                        start_2, stop_2, stop_2 - start_2 + 1, dtype=cp.int32
                    )

                for i in range(len(uprow) - border):
                    idx_top_left = uprow[i]
                    idx_top_right = uprow[(i + 1) % len(uprow)]
                    idx_bottom = downrow[(i + 1) % len(downrow)]
                    downtris.append([idx_top_left, idx_top_right, idx_bottom])

                for i in range(len(downrow) - border):
                    idx_top = uprow[i % len(uprow)]
                    idx_down_left = downrow[i]
                    idx_down_right = downrow[(i + 1) % len(downrow)]
                    uptris.append([idx_top, idx_down_left, idx_down_right])

        elif octants == 2:
            uptris = []
            downtris = []
            for n in range(M):
                if n == 0:
                    uprow = cp.array([0])
                    downrow = cp.linspace(1, 2, 2, dtype=cp.int32)
                    start_2 = 1
                    stop_2 = 2
                    for i in range(octants):
                        uptris.append(
                            cp.array(
                                [
                                    cp.array(0),
                                    downrow[i],
                                    downrow[(i + 1) % octants],
                                ]
                            )
                        )
                    continue
                else:
                    start = 1 * start_2
                    stop = 1 * stop_2
                    start_2 = stop + 1
                    stop_2 = start_2 + (n + 1) * octants - 1
                    uprow = cp.linspace(start, stop, stop - start + 1, dtype=cp.int32)
                    downrow = cp.linspace(
                        start_2, stop_2, stop_2 - start_2 + 1, dtype=cp.int32
                    )

                for k in range(2):
                    for i in range(1, len(uprow) // octants + 1):
                        offset_up = k * n
                        offset_low = k * (n + 1)
                        idx_top_left = uprow[(i - 1 + offset_up) % len(uprow)]
                        idx_top_right = uprow[(i + offset_up) % len(uprow)]
                        idx_bottom = downrow[(i + offset_low) % len(downrow)]
                        downtris.append([idx_top_left, idx_top_right, idx_bottom])

                    for i in range(len(downrow) // octants):
                        offset_up = k * n
                        offset_low = k * (n + 1)
                        idx_top = uprow[(i + offset_up) % len(uprow)]
                        idx_down_left = downrow[(i + offset_low) % len(downrow)]
                        idx_down_right = downrow[(i + 1 + offset_low) % len(downrow)]
                        uptris.append([idx_top, idx_down_left, idx_down_right])

        elif octants == 4:
            uptris = []
            downtris = []
            for n in range(M):
                if n == 0:
                    uprow = cp.array([0])
                    downrow = cp.linspace(1, 4, 4, dtype=cp.int32)
                    start_2 = 1
                    stop_2 = 4
                    for i in range(octants):
                        uptris.append(
                            cp.array(
                                [
                                    cp.array(0),
                                    downrow[i],
                                    downrow[(i + 1) % octants],
                                ]
                            )
                        )
                    continue
                else:
                    start = 1 * start_2
                    stop = 1 * stop_2
                    start_2 = stop + 1
                    stop_2 = start_2 + (n + 1) * octants - 1
                    uprow = cp.linspace(start, stop, stop - start + 1, dtype=cp.int32)
                    downrow = cp.linspace(
                        start_2, stop_2, stop_2 - start_2 + 1, dtype=cp.int32
                    )

                for k in range(4):
                    for i in range(1, len(uprow) // octants + 1):
                        offset_up = k * n
                        offset_low = k * (n + 1)
                        idx_top_left = uprow[(i - 1 + offset_up) % len(uprow)]
                        idx_top_right = uprow[(i + offset_up) % len(uprow)]
                        idx_bottom = downrow[(i + offset_low) % len(downrow)]
                        downtris.append([idx_top_left, idx_top_right, idx_bottom])

                    for i in range(len(downrow) // octants):
                        offset_up = k * n
                        offset_low = k * (n + 1)
                        idx_top = uprow[(i + offset_up) % len(uprow)]
                        idx_down_left = downrow[(i + offset_low) % len(downrow)]
                        idx_down_right = downrow[(i + 1 + offset_low) % len(downrow)]
                        uptris.append([idx_top, idx_down_left, idx_down_right])

        elif octants == 8:
            uptris = []
            downtris = []
            for hemi in range(2):
                for n in range(M):
                    if n == 0:
                        if hemi == 0:
                            hemi_offset = 0
                        else:
                            hemi_offset = stop_2 + 1
                            replace_line = downrow.copy()
                        uprow = cp.array([0 + hemi_offset])
                        downrow = cp.linspace(
                            1 + hemi_offset, 4 + hemi_offset, 4, dtype=cp.int32
                        )
                        start_2 = 1 + hemi_offset
                        stop_2 = 4 + hemi_offset
                        for i in range(4):
                            uptris.append(
                                cp.array(
                                    [
                                        cp.array(0 + hemi_offset),
                                        downrow[i],
                                        downrow[(i + 1) % 4],
                                    ]
                                )
                            )
                        continue
                    else:
                        start = 1 * start_2
                        stop = 1 * stop_2
                        start_2 = stop + 1
                        stop_2 = start_2 + (n + 1) * 4 - 1
                        uprow = cp.linspace(
                            start, stop, stop - start + 1, dtype=cp.int32
                        )
                        if n == M - 1 and hemi == 1:
                            downrow = replace_line
                        else:
                            downrow = cp.linspace(
                                start_2,
                                stop_2,
                                stop_2 - start_2 + 1,
                                dtype=cp.int32,
                            )

                    for k in range(4):
                        for i in range(1, len(uprow) // 4 + 1):
                            offset_up = k * n
                            offset_low = k * (n + 1)
                            idx_top_left = uprow[(i - 1 + offset_up) % len(uprow)]
                            idx_top_right = uprow[(i + offset_up) % len(uprow)]
                            idx_bottom = downrow[(i + offset_low) % len(downrow)]
                            downtris.append([idx_top_left, idx_top_right, idx_bottom])

                        for i in range(len(downrow) // 4):
                            offset_up = k * n
                            offset_low = k * (n + 1)
                            idx_top = uprow[(i + offset_up) % len(uprow)]
                            idx_down_left = downrow[(i + offset_low) % len(downrow)]
                            idx_down_right = downrow[
                                (i + 1 + offset_low) % len(downrow)
                            ]
                            uptris.append([idx_top, idx_down_left, idx_down_right])

        uptris = cp.array(uptris)
        downtris = cp.array(downtris)
        indices = cp.vstack([uptris, downtris])

        def spherical_area(a, b, c):
            """
            Calculate the area of a spherical triangle on the unit sphere.

            Parameters
            ----------
            a : cp.array
                First vertex as Cartesian unit vector.
            b : cp.array
                Second vertex as Cartesian unit vector.
            c : cp.array
                Third vertex as Cartesian unit vector.

            Returns
            -------
            float
                Spherical triangle area in steradians.
            """
            t = abs(cp.inner(a, cp.cross(b, c)))
            t /= 1 + cp.inner(a, b) + cp.inner(b, c) + cp.inner(a, c)
            return 2 * cp.arctan(t)

        # Correct the triangles for area calculation if border is False
        if octants == 8:
            border = True
        if not border:
            idx_dt = indices[:, 0] >= indices[:, 1]
            idx_ut = indices[:, 1] >= indices[:, 2]

        areas = []
        for i in range(len(indices)):
            if not border and idx_dt[i]:
                coords = xyz[indices[i, 1]]
                coords = cartesian_to_spherical(*coords)
                coords[2] = phi_max
                coords = spherical_to_cartesian(coords[1], coords[2])

                p_1 = cp.array(xyz[indices[i, 0]])
                p_2 = coords
                p_3 = cp.array(xyz[indices[i, 2]])

            elif not border and idx_ut[i]:
                coords_1 = cp.array(xyz[indices[i, 0]])
                coords_1 = cartesian_to_spherical(*coords_1)
                coords_1[2] = phi_max
                coords_1 = spherical_to_cartesian(coords_1[1], coords_1[2])

                coords_2 = xyz[indices[i, 2]]
                coords_2 = cartesian_to_spherical(*coords_2)
                coords_2[2] = phi_max
                coords_2 = spherical_to_cartesian(coords_2[1], coords_2[2])
                p_1 = coords_1
                p_2 = cp.array(xyz[indices[i, 1]])
                p_3 = coords_2
            else:
                p_1 = cp.array(xyz[indices[i, 0]])
                p_2 = cp.array(xyz[indices[i, 1]])
                p_3 = cp.array(xyz[indices[i, 2]])

            areas.append(spherical_area(p_1, p_2, p_3))

        indices = cp.hstack([indices, cp.array([areas]).T])

        return indices


def spherical_to_cartesian(
    theta: cp.array, phi: cp.array, r: cp.array = 1.0
) -> cp.array:
    r"""
    Transform spherical coordinates to Cartesian coordinates.

    .. math::

        x &= r\sin(\theta)\cos(\phi) \\
        y &= r\sin(\theta)\sin(\phi) \\
        z &= r\cos(\theta)

    Parameters
    ----------
    theta : cp.array
        Elevation angles in radians.
    phi : cp.array
        Azimuth angles in radians.
    r : cp.array or float, optional
        Radius values. Default is ``1.0``.

    Returns
    -------
    cp.array
        Cartesian coordinates with shape ``(N, 3)``.
    """
    x = r * cp.sin(theta) * cp.cos(phi)
    y = r * cp.sin(theta) * cp.sin(phi)
    z = r * cp.cos(theta)
    return cp.array([x, y, z]).T


def cartesian_to_spherical(x: cp.array, y: cp.array, z: cp.array) -> cp.array:
    r"""
    Transform Cartesian coordinates to spherical coordinates.

    .. math::

        r &= \sqrt{x^2 + y^2 + z^2} \\
        \theta &= \arccos\left(\frac{z}{r}\right) \\
        \phi &= \arctan2(y, x)

    Parameters
    ----------
    x : cp.array
        X components.
    y : cp.array
        Y components.
    z : cp.array
        Z components.

    Returns
    -------
    cp.array
        Spherical coordinates with shape ``(N, 3)`` as
        ``(radius, elevation, azimuth)``.

    Notes
    -----
    This convention uses a right-handed coordinate system.
    """
    r = cp.sqrt(x**2 + y**2 + z**2)
    theta = cp.arccos(z / r)
    phi = cp.arctan2(y, x)
    return cp.array([r, theta, phi]).T
