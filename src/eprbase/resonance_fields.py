#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Resonance field calculation utilities for EPR simulations.

This module provides the :class:`ResonanceFields` class to calculate resonance
fields, intensities, linewidths, and transition indices for EPR spectra simulations.
Uses adaptive spline interpolation and transition probability analysis.

© M. Sc. Florian Quintes, 2026.
@contact: florian.quintes@pc.uni.freiburg.de
@author: Florian Quintes
"""

from scipy.interpolate import interp1d, CubicHermiteSpline
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from warnings import warn
from scipy.constants import physical_constants

mu_b, *_ = physical_constants["Bohr magneton in Hz/T"]


class ResonanceFields:
    """
    Resonance field calculator for EPR simulations.

    Handles the calculation of resonance fields, intensities, linewidths, and
    transition indices using adaptive spline interpolation and transition
    probability analysis.

    Parameters
    ----------
    Hamiltonian : object
        Hamiltonian object for energy calculations
    Grid : object
        Grid object for orientation sampling
    b_field : np.array
        Magnetic field range
    nu : np.array
        Frequency range
    rho : np.array
        Density matrix
    testing : bool, optional
        Enable testing mode, by default False
    """

    def __init__(
        self,
        Hamiltonian: object,
        Grid: object,
        b_field: np.array,
        nu: np.array,
        rho: np.array,
        testing: bool = False,
    ):
        """
        Initialize the resonance field calculator.

        Parameters
        ----------
        Hamiltonian : object
            Hamiltonian object for energy calculations
        Grid : object
            Grid object for orientation sampling
        b_field : np.array
            Magnetic field range
        nu : np.array
            Frequency range
        rho : np.array
            Density matrix
        testing : bool, optional
            Enable testing mode, by default False
        """
        self._ham = Hamiltonian
        self._proj = self._ham.get_proj()
        self._grid = Grid.get_grid(Grid._symmetry)
        self._field = b_field
        self._rho = rho
        self._nu = nu

        # Used for adaptive Bisection / Error estimation
        self._tau_B = np.ptp(self._field) / (self._field.size - 1)
        self._eigvec_field = np.array([])
        self._eigvec_vec = np.array([])

        self._pop_trshld = 1e-5
        self._trans_prob_trshld = 1e-5
        self._int_trshld = 1e-5

        self._testing = testing

    def get_res_fields(self) -> list[np.array, np.array, np.array, np.array]:
        """
        Calculate resonance fields for all grid points.

        Returns
        -------
        res_fields_t : list of np.array
            Resonance fields for each grid point
        intensities_t : list of np.array
            Intensities for each transition
        widths_t : list of np.array
            Linewidths for each transition
        transition_t : list of np.array
            Transition indices for each transition
        """
        res_fields_t, intensities_t, widths_t, transition_t = [], [], [], []
        if not hasattr(self, "_transitions"):
            self._get_transitions()

        for i in range(self._grid.shape[0]):
            (
                res_fields,
                intensities,
                widths,
                transition,
            ) = self._get_single_res_fields(i)
            res_fields_t.append(res_fields)
            intensities_t.append(intensities)
            widths_t.append(widths)
            transition_t.append(transition)
        return res_fields_t, intensities_t, widths_t, transition_t

    def res_field_plot(self, point: int) -> None:
        """
        Plot energy level diagram with resonance fields.

        Parameters
        ----------
        point : int
            Grid point index to plot

        Returns
        -------
        tuple
            (res_fields, intensities, transition, energy_levels, field)
        """
        _, theta, phi = self._grid[point]
        energy_levels, pop, trans_prob = self._adaptive_spline(theta, phi)
        print(energy_levels)
        idx = np.tril_indices(energy_levels.c.shape[2], k=-1)
        transition = np.column_stack(idx)

        transition = self._filter_at_center(trans_prob, pop, transition)
        (
            res_fields,
            intensities,
            widths,
            transition,
        ) = self._get_single_res_fields(point)

        plt.figure()
        plt.yscale("symlog")
        plt.plot(
            self._field / mu_b * 1e3,
            energy_levels(self._field),
        )
        for i, res_field in enumerate(res_fields):
            field = res_field / mu_b * 1e3
            start = energy_levels(res_field)[transition[i][0]]
            end = energy_levels(res_field)[transition[i][1]]
            if intensities[i] < 0:
                start, end = end, start
            delta = end - start

            plt.arrow(
                field,
                start,
                0,
                delta,
                color="gray",
                head_length=abs(end) * 0.7,
                head_width=0.02,
                width=abs(intensities[i]) * 0.001,
                length_includes_head=True,
            )
        plt.show()
        return res_fields, intensities, transition, energy_levels, self._field

    def levels_plot(self, point: int, bisections: bool = False) -> None:
        """
        Plot energy levels with population indication.

        Parameters
        ----------
        point : int
            Grid point index to plot
        bisections : bool, optional
            Show bisection points, by default False
        """
        _, theta, phi = self._grid[point]
        energy_levels, pop, trans_prob = self._adaptive_spline(theta, phi)
        plt.figure()
        levels = energy_levels(self._field)
        center = (self._field.max() + self._field.min()) / 2
        width = pop(center)
        width /= width.max()
        width[width < 1e-1] = 0
        plt.yscale("symlog")
        for i in range(levels.shape[1]):
            if width[i] > 0:
                plt.plot(self._field * 1e3 / mu_b, levels[:, i], linewidth=width[i])
            else:
                plt.plot(self._field * 1e3 / mu_b, levels[:, i], linestyle="--")

        if bisections:
            plt.vlines(
                self._centers[:, :, 0] * 1e3 / mu_b,
                4 * levels.min(),
                4 * levels.max(),
                color="gray",
            )
        plt.ylim(4 * levels.min(), 4 * levels.max())
        plt.show()

    def _get_transitions(self):
        # TODO: Ordentlich machen
        """
        Calculate transition indices for all grid points.

        Notes
        -----
        Sets self._transitions attribute.
        """
        field = np.array([(self._field.min() + self._field.max()) / 2])

        for i in range(self._grid.shape[0]):
            _, theta, phi = self._grid[i]
            knots, (eigvec_field, eigvec_vec) = self._get_energies_gradients(
                field, theta, phi
            )

            if i == 0:
                idx = np.tril_indices(eigvec_vec.shape[1], k=-1)
                transitions = np.column_stack(idx)

            trans_prob = self._get_transition_probabilities(eigvec_vec)
            pop = np.linalg.multi_dot(
                [eigvec_vec[0].T, self._rho, np.linalg.inv(eigvec_vec[0].T)]
            )
            pop = np.diag(pop.real)
            delta_pop = pop[transitions[:, 0]] - pop[transitions[:, 1]]

            trans = transitions[abs(delta_pop) > self._pop_trshld]
            trans = trans[
                trans_prob[0, trans[:, 0], trans[:, 1]] > self._trans_prob_trshld
            ]

            if i == 0:
                trans_list = trans.copy()
            else:
                trans_list = np.vstack([trans_list, trans])

        transitions = np.unique(trans_list, axis=0)

        energies = knots[0, :, 1]
        start = transitions[:, 0]
        end = transitions[:, 1]
        deltas = abs(energies[end] - energies[start])
        delta_energies_filter = np.logical_and(
            deltas >= 1.8 * self._field.min(),
            deltas <= 2.2 * self._field.max(),
        )
        self._transitions = transitions[delta_energies_filter]

    def _get_single_res_fields(self, grid_point: int) -> np.array:
        """
        Calculate resonance fields for a single grid point.

        Parameters
        ----------
        grid_point : int
            Index of the grid point

        Returns
        -------
        res_fields : np.array
            Resonance fields for the grid point
        intensities : np.array
            Intensities for each transition
        widths : np.array
            Linewidths for each transition
        transition : np.array
            Transition indices for each transition
        """
        _, theta, phi = self._grid[grid_point]
        self._grid_point = grid_point

        energy_levels, pop, trans_prob = self._adaptive_spline(theta, phi)

        idx = np.tril_indices(energy_levels.c.shape[2], k=-1)
        transition = np.column_stack(idx)

        # transition = self._filter_at_center(trans_prob, pop, transition)
        # transition = self._transitions.copy()
        delta_energy = self._get_delta_splines(energy_levels, transition)
        res_fields, delta_energy, transition = self._find_res_fields(
            delta_energy, transition
        )

        intensities = self._get_intensities(res_fields, trans_prob, pop, transition)

        # (
        #     res_fields,
        #     intensities,
        #     delta_energy,
        #     transition,
        # ) = self._filter_by_intensity(
        #     res_fields, intensities, delta_energy, transition
        # )

        (
            res_fields,
            intensities,
            delta_energy,
            transition,
        ) = self._filter_by_position(res_fields, intensities, delta_energy, transition)

        widths = self._get_linewidths(res_fields, delta_energy)

        return res_fields, intensities, widths, transition

    def _find_res_fields(
        self, delta_splines: CubicHermiteSpline, transitions: np.array
    ) -> np.array:
        """
        Find resonance fields for all transitions.

        Parameters
        ----------
        delta_splines : CubicHermiteSpline
            Energy difference splines
        transitions : np.array
            Transition indices

        Returns
        -------
        res_fields : np.array
            Resonance fields for each transition
        delta_splines : CubicHermiteSpline
            Updated energy difference splines
        transitions : np.array
            Updated transition indices
        """
        fields_1 = delta_splines.solve(self._nu)
        fields_2 = delta_splines.solve(-self._nu)
        res_fields = [
            np.sort(np.append(fields_1[i], fields_2[i])) for i in range(len(fields_1))
        ]

        if not res_fields:
            return np.array(res_fields), delta_splines, transitions

        trans = []
        fields = []
        coeff = []
        for i in range(len(res_fields)):
            trans.append(list(transitions[i]) * res_fields[i].size)
            fields.append(list(res_fields[i]))
            for _ in range(res_fields[i].size):
                coeff.append(delta_splines.c[:, :, i])

        transition = np.array(
            [t for sec in trans for t in sec], dtype=transitions.dtype
        )
        transitions = transition.reshape((transition.size // 2, 2))
        res_fields = np.array([f for sec in fields for f in sec])
        coeff = np.array(coeff)
        delta_splines.c = np.einsum("abc -> bca", coeff)

        return res_fields, delta_splines, transitions

    def _adaptive_spline(
        self, theta: float, phi: float
    ) -> list[CubicHermiteSpline, object, object, object]:
        """
        Calculate energy levels using adaptive spline interpolation.

        Parameters
        ----------
        theta : float
            Theta angle
        phi : float
            Phi angle

        Returns
        -------
        energy_levels : CubicHermiteSpline
            Energy level splines
        population : object
            Population interpolator
        trans_prob : object
            Transition probability interpolator
        """
        min_field, max_field = self._field.min(), self._field.max()
        points = np.array([min_field, max_field])
        knots, (eigvec_field, eigvec_vec) = self._get_energies_gradients(
            points, theta, phi
        )
        segments = np.stack([knots[0], knots[1]], axis=1)[np.newaxis, :, :, :]
        centers = np.array([])
        converged = np.array([False])

        n = 1
        while not np.all(converged) or n == 1:
            # Update 'segment' list
            if n > 1:
                segments = self._get_segments(knots, converged)

            # Add center in every non converged  segment
            new_centers = self._get_new_centers(segments)

            # Determine E_exact and gradients for each new center point
            eval_centers, eigvecs = self._get_energies_gradients(
                new_centers, theta, phi
            )  # eval_centers: (field, energy, gradient), eigvecs:(field, vecs)

            eigvec_field = np.append(eigvec_field, eigvecs[0])
            eigvec_vec = np.vstack([eigvec_vec, eigvecs[1]])

            # Determine which segment is converged
            converged = self._get_error_estimation(segments, eval_centers)
            if n == 4:
                converged = np.ones(converged.size, dtype=np.bool_)

            # Add converged segments center to 'centers' list
            centers = self._get_converged_centers(centers, eval_centers, converged)

            # Add non converged segments center to 'knots' list, append
            # converged knots to centers
            knots, centers = self._get_knots(knots, centers, eval_centers, converged)

            n += 1

        # If each segment is converged:
        splines = self._get_splines(centers)
        self._centers = centers.copy()
        sorting = np.argsort(eigvec_field)
        eigvec_field = eigvec_field[sorting]
        eigvec_vec = eigvec_vec[sorting]

        trans_prob = self._get_transition_probabilities(eigvec_vec)
        trans_prob = self._get_trans_prob_interp(eigvec_field, trans_prob)

        population = self._get_pop_interp(eigvec_field, eigvec_vec)

        return splines, population, trans_prob

    def _get_segments(self, knots: np.array, converged: np.array) -> np.array:
        """
        Get non-converged segments for next iteration.

        Parameters
        ----------
        knots : np.array
            Energy level data points
        converged : np.array
            Convergence flags

        Returns
        -------
        np.array
            Non-converged segments
        """
        offset = np.where(~converged)[0].min()
        needed = ~converged[offset:]
        start = []
        end = []
        i = 0
        j = 0
        while i < needed.size:
            if needed[i]:
                start.append(knots[j])
                start.append(knots[j + 1])
                end.append(knots[j + 1])
                end.append(knots[j + 2])
                i += 1
                j += 2
                done = False
            else:
                if not done:
                    j += 1
                    done = True
                i += 1

        segments = np.stack([start, end], axis=2)

        return segments

    def _get_energies_gradients(
        self, field: np.array, theta: float, phi: float
    ) -> list[np.array, tuple[np.array, np.array]]:
        """
        Calculate energies and gradients for field points.

        Parameters
        ----------
        field : np.array
            Field points
        theta : float
            Theta angle
        phi : float
            Phi angle

        Returns
        -------
        knots : np.array
            Energy level data points
        eigenvector : tuple
            (field points, eigenvector matrices)
        """
        size = field.size
        theta = np.full(size, theta)
        phi = np.full(size, phi)

        energies, eigvec = self._ham.get_eigen(field, theta, phi)
        gradients = self._ham.get_field_gradients(field, theta, phi)

        eigvec_field = field.copy()

        field = np.repeat(field, energies.shape[1]).reshape(
            (field.size, energies.shape[1])
        )

        knots = np.stack([field, energies, gradients], axis=-1)

        return knots, (eigvec_field, eigvec)

    def _get_error_estimation(self, segment: np.array, center: np.array) -> bool:
        r"""
        Estimate error for spline approximation.

        .. math::

           \tilde{\delta}_B = \max_u \left| \frac{E_u(B_3)-\tilde{E}_u(B_3)}
           {\delta E_u(B_3) /\delta B}\right|

        Parameters
        ----------
        segment : np.array
            Spline segment data
        center : np.array
            Exact center values

        Returns
        -------
        bool
            True if error is within tolerance
        """
        delta_B = segment[:, :, 1, 0] - segment[:, :, 0, 0]
        sum_E = segment[:, :, 0, 1] + segment[:, :, 1, 1]
        delta_grad = segment[:, :, 0, 2] - segment[:, :, 1, 2]
        E_est = 0.5 * sum_E + delta_B / 8 * delta_grad
        B_error = np.max(np.abs((center[:, :, 1] - E_est) / center[:, :, 2]), axis=1)
        return B_error <= self._tau_B

    def _get_splines(self, knots: np.array) -> object:
        """
        Create cubic spline interpolators for energy levels.

        Parameters
        ----------
        knots : np.array
            Energy level data points

        Returns
        -------
        CubicHermiteSpline
            Energy level splines
        """
        knots = knots[knots[:, 0, 0].argsort()]  # sort along field axis
        field = knots[:, 0, 0]
        energies = knots[:, :, 1]
        gradients = knots[:, :, 2]
        splines = CubicHermiteSpline(
            field, energies, gradients, axis=0, extrapolate=True
        )

        return splines

    def _get_eigvec_interp(self, field: np.array, eigvecs: np.array) -> object:
        """
        Create eigenvector interpolator (deprecated).

        Parameters
        ----------
        field : np.array
            Field points
        eigvecs : np.array
            Eigenvectors

        Returns
        -------
        object
            Linear interpolator
        """
        warn(
            "This function is deprecated due to errors!",
            DeprecationWarning,
            stacklevel=2,
        )
        return interp1d(field, eigvecs, axis=0, fill_value="extrapolate")

    def _get_trans_prob_interp(self, field: np.array, trans_prob: np.array) -> object:
        """
        Create transition probability interpolator.

        Parameters
        ----------
        field : np.array
            Field points
        trans_prob : np.array
            Transition probabilities

        Returns
        -------
        object
            Linear interpolator
        """
        return interp1d(field, trans_prob, axis=0, fill_value="extrapolate")

    def _get_pop_interp(self, field: np.array, eigvecs: np.array) -> object:
        """
        Create population interpolator.

        Parameters
        ----------
        field : np.array
            Field points
        eigvecs : np.array
            Eigenvectors

        Returns
        -------
        object
            Linear interpolator
        """
        eigvecs_T = np.einsum("aij -> aji", eigvecs)
        eigvecs_inv = np.linalg.inv(eigvecs_T)

        pop = np.empty(eigvecs.shape)
        for i in range(eigvecs.shape[0]):
            pop[i] = np.linalg.multi_dot([eigvecs_T[i], self._rho, eigvecs_inv[i]])

        pop = np.einsum("ajj -> aj", pop).real

        return interp1d(field, pop, axis=0, fill_value="extrapolate")

    def _get_new_centers(self, segments: np.array) -> np.array:
        """
        Calculate new center points for non-converged segments.

        Parameters
        ----------
        segments : np.array
            Non-converged segments

        Returns
        -------
        np.array
            New center points
        """
        centers = (segments[:, 0, 1, 0] + segments[:, 0, 0, 0]) / 2
        return centers

    def _get_converged_centers(
        self, centers: np.array, new_centers: np.array, converged: np.array
    ) -> np.array:
        """
        Get converged center points.

        Parameters
        ----------
        centers : np.array
            Previous centers
        new_centers : np.array
            New center points
        converged : np.array
            Convergence flags

        Returns
        -------
        np.array
            Updated center points
        """
        if centers.size == 0:
            return new_centers[converged]

        centers = np.vstack([centers, new_centers[converged]])
        return centers

    def _get_knots(
        self,
        knots: np.array,
        centers: np.array,
        new_centers: np.array,
        converged: np.array,
    ) -> list[np.array, np.array]:
        """
        Prepare knots for next iteration.

        Parameters
        ----------
        knots : np.array
            Current knots
        centers : np.array
            Converged centers
        new_centers : np.array
            New center points
        converged : np.array
            Convergence flags

        Returns
        -------
        knots : np.array
            Updated knots
        centers : np.array
            Updated centers
        """
        needed = new_centers[~converged]
        k = []
        c = []
        j = 0
        for i in range(converged.size):
            if not converged[i]:
                k.append(knots[i])
                k.append(needed[j])
                j += 1
                if i + 1 < converged.size:
                    if converged[i + 1]:
                        k.append(knots[i + 1])
                else:
                    k.append(knots[i + 1])
            else:
                if i == 0:
                    c.append(knots[i])
                    if i == converged.size - 1:
                        c.append(knots[i + 1])
                elif converged[i - 1]:
                    c.append(knots[i])
                    if i == converged.size - 1:
                        c.append(knots[i + 1])
                else:
                    pass

        knots = np.array(k)
        if c:
            centers = np.vstack([centers, np.array(c)])

        return knots, centers

    def _get_transition_probabilities(self, eigvecs: np.array) -> np.array:
        """
        Calculate transition probabilities.

        Parameters
        ----------
        eigvecs : np.array
            Eigenvector matrices

        Returns
        -------
        np.array
            Transition probabilities
        """
        eigvecs_T = np.einsum("aij -> aji", eigvecs)
        trans_prob = np.empty(eigvecs.shape)

        for i in range(trans_prob.shape[0]):
            trans_prob[i] = np.tril(
                abs(
                    np.linalg.multi_dot(
                        [np.conjugate(eigvecs_T[i]), self._proj, eigvecs[i]]
                    )
                )
                ** 2,
                k=-1,
            )

        return trans_prob

    def _get_intensities(
        self,
        field: list,
        trans_prob: object,
        population: object,
        transitions: np.array,
    ) -> np.array:
        """
        Calculate transition intensities.

        Parameters
        ----------
        field : np.array
            Resonance fields
        trans_prob : object
            Transition probability interpolator
        population : object
            Population interpolator
        transitions : np.array
            Transition indices

        Returns
        -------
        np.array
            Transition intensities
        """
        delta_pop = self._get_delta_pop(field, population, transitions)
        trans = trans_prob(field)

        i = range(transitions.shape[0])
        tp = trans[i, transitions[i, 0], transitions[i, 1]]
        intensities = delta_pop * tp
        return intensities

    def _get_delta_pop(
        self, field: np.array, population: object, transitions: np.array
    ) -> np.array:
        """
        Calculate population differences for transitions.

        Parameters
        ----------
        field : np.array
            Resonance fields
        population : object
            Population interpolator
        transitions : np.array
            Transition indices

        Returns
        -------
        np.array
            Population differences
        """
        pops = population(field)
        i = range(field.size)
        delta_pop = pops[i, transitions[:, 0]] - pops[i, transitions[:, 1]]
        return delta_pop

    def _filter_at_center(
        self, trans_prob: object, population: object, transitions: np.array
    ) -> np.array:
        """
        Filter transitions at center field.

        Parameters
        ----------
        trans_prob : object
            Transition probability interpolator
        population : object
            Population interpolator
        transitions : np.array
            Transition indices

        Returns
        -------
        np.array
            Filtered transitions
        """
        B_center = np.ones(transitions.shape[0]) * (
            (self._field.min() + self._field.max()) / 2
        )

        # Filter by population
        delta_pop = self._get_delta_pop(B_center, population, transitions)
        transitions = transitions[abs(delta_pop) > self._pop_trshld]

        # Filter by transition probability
        trans_probs = trans_prob(B_center[0])
        transitions = transitions[
            trans_probs[transitions[:, 0], transitions[:, 1]] > self._trans_prob_trshld
        ]

        return transitions

    def _filter_by_intensity(
        self,
        field: np.array,
        intensities: np.array,
        delta_E: CubicHermiteSpline,
        transitions: np.array,
    ) -> list[np.array, np.array, np.array]:
        """
        Filter transitions by intensity.

        Parameters
        ----------
        field : np.array
            Resonance fields
        intensities : np.array
            Transition intensities
        delta_E : CubicHermiteSpline
            Energy difference splines
        transitions : np.array
            Transition indices

        Returns
        -------
        field : np.array
            Filtered fields
        intensities : np.array
            Filtered intensities
        delta_E : CubicHermiteSpline
            Updated splines
        transitions : np.array
            Filtered transitions
        """
        idx = abs(intensities) > self._int_trshld
        delta_E.c = delta_E.c[:, :, idx]
        return field[idx], intensities[idx], delta_E, transitions[idx]

    def _filter_by_position(
        self,
        field: np.array,
        intensities: np.array,
        delta_E: CubicHermiteSpline,
        transitions: np.array,
    ) -> list[np.array, np.array, np.array]:
        # TODO: max_spread einführen!
        """
        Filter transitions by position.

        Parameters
        ----------
        field : np.array
            Resonance fields
        intensities : np.array
            Transition intensities
        delta_E : CubicHermiteSpline
            Energy difference splines
        transitions : np.array
            Transition indices

        Returns
        -------
        field : np.array
            Filtered fields
        intensities : np.array
            Filtered intensities
        delta_E : CubicHermiteSpline
            Updated splines
        transitions : np.array
            Filtered transitions
        """
        max_spread = 0.5 * 3
        if not self._testing:  # TODO
            max_spread *= 1e-3 * mu_b
        idx_1 = field > (self._field.min() - max_spread)
        idx_2 = field < (self._field.max() + max_spread)
        idx = idx_1 & idx_2
        delta_E.c = delta_E.c[:, :, idx]
        return field[idx], intensities[idx], delta_E, transitions[idx]

    def _get_delta_splines(
        self, splines: CubicHermiteSpline, transitions: np.array
    ) -> CubicHermiteSpline:
        """
        Construct the delta functions of each splines pair.

        Parameters
        ----------
        splines : CubicHermiteSpline
            Spline functions.
        transitions : np.array, (M, 2)
            Indices for each transition.

        Returns
        -------
        CubicHermiteSpline
            Delta spline functions.

        """
        delta_splines = deepcopy(splines)
        coeff = splines.c

        a, n = coeff.shape[1], coeff.shape[2]
        b = int(n * (n - 1) / 2)
        delta_coeff = np.empty((4, a, b))

        delta_coeff = coeff[:, :, transitions[:, 0]] - coeff[:, :, transitions[:, 1]]
        delta_splines.c = delta_coeff

        return delta_splines

    def _get_linewidths(self, field: np.array, delta_E: CubicHermiteSpline) -> np.array:
        """
        Calculate linewidths for transitions.

        Parameters
        ----------
        field : np.array
            Resonance fields
        delta_E : CubicHermiteSpline
            Energy difference splines

        Returns
        -------
        np.array
            Linewidths for each transition
        """
        linewidths = 1 / np.diag(delta_E.derivative()(field))
        return linewidths
