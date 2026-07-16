#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
© M. Sc. Florian Quintes, 2021-2022.

@contact: florian.quintes@pc.uni.freiburg.de

@author: Florian Quintes
"""
from scipy.interpolate import interp1d, PPoly  # Ersetzen
from copy import deepcopy
import numpy as np
import cupy as cp
from cupyx.scipy.interpolate import CubicHermiteSpline
import matplotlib.pyplot as plt
from scipy.constants import physical_constants
from time import time

mu_b, *_ = physical_constants["Bohr magneton in Hz/T"]

CUPY_FLOAT = cp.float32
CUPY_CMPLX = cp.complex64


class ResonanceFields:
    def __init__(
        self,
        Hamiltonian: object,
        Grid: object,
        b_field: cp.array,
        nu: cp.array,
        rho: cp.array,
        testing: bool = False,
    ):
        self._ham = Hamiltonian
        self._proj = self._ham.get_proj().astype(CUPY_CMPLX)
        self._grid = Grid.get_grid(Grid._symmetry).astype(CUPY_FLOAT)
        self._field = b_field.astype(CUPY_FLOAT)
        self._min_field = self._field.min()
        self._max_field = self._field.max()
        self._rho = cp.array(rho, dtype=CUPY_CMPLX)
        self._nu = nu

        # Used for adaptive Bisection / Error estimation
        self._tau_B = cp.ptp(self._field) / (self._field.size - 1)
        self._eigvec_field = cp.array([], dtype=CUPY_FLOAT)
        self._eigvec_vec = cp.array([], dtype=CUPY_CMPLX)

        self._pop_trshld = 1e-5
        self._trans_prob_trshld = 1e-5
        self._int_trshld = 1e-3

        self._init_idx_managers()

        self._testing = testing

    def get_res_fields(self) -> [cp.array, cp.array, cp.array, cp.array]:
        """
        Get the resonance fields.

        Returns
        -------
        res_fields : cp.array, (N,)
            Resonance fields.
        intensities : cp.array, (N,)
            Intensity of the transition.
        width : cp.array, (N,)
            Gaussian linewidth.
        transition : cp.array, (N, 2)
            Level indices for each transition. [0, 1] represents the transition
            from the lowest level to the second lowest. [1, 0] the opposite
            direction.

        """
        start = time()
        res_fields_t, intensities_t, widths_t, transition_t = [], [], [], []
        if not hasattr(self, "_transitions"):
            self._get_transitions()
        print("transitions: ", time() - start)
        print("n_transitions: ", self._transitions.shape[0])

        start = time()
        all_energy, all_pop, all_trans_prob = self._adaptive_spline(
            self._grid[:, 1], self._grid[:, 2]
        )
        print("splines: ", time() - start)

        start = time()
        for i in range(self._grid.shape[0]):
            (
                res_fields,
                intensities,
                widths,
                transition,
            ) = self._get_multi_single_res_fields(
                all_energy[i], all_pop[i], all_trans_prob[i]
            )
            res_fields_t.append(res_fields)
            intensities_t.append(intensities)
            widths_t.append(widths)
            transition_t.append(transition)
        print("res_fields: ", time() - start)

        (res_fields_t, intensities_t, widths_t, transition_t) = (
            self._sanitize_results(
                res_fields_t, intensities_t, widths_t, transition_t
            )
        )

        return res_fields_t, intensities_t, widths_t, transition_t

    def res_field_plot(self, point: int) -> None:
        """
        Plot the levels diagram and mark each resonance field.

        Parameters
        ----------
        point : int
            Orientation index used for the grid point.

        Returns
        -------
        None.

        """
        _, theta, phi = self._grid[point]
        energy_levels, pop, trans_prob = self._adaptive_spline(theta, phi)
        idx = cp.tril_indices(energy_levels.c.shape[2], k=-1)
        transition = cp.column_stack(idx)

        transition = self._filter_at_center(trans_prob, pop, transition)
        (
            res_fields,
            intensities,
            widths,
            transition,
        ) = self._get_single_res_fields(point)

        plt.figure()
        plt.yscale("symlog")
        plt.plot(self._field / mu_b * 1e3, energy_levels(self._field))
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

    def levels_plot(self, point: int, bisections: bool = False) -> None:
        """
        Plot the energy levels.

        The linewidth corresponds to the population of the energy level. Dashed
        lines show unpopulated levels.

        Parameters
        ----------
        point : int
            Orientation index used for the grid point.
        bisections : bool, optional
            If True, the field points used for the iterative bisection
            algorithm will be plotted as vertical lines. The default is False.

        Returns
        -------
        None

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
                plt.plot(
                    self._field * 1e3 / mu_b, levels[:, i], linewidth=width[i]
                )
            else:
                plt.plot(
                    self._field * 1e3 / mu_b, levels[:, i], linestyle="--"
                )

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
        # Die Ergebnisse hiervon können für adaptive spline genutzt werden!
        """Get all transition indices."""
        field = cp.array(
            [(self._field.min() + self._field.max()) / 2], dtype=CUPY_FLOAT
        )

        theta, phi = self._grid[:, 1], self._grid[:, 2]
        field = cp.repeat(field, theta.size)

        knots, (eigvec_field, eigvec_vec) = self._get_energies_gradients(
            field, theta, phi, cp.ones(theta.size, dtype=cp.uint32)
        )

        trans_prob = self._get_transition_probabilities(eigvec_vec)

        for i in range(theta.size):
            if i == 0:
                idx = cp.tril_indices(eigvec_vec.shape[1], k=-1)
                transitions = cp.column_stack(idx)

            pop = cp.dot(
                eigvec_vec[0].T,
                cp.dot(self._rho, cp.linalg.inv(eigvec_vec[0].T)),
            )
            pop = cp.diag(pop.real)
            delta_pop = pop[transitions[:, 0]] - pop[transitions[:, 1]]

            trans = transitions[abs(delta_pop) > self._pop_trshld]
            trans = trans[
                trans_prob[0, trans[:, 0], trans[:, 1]]
                > self._trans_prob_trshld
            ]

            if i == 0:
                trans_list = trans.copy()
            else:
                trans_list = cp.vstack([trans_list, trans])

        # TODO: Ändern, sollte cp.unique schneller werden.
        transitions = cp.array(
            np.unique(trans_list.get(), axis=0), dtype=cp.uint32
        )

        energies = knots[0, :, 1]
        start = transitions[:, 0]
        end = transitions[:, 1]
        deltas = abs(energies[end] - energies[start])
        delta_energies_filter = cp.logical_and(
            deltas >= 1.8 * self._field.min(),
            deltas <= 2.2 * self._field.max(),
        )
        self._transitions = transitions[delta_energies_filter]

    def _get_single_res_fields(
        self, grid_point: int
    ) -> [cp.array, cp.array, cp.array, cp.array]:
        """
        Get the resonance fields for one grid point.

        Parameters
        ----------
        grid_point : int
            Index of the angles in self._grid.

        Returns
        -------
        res_fields : cp.array, (N,)
            Resonance fields for one angle on the sphere.
        intensities : cp.array, (N,)
            Intensity for each resonance field.
        widths : cp.array, (N,)
            Linewidth for each resonance field.
        transition : cp.array, (N, 2)
            Indices for the energy levels of each transition.

        """
        start = time()
        _, theta, phi = self._grid[grid_point]
        self._grid_point = grid_point
        start_ = time()
        energy_levels, pop, trans_prob = self._adaptive_spline(theta, phi)
        end_ = time()

        # idx = cp.tril_indices(energy_levels.c.shape[2], k=-1)
        # transition = cp.column_stack(idx)

        # transition = self._filter_at_center(trans_prob, pop, transition)
        transition = self._transitions.copy()
        delta_energy = self._get_delta_splines(energy_levels[0], transition)
        res_fields, delta_energy, transition = self._find_res_fields(
            delta_energy, transition
        )

        intensities = self._get_intensities(
            res_fields, trans_prob[0], pop[0], transition
        )

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
        ) = self._filter_by_position(
            res_fields, intensities, delta_energy, transition
        )

        widths = self._get_linewidths(res_fields, delta_energy)
        print((end_ - start_) / (time() - start), time() - start)

        return res_fields, intensities, widths, transition

    def _find_res_fields(
        self, delta_splines: CubicHermiteSpline, transitions: cp.array
    ) -> cp.array:
        """
        Get the resonance fields for each possible transition.

        Parameters
        ----------
        delta_splines : CubicHermiteSpline
            Spline representation of the energy difference between levels.
        transitions : cp.array, (M, 2)
            Indices for each transition.

        Returns
        -------
        res_fields : cp.array, (N,)
            Resonance field for each transition.
        transition : cp.array, (N, 2)
            Indices for each transition.

        """
        # TODO: Uncomment CuPy code and delete SciPy Code when .solve() is
        # implemented.
        # fields_1 = delta_splines.solve(self._nu)
        # fields_2 = delta_splines.solve(-self._nu)
        cpu_pp = PPoly(
            delta_splines.c.get(),
            delta_splines.x.get(),
            delta_splines.extrapolate,
            delta_splines.axis,
        )

        fields_1 = cpu_pp.solve(self._nu)
        fields_2 = cpu_pp.solve(-self._nu)

        res_fields = [
            cp.sort(
                cp.append(
                    cp.array(fields_1[i], dtype=CUPY_FLOAT),
                    cp.array(fields_2[i], dtype=CUPY_FLOAT),
                )
            )
            for i in range(len(fields_1))
        ]

        if not res_fields:
            return (
                cp.array(res_fields, dtype=CUPY_FLOAT),
                delta_splines,
                transitions,
            )

        trans = []
        fields = []
        coeff = []
        for i in range(len(res_fields)):
            trans.append(list(transitions[i]) * res_fields[i].size)
            fields.append(list(res_fields[i]))
            for _ in range(res_fields[i].size):
                coeff.append(delta_splines.c[:, :, i])

        transition = cp.array(
            [t for sec in trans for t in sec], dtype=cp.uint32
        )
        transitions = transition.reshape((transition.size // 2, 2))
        res_fields = cp.array([f for sec in fields for f in sec])
        coeff = cp.array(coeff)  # , dtype=CUPY_FLOAT)
        delta_splines.c = cp.einsum("abc -> bca", coeff)  # , dtype=CUPY_FLOAT)

        return res_fields, delta_splines, transitions

    def _get_splines(self, knots: cp.array) -> object:
        """
        Get the splines for each energy level as a CubicHermiteSpline object.

        Parameters
        ----------
        knots : cp.array, (M, N, 3)
            Calculated knots with their energies and gradients.

        Returns
        -------
        object
            scipy.interpolate.CubicHermiteSpline object.

        """
        knots = knots[knots[:, 0, 0].argsort()]  # sort along field axis
        field = knots[:, 0, 0]
        energies = knots[:, :, 1]
        gradients = knots[:, :, 2]
        splines = CubicHermiteSpline(
            field.astype(CUPY_FLOAT),
            energies.astype(CUPY_FLOAT),
            gradients.astype(CUPY_FLOAT),
            axis=0,
            extrapolate=True,
        )

        return splines

    def _get_trans_prob_interp(
        self, field: cp.array, trans_prob: cp.array
    ) -> object:
        """
        Interpolate the transition probability along the field axis.

        Parameters
        ----------
        field : cp.array, (M,)
            Magnetic field points.
        trans_prob : cp.array, (M, N, N)
            Corresponding transition probabilities.

        Returns
        -------
        object
            Linear interpolator for the populations of each energy level.

        """
        # TODO: interp1d ersetzen zu CuPy
        return interp1d(
            field.get(), trans_prob.get(), axis=0, fill_value="extrapolate"
        )

    def _get_pop_interp(self, field: cp.array, eigvecs: cp.array) -> object:
        """
        Interpolate the populations along the field axis.

        Parameters
        ----------
        field : cp.array, (M,)
            Magnetic field points.
        eigvecs : cp.array, (M, M)
            Corresponding eigenvectors.

        Returns
        -------
        object
            Linear interpolator for the populations of each energy level.

        """
        eigvecs_T = cp.einsum("aij -> aji", eigvecs, dtype=CUPY_CMPLX)
        eigvecs_inv = cp.linalg.inv(eigvecs_T)

        pop = cp.empty(eigvecs.shape, dtype=CUPY_FLOAT)
        for i in range(eigvecs.shape[0]):
            pop[i] = cp.dot(
                eigvecs_T[i], cp.dot(self._rho, eigvecs_inv[i])
            ).real

        pop = cp.einsum("ajj -> aj", pop).real

        # TODO: interp1d ersetzen zu CuPy
        return interp1d(
            field.get(), pop.get(), axis=0, fill_value="extrapolate"
        )

    def _get_transition_probabilities(self, eigvecs: cp.array) -> cp.array:
        """
        Get the transition probabilties matrix.

        The matrix contains the transitions probabilities for each eigenvector
        combination. The probability for the transition eigenvector[i] ->
        eigenvector[j] is the element [i, j] of the  matrix. Due to symmetry,
        only the lower triangle is returned.

        Parameters
        ----------
        eigvecs : cp.array, (N, M, M)
            Eigenvector matrices returned by e. g. cp.linalg.eigh().

        Returns
        -------
        trans_prob : cp.array, (N, M, M)
            Transition probabilities.

        """
        eigvecs_T = cp.einsum("aij -> aji", eigvecs, dtype=CUPY_CMPLX)
        trans_prob = cp.empty(eigvecs.shape, dtype=CUPY_FLOAT)

        for i in range(trans_prob.shape[0]):
            trans_prob[i] = cp.tril(
                abs(
                    cp.dot(
                        cp.conjugate(eigvecs_T[i]),
                        cp.dot(self._proj, eigvecs[i]),
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
        transitions: cp.array,
    ) -> cp.array:
        """
        Calculate the intensities for the given resonance fields.

        Parameters
        ----------
        field : cp.array, (M,)
            Resonance field for each transition.
        trans_prob : object
            Linear interpolator for the transition probabilities.
        population : object
            Linear interpolator for the populations of each energy level along
            the field axis.
        transitions : cp.array, (M, 2)
            Indices for each transition.

        Returns
        -------
        intensities : cp.array, (M,)
            Peak intensity at the resonance field for each transition.

        """
        delta_pop = self._get_delta_pop(field, population, transitions)

        # TODO: nach interp1d ersetzen zu CuPy kann cp.array und get() entfernt
        # werden
        trans = cp.array(trans_prob(field.get()), dtype=CUPY_FLOAT)

        i = range(transitions.shape[0])
        tp = trans[i, transitions[i, 0], transitions[i, 1]]
        return delta_pop * tp

    def _get_delta_pop(
        self, field: cp.array, population: object, transitions: cp.array
    ) -> cp.array:
        """
        Get the population differences for each transition.

        Parameters
        ----------
        field : cp.array, (M,)
            Resonance field for each transition.
        population : object
            Linear interpolator for the populations of each energy level along
            the field axis.
        transitions : cp.array, (M, 2)
            Indices for each transition.

        Returns
        -------
        cp.array, (M,)
            Population difference for each transition.

        """
        # TODO: nach interp1d ersetzen zu CuPy kann cp.array und get() entfernt
        # werden
        pops = cp.array(population(field.get()), dtype=CUPY_FLOAT)
        i = range(field.size)
        return pops[i, transitions[:, 0]] - pops[i, transitions[:, 1]]

    def _sanitize_results(
        self,
        res_fields: list,
        intensities: list,
        widths: list,
        transition: list,
    ) -> [list, list, list, list]:
        """
        Sanitize the results.

        Remove all transitions where not at least one angle has an intensity
        above self._int_thrsld. Also adds missing transitions if an angle has
        no resonance field for the particular transition. The value of those
        resonance fields will be None.

        Sanitizing the results will decrease further computation time and
        memory consumption.

        Parameters
        ----------
        res_fields : list
            List of res_fields for each angle.
        intensities : list
            List of intensities for each angle.
        widths : list
            List of linewidths for each angle.
        transition : list
            List of transitions for each angle.

        Returns
        -------
        res_fields : list
            List of res_fields for each angle.
        intensities : list
            List of intensities for each angle.
        widths : list
            List of linewidths for each angle.
        transition : list
            List of transitions for each angle.

        """
        transition_count = cp.zeros(
            (len(res_fields), self._transitions.shape[0]), dtype=cp.uint32
        )
        for i in range(len(res_fields)):
            for j in range(self._transitions.shape[0]):
                transition_count[i, j] = cp.sum(
                    cp.all(transition[i] == self._transitions[j], axis=1)
                )

        if not (transition_count == 1).all():
            # Add missing transitions
            fill = cp.array([None], dtype=CUPY_FLOAT)
            missing = cp.argwhere(transition_count == 0).get()
            for i, j in missing:
                mid_idx = transition_count[i, :j].sum()

                res_fields[i] = cp.concatenate(
                    [res_fields[i][:mid_idx], fill, res_fields[i][mid_idx:]],
                    dtype=CUPY_FLOAT,
                )
                intensities[i] = cp.concatenate(
                    [intensities[i][:mid_idx], fill, intensities[i][mid_idx:]],
                    dtype=CUPY_FLOAT,
                )
                widths[i] = cp.concatenate(
                    [widths[i][:mid_idx], fill, widths[i][mid_idx:]],
                    dtype=CUPY_FLOAT,
                )

                empty = cp.array([], dtype=cp.uint32).reshape((0, 2))
                if transition[i][:mid_idx].size == 0:
                    first_part = empty
                else:
                    first_part = transition[i][:mid_idx]
                if transition[i][mid_idx:].size == 0:
                    second_part = empty
                else:
                    second_part = transition[i][mid_idx:]

                transition[i] = cp.concatenate(
                    [
                        first_part,
                        cp.atleast_2d(self._transitions[j]),
                        second_part,
                    ],
                    dtype=cp.uint32,
                )

                transition_count[i, j] = 1

        # Filter by intensity
        mask = cp.zeros(self._transitions.shape[0], dtype=cp.bool_)
        ends = cp.cumsum(transition_count, axis=1)
        starts = cp.roll(ends, 1)
        starts[:, 0] = 0

        for i in range(mask.size):
            trans = self._transitions[i]
            treshold_reached = False
            j = 0
            while not treshold_reached and j < transition_count.shape[0]:
                if (
                    cp.abs(intensities[j][starts[j, i] : ends[j, i]])
                    >= self._int_trshld
                ).any():
                    treshold_reached = True
                else:
                    j += 1

            mask[i] = treshold_reached

        if not cp.all(mask):
            start_idx = starts[:, ~mask]
            end_idx = ends[:, ~mask]
            for i in range(transition_count.shape[0]):
                idx_remove = cp.concatenate(
                    [
                        cp.linspace(
                            start, end - 1, int(end - start), dtype=cp.uint32
                        )
                        for start, end in zip(start_idx[i], end_idx[i])
                    ]
                ).tolist()

                res_fields[i] = cp.delete(res_fields[i], idx_remove)
                intensities[i] = cp.delete(intensities[i], idx_remove)
                widths[i] = cp.delete(widths[i], idx_remove)
                transition[i] = cp.delete(transition[i], idx_remove, axis=0)

        return res_fields, intensities, widths, transition

    def _filter_at_center(
        self, trans_prob: object, population: object, transitions: cp.array
    ) -> cp.array:
        """
        Filter the transitions by their transition rates at B_center.

        Parameters
        ----------
        trans_prob : object
            Interpolator for the transition probabilities.
        population : object
            Linear interpolator for the populations of each energy level along
            the field axis.
        transitions : cp.array, (M, 2)
            Indices for each transition.

        Returns
        -------
        transitions : cp.array, (N, 2)
            Indices for each transition.

        """
        B_center = cp.ones(transitions.shape[0], dtype=CUPY_FLOAT) * (
            (self._field.min() + self._field.max()) / 2
        )

        # Filter by population
        delta_pop = self._get_delta_pop(B_center, population, transitions)
        transitions = transitions[abs(delta_pop) > self._pop_trshld]

        # Filter by transition probability
        # TODO: trans_prob(B_center[0]) , wenn interp1d in CuPy
        trans_probs = cp.array(trans_prob(B_center[0].get()), dtype=CUPY_FLOAT)
        transitions = transitions[
            trans_probs[transitions[:, 0], transitions[:, 1]]
            > self._trans_prob_trshld
        ]

        return transitions

    def _filter_by_intensity(
        self,
        field: cp.array,
        intensities: cp.array,
        delta_E: CubicHermiteSpline,
        transitions: cp.array,
    ) -> [cp.array, cp.array, cp.array]:
        """
        Filter transitions by their intensity.

        Parameters
        ----------
        field : cp.array, (M,)
            Resonance field for each transition.
        intensities : cp.array, (M,)
            Peak intensity at the resonance field for each transition.
        delta_E : CubicHermiteSpline
            Delta spline functions.
        transitions : cp.array, (M, 2)
            Indices for each transition.

        Returns
        -------
        field : cp.array, (N,)
            Resonance field for each transition.
        intensities : cp.array, (N,)
            Peak intensity at the resonance field for each transition.
        delta_E : CubicHermiteSpline
            Delta spline functions.
        transitions : cp.array, (N, 2)
            Indices for each transition.

        """
        idx = abs(intensities) > self._int_trshld
        delta_E.c = delta_E.c[:, :, idx]
        return field[idx], intensities[idx], delta_E, transitions[idx]

    def _filter_by_position(
        self,
        field: cp.array,
        intensities: cp.array,
        delta_E: CubicHermiteSpline,
        transitions: cp.array,
    ) -> [cp.array, cp.array, cp.array]:
        # TODO: max_spread einführen!
        """
        Filter transitions by their field position.

        Parameters
        ----------
        field : cp.array, (M,)
            Resonance field for each transition.
        intensities : cp.array, (M,)
            Peak intensity at the resonance field for each transition.
        delta_E : CubicHermiteSpline
            Delta spline functions.
        transitions : cp.array, (M, 2)
            Indices for each transition.

        Returns
        -------
        field : cp.array, (N,)
            Resonance field for each transition.
        intensities : cp.array, (N,)
            Peak intensity at the resonance field for each transition.
        delta_E : CubicHermiteSpline
            Delta spline functions.
        transitions : cp.array, (N, 2)
            Indices for each transition.

        """
        max_spread = (0.5**2 / cp.log(2, dtype=CUPY_FLOAT)) * 3
        max_spread *= 1e-3 * mu_b
        idx_1 = field > (self._min_field - max_spread)
        idx_2 = field < (self._max_field + max_spread)
        idx = idx_1 & idx_2
        delta_E.c = delta_E.c[:, :, idx]
        return field[idx], intensities[idx], delta_E, transitions[idx]

    def _get_delta_splines(
        self, splines: CubicHermiteSpline, transitions: cp.array
    ) -> CubicHermiteSpline:
        """
        Construct the delta functions of each splines pair.

        Parameters
        ----------
        splines : CubicHermiteSpline
            Spline functions.
        transitions : cp.array, (M, 2)
            Indices for each transition.

        Returns
        -------
        CubicHermiteSpline
            Delta spline functions.

        """
        delta_splines = deepcopy(splines)
        coeff = splines.c

        a = coeff.shape[1]
        b = transitions.shape[0]
        delta_coeff = cp.empty((4, a, b))

        delta_coeff = (
            coeff[:, :, transitions[:, 0]] - coeff[:, :, transitions[:, 1]]
        )

        delta_splines.c = delta_coeff

        return delta_splines

    def _get_linewidths(
        self, field: cp.array, delta_E: CubicHermiteSpline
    ) -> cp.array:
        """
        Calculate the gaussian linewidth for each resonance field.

        Parameters
        ----------
        field : cp.array, (M,)
            Resonance field for each transition.
        delta_E : CubicHermiteSpline
            Delta spline functions.

        Returns
        -------
        linewidths : cp.array, (M,)
            Linewidth for each resonance field.

        """
        linewidths = 1 / cp.diag(
            delta_E.derivative()(field).astype(CUPY_FLOAT)
        )

        return linewidths

    def _get_start_segments(self, n_angles: int) -> cp.array:
        """
        Get the start segments of each angle for the adaptive spline algorithm.

        Parameters
        ----------
        n_angles : int
            Number of angles.

        Returns
        -------
        cp.array, (n_angles,)
            Array containing the start segment for each angle.

        """
        return cp.linspace(
            2 * n_angles, 3 * n_angles - 1, n_angles, dtype=cp.uint32
        )

    def _get_start_points(
        self, theta: cp.array, phi: cp.array
    ) -> [cp.array, cp.array]:
        """
        Get the eigenvalues and eigenvectors for the initial points.

        Parameters
        ----------
        theta : cp.array, (N,)
            Theta angles.
        phi : cp.array, (N,)
            Phi angles.

        Returns
        -------
        knots : cp.array, (2*N, M, 3)
            Evaluated segments for each angle.
        eigvec_field : cp.array, (2*N,)
            Field values for all eigenvectors
        eigvec_vec : cp.array, (2*N, M, M)
            Eigenvectors for all angles.

        """
        fields = cp.tile(
            cp.array([self._min_field, self._max_field], dtype=CUPY_FLOAT),
            theta.size,
        )
        knots, (eigvec_field, eigvec_vec) = self._get_energies_gradients(
            fields, theta, phi, self._n_new_centers
        )
        return knots, eigvec_field, eigvec_vec

    def _get_interpolated_values(
        self, values: cp.array, idx_left: cp.array, idx_right: cp.array
    ) -> cp.array:
        r"""
        Get the estimated energies at the center of the spline segments.

        Those estimated energies will be compared to the exact values to
        determine if the adaptive spline algorithm is converged.

        .. math::

            \tilde{E}_u(B_3) &= \frac{1}{2}\left[E_u(B_1)+E_u(B_2)\right] +
            \frac{\Delta B}{8}\left[\frac{\delta E_u(B_1)}{\delta B}
                                    -\frac{\delta E_u(B_2)}{\delta B}\right]\\
            \textrm{with:}\\
            B_3 &= \frac{B_1 + B_2}{2}

        Parameters
        ----------
        values : cp.array, (N, M, 3)
            Start and end points of the local spline [B1, B2] with their
            energies and gradients. Start and end points are alternating. E. g.
            values[0] is start, values[1] is end point of the first segment.

        Returns
        -------
        E_interp : cp.array, (N/2, M)
            Interpolated energies for each segment.

        """
        delta_B = values[idx_right][:, :, 0] - values[idx_left][:, :, 0]
        sum_E = values[idx_left][:, :, 1] + values[idx_right][:, :, 1]
        delta_grad = values[idx_right][:, :, 2] - values[idx_left][:, :, 2]
        E_interp = 0.5 * sum_E + delta_B / 8 * delta_grad
        return E_interp

    def _get_spline_error(
        self, calculated: cp.array, expected: cp.array
    ) -> cp.array:
        r"""
        Get the maximum field error for each segment.

        .. math::

            \tilde{\delta}_B = \max_u \lvert \frac{E_u(B_3)-\tilde{E}_u(B_3)}
            {\delta E_u(B_3) /\delta B}\rvert

        Parameters
        ----------
        calculated : cp.array, (N, M, 3)
            Calculated exact energies and gradients for the center of each
            segment.
        expected : cp.array, (N, M)
            Interpolated energies for the center of each segment.

        Returns
        -------
        error : cp.array, (N,)
            Maximum field error for each segment.

        """
        error = cp.max(
            cp.abs((calculated[:, :, 1] - expected) / calculated[:, :, 2]),
            axis=1,
        )
        return error

    def _get_new_indexes(self) -> cp.array:
        """
        Get the indexes of the new points.

        Returns
        -------
        cp.array, (N,)
            Indexes of the new points.

        """
        start_idx = cp.sum(self._n_tot_centers).get()
        n_idx = cp.sum(self._n_new_centers).get()
        return cp.linspace(
            start_idx, start_idx + n_idx - 1, n_idx, dtype=cp.uint32
        )

    def _get_start_neighbors(self, N: int) -> cp.array:
        """
        Get the indexes of the left and right neighbors at the start.

        Parameters
        ----------
        N : int
            Number of grid angles.

        Returns
        -------
        cp.array, (N, 2)
            Index of the left and right neighbor for each angle.

        """
        return cp.linspace(0, 2 * N - 1, 2 * N, dtype=cp.uint32).reshape(
            (N, 2)
        )

    def _set_idx_man_map(self) -> None:
        """
        Set the map between indexes of the new points and the index managers.

        The result will be saved in self._idx_man_map

        Returns
        -------
        None

        """
        self._idx_man_map = cp.empty(
            self._n_new_centers.sum().get(), dtype=cp.uint32
        )

        pos = 0
        for j, n in enumerate(self._n_new_centers):
            i = 0
            while i < n:
                self._idx_man_map[pos] = j
                pos += 1
                i += 1

    def _update_idx_managers(self) -> None:
        """
        Update the index managers.

        This function inserts the indices of the new segments between their
        neighbors. self._idx_managers will be updated.

        Returns
        -------
        None

        """
        for i, man_id in enumerate(self._idx_man_map):
            self._idx_managers[man_id.get()].add_index(
                self._segments[i],
                self._neighbors[i, 0],
                self._neighbors[i, 1],
            )

    def _set_new_indexes(self, converged: cp.array) -> None:
        """
        Set the indexes of the new segments and their neighbors.

        Parameters
        ----------
        converged : cp.array, (N,)
            Boolean array, which contains the convergence status of each
            segment.

        Returns
        -------
        None

        """
        # Construct new neighbors
        left = cp.column_stack(
            (self._neighbors[~converged, 0], self._segments[~converged])
        )
        right = cp.column_stack(
            (self._segments[~converged], self._neighbors[~converged, 1])
        )
        new_neighbors = cp.vstack((left, right))

        size = new_neighbors.shape[0]
        order = (
            cp.linspace(0, size - 1, size, dtype=cp.uint32)
            .reshape((2, size // 2))
            .T.flatten()
        )
        self._neighbors = new_neighbors[order]

        # Construct new segments
        start = self._n_tot_centers.sum()
        n_new_centers = 2 * int((~converged).sum())
        new_idx = cp.linspace(
            start, start + n_new_centers - 1, n_new_centers, dtype=cp.uint32
        )
        self._segments = new_idx

    def _set_n_new_centers(self, converged: cp.array) -> None:
        """
        Update self._n_new_centers.

        Parameters
        ----------
        converged : cp.array, (N,)
            Boolean array, which contains the convergence status of each
            segment.

        Returns
        -------
        None

        """
        if (~converged).sum() > 0:
            self._n_new_centers = 2 * cp.bincount(
                self._idx_man_map[~converged], minlength=self._minlength
            ).astype(cp.uint32)
        else:
            self._n_new_centers[:] = 0

    def _evaluate_segments(
        self, values: cp.array, theta, phi
    ) -> [cp.array, cp.array]:
        """
        Evaluate all segments.

        Calculate all eigenvalues and eigenvectors for the new segments and
        determine the convergence status.

        Parameters
        ----------
        values : cp.array, (A, M, 3)
            Eigenvalues and gradients of all calculated segments.

        Returns
        -------
        exact : cp.array, (B, M, 3)
            Field point, eigenvalues and gradients for each new segment.
        new_eigvec : cp.array, (B, M, M)
            Eigenvectors of the new segments.

        """
        # theta, phi = self._grid[:, 1], self._grid[:, 2]

        # Get left and right borders; Get new field points
        left_idx, right_idx = self._neighbors[:, 0], self._neighbors[:, 1]
        b_left = values[left_idx, 0, 0]
        b_right = values[right_idx, 0, 0]
        b_mid = (b_left + b_right) / 2
        # Get exact and interpolated values for all new field points
        exact, new_eigvec = self._get_energies_gradients(
            b_mid, theta, phi, self._n_new_centers
        )
        interp = self._get_interpolated_values(values, left_idx, right_idx)

        # Error estimation
        error = self._get_spline_error(exact, interp)

        # Convergence determination
        converged = cp.asarray(error <= self._tau_B, dtype=cp.bool_)

        # Update index managers
        self._update_idx_managers()

        # Update n_tot_centers
        self._n_tot_centers += self._n_new_centers

        # Update n_new_centers
        self._set_n_new_centers(converged)

        # Set the new indexes for the new segments and new neighbors
        self._set_new_indexes(converged)

        return exact, new_eigvec

    def _adaptive_spline(
        self, theta: cp.array, phi: cp.array
    ) -> [CubicHermiteSpline, object, object]:
        """
        Get a cubic spline representation for each energy level.

        This function uses the adaptive bisection algorithm. All angles get
        evaluated in parallel.

        Parameters
        ----------
        theta : cp.array, (N,)
            Angles in radian. Used for the setup of the hamiltonian.
        phi : cp.array, (N,)
            Angles in radian. Used for the setup of the hamiltonian.

        Returns
        -------
        CubicHermiteSpline
            Spline representation of the energy levels.
        object
            Linear interpolator for the populations.
        object
            Linear interpolator for the transition probabilities.

        """
        theta, phi = cp.array([theta]), cp.array([phi])  # TODO: entfernen
        self._minlength = theta.size
        # Arrays to distinguish between different angles
        self._n_new_centers = cp.full(theta.size, 2, dtype=cp.uint32)
        self._n_tot_centers = cp.zeros(theta.size, dtype=cp.uint32)

        # Set start values
        points, eigvec_field, eigvec_vec = self._get_start_points(theta, phi)

        # Set start segments
        self._segments = self._get_start_segments(theta.size)
        self._neighbors = self._get_start_neighbors(theta.size)
        self._n_tot_centers += self._n_new_centers
        self._n_new_centers[:] = 1

        n = 0
        while self._segments.size > 0:
            # Set index manager map
            self._set_idx_man_map()

            # Evaluate segments
            new_points, new_eigvec = self._evaluate_segments(
                points, theta, phi
            )

            # Safe new points
            points = cp.vstack([points, new_points])
            eigvec_field = cp.append(eigvec_field, new_eigvec[0])
            eigvec_vec = cp.vstack([eigvec_vec, new_eigvec[1]])
            n += 1
            if n > 4:
                break

        # Irrelevanter Zeitanteil (nur 11% der Laufzeit)
        # If each segment is converged:
        all_splines = []
        all_population = []
        all_trans_prob = []
        for i in range(theta.size):
            sorting = self._idx_managers[i].order()

            field = eigvec_field[sorting]
            vec = eigvec_vec[sorting]

            splines = self._get_splines(points[sorting])
            sorting = cp.argsort(field)
            field = field[sorting]
            vec = vec[sorting]

            trans_prob = self._get_transition_probabilities(vec)
            trans_prob = self._get_trans_prob_interp(field, trans_prob)

            population = self._get_pop_interp(field, vec)

            all_splines.append(splines)
            all_population.append(population)
            all_trans_prob.append(trans_prob)

        return all_splines, all_population, all_trans_prob

    def _get_multi_single_res_fields(
        self, energy_levels, pop, trans_prob
    ) -> cp.array:
        """
        Get the resonance fields for one grid point.

        Parameters
        ----------
        grid_point : int
            Index of the angles in self._grid.

        Returns
        -------
        res_fields : cp.array, (N, 4)
            Resonance fields for one angle on the sphere.

        """
        transition = self._transitions.copy()
        delta_energy = self._get_delta_splines(energy_levels, transition)
        res_fields, delta_energy, transition = self._find_res_fields(
            delta_energy, transition
        )

        intensities = self._get_intensities(
            res_fields, trans_prob, pop, transition
        )

        (
            res_fields,
            intensities,
            delta_energy,
            transition,
        ) = self._filter_by_position(
            res_fields, intensities, delta_energy, transition
        )

        widths = self._get_linewidths(res_fields, delta_energy)

        return res_fields, intensities, widths, transition

    def _get_energies_gradients(
        self,
        field: cp.array,
        theta: cp.array,
        phi: cp.array,
        n_points: cp.array,
    ) -> [cp.array, tuple[cp.array, cp.array]]:
        """
        Get the energies and gradients for the given field points.

        Parameters
        ----------
        field : cp.array, (L,)
            Field points.
        theta : cp.array, (N,)
            Angles on the grid.
        phi : cp.array, (N,)
            Angles on the grid.
        n_points: cp.array, (N,)
            Number of field points per orientation. Used to map the correct
            angles to each new field point.

        Returns
        -------
        knots : cp.array, (L, M, 3)
            Calculated knots with their energies and gradients.
        eigenvector : tuple, (cp.array, cp.array)
            Field points and corresponding eigenvector matrices from
            cp.linalg.eigh().

        """
        theta = cp.repeat(theta, n_points.tolist())
        phi = cp.repeat(phi, n_points.tolist())

        energies, eigvec = self._ham.get_eigen(field, theta, phi)
        gradients = self._ham.get_field_gradients(field, theta, phi)

        eigvec_field = field.copy()

        field = cp.repeat(field, energies.shape[1]).reshape(
            (field.size, energies.shape[1])
        )

        knots = cp.stack([field, energies, gradients], axis=-1)

        return knots, (eigvec_field, eigvec)

    def _init_idx_managers(self) -> None:
        """Initalize the IndexManager for each orientation."""
        n_theta = self._grid.shape[0]
        idx_left = 2 * cp.linspace(0, n_theta - 1, n_theta, dtype=cp.uint32)
        self._idx_managers = [
            IndexManager(idx, idx + 1) for idx in idx_left.get()
        ]


class IndexManager:
    def __init__(self, left_boundary, right_boundary):
        """
        Initialisiert die Klasse mit festen Grenzen.
        :param left_boundary: Der linke feste Grenz-Indize
        :param right_boundary: Der rechte feste Grenz-Indize
        """
        if left_boundary >= right_boundary:
            raise ValueError(
                "Left boundary must be smaller than right boundary."
            )

        self.indices = {
            left_boundary: {"left": None, "right": int(right_boundary)},
            right_boundary: {"left": int(left_boundary), "right": None},
        }
        self.left_boundary = int(left_boundary)
        self.right_boundary = int(right_boundary)

    def add_index(self, index, left_neighbor, right_neighbor):
        """
        Fügt einen neuen Indize hinzu und sortiert ihn entsprechend ein.
        :param index: Der neue Indize
        :param left_neighbor: Der linke Nachbar des neuen Indizes
        :param right_neighbor: Der rechte Nachbar des neuen Indizes
        """
        if int(index) in self.indices:
            raise ValueError(f"Index {index} already exists.")

        if (
            int(left_neighbor) not in self.indices
            or int(right_neighbor) not in self.indices
        ):
            raise ValueError(
                "Both neighbors must already exist in the structure."
            )

        if (
            self.indices[int(left_neighbor)]["right"] != right_neighbor
            or self.indices[int(right_neighbor)]["left"] != left_neighbor
        ):
            raise ValueError("The given neighbors are not adjacent.")

        # Update neighbors
        self.indices[int(left_neighbor)]["right"] = int(index)
        self.indices[int(right_neighbor)]["left"] = int(index)

        # Add the new index
        self.indices[int(index)] = {
            "left": int(left_neighbor),
            "right": int(right_neighbor),
        }

    def get_neighbors(self, index):
        """
        Gibt die Nachbarn eines Indizes zurück.
        :param index: Der Indize, dessen Nachbarn abgefragt werden
        :return: Ein Tupel (linker Nachbar, rechter Nachbar)
        """
        if int(index) not in self.indices:
            raise ValueError(f"Index {index} does not exist.")

        neighbors = self.indices[int(index)]
        return neighbors["left"], neighbors["right"]

    def order(self):
        order = []
        current = self.left_boundary

        while current is not None:
            order.append(current)
            current = self.indices[current]["right"]
        return cp.array(order, dtype=cp.uint32)

    def __str__(self):
        """
        Gibt die aktuelle Reihenfolge der Indizes als String zurück.
        """
        order = []
        current = self.left_boundary

        while current is not None:
            order.append(current)
            current = self.indices[current]["right"]

        return " -> ".join(map(str, order))
