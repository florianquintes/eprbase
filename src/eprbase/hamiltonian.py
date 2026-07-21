#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hamiltonian construction for EPR simulations of radical pairs.

This module provides the :class:`Hamiltonian` class to set up and diagonalize
the spin Hamiltonian, including Electron-Zeeman (EZ), Hyperfine (HFI),
Dipolar (DIP), and Exchange (EX) interactions.

© M. Sc. Florian Quintes, 2026.
@contact: florian.quintes@pc.uni.freiburg.de
@author: Florian Quintes
"""

import numpy as np
from functools import cache


class Hamiltonian:
    """
    Spin Hamiltonian for coupled radical pairs.

    Handles the construction and caching of the total spin Hamiltonian matrix,
    including Zeeman, hyperfine, dipolar, and exchange terms. Provides methods
    to retrieve the Hamiltonian matrix and its eigenvalues/eigenvectors for
    given magnetic fields and orientations.

    Attributes
    ----------
    _EZ, _HFI, _DIP, _SI, _matrix : ndarray or None
        Internal caches for interaction tensors and the full Hamiltonian.
    _eigenvalues, _eigenvectors : ndarray or None
        Internal caches for eigenvalues and eigenvectors of the Hamiltonian.
    _multiplicity : int
        Total spin multiplicity of the system.
    """

    def __init__(self) -> None:
        """
        Initialize the Hamiltonian object with default parameters.

        Sets up coupled electron spins, initializes projection operators, and
        flags all interaction terms as changed.
        """
        self._EZ = None
        self._HFI = None
        self._DIP = None
        self._SI = None
        self._matrix = None
        self._eigenvalues = None
        self._eigenvectors = None
        self._multiplicity = 1
        self._init_coupled_electron_spins()
        self.set_exchange(0)
        self.set_dipolar(0, 0)
        self._changed_g = True
        self._changed_hfi = True
        self._changed_dip = True
        self._changed_ex = True
        self._field = np.array([])
        self._phi = np.array([])
        self._theta = np.array([])
        self._set_proj()
        self._symmetry = None

    def _set_proj(self) -> None:
        """
        Set up the projection operator (S_x/S_y).

        Notes
        -----
        Updates the internal ``self._proj`` attribute in place.
        """
        self._proj = np.kron(self._S[0], np.eye(self._multiplicity))

    def get_proj(self) -> np.array:
        """
        Return the projection operator.

        Returns
        -------
        np.array
            The projection operator matrix.
        """
        return self._proj

    def get_field_gradients(
        self, field: np.array, theta: np.array, phi: np.array
    ) -> np.array:
        """
        Calculate the gradients of energy levels along the magnetic field axis.

        Parameters
        ----------
        field : np.array, shape (N,)
            Magnetic field values.
        theta : np.array, shape (N,)
            Theta angles in radians.
        phi : np.array, shape (N,)
            Phi angles in radians.

        Returns
        -------
        np.array, shape (N, M)
            Gradient along the field for each energy level.
        """
        eigvec = self.get_eigenvectors(field, theta, phi)
        self.set_EZ(theta, phi)
        EZ = np.kron(self._EZ, np.eye(self._multiplicity))
        grads = np.empty((field.size, EZ.shape[1]))
        for i in range(field.size):
            grads[i] = np.diag(
                np.linalg.multi_dot([np.conjugate(eigvec[i].T), EZ[i], eigvec[i]])
            )
        return grads

    def set_g(self, g: np.array) -> None:
        """
        Set the principal values of the g tensor for each electron.

        Parameters
        ----------
        g : np.array
            Principal g-values. Must be provided for each electron in the
            coupled system.
        """
        self._g = np.array(np.atleast_2d(g))[:, :, np.newaxis] * np.eye(3)
        self._changed_g = True

    def set_EZ(self, theta: np.array, phi: np.array) -> None:
        r"""
        Set up the Electron-Zeeman (EZ) interaction Hamiltonian.

        .. math::

           \hat{\mathcal{H}}_{\mathrm{ez}} = -\sum_{i = x,y,z}{g_{iz} \cdot
           \hat{S}_i}

        Parameters
        ----------
        theta : np.array
            Theta angles in radians.
        phi : np.array
            Phi angles in radians.
        """
        self._EZ = np.zeros((theta.size, 4, 4), dtype=np.complex128)
        S = np.array([self._S1, self._S2])
        for i in range(len(self._g)):
            g_rot = rotate_tensor(self._g[i], phi, theta)
            self._EZ -= (
                g_rot[:, :, 2, np.newaxis, np.newaxis] * S[i][np.newaxis, :, :]
            ).sum(axis=1)

    def set_Nuc(self, A: np.array, spin: np.array, acc_len: int = 0) -> None:
        r"""
        Set the nuclei coupling with the radical pair and precalculate S*I.

        Precalculates the product of electron spin (S) and nuclear spin (I)
        matrices for the hyperfine coupling.

        .. math::

           SI_{mn} = S_m \cdot I_n

        Parameters
        ----------
        A : np.array
            Hyperfine coupling tensors.
        spin : np.array of float
            Nuclei spin numbers. First the ones for the acceptor electron,
            then all for the donor electron.
        acc_len : int, optional
            Number of nuclei which couple to the acceptor electron. The
            default is 0.
        """
        self._A = A
        if sum(spin) == 0:
            self._multiplicity = 1
        else:
            self._multiplicity = int((2 * np.array(spin) + 1).prod())
        self._changed_hfi = True
        self._set_proj()

        if sum(spin) == 0:
            self._SI = None
            return

        spin_matrices = self._get_coupled_spin_matrices(0.5, 0.5, *spin)
        S1 = spin_matrices[0]
        S2 = spin_matrices[1]
        I_ = spin_matrices[2:]

        self._SI = np.zeros((I_.shape[0], 3, 3, *S1[0].shape), dtype=np.complex128)
        path, _ = np.einsum_path("ikl, jlm-> ijkm", S1, I_[0], optimize=True)
        for i in range(acc_len):
            self._SI[i] = np.einsum("ikl, jlm-> ijkm", S1, I_[i], optimize=path)

        for i in range(acc_len, I_.shape[0]):
            self._SI[i] = np.einsum("ikl, jlm-> ijkm", S2, I_[i], optimize=path)

    def set_HFI(self, theta: np.array, phi: np.array):
        r"""
        Set up the Hyperfine (HFI) Hamiltonian for multiple nuclei.

        .. math::

            \hat{\mathcal{H}}_{\mathrm{HF}} &= \sum_i{\mathbf{
                \overrightarrow{S}A_i\overrightarrow{I_i}}}\\
                &= \sum_i\sum_{m}\sum_{n}a_{i,mn}\cdot\overrightarrow{S}_{m}
                \cdot\overrightarrow{I}_n

        with:

        .. math::

            m, n \in\{x, y, z\}

                
        Parameters
        ----------
        theta : np.array
            Theta angles in radians.
        phi : np.array
            Phi angles in radians.
        """
        if self._SI is None or self._multiplicity == 1:
            self._HFI = np.zeros((theta.size, 4, 4), dtype=np.complex128)
            return

        self._HFI = np.zeros(
            (theta.size, self._SI.shape[-1], self._SI.shape[-1]),
            dtype=np.complex128,
        )

        A_rot = np.zeros((self._A.shape[0], theta.size, *self._A.shape[1:]))
        for i, A in enumerate(self._A):
            A_rot[i] = rotate_tensor(A, phi, theta)

        # a = nuc, b = angle, m&n = A values, i&j = spin matrices
        self._HFI = np.einsum(
            "abmnij, abmnij->bij",
            A_rot[:, :, :, :, np.newaxis, np.newaxis],
            self._SI[:, np.newaxis, :, :, :, :],
        )

    def set_exchange(self, J_ex: float) -> None:
        r"""
        Set up the Hamiltonian for the exchange coupling.

        .. math::

        \hat{\mathcal{H}}_{\mathrm{ex}} = -2J\cdot \hat{S}_1\cdot \hat{S}_2

        Parameters
        ----------
        J_ex : float
            Exchange coupling constant.
        """
        self._changed_ex = True
        self._exchange = J_ex * (np.eye(self._S1S2.shape[0]) * 0.5 + 2 * self._S1S2)

    def set_dipolar(self, D: float, E: float) -> None:
        r"""
        Set up the Dipolar / Zero-Field Splitting (ZFS) tensor.

        .. math::

        \mathbf{D} = \begin{bmatrix}
        -D+E & 0   & 0 \\
        0   & -D-E & 0 \\
        0   & 0   & 2\cdot D
        \end{bmatrix}

        Parameters
        ----------
        D : float
            D value of the zero field splitting.
        E : float
            E value of the zero field splitting.
        """
        self._changed_dip = True
        self._dipolar = np.array([-D + E, -D - E, 2 * D], dtype=np.complex128) * np.eye(
            3, dtype=np.complex128
        )

    def set_DIP(self, theta, phi) -> None:
        """
        Set up the Dipolar interaction Hamiltonian.

        Rotates the dipolar tensor according to the given angles and calculates
        the interaction term.

        Parameters
        ----------
        theta : np.array
            Theta angles in radians.
        phi : np.array
            Phi angles in radians.
        """
        if self._dipolar is None:
            return np.zeros((theta.size, 4, 4), dtype=np.complex128)

        D_rot = rotate_tensor(self._dipolar, phi, theta)
        self._DIP = (
            (D_rot[:, :, :, np.newaxis, np.newaxis] * self._SS[np.newaxis, :, :, :, :])
            .sum(axis=1)
            .sum(axis=1)
        )

    def get_symmetry() -> str:
        # TODO: Funktion erstellen und Tests schreiben.
        """
        Get the SO(3) point group symmetry of the Hamiltonian.

        Returns
        -------
        str
            The SO(3) group identifier. Currently defaults to ``"Ci"``.
        """
        return "Ci"

    def get(self, field, theta, phi) -> np.array:
        """
        Calculate and return the total Hamiltonian matrix.

        Caches the result if neither the interaction parameters nor the
        field/orientation have changed since the last call.

        Parameters
        ----------
        field : np.array, shape (N,)
            Magnetic field values.
        theta : np.array, shape (N,)
            Theta angles in radians.
        phi : np.array, shape (N,)
            Phi angles in radians.

        Returns
        -------
        np.array, shape (N, M, M)
            The total Hamiltonian matrix for each orientation/field point.
        """
        same_theta = np.array_equal(theta, self._theta)
        same_phi = np.array_equal(phi, self._phi)
        same_angles = same_theta and same_phi
        same_field = np.array_equiv(field, self._field)
        hfi_dim = np.eye(self._multiplicity)

        if (
            same_field
            and same_angles
            and (
                not (
                    self._changed_g
                    | self._changed_hfi
                    | self._changed_dip
                    | self._changed_ex
                )
            )
        ):
            return self._matrix

        if self._changed_g or not same_angles:
            self.set_EZ(theta, phi)
        if self._changed_hfi or not same_angles:
            self.set_HFI(theta, phi)
        if self._changed_dip or not same_angles:
            self.set_DIP(theta, phi)

        if not same_angles:
            self._theta = theta
            self._phi = phi

        EZ = np.kron(field[:, np.newaxis, np.newaxis] * self._EZ, hfi_dim)
        HFI = self._HFI
        DIP = np.kron(self._DIP, hfi_dim)
        EX = np.kron(self._exchange, hfi_dim)

        self._matrix = EZ + HFI + DIP + EX[np.newaxis, :, :]
        self._eigenvalues = None
        self._eigenvectors = None

        return self._matrix

    def get_eigen(self, field, theta, phi) -> tuple[np.array, np.array]:
        """
        Return the eigenvalues and eigenvectors of the Hamiltonian.

        Parameters
        ----------
        field : np.array, shape (N,)
            Magnetic field values.
        theta : np.array, shape (N,)
            Theta angles in radians.
        phi : np.array, shape (N,)
            Phi angles in radians.

        Returns
        -------
        eigenvalues : np.array, shape (N, M)
            Eigenvalues of the Hamiltonian.
        eigenvectors : np.array, shape (N, M, M)
            Corresponding eigenvectors.
        """
        self.get(field, theta, phi)
        if self._eigenvectors is None:
            self._eigenvalues, self._eigenvectors = np.linalg.eigh(
                np.round(self._matrix, decimals=1)
            )
        return self._eigenvalues, self._eigenvectors

    def get_eigenvalues(self, field, theta, phi) -> np.array:
        """
        Return only the eigenvalues of the Hamiltonian.

        Parameters
        ----------
        field : np.array, shape (N,)
            Magnetic field values.
        theta : np.array, shape (N,)
            Theta angles in radians.
        phi : np.array, shape (N,)
            Phi angles in radians.

        Returns
        -------
        np.array, shape (N, M)
            Eigenvalues of the Hamiltonian.
        """
        self.get(field, theta, phi)
        if self._eigenvalues is None:
            self._eigenvalues = np.linalg.eigvalsh(np.round(self._matrix, decimals=1))
        return self._eigenvalues

    def get_eigenvectors(self, field, theta, phi) -> np.array:
        """
        Return only the eigenvectors of the Hamiltonian.

        Parameters
        ----------
        field : np.array, shape (N,)
            Magnetic field values.
        theta : np.array, shape (N,)
            Theta angles in radians.
        phi : np.array, shape (N,)
            Phi angles in radians.

        Returns
        -------
        np.array, shape (N, M, M)
            Eigenvectors of the Hamiltonian.
        """
        self.get(field, theta, phi)
        if self._eigenvectors is None:
            self._eigenvalues, self._eigenvectors = np.linalg.eigh(
                np.round(self._matrix, decimals=1)
            )
        return self._eigenvectors

    @classmethod
    @cache
    def _init_coupled_electron_spins(self) -> np.array:
        """
        Initialize spin matrices for the coupled electron system.

        Calculates and caches the total spin vectors and their products
        (e.g., :math:`S_1 \cdot S_2`) for the two coupled electrons.
        """
        sigma_x, sigma_y, sigma_z = self._get_spin_matrices(0.5)

        pauli = np.array([sigma_x, sigma_y, sigma_z])
        self._S1 = np.kron(pauli, np.eye(2))
        self._S2 = np.kron(np.eye(2), pauli)
        self._S = self._S1 + self._S2
        self._S1S2 = (self._S1 @ self._S2).sum(axis=0)
        self._SS = np.einsum("ikl, jlm ->ijkm", self._S, self._S)

    @classmethod
    @cache
    def _get_spin_matrices(self, S: float = 0.5) -> list[np.array, np.array, np.array]:
        r"""
        Calculate the spin matrices for a given spin quantum number.

        Uses the standard ladder operator approach:

        .. math::

            \hat{S}_x &= \frac{1}{2} \left( \hat{S}_{+}
                                             + \hat{S}_{-} \right)\\
            \hat{S}_y &= \frac{1}{2i} \left( \hat{S}_{+}
                                              - \hat{S}_{-} \right)\\
            \hat{S}_{z} &= \frac{1}{2} \left[ \hat{S}_{+}, \hat{S}_{-} \right]

        Note, that :math:`\hat{S}_{z}` can be determined using:

        .. math::

            \hat{S}_{z_{j, k}} = \delta_{j, k}\cdot (2 \cdot S + 1 - (j-1))

        To get the so-called Ladder-Operators, you can use:

        .. math::

            \hat{S}_{+_{j, k}} &= \delta_{j+1, k}\cdot \sqrt{S(S+1)-n_j}\\
            \hat{S}_{-_{j, k}} &= \delta_{j, k+1}\cdot \sqrt{S(S+1)-n_j}\\

        with:

        .. math::

            n_j &= {m_j'\cdot m_j}\\
            m' &= \{ S, ..., -S+1 \}\\
            m &= \{ S-1, ..., -S \}


        Parameters
        ----------
        S : float or int, optional
            Spin quantum number. The default is 0.5.

        Returns
        -------
        s_x : np.array
            :math:`\\hat{S}_{x}` spin matrix.
        s_y : np.array
            :math:`\\hat{S}_{y}` spin matrix.
        s_z : np.array
            :math:`\\hat{S}_{z}` spin matrix.

        """
        # Multiplicity M
        M = int(2 * S + 1)

        # Values for the Ladder Operators
        m_1 = np.linspace(S, -S + 1, M - 1)
        m_2 = np.linspace(S - 1, -S, M - 1)
        m_m = np.sqrt(S * (S + 1) - m_1 * m_2)

        # Ladder Operators S_+ and S_-
        s_p = np.eye(M, k=1, dtype=np.complex128)
        s_m = np.eye(M, k=-1, dtype=np.complex128)
        s_p[s_p == 1] = m_m
        s_m[s_m == 1] = m_m

        # Spin Matrices
        s_x = 0.5 * (s_p + s_m)
        s_y = -0.5j * (s_p - s_m)
        s_z = np.linspace(S, -S, M) * np.eye(M, dtype=np.complex128)

        return s_x, s_y, s_z

    @classmethod
    @cache
    def _get_coupled_spin_matrices(self, *spins: float) -> np.array:
        r"""
        Calculate spin matrices for a system of multiple coupled spins.

        All spins are coupled into the same product basis:

        .. math::

            S_i  =  E(a) \otimes \sigma(s_i) \otimes E(b)

        with:

        .. math::

            E &\equiv \mathbf{Id}\\
            \sigma(s_i) &\equiv \text{uncoupled spin matrix}\\
            a &= \prod_{k = 1}^{i}\mathrm{dim}(s_k)\\
            b &= \prod_{k = i+1}^{n}\mathrm{dim}(s_k)

        Example:

        With the following spins:

        .. math::

            s_1 &= 0.5\\
            s_2 &= 1.0\\
            s_3 &= 1.5

        the spin matrices are:

        .. math::

            S_1 &= E(1) \otimes \sigma(0.5) \otimes E(12)\\
            S_2 &= E(2) \otimes \sigma(1.0) \otimes E(4)\\
            S_3 &= E(6) \otimes \sigma(1.5) \otimes E(1)


        Parameters
        ----------
        *spins : float
            Spin quantum numbers for all coupled spins.

        Returns
        -------
        np.array
            Array of spin vectors for each coupled spin, in the same order
            as the input spins.
        """
        spins = np.array(spins)
        dims = (2 * spins + 1).astype(int)
        dim_total = dims.prod()
        spin_matrices = np.empty(
            (spins.size, 3, dim_total, dim_total), dtype=np.complex128
        )

        for i in range(spins.size):
            x, y, z = self._get_spin_matrices(spins[i])
            pauli_mat = np.array([x, y, z])

            dim_lhs = dims[:i].prod()
            dim_rhs = dims[i + 1 :].prod()
            spin_matrices[i] = np.kron(
                np.kron(np.eye(dim_lhs), pauli_mat), np.eye(dim_rhs)
            )

        return spin_matrices


def rotate_tensor(
    tensor: np.array, phi: np.array, theta: np.array, psi: np.array = None
) -> np.array:
    r"""
    Rotate a tensor using Euler transformation in y-convention.

    Performs an orthogonal similarity transformation of the tensor:

    .. math::

       T' = O^{-1} \cdot T \cdot O

    with:

        .. math::

            O^{-1} = O^T

    where :math:`O` is the Euler matrix of the SO(3) group in
    y-convention.

    Parameters
    ----------
    tensor : np.array
        Tensor to be rotated. Can be 2D or 3D.
    phi : float or np.array
        Phi angle(s) in radians.
    theta : float or np.array
        Theta angle(s) in radians.
    psi : float or np.array, optional
        Psi angle(s) in radians. If None, defaults to zero.

    Returns
    -------
    np.array
        The rotated tensor.
    """
    if psi is None:
        psi = np.zeros(phi.size)
    # Allocations
    cosphi = np.cos(phi)
    sinphi = np.sin(phi)
    costhet = np.cos(theta)
    sinthet = np.sin(theta)
    cospsi = np.cos(psi)
    sinpsi = np.sin(psi)
    eulermatrix = np.zeros((phi.size, 3, 3))
    # Set up the full 3-dimensional Euler matrix
    eulermatrix[:, 0, 0] = cosphi * costhet * cospsi - sinphi * sinpsi
    eulermatrix[:, 0, 1] = -cosphi * costhet * sinpsi - sinphi * cospsi
    eulermatrix[:, 0, 2] = cosphi * sinthet
    eulermatrix[:, 1, 0] = sinphi * costhet * cospsi + cosphi * sinpsi
    eulermatrix[:, 1, 1] = -sinphi * costhet * sinpsi + cosphi * cospsi
    eulermatrix[:, 1, 2] = sinphi * sinthet
    eulermatrix[:, 2, 0] = -sinthet * cospsi
    eulermatrix[:, 2, 1] = sinthet * sinpsi
    eulermatrix[:, 2, 2] = costhet

    if tensor.ndim == 2:
        rot_1 = np.einsum("ij, ajk -> aik", tensor, eulermatrix)
    elif tensor.ndim == 3:
        rot_1 = np.einsum("aij, ajk -> aik", tensor, eulermatrix)
    else:
        raise ValueError("Tensor has wrong dimensions!")
    rotatedTensor = np.einsum("aji, ajk -> aik", eulermatrix, rot_1)

    return rotatedTensor
