#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
© M. Sc. Florian Quintes, 2021-2022.

@contact: florian.quintes@pc.uni.freiburg.de

@author: Florian Quintes
"""

import numpy as np
import cupy as cp

CUPY_FLOAT = cp.float32
CUPY_CMPLX = cp.complex64


class Hamiltonian:
    def __init__(self) -> None:
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
        self._field = cp.array([], dtype=CUPY_FLOAT)
        self._phi = cp.array([], dtype=CUPY_FLOAT)
        self._theta = cp.array([], dtype=CUPY_FLOAT)
        self._set_proj()
        self._symmetry = None

    def _set_proj(self):
        """Set up the projection operator (S_x/S_y)."""
        self._proj = cp.kron(self._S[0], cp.eye(self._multiplicity, dtype=CUPY_CMPLX))

    def get_proj(self):
        """Get the projection operator."""
        return self._proj

    def get_field_gradients(
        self, field: cp.array, theta: cp.array, phi: cp.array
    ) -> cp.array:
        """
        Get the gradients for each energy level along the field axis.

        Parameters
        ----------
        field : cp.array, (N,)
            Magnetic field point.
        theta : cp.array, (N,)
            Theta angles.
        phi : cp.array, (N,)
            Phi angles.

        Returns
        -------
        grads : cp.array, (N, M)
            Gradient along the field for each energy level.

        """
        eigvec = self.get_eigenvectors(field, theta, phi)
        self.set_EZ(theta, phi)
        EZ = cp.kron(self._EZ, cp.eye(self._multiplicity, dtype=CUPY_CMPLX))
        grads = cp.empty((field.size, EZ.shape[1]), dtype=CUPY_FLOAT)
        for i in range(field.size):
            grads[i] = cp.diag(
                cp.dot(cp.conjugate(eigvec[i].T), cp.dot(EZ[i], eigvec[i]))
            ).real
        return grads

    def set_g(self, g: cp.array) -> None:
        """Set the principal values of the g tensor for each electron."""
        self._g = cp.array(cp.atleast_2d(g), dtype=CUPY_FLOAT)[
            :, :, cp.newaxis
        ] * cp.eye(3, dtype=CUPY_FLOAT)
        self._changed_g = True

    def set_EZ(self, theta: cp.array, phi: cp.array):
        r"""
        Set up a Hamiltonian for the Electron-Zeeman interaction.

        .. math::

            \hat{\mathcal{H}}_{\mathrm{ez}} = -\sum_{i = x,y,z}{g_{iz} \cdot
                                                                \hat{S}_i}

        Parameters
        ----------
        theta : cp.array
            Angle in radian.
        phi : cp.array
            Angle in radian.

        """
        self._EZ = cp.zeros((theta.size, 4, 4), dtype=CUPY_CMPLX)
        S = cp.array([self._S1, self._S2], dtype=CUPY_CMPLX)
        for i in range(len(self._g)):
            g_rot = rotate_tensor(self._g[i], phi, theta)
            self._EZ -= (
                g_rot[:, :, 2, cp.newaxis, cp.newaxis] * S[i][cp.newaxis, :, :]
            ).sum(axis=1)

    def set_Nuc(self, A: cp.array, spin: cp.array, acc_len: int = 0) -> None:
        r"""
        Set the nuclei which couple with the radical pair.

        Precalculate the product of S and I for the hyperfine coupling.

        .. math::
            SI_{mn} = S_m\cdot I_n

        with:

        .. math::

            m, n \in\{x, y, z\}

        Calculations will be done for all S_i - I_j hyperfine interactions.

        Parameters
        ----------
        spin : cp.array, float
            Nuclei spin numbers. First the ones for the acceptor electron, then
            all for the donor electron.
        acc_len : int, optional
            Number of nuclei which couple to the acceptor electron. The default
            is 0.

        """
        if isinstance(spin, list):
            spin = cp.array(spin, dtype=CUPY_FLOAT)

        self._A = A
        if sum(spin) == 0:
            self._multiplicity = 1
        else:
            self._multiplicity = int((2 * cp.array(spin, dtype=CUPY_FLOAT) + 1).prod())
        self._changed_hfi = True
        self._set_proj()

        if sum(spin) == 0:
            self._SI = None
            return

        spin_matrices = self._get_coupled_spin_matrices(0.5, 0.5, *spin.tolist())
        S1 = spin_matrices[0]
        S2 = spin_matrices[1]
        I_ = spin_matrices[2:]

        self._SI = cp.zeros((I_.shape[0], 3, 3, *S1[0].shape), dtype=CUPY_CMPLX)
        path, _ = np.einsum_path(
            "ikl, jlm-> ijkm", S1.get(), I_[0].get(), optimize=True
        )
        for i in range(acc_len):
            self._SI[i] = cp.einsum(
                "ikl, jlm-> ijkm", S1, I_[i], optimize=path, dtype=CUPY_CMPLX
            )

        for i in range(acc_len, I_.shape[0]):
            self._SI[i] = cp.einsum(
                "ikl, jlm-> ijkm", S2, I_[i], optimize=path, dtype=CUPY_CMPLX
            )

    def set_HFI(self, theta: cp.array, phi: cp.array):
        r"""
        Set up the hyperfine Hamiltonian for multiple nuclei with one electron.

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
        theta : cp.array
            Angle in radian.
        phi : cp.array
            Angle in radian.

        """
        if self._SI is None or self._multiplicity == 1:
            self._HFI = cp.zeros((theta.size, 4, 4), dtype=CUPY_CMPLX)
            return

        self._HFI = cp.zeros(
            (theta.size, self._SI.shape[-1], self._SI.shape[-1]),
            dtype=CUPY_CMPLX,
        )

        A_rot = cp.zeros(
            (self._A.shape[0], theta.size, *self._A.shape[1:]),
            dtype=CUPY_FLOAT,
        )
        for i, A in enumerate(self._A):
            A_rot[i] = rotate_tensor(A, phi, theta)

        # a = nuc, b = angle, m&n = A values, i&j = spin matrices
        self._HFI = cp.einsum(
            "abmnij, abmnij->bij",
            A_rot[:, :, :, :, cp.newaxis, cp.newaxis],
            self._SI[:, cp.newaxis, :, :, :, :],
            dtype=CUPY_CMPLX,
        )

    def set_exchange(self, J_ex: float) -> None:
        r"""
        Set up the Hamiltonian for the exchange coupling.

        .. math::

            \hat{\mathcal{H}}_{\mathrm{ex}} = -2J\cdot \hat{S}_1\cdot \hat{S}_2

        Parameters
        ----------
        J_ex : float
            Exchange coupling.

        """
        self._changed_ex = True
        self._exchange = J_ex * (
            cp.eye(self._S1S2.shape[0], dtype=CUPY_CMPLX) * 0.5 + 2 * self._S1S2
        )

    def set_dipolar(self, D: float, E: float) -> None:
        r"""
        Set up the D/ZFS tensor.

        .. math::

            \mathbf{D} =\begin{bmatrix}
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
        self._dipolar = cp.array([-D + E, -D - E, 2 * D], dtype=CUPY_FLOAT) * cp.eye(
            3, dtype=CUPY_FLOAT
        )

    def set_DIP(self, theta, phi) -> None:
        if self._dipolar is None:
            return cp.zeros((theta.size, 4, 4), dtype=CUPY_CMPLX)

        D_rot = rotate_tensor(self._dipolar, phi, theta)
        self._DIP = (
            (D_rot[:, :, :, cp.newaxis, cp.newaxis] * self._SS[cp.newaxis, :, :, :, :])
            .sum(axis=1)
            .sum(axis=1)
        )

    def get_symmetry(self) -> str:
        # TODO: Funktion erstellen und Tests schreiben.
        """
        Get the SO(3) group of the hamiltonian.

        Returns
        -------
        symmetry: str
            SO(3) group.

        """
        return "Ci"

    def get(self, field, theta, phi):
        if field.dtype != CUPY_FLOAT:
            field = field.astype(CUPY_FLOAT)
        same_theta = cp.array_equal(theta, self._theta)
        same_phi = cp.array_equal(phi, self._phi)
        same_angles = same_theta and same_phi
        same_field = cp.array_equiv(field, self._field)
        hfi_dim = cp.eye(self._multiplicity, dtype=CUPY_FLOAT)

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

        EZ = cp.kron(field[:, cp.newaxis, cp.newaxis] * self._EZ, hfi_dim)
        HFI = self._HFI
        DIP = cp.kron(self._DIP, hfi_dim)
        EX = cp.kron(self._exchange, hfi_dim)

        self._matrix = EZ + HFI + DIP + EX[cp.newaxis, :, :]
        self._eigenvalues = None
        self._eigenvectors = None

        return self._matrix

    def get_eigen(self, field, theta, phi):
        self.get(field, theta, phi)
        if self._eigenvectors is None:
            self._eigenvalues, self._eigenvectors = cp.linalg.eigh(
                cp.round(self._matrix, decimals=1)
            )
        return self._eigenvalues, self._eigenvectors

    def get_eigenvalues(self, field, theta, phi):
        self.get(field, theta, phi)
        if self._eigenvalues is None:
            self._eigenvalues = cp.linalg.eigvalsh(cp.round(self._matrix, decimals=1))
        return self._eigenvalues

    def get_eigenvectors(self, field, theta, phi):
        self.get(field, theta, phi)
        if self._eigenvectors is None:
            self._eigenvalues, self._eigenvectors = cp.linalg.eigh(
                cp.round(self._matrix, decimals=1)
            )
        return self._eigenvectors

    @classmethod
    @cp.memoize()
    def _init_coupled_electron_spins(self) -> cp.array:
        sigma_x, sigma_y, sigma_z = self._get_spin_matrices(0.5)

        pauli = cp.array([sigma_x, sigma_y, sigma_z], dtype=CUPY_CMPLX)
        self._S1 = cp.kron(pauli, cp.eye(2, dtype=CUPY_CMPLX))
        self._S2 = cp.kron(cp.eye(2, dtype=CUPY_CMPLX), pauli)
        self._S = self._S1 + self._S2
        self._S1S2 = (self._S1 @ self._S2).sum(axis=0)
        self._SS = cp.einsum("ikl, jlm ->ijkm", self._S, self._S)

    @classmethod
    @cp.memoize()
    def _get_spin_matrices(self, S: float = 0.5) -> list[cp.array, cp.array, cp.array]:
        r"""
        Get the spin matrices for a given spin.

        Calculates the spin matrices for a given spin using:

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
        s_x : cp.array
            :math:`\\hat{S}_{x}` spin matrix.
        s_y : cp.array
            :math:`\\hat{S}_{y}` spin matrix.
        s_z : cp.array
            :math:`\\hat{S}_{z}` spin matrix.

        """
        # Multiplicity M
        M = int(2 * S + 1)

        # Values for the Ladder Operators
        m_1 = cp.linspace(S, -S + 1, M - 1, dtype=CUPY_CMPLX)
        m_2 = cp.linspace(S - 1, -S, M - 1, dtype=CUPY_CMPLX)
        m_m = cp.sqrt(S * (S + 1) - m_1 * m_2, dtype=CUPY_FLOAT)

        # Ladder Operators S_+ and S_-
        s_p = cp.eye(M, k=1, dtype=CUPY_CMPLX)
        s_m = cp.eye(M, k=-1, dtype=CUPY_CMPLX)
        s_p[s_p == 1] = m_m
        s_m[s_m == 1] = m_m

        # Spin Matrices
        s_x = 0.5 * (s_p + s_m)
        s_y = -0.5j * (s_p - s_m)
        s_z = cp.linspace(S, -S, M, dtype=CUPY_CMPLX) * cp.eye(M, dtype=CUPY_CMPLX)

        return s_x, s_y, s_z

    @classmethod
    @cp.memoize()
    def _get_coupled_spin_matrices(self, *spins: float) -> cp.array:
        r"""
        Calculate all spin matrices for a coupled system.

        All spins are coupled into the same product basis.

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
            All spin quantum numbers for all coupled spins.

        Returns
        -------
        spin_matrices : cp.array
            Spin vectors for each coupled spin. Same order as corresponding
            spin.

        """
        spins = cp.array(spins)
        dims = (2 * spins + 1).astype(int)
        dim_total = dims.prod().get()
        spin_matrices = cp.empty(
            (spins.size, 3, dim_total, dim_total), dtype=CUPY_CMPLX
        )

        for i in range(spins.size):
            x, y, z = self._get_spin_matrices(float(spins[i]))
            pauli_mat = cp.array([x, y, z], dtype=CUPY_CMPLX)

            dim_lhs = dims[:i].prod().get()
            dim_rhs = dims[i + 1 :].prod().get()
            spin_matrices[i] = cp.kron(
                cp.kron(cp.eye(dim_lhs, dtype=CUPY_CMPLX), pauli_mat),
                cp.eye(dim_rhs, dtype=CUPY_CMPLX),
            )

        return spin_matrices


def rotate_tensor(
    tensor: cp.array, phi: cp.array, theta: cp.array, psi: cp.array = None
) -> cp.array:
    r"""
    Algorithm:
    Euler transformation using y-convention. The euler matrix is set up
    with the given angles. Phi and theta is necessary, psi is optional.
    The euler matrix O of the SO(3) Group in y-convention is set up in already
    multiplicated form. Than the orthogonal similarity transformation of the
    tensor T is carried out:

    .. math::

        T' = O^{-1}\cdot T \cdot O

    with:

    .. math::

        O^{-1} = O^T


    Parameters
    ----------
    tensor : cp.array
        Tensor which should be rotated using Euler transformation
        (y-convention).
    phi : float
        Phi angle in radian for transformation.
    theta : float
        Theta angle in radian for transformation.
    psi : float, optional
        Psi angle in radian for transformation. The default is None.

    Returns
    -------
    rotatedTensor : cp.array
        Rotated tensor.

    """
    if psi is None:
        psi = cp.zeros(phi.size, dtype=CUPY_FLOAT)
    # Allocations
    cosphi = cp.cos(phi, dtype=CUPY_FLOAT)
    sinphi = cp.sin(phi, dtype=CUPY_FLOAT)
    costhet = cp.cos(theta, dtype=CUPY_FLOAT)
    sinthet = cp.sin(theta, dtype=CUPY_FLOAT)
    cospsi = cp.cos(psi, dtype=CUPY_FLOAT)
    sinpsi = cp.sin(psi, dtype=CUPY_FLOAT)
    eulermatrix = cp.zeros((phi.size, 3, 3), dtype=CUPY_FLOAT)
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
        rot_1 = cp.einsum("ij, ajk -> aik", tensor, eulermatrix)
    elif tensor.ndim == 3:
        rot_1 = cp.einsum("aij, ajk -> aik", tensor, eulermatrix)
    else:
        raise ValueError("Tensor has wrong dimensions!")
    rotatedTensor = cp.einsum("aji, ajk -> aik", eulermatrix, rot_1)

    return rotatedTensor
