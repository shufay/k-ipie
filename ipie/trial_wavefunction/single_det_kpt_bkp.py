import time
from typing import Optional, Tuple

import numpy
from numba import jit
import plum

from ipie.config import CommType, config, MPI
from ipie.estimators.utils import gabk_spin
from ipie.hamiltonians.kpt_hamiltonian import KptComplexChol, KptComplexCholSymm, KptISDF
from ipie.hamiltonians.kpt_chunked import KptComplexCholChunked
from ipie.walkers.uhf_walkers import UHFWalkers
from ipie.propagation.force_bias import construct_force_bias_kpt_batch_single_det, construct_force_bias_kptsymm_batch_single_det,construct_force_bias_kptisdf_batch_single_det, construct_force_bias_kptsymm_batch_single_det_chunked
from ipie.trial_wavefunction.half_rotate import half_rotate_generic, half_rotate_chunked, half_rotate_isdf
from ipie.propagation.overlap import calc_overlap_single_det_kpt
from ipie.trial_wavefunction.wavefunction_base import TrialWavefunctionBase
from ipie.estimators.greens_function_kpt_single_det import greens_function_kpt_single_det, greens_function_kpt_single_det_batch
from ipie.utils.backend import arraylib as xp
from ipie.utils.mpi import MPIHandler
from typing import Union

@jit(nopython=True, fastmath=True)
def _kpt_chol_ecoul_kernel_uhf(
        rchola, rcholb, rcholbara, rcholbarb, Ghalfa, Ghalfb, GhalfaT, GhalfbT, igamma):
    """Compute coulomb contribution for real rchol with UHF trial for a single walker.

    Parameters
    ----------
    rchola : :class:`numpy.ndarray`
        Half-rotated cholesky (alpha).
    rcholb : :class:`numpy.ndarray`
        Half-rotated cholesky (beta).
    Ghalfa : :class:`numpy.ndarray`
        Walker's half-rotated "green's function" shape is nalpha  x nbasis.
    Ghalfb : :class:`numpy.ndarray`
        Walker's half-rotated "green's function" shape is nbeta x nbasis.
    igamma : int
        kpt index of the gamma point.

    Returns
    -------
    ecoul : :class:`numpy.ndarray`
        coulomb contribution for all walkers.
    """
    # sort out cupy later
    zeros = numpy.zeros
    dot = numpy.dot

    # shape of rchola: (naux, nk, nocc, nk, nbsf) (gamma, k, i, q, p)
    # shape of Ghalf: (nk, nocc, nbsf)
    naux = rchola.shape[0]
    nk = rchola.shape[1]
    nbsf = rchola.shape[-1]
    nocca = rchola.shape[2]
    noccb = rcholb.shape[2]

    rchola = rchola.transpose(3, 1, 2, 0, 4).copy() # (q, k, gamma, i, p)
    rcholb = rcholb.transpose(3, 1, 2, 0, 4).copy()
    rcholbara = rcholbara.transpose(3, 1, 2, 0, 4).copy()
    rcholbarb = rcholbarb.transpose(3, 1, 2, 0, 4).copy()
    X = zeros((naux), dtype=numpy.complex128)
    Xbar = zeros((naux), dtype=numpy.complex128)

    for ik in range(nk):
        La = rchola[igamma, ik].reshape(naux, nocca*nbsf)
        Lb = rcholb[igamma, ik].reshape(naux, noccb*nbsf)
        Lbara = rcholbara[igamma, ik].reshape(naux, nocca*nbsf)
        Lbarb = rcholbarb[igamma, ik].reshape(naux, noccb*nbsf)

        Ghalfa_k = Ghalfa[ik].reshape(nocca*nbsf)
        GhalfTa_k = GhalfaT[ik].reshape(nocca*nbsf)
        Ghalfb_k = Ghalfb[ik].reshape(noccb*nbsf)
        GhalfTb_k = GhalfbT[ik].reshape(noccb*nbsf)
        X += La @ Ghalfa_k + Lb @ Ghalfb_k
        Xbar += Lbara @ GhalfTa_k + Lbarb @ GhalfTb_k
    
    ecoul = 0.5 * dot(X, Xbar) / nk
    return ecoul

@jit(nopython=True, fastmath=True)
def _kpt_symmchol_ecoul_kernel_uhf(rchola, rcholb, rcholbara, rcholbarb, Ghalfa, Ghalfb, GhalfaT, GhalfbT, kpq_mat, Sset, Qplus):
    """Compute coulomb contribution for real rchol with UHF trial for a single walker.

    Parameters
    ----------
    rchola : :class:`numpy.ndarray`
        Half-rotated cholesky (alpha).
    rcholb : :class:`numpy.ndarray`
        Half-rotated cholesky (beta).
    Ghalfa : :class:`numpy.ndarray`
        Walker's half-rotated "green's function" shape is nalpha  x nbasis.
    Ghalfb : :class:`numpy.ndarray`
        Walker's half-rotated "green's function" shape is nbeta x nbasis.

    Returns
    -------
    ecoul : :class:`numpy.ndarray`
        coulomb contribution for all walkers.
    """
    # sort out cupy later
    zeros = numpy.zeros
    dot = numpy.dot
    multiply = numpy.multiply

    # shape of rchola: (nq, nk, nocc, naux, nbsf) (q, k, i, gamma, p)
    # shape of Ghalf: (nk, nk, nw, nocc, nbsf)
    unique_nq = len(Sset) + len(Qplus)
    nbsf = rchola.shape[4]
    naux = rchola.shape[3]
    nk = rchola.shape[1]
    nocca = rchola.shape[2]
    noccb = rcholb.shape[2]
    rchola = rchola.transpose(0, 1, 3, 2, 4).copy()
    rcholb = rcholb.transpose(0, 1, 3, 2, 4).copy()
    rcholbara = rcholbara.transpose(0, 1, 3, 2, 4).copy()
    rcholbarb = rcholbarb.transpose(0, 1, 3, 2, 4).copy()
    X = zeros((unique_nq, naux), dtype=numpy.complex128)
    Xbar = zeros((unique_nq, naux), dtype=numpy.complex128)
    for iq in range(len(Sset)):
        iq_real = Sset[iq]
        Xq = X[iq]
        Xbarq = Xbar[iq]
        for ik in range(nk):
            ik_pq = kpq_mat[iq_real, ik]
            La = rchola[iq, ik].reshape(naux,nocca*nbsf)
            Lb = rcholb[iq, ik].reshape(naux,noccb*nbsf)
            Lbara = rcholbara[iq, ik].reshape(naux,nocca*nbsf)
            Lbarb = rcholbarb[iq, ik].reshape(naux,noccb*nbsf)
            Ghalfa_k_kpq = Ghalfa[ik, ik_pq].reshape(nocca*nbsf)
            GhalfTa_k_kpq = GhalfaT[ik, ik_pq].reshape(nocca*nbsf)
            Ghalfb_k_kpq = Ghalfb[ik, ik_pq].reshape(noccb*nbsf)
            GhalfTb_k_kpq = GhalfbT[ik, ik_pq].reshape(noccb*nbsf)
            Xq += La @ Ghalfa_k_kpq + Lb @ Ghalfb_k_kpq 
            Xbarq += Lbara @ GhalfTa_k_kpq + Lbarb @ GhalfTb_k_kpq

    for iq in range(len(Sset), len(Sset) + len(Qplus)):
        iq_real = Qplus[iq - len(Sset)]
        Xq = X[iq]
        Xbarq = Xbar[iq]
        for ik in range(nk):
            ik_pq = kpq_mat[iq_real, ik]
            La = rchola[iq, ik].reshape(naux,nocca*nbsf)
            Lb = rcholb[iq, ik].reshape(naux,noccb*nbsf)
            Lbara = rcholbara[iq, ik].reshape(naux,nocca*nbsf)
            Lbarb = rcholbarb[iq, ik].reshape(naux,noccb*nbsf)
            Ghalfa_k_kpq = Ghalfa[ik, ik_pq].reshape(nocca*nbsf)
            GhalfTa_k_kpq = GhalfaT[ik, ik_pq].reshape(nocca*nbsf)
            Ghalfb_k_kpq = Ghalfb[ik, ik_pq].reshape(noccb*nbsf)
            GhalfTb_k_kpq = GhalfbT[ik, ik_pq].reshape(noccb*nbsf)
            Xq += sqrt(2) * (La @ Ghalfa_k_kpq + Lb @ Ghalfb_k_kpq)
            Xbarq += sqrt(2) * (Lbara @ GhalfTa_k_kpq + Lbarb @ GhalfTb_k_kpq)

    X = X.transpose(1, 0, 2).copy()
    Xbar = Xbar.transpose(1, 0, 2).copy()
    X = X.reshape(nwalkers, naux * unique_nq)
    Xbar = Xbar.reshape(nwalkers, naux * unique_nq)
    ecoul = 0.5 * dot(X, Xbar) / nk
    return ecoul

@jit(nopython=True, fastmath=True)
def _kpt_chol_exx_kernel(rchol, Ghalf, kpq_mat, mq_vec):
    """Compute coulomb contribution for complex rchol with RHF trial for a single walker.

    Parameters
    ----------
    rchol : :class:`numpy.ndarray`
        Half-rotated cholesky.
    Ghalf : :class:`numpy.ndarray`
        Walker's half-rotated "green's function" shape is nalpha  x nbasis
    kpq_mat : :class:`numpy.ndarray`
        all k + q in fractional coordinates.
    mq_vec : :class:`numpy.ndarray`
        all -q in fractional coordinates.

    Returns
    -------
    ecoul : :class:`numpy.ndarray`
        coulomb contribution for all walkers.
    """
    # sort out cupy later
    zeros = numpy.zeros
    dot = numpy.dot

    # shape of rchol: (naux, nk, nocc, nk, nbsf) (gamma, k, i, q, p)
    # shape of Ghalf: (nk, nocc, nk, nbsf)
    naux = rchol.shape[0]
    nocc = rchol.shape[2]
    nk = rchol.shape[1]
    exx = 0.j
    GhalfT = Ghalf.transpose(0, 2, 1)

    T1 = zeros((naux, nocc, nocc), dtype=numpy.complex128)
    T2 = zeros((naux, nocc, nocc), dtype=numpy.complex128)

    for iq in range(nk):
        for ik in range(nk):
            ik_pq = kpq_mat[ik, iq]
            i_mq = mq_vec[iq]

            for g in range(naux):
                T1[g] = dot(rchol[g, ik, :, iq, :], GhalfT[ik_pq])
                T2[g] = dot(rchol[g, ik_pq, :, i_mq, :], GhalfT[ik])
                exx += -numpy.trace(dot(T1[g], T2[g]))

    return 0.5 * exx / nk

# class for UHF trial
class KptSingleDet(TrialWavefunctionBase):
    def __init__(self, wavefunction, nkpts, num_elec, num_basis, handler=MPIHandler(), verbose=False):
        assert isinstance(wavefunction, numpy.ndarray)
        assert len(wavefunction.shape) == 3 # nkpts, nbasis, nocc
        super().__init__(wavefunction, num_elec, num_basis, verbose=verbose)
        if verbose:
            print("# Parsing input options for trial_wavefunction.MultiSlater.")
        self.psi = wavefunction
        self.num_elec = num_elec
        self.nk = nkpts
        self._num_dets = 1
        self._max_num_dets = 1
        imag_norm = numpy.sum(self.psi.imag.ravel() * self.psi.imag.ravel())
        if imag_norm <= 1e-8:
            # print("# making trial wavefunction MO coefficient real")
            self.psi = numpy.array(self.psi.real, dtype=numpy.float64)

        self.psi0a = self.psi[:, :, : self.nalpha]
        self.psi0b = self.psi[:, :, self.nalpha :]
        self.G, self.Ghalf = gabk_spin(self.psi, self.psi, self.nalpha, self.nbeta)
        self.handler = handler

        self.psi0a = numpy.ascontiguousarray(self.psi0a)
        self.psi0b = numpy.ascontiguousarray(self.psi0b)

    def build(self) -> None:
        pass

    @property
    def num_dets(self) -> int:
        return 1

    @num_dets.setter
    def num_dets(self, ndets: int) -> None:
        raise RuntimeError("Cannot modify number of determinants in SingleDet trial.")
    
    @plum.dispatch
    def calculate_energy(
            self, 
            system, 
            hamiltonian: KptComplexChol) -> numpy.ndarray:
        if self.verbose:
            print("# Computing trial wavefunction energy.")

        start = time.time()
        nk = hamiltonian.nk
        nalpha = self.nalpha
        nbeta = self.nbeta
        nbasis = hamiltonian.nbasis
    
        Ghalfa = self.Ghalf[0]
        Ghalfb = self.Ghalf[1]
        GhalfaT = Ghalfa.transpose(0, 2, 1)
        GhalfbT = Ghalfb.transpose(0, 2, 1)

        self.e1b = (numpy.sum(Ghalfa * self._rH1a)
                    + numpy.sum(Ghalfb * self._rH1b)) / nk + hamiltonian.ecore
        
        self.ej = _kpt_chol_ecoul_kernel_uhf(
            self._rchola, self._rcholb, self._rcholbara, self._rcholbarb, 
            Ghalfa, Ghalfb, GhalfaT, GhalfbT, hamiltonian.igamma)

        exxa = _kpt_chol_exx_kernel(self._rchola, Ghalfa, hamiltonian.ikpq_mat, hamiltonian.imq_vec) 
        exxb = _kpt_chol_exx_kernel(self._rcholb, Ghalfb, hamiltonian.ikpq_mat, hamiltonian.imq_vec)
        self.ek = exxa + exxb
        self.e2b = self.ej + self.ek # minus sign included in ek.
        self.energy = self.e1b + self.e2b
        
        if self.verbose:
            print(
                "# (E, E1B, E2B): (%13.8e, %13.8e, %13.8e)"
                % (self.energy.real, self.e1b.real, self.e2b.real))
            print(
                "# (EJ, EK): (%13.8e, %13.8e)"
                % (self.ej.real, self.ek.real))
            print(f"# Time to evaluate local energy: {time.time() - start} s")

    @plum.dispatch
    def half_rotate(
        self: "KptSingleDet",
        hamiltonian: KptComplexChol,
        comm: Optional[CommType] = MPI.COMM_WORLD,
    ):
        num_dets = 1
        orbsa = self.psi0a.reshape((num_dets, self.nk, self.nbasis, self.nalpha))
        orbsb = self.psi0b.reshape((num_dets, self.nk, self.nbasis, self.nbeta))
        rot_1body, rot_chol = half_rotate_generic(
            self,
            hamiltonian,
            comm,
            orbsa,
            orbsb,
            ndets=num_dets,
            verbose=self.verbose,
        )
        # Single determinant functions do not expect determinant index, so just
        # grab zeroth element.
        self._rH1a = rot_1body[0][0]
        self._rH1b = rot_1body[1][0]
        self._rchola = rot_chol[0][0]
        self._rcholb = rot_chol[1][0]
        self._rcholbara = rot_chol[2][0]
        self._rcholbarb = rot_chol[3][0]
        self.half_rotated = True

    @plum.dispatch
    def half_rotate(
        self: "KptSingleDet",
        hamiltonian: KptComplexCholSymm,
        comm: Optional[CommType] = MPI.COMM_WORLD,
    ):
        num_dets = 1
        orbsa = self.psi0a.reshape((num_dets, self.nk, self.nbasis, self.nalpha))
        orbsb = self.psi0b.reshape((num_dets, self.nk, self.nbasis, self.nbeta))
        rot_1body, rot_chol = half_rotate_generic(
            self,
            hamiltonian,
            comm,
            orbsa,
            orbsb,
            ndets=num_dets,
            verbose=self.verbose,
        )
        # Single determinant functions do not expect determinant index, so just
        # grab zeroth element.
        self._rH1a = rot_1body[0][0]
        self._rH1b = rot_1body[1][0]
        self._rchola = rot_chol[0][0]
        self._rcholb = rot_chol[1][0]
        self._rcholbara = rot_chol[2][0]
        self._rcholbarb = rot_chol[3][0]
        self.half_rotated = True

    @plum.dispatch
    def half_rotate(
        self: "KptSingleDet",
        hamiltonian: KptComplexCholChunked,
        comm: Optional[CommType] = MPI.COMM_WORLD,
    ):
        num_dets = 1
        orbsa = self.psi0a.reshape((num_dets, self.nk, self.nbasis, self.nalpha))
        orbsb = self.psi0b.reshape((num_dets, self.nk, self.nbasis, self.nbeta))
        rot_1body, rot_chol = half_rotate_chunked(
            self,
            hamiltonian,
            comm,
            orbsa,
            orbsb,
            ndets=num_dets,
            verbose=self.verbose,
        )
        # Single determinant functions do not expect determinant index, so just
        # grab zeroth element.
        self._rH1a = rot_1body[0][0]
        self._rH1b = rot_1body[1][0] if self.nbeta > 0 else None
        
        self._rchola_chunk = rot_chol[0][0]
        self._rcholb_chunk = rot_chol[1][0] if self.nbeta > 0 else None
        self._rcholbara_chunk = rot_chol[2][0]
        self._rcholbarb_chunk = rot_chol[3][0] if self.nbeta > 0 else None
        self.half_rotated = True

    @plum.dispatch
    def half_rotate(
        self: "KptSingleDet",
        hamiltonian: KptISDF,
        comm: Optional[CommType] = MPI.COMM_WORLD,
    ):
        num_dets = 1
        orbsa = self.psi0a.reshape((num_dets, self.nk, self.nbasis, self.nalpha))
        orbsb = self.psi0b.reshape((num_dets, self.nk, self.nbasis, self.nbeta))
        rot_1body, rot_cgto = half_rotate_isdf(
            self,
            hamiltonian,
            comm,
            orbsa,
            orbsb,
            ndets=num_dets,
            verbose=self.verbose,
        )
        # Single determinant functions do not expect determinant index, so just
        # grab zeroth element.
        self._rH1a = rot_1body[0][0]
        self._rH1b = rot_1body[1][0]
        self._rcgtoa = rot_cgto[0][0]
        self._rcgtob = rot_cgto[1][0]
        self.half_rotated = True

    def calc_overlap(self, walkers: "UHFWalkers") -> xp.ndarray:
        return calc_overlap_single_det_kpt(walkers, self)

    def calc_greens_function(self, walkers, build_full: bool = False) -> xp.ndarray:
        if config.get_option("use_gpu"):
            return greens_function_kpt_single_det_batch(walkers, self, build_full=build_full)
        else:
            return greens_function_kpt_single_det(walkers, self, build_full=build_full)

    @plum.dispatch
    def calc_force_bias(
        self,
        hamiltonian: KptComplexChol,
        walkers: UHFWalkers,
        mpi_handler: MPIHandler,
    ) -> Tuple[xp.ndarray, xp.ndarray]:
        if hamiltonian.chunked:
            raise NotImplementedError
        else:
            return construct_force_bias_kpt_batch_single_det(hamiltonian, walkers, self)
        
    @plum.dispatch
    def calc_force_bias(
        self,
        hamiltonian: Union[KptComplexCholSymm, KptComplexCholChunked],
        walkers: UHFWalkers,
        mpi_handler: MPIHandler,
    ) -> Tuple[xp.ndarray, xp.ndarray]:
        if hamiltonian.chunked:
            return construct_force_bias_kptsymm_batch_single_det_chunked(hamiltonian, walkers, self, mpi_handler)
        else:
            return construct_force_bias_kptsymm_batch_single_det(hamiltonian, walkers, self)

    @plum.dispatch
    def calc_force_bias(
        self,
        hamiltonian: KptISDF,
        walkers: UHFWalkers,
        mpi_handler: MPIHandler,
    ) -> Tuple[xp.ndarray, xp.ndarray]:
        if hamiltonian.chunked:
            raise NotImplementedError
        else:
            return construct_force_bias_kptisdf_batch_single_det(hamiltonian, walkers, self)


