from ipie.utils.backend import arraylib as xp
from ipie.utils.backend import synchronize
from numba import jit
import numpy

def greens_function_kpt_single_det(walker_batch, trial, build_full=False):
    """Compute walker's green's function.

    Parameters
    ----------
    walker_batch : object
        SingleDetWalkerBatch object.
    trial : object
        Trial wavefunction object.
    Returns
    -------
    det : float64 / complex128
        Determinant of overlap matrix.
    """
    def pad_empty_kpts(Ghalf, nelec_per_k, noccs):
        if Ghalf is None: 
            return None
        
        nk = noccs.shape[-1]
        nk_occ = xp.sum(noccs)
        nbsf = Ghalf.shape[-1] // nk
        _Ghalf = Ghalf.reshape((nk_occ, -1, nk, nbsf))
        Ghalf = xp.zeros((nk, nelec_per_k, nk, nbsf), dtype=_Ghalf.dtype)
        Ghalf[noccs>0] = _Ghalf
        return Ghalf.reshape((-1, nk*nbsf))

    nup = trial.nalpha
    ndown = trial.nbeta
    nbsf = trial.nbasis
    nk = trial.nk
    noccs = walker_batch.noccs

    if noccs is not None:
        phia = walker_batch.remove_empty_kpts(walker_batch.phia, nk*nup, noccs[0])
        phib = walker_batch.remove_empty_kpts(walker_batch.phib, nk*ndown, noccs[1])

    phia = numpy.ascontiguousarray(
            phia.reshape(walker_batch.nwalkers, nk, nbsf, -1, nup))
    nk_occ = phia.shape[3]

    if ndown > 0:
        phib = numpy.ascontiguousarray(
                phib.reshape(walker_batch.nwalkers, nk, nbsf, -1, ndown))
    else:
        phib = None

    det = []
    for iw in range(walker_batch.nwalkers):
        ovlpt = numpy.zeros((nk_occ, nup, nk, nup), dtype=numpy.complex128)
        for ik1 in range(nk_occ):
            for ik2 in range(nk):
                if noccs[0, ik2] > 0:
                    ovlpt[ik1, :, ik2, :] = numpy.dot(
                            phia[iw, ik2, :, ik1, :].T, trial.psi0a[ik2].conj())
        
        ovlpt = ovlpt[:, :, noccs[0]>0].reshape(nk_occ*nup, nk_occ*nup)
        ovlpinvt = numpy.linalg.inv(ovlpt)
        phia_iw = xp.ascontiguousarray(phia[iw].reshape(nk*nbsf, nk_occ*nup))
        Ghalfa_iw = numpy.dot(ovlpinvt, phia_iw.T)
        walker_batch.Ghalfa[iw] = pad_empty_kpts(Ghalfa_iw, nup, noccs[0])
        Ghalfa_reshaped = walker_batch.Ghalfa[iw].reshape(nk, nup, nk, nbsf)
        Ga = numpy.zeros((nk, nbsf, nk, nbsf), dtype=numpy.complex128)
        if not trial.half_rotated or build_full:
            for ik1 in range(nk):
                for ik2 in range(nk):
                    Ga[ik1, :, ik2, :] = numpy.dot(
                                            trial.psi0a[ik1].conj(), 
                                            Ghalfa_reshaped[ik1, :, ik2, :])
            walker_batch.Ga[iw] = Ga.reshape(nk * nbsf, nk * nbsf)
        sign_a, log_ovlp_a = xp.linalg.slogdet(ovlpt)
        sign_b, log_ovlp_b = 1.0, 0.0
        if ndown > 0 and not walker_batch.rhf:
            nk_occ = phib.shape[3]
            ovlpt = numpy.zeros((nk_occ, ndown, nk, ndown), dtype=numpy.complex128)
            for ik1 in range(nk_occ):
                for ik2 in range(nk):
                    ovlpt[ik1, :, ik2, :] = numpy.dot(
                            phib[iw, ik2, :, ik1, :].T, trial.psi0b[ik2].conj())
            ovlpt = ovlpt[:, :, noccs[1]>0].reshape(nk_occ*ndown, nk_occ*ndown)
            sign_b, log_ovlp_b = xp.linalg.slogdet(ovlpt)
            ovlpinvt = numpy.linalg.inv(ovlpt)
            phib_iw = xp.ascontiguousarray(phib[iw].reshape(nk*nbsf, nk_occ*ndown))
            Ghalfb_iw = numpy.dot(ovlpinvt, phib_iw.T)
            walker_batch.Ghalfb[iw] = pad_empty_kpts(Ghalfb_iw, ndown, noccs[1])
            Ghalfb_reshaped = walker_batch.Ghalfb[iw].reshape(nk, ndown, nk, nbsf)
            Gb = numpy.zeros((nk, nbsf, nk, nbsf), dtype=numpy.complex128)
            if not trial.half_rotated or build_full:
                for ik1 in range(nk):
                    for ik2 in range(nk):
                        Gb[iw, ik1, :, ik2, :] = numpy.dot(
                                                    trial.psi0b[ik1].conj(), 
                                                    Ghalfb_reshaped[ik1, :, ik2, :])
                walker_batch.Gb[iw] = Gb.reshape(nk * nbsf, nk * nbsf)
            det += [sign_a * sign_b * xp.exp(log_ovlp_a + log_ovlp_b - walker_batch.log_shift[iw])]
        elif ndown > 0 and walker_batch.rhf:
            det += [sign_a * sign_a * xp.exp(log_ovlp_a + log_ovlp_a - walker_batch.log_shift[iw])]
        elif ndown == 0:
            det += [sign_a * xp.exp(log_ovlp_a - walker_batch.log_shift[iw])]

    det = xp.array(det, dtype=xp.complex128)

    synchronize()

    return det


def greens_function_kpt_single_det_batch(walker_batch, trial, build_full=False):
    """Compute walker's green's function using only batched operations.

    Parameters
    ----------
    walker_batch : object
        SingleDetWalkerBatch object.
    trial : object
        Trial wavefunction object.
    Returns
    -------
    ot : float64 / complex128
        Overlap with trial.
    """
    nup = trial.nalpha
    ndown = trial.nbeta
    nbsf = trial.nbasis
    nk = trial.nk
    phia = xp.ascontiguousarray(
            walker_batch.phia.reshape(walker_batch.nwalkers, nk, nbsf, nk, nup))
    if ndown > 0:
        phib = xp.ascontiguousarray(
                walker_batch.phib.reshape(walker_batch.nwalkers, nk, nbsf, nk, ndown))
    else:
        phib = None
    
    ovlp_a = xp.einsum("wlpki, lpj->wkilj", phia, trial.psi0a.conj(), optimize=True)
    ovlp_a = ovlp_a.reshape(walker_batch.nwalkers, nk * nup, nk * nup)
    ovlp_inv_a = xp.linalg.inv(ovlp_a)
    sign_a, log_ovlp_a = xp.linalg.slogdet(ovlp_a)

    # walker_batch.Ghalfa = xp.einsum("wij,wmj->wim", ovlp_inv_a, walker_batch.phia, optimize=True)
    walker_batch.Ghalfa = xp.matmul(ovlp_inv_a, walker_batch.phia.transpose(0, 2, 1))
    if not trial.half_rotated or build_full:
        Ga = xp.einsum(
            "kpi,wkilq->wkplq", trial.psi0a.conj(), walker_batch.Ghalfa, optimize=True
        )
        walker_batch.Ga = Ga.reshape(walker_batch.nwalkers, nk, nbsf, nk, nbsf)

    if ndown > 0 and not walker_batch.rhf:
        ovlp_b = xp.einsum("wlpki, lpj->wkilj", phib, trial.psi0b.conj(), optimize=True)
        ovlp_b = ovlp_b.reshape(walker_batch.nwalkers, nk * ndown, nk * ndown)
        ovlp_inv_b = xp.linalg.inv(ovlp_b)
        sign_b, log_ovlp_b = xp.linalg.slogdet(ovlp_b)

        walker_batch.Ghalfb = xp.matmul(ovlp_inv_b, walker_batch.phib.transpose(0, 2, 1))
        if not trial.half_rotated or build_full:
            Gb = xp.einsum(
                "kpi,wkilq->wkplq", trial.psi0b.conj(), walker_batch.Ghalfb, optimize=True
            )
            walker_batch.Gb = Gb.reshape(walker_batch.nwalkers, nk, nbsf, nk, nbsf)
        ot = sign_a * sign_b * xp.exp(log_ovlp_a + log_ovlp_b - walker_batch.log_shift)
    elif ndown > 0 and walker_batch.rhf:
        ot = sign_a * sign_a * xp.exp(log_ovlp_a + log_ovlp_a - walker_batch.log_shift)
    elif ndown == 0:
        ot = sign_a * xp.exp(log_ovlp_a - walker_batch.log_shift)

    synchronize()

    return ot
