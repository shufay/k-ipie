"""
Generates the integrals in a .h5 file using pyqcpbc.
Run with:
    `python -u pyqcpbc_generation.py -i $input_file_name`
"""

import pyqcpbc
from pyqcpbc import SCF, INTEGRAL, KSCF
import numpy
import argparse
from eval_local_energy import get_Ghalf, half_rotate_chol_symm, kpt_symmchol_ecoul_kernel_rhf, kpt_symmchol_exx_kernel
import h5py

def BZ_to_1BZ(kpts):
    """
    Map k-points to the first Brillouin zone.
    kpts: (nkpts, 3) array of k-points in fractional coordinates
    """
    kpts = numpy.where(numpy.abs(kpts - 0.5) < 1e-8, 0.5 - 1e-12, kpts)
    kpts = numpy.where(numpy.abs(kpts + 0.5) < 1e-8, -0.5 - 1e-12, kpts)
    kpts = numpy.floor(0.5 - kpts) + kpts
    return kpts

def find_translated_index_batched(kpts, q_vec, tol=1e-6):
    """
    Find the index of the k-point that is translated by trs_vector for the whole k point lists
    kpts: (nkpts, 3) array of k-points in fractional coordinates
    trs_vector: (3,) array of the translation vector in fractional coordinates
    """
    # assert numpy.max(numpy.abs(kpts)) < 0.5
    # here we do not do sanity check, make sure the kpts are in the first BZ
    idxlis = []
    kpts_translated = kpts + q_vec
    fbz_kpts_trs = BZ_to_1BZ(kpts_translated)

    for i in range(fbz_kpts_trs.shape[0]):
        kpt = fbz_kpts_trs[i]
        for j in range(len(kpts)):
            if numpy.allclose(kpt, kpts[j], atol=tol):
                idxlis.append(j)
    idxlis = numpy.array(idxlis, dtype=numpy.int64)
    return idxlis

def construct_kpq(kpts, tol=1e-6):
    """
    Construct the kpq matrix
    kpts: (nk, 3) array of k-points in fractional coordinates
    """
    nk = len(kpts)
    kpq_mat = numpy.zeros((nk, nk), dtype=numpy.int64)
    for iq in range(nk):
        qvec = kpts[iq]
        idx_kpq = find_translated_index_batched(kpts, qvec)
        kpq_mat[iq] = idx_kpq
    return kpq_mat


def find_gamma_index(kpts, tol=1e-6):
    """
    Find the index of the gamma point
    kpts: (nk, 3) array of k-points in fractional coordinates
    """
    for i in range(len(kpts)):
        if numpy.allclose(kpts[i], [0.0, 0.0, 0.0], atol=tol):
            return i
    return None

def find_inverted_index_batched(kpts, tol=1e-6):
    """
    Find the index of the k-point that is transformed to -k for
    kpts: (nkpts, 3) array of k-points in fractional coordinates
    trs_vector: (3,) array of the translation vector in fractional coordinates
    """
    # assert numpy.max(numpy.abs(kpts)) < 0.5
    # here we do not do sanity check, make sure the kpts are in the first BZ
    idxlis = []
    mkpts = -kpts
    fbz_mkpts = BZ_to_1BZ(mkpts)
    for i in range(fbz_mkpts.shape[0]):
        fbz_mkpt = fbz_mkpts[i]
        for j in range(len(kpts)):
            if numpy.allclose(fbz_mkpt, kpts[j], atol=tol):
                idxlis.append(j)
    idxlis = numpy.array(idxlis, dtype=numpy.int64)
    return idxlis

def find_self_inverse_set(kpts):
    """
    Find the set of k-points that are self-inverse
    kpts: (nk, 3) array of k-points in fractional coordinates
    """
    mq_vec = find_inverted_index_batched(kpts)
    nk = kpts.shape[0]
    self_inv_set = []
    for ik in range(nk):
        if mq_vec[ik] == ik:
            self_inv_set.append(ik)
    return numpy.array(self_inv_set)

def find_idx_k_mod_neg(mq):
    """
    Find the union of S and Q+ set
    """
    nk = mq.shape[0]
    smaller_indices = numpy.where(numpy.arange(nk) < mq, numpy.arange(nk), mq)
    unique_indices = numpy.unique(smaller_indices)
    return unique_indices

def find_Qplus(kpts):
    """
    Find the set of k-points that are not self-inverse mod inversion
    """
    mq_vec = find_inverted_index_batched(kpts)
    nk = kpts.shape[0]
    unique_indices = find_idx_k_mod_neg(mq_vec)
    Sset = find_self_inverse_set(kpts)
    Qplus = numpy.setdiff1d(unique_indices, Sset)
    return Qplus

parser = argparse.ArgumentParser(
        description="A script to process an input file."
    )
    
# Add the -i/--input argument
parser.add_argument(
    '-i', '--input',
    type=str,
    required=True,
    help='Path to the input file'
)

# Parse the arguments
args = parser.parse_args()

filepath = "/scratch/bcra/jzhang33/qcpbc_calc/C_cohesive_pyqcpbc/"
with open(filepath + "input/" + args.input, 'r') as f:
    gpt_inp = f.read()

qcpbc = pyqcpbc.prog(gpt_inp, verbose = True)
mf = KSCF.kscf(qcpbc)
mf.run()
energy = mf.energy
print("total energy is {}".format(energy))

# verify energy using AO integrals
T = INTEGRAL.eval.kinetic(qcpbc)
Vpp = INTEGRAL.eval.vpp(qcpbc)

nbsf = T.shape[1]
nk = T.shape[-1]

chol_ao = INTEGRAL.eval.ao_chol(qcpbc, tol=1e-5, verbose=True, sym=True)

# pad chol_ao with zeros
max_cols = max(A.shape[1] for A in chol_ao)

padded_arrays = [numpy.pad(A, ((0, 0), (0, max_cols - A.shape[1])), mode='constant', constant_values=0.) for A in chol_ao]

chol_ao = numpy.array(padded_arrays)
nchol = chol_ao.shape[-1]

chol_ao = chol_ao.reshape(-1, nk, nbsf, nbsf, nchol) # q, k, m, n, X

kpts = mf.get_kpoints()
bmat = qcpbc.get_b()
kpts_frac = numpy.linalg.inv(bmat) @ kpts
kpts_frac = kpts_frac.T
Sset = find_self_inverse_set(kpts_frac)
Qplus = find_Qplus(kpts_frac)
unique_kpts = numpy.concatenate((Sset, Qplus))

dm = numpy.array(mf.get_dm())[0]  # only one spin component
nocc = qcpbc.nocca
hcore = T + Vpp

E1 = 2.0 * numpy.einsum("mnk, nmk ->", hcore, dm, optimize=True) / nk
Enuc = INTEGRAL.eval.get_enuc(qcpbc)
Emadelung = INTEGRAL.eval.madelung(qcpbc)

igamma = find_gamma_index(kpts_frac[unique_kpts])
chol_ao_gamma = chol_ao[igamma]

EJ = 2.0 * numpy.einsum("kmnX, qslX, nmk, slq->", chol_ao_gamma, chol_ao_gamma.conj(), dm, dm, optimize=True) / nk**2

print("EJ is {}".format(EJ))
kpq_mat = construct_kpq(kpts_frac)

EK = 0.0
for iq in range(len(Sset)):
    for ik in range(nk):
        iq_real = Sset[iq]
        iqpk = kpq_mat[iq_real, ik]
        aochol_kq = chol_ao[iq, ik].copy()
        EK += -numpy.einsum("mnX,slX,nl,sm->", aochol_kq, aochol_kq.conj(), dm[:, :, iqpk], dm[:, :, ik], optimize=True) / nk**2

for iq in range(len(Sset), len(Sset) + len(Qplus)):
    for ik in range(nk):
        iq_real = Qplus[iq - len(Sset)]
        iqpk = kpq_mat[iq_real, ik]
        aochol_kq = chol_ao[iq, ik].copy()
        EK += -2.0 * numpy.einsum("mnX,slX,nl,sm->", aochol_kq, aochol_kq.conj(), dm[:, :, iqpk], dm[:, :, ik], optimize=True) / nk**2

EHF = Enuc + E1 + EJ + EK + Emadelung
assert numpy.isclose(EHF, mf.energy, atol=1e-3)

# verify energy using MO integrals
mo = mf.get_mo()[0]
max_cols = max(A.shape[1] for A in mo)

padded_arrays = [numpy.pad(A, ((0, 0), (0, max_cols - A.shape[1])), mode='constant', constant_values=0.) for A in mo]
mo = numpy.array(padded_arrays)
nmo = mo.shape[-1]
print(mo.shape)

hcore_mo = numpy.zeros((nmo, nmo, nk), dtype=numpy.complex128)
for i in range(nk):
    hcore_mo[:, :, i] = mo[i].T.conj() @ hcore[:, :, i] @ mo[i]

# cholmo = INTEGRAL.eval.mo_chol(qcpbc, mo, tol=1e-10, verbose=True)
chol_mo = numpy.zeros((len(unique_kpts), nk, nmo, nmo, nchol), dtype=numpy.complex128)
for iq in range(unique_kpts.shape[0]):
    for ik in range(nk):
        iq_real = unique_kpts[iq]
        iqpk = kpq_mat[iq_real, ik]
        # chol_mo[iq, ik] = mo[ik].T.conj() @ chol_ao[iq, ik] @ mo[iqpk]
        chol_mo[iq, ik] = numpy.einsum("mp, mnX, nr-> prX", mo[ik].conj(), chol_ao[iq, ik], mo[iqpk], optimize=True)
# print(chol_mo.shape)

E1 = 2.0 * numpy.einsum("iik -> ", hcore_mo[:nocc, :nocc], optimize=True) / nk
cholmo_gamma = chol_mo[igamma]

EJ = 2.0 * numpy.einsum("kiiX, qjjX ->", cholmo_gamma[:, :nocc, :nocc], cholmo_gamma[:, :nocc, :nocc].conj(), optimize=True) / nk**2
EK = 0.0
for iq in range(len(Sset)):
    for ik in range(nk):
        iq_real = Sset[iq]
        iqpk = kpq_mat[iq_real, ik]
        mochol_kq = chol_mo[iq, ik].copy()
        EK += -numpy.einsum("ijX,ijX->", mochol_kq[:nocc, :nocc], mochol_kq[:nocc, :nocc].conj(), optimize=True) / nk**2

for iq in range(len(Sset), len(Sset) + len(Qplus)):
    for ik in range(nk):
        iq_real = Qplus[iq - len(Sset)]
        iqpk = kpq_mat[iq_real, ik]
        mochol_kq = chol_mo[iq, ik].copy()
        EK += -2.0 * numpy.einsum("ijX,ijX->", mochol_kq[:nocc, :nocc], mochol_kq[:nocc, :nocc].conj(), optimize=True) / nk**2

EHF = Enuc + E1 + EJ + EK + Emadelung
assert numpy.isclose(EHF, mf.energy, atol=1e-3)

chol_mo = chol_mo.transpose(4, 1, 2, 0, 3).copy() # (nq, nk, nmo, nmo, nchol) -> (nchol, nk, nmo, nq, nmo).

savefilename = args.input.split(".")[0] + "_nolindep.h5"
with h5py.File(filepath + savefilename, 'w') as f:
    f['hcore'] = hcore_mo.transpose(2, 0, 1) # nk, nmo, nmo
    f['chol'] = chol_mo / numpy.sqrt(nk)
    f['kpoints'] = kpts_frac
    f['e0'] = Enuc + Emadelung
