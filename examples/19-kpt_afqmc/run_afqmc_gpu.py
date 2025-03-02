import sys
import h5py
import numpy as np

# Need to flag that we want to use GPU before **any** ipie modules are imported
from ipie.config import config
config.update_option("use_gpu", True)

from ipie.hamiltonians.kpt_chunked import KptComplexCholChunked
from ipie.qmc.afqmc import AFQMC
from ipie.systems.generic import Generic
from ipie.trial_wavefunction.single_det_kpt import KptSingleDet
from ipie.walkers.uhf_walkers import UHFWalkers
from ipie.utils.mpi import MPIHandler

import os
from ipie.utils.backend import arraylib as xp

try:
    import cupy
    from mpi4py import MPI
except ImportError:
    sys.exit(0)

gpu_number_per_node = 1
nmembers = 1
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

xp.cuda.Device(rank % gpu_number_per_node).use()

nsteps = 25
nblocks = 400
timestep = 0.0025
rng_seed = 0

filepath = '/n/netscratch/joonholee_lab/Lab/jhzhang/diamond_coh_gthhfrev_gthcc/3.56/dz/555/'
filename = 'C_555_dz.h5'
#filepath = './'
#filename = 'SiC_ao.h5'
with h5py.File(filepath+filename, 'r') as fa:
    _e0 = fa["e0"]
    e0 = np.empty(_e0.shape, dtype=np.float64)
    _e0.read_direct(e0)

    hcore = fa["hcore"][()]
    kpoints = fa["kpoints"][()]
    #chol_chunk = fa["chol"][()]


if rank == 0:
    print(f"finished reading hcore")
srank = rank % nmembers

from ipie.utils.mpi import MPIHandler, make_splits_displacements
handler = MPIHandler(nmembers=nmembers)
print(f"rank: {rank}, srank: {srank}")

num_basis = hcore.shape[-1]
with h5py.File(filepath + f"chol_{srank}.h5") as fa:
    chol_chunk = fa["chol"][()]

chunked_chols = chol_chunk.shape[0]
num_chol = handler.scomm.allreduce(chunked_chols, op=MPI.SUM)

split_size = make_splits_displacements(num_chol, nmembers)[0]
assert chunked_chols == split_size[srank]

neleca, nelecb = 4, 4
system = Generic(nelec=(neleca, nelecb))
ham = KptComplexCholChunked(np.array([hcore, hcore]), kpoints, chol=None, chol_chunk=chol_chunk, ecore=e0, handler=handler)

num_basis = ham.nbasis
nk = ham.nk
neleca, nelecb = 4, 4
if rank == 0:
    print(f"# num_basis: {num_basis}, nk: {nk}, neleca: {neleca}, nelecb: {nelecb}")
# Build uhf trial to compute the force bias, in a less accurate but cheaper way, tricky.
psi_a = np.zeros((nk, num_basis, neleca), dtype=np.complex128)
psi_b = np.zeros((nk, num_basis, nelecb), dtype=np.complex128)
phi_a = np.zeros((nk, num_basis, nk, neleca), dtype=np.complex128)
phi_b = np.zeros((nk, num_basis, nk, nelecb), dtype=np.complex128)

for ik in range(nk):
    psi_a[ik] = np.eye(num_basis, neleca, dtype=np.complex128)
    psi_b[ik] = np.eye(num_basis, nelecb, dtype=np.complex128)

for ik1 in range(nk):
    phi_a[ik1, :, ik1, :] = np.eye(num_basis, neleca, dtype=np.complex128)
    phi_b[ik1, :, ik1, :] = np.eye(num_basis, nelecb, dtype=np.complex128)

phia = phi_a.reshape(nk*num_basis, nk*neleca)
phib = phi_b.reshape(nk*num_basis, nk*nelecb)

trial = KptSingleDet(np.concatenate([psi_a, psi_b], axis=2), nk, (neleca, nelecb), num_basis, handler=handler)
trial.build()
trial.half_rotate(ham)

num_walkers = 24
walkers = UHFWalkers(np.hstack([phia, phib]), nk * system.nup, nk * system.ndown, nk * ham.nbasis, num_walkers, mpi_handler=handler)
walkers.build(trial)

    
afqmc = AFQMC.build(
        (neleca, nelecb),
        ham,
        trial,
        walkers,
        num_walkers,
        rng_seed,
        nsteps,
        nblocks,
        timestep,
        stabilize_freq=1,
        pop_control_freq=1,
        mpi_handler=handler,
        verbose=True)
afqmc.run()
afqmc.finalise(verbose=True)
