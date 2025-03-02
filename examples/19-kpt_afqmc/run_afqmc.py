import time
import h5py
import numpy as np
from ipie.qmc.afqmc import AFQMC
from ipie.systems.generic import Generic
from ipie.trial_wavefunction.single_det_kpt import KptSingleDet
from ipie.walkers.uhf_walkers import UHFWalkers
from ipie.hamiltonians.kpt_hamiltonian import KptComplexChol, KptComplexCholSymm
from ipie.hamiltonians.utils import get_kpt_hamiltonian, get_kpt_integrals
from ipie.estimators.local_energy_kpt_sd  import local_energy_kpt_single_det_uhf
from ipie.utils.mpi import MPIHandler, get_shared_comm, get_shared_array, have_shared_mem

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

verbose = True if rank == 0 else False
scomm = get_shared_comm(comm, verbose=verbose)

handler = MPIHandler()

# read hamiltonian in h5 file
#ham = get_kpt_hamiltonian('/n/netscratch/joonholee_lab/Lab/jhzhang/afqmc_calc_integrals/kpt_integrals/C_222_dz.h5', scomm, verbose=True)
#ham = get_kpt_hamiltonian('./SiC_nolindep.h5', scomm, verbose=True)

start = time.time()
filename = './SiC_ao.h5'
hcore, chol, kpts, enuc = get_kpt_integrals(filename, comm=scomm, verbose=verbose)

if verbose:
    print(f"# Time to read integrals: {time.time() - start:.6f}")

nbsf = hcore.shape[-1]
nchol = chol.shape[0]
shmem = have_shared_mem(scomm)
ham = KptComplexCholSymm(hcore, chol, kpts, enuc, verbose=verbose)

num_basis = ham.nbasis
nk = ham.nk
neleca, nelecb = 4, 4

# Build uhf trial to compute the force bias, in a less accurate but cheaper way, tricky.
with h5py.File(filename, 'r') as h5:
    mo_coeff = np.array(h5['mo_coeff'][()])
    mo_occ = np.array(h5['mo_occ'][()])

psi_a = np.zeros((nk, num_basis, neleca), dtype=np.complex128)
psi_b = np.zeros((nk, num_basis, nelecb), dtype=np.complex128)
phi_a = np.zeros((nk, num_basis, nk, neleca), dtype=np.complex128)
phi_b = np.zeros((nk, num_basis, nk, nelecb), dtype=np.complex128)

for ik in range(nk):
    nocca, noccb = mo_occ[:, ik]
    psi_a[ik] = mo_coeff[0, ik, :, :nocca].copy()
    psi_b[ik] = mo_coeff[1, ik, :, :noccb].copy()

for ik in range(nk):
    phi_a[ik, :, ik, :] = psi_a[ik].copy()
    phi_b[ik, :, ik, :] = psi_b[ik].copy()

phia = phi_a.reshape(nk*num_basis, nk*neleca)
phib = phi_b.reshape(nk*num_basis, nk*nelecb)

system = Generic(nelec=(neleca, nelecb))

trial = KptSingleDet(np.concatenate([psi_a, psi_b], axis=2), nk, (neleca, nelecb), num_basis)
trial.build()
trial.half_rotate(ham, scomm)
print(f'\n# Trial energy = {trial.calculate_energy(system, ham)}')

num_walkers = 1
walkers = UHFWalkers(np.hstack([phia, phib]), nk * system.nup, nk * system.ndown, nk * ham.nbasis, num_walkers, mpi_handler=handler)

local_energy = local_energy_kpt_single_det_uhf(system, ham, walkers, trial)
print(local_energy)
exit()

seed = 0     
afqmc = AFQMC.build(
        (neleca, nelecb),
        ham,
        trial,
        walkers=walkers,
        num_walkers=num_walkers,
        seed=seed,
        num_steps_per_block=25,
        num_blocks=200,
        timestep=0.005,
        stabilize_freq=3,
        pop_control_freq=3,
        mpi_handler=handler,
        verbose=verbose)

afqmc.run()
afqmc.finalise(verbose=verbose)
