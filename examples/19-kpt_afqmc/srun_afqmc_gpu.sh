#!/bin/bash

#SBATCH -p gpu_requeue
#SBATCH -N 1                 # The number of nodes to request
#SBATCH --gres=gpu:1
#SBATCH --mem=200G           # The memory the job will use per cpu core
#SBATCH --time=0-1:00:00     # The time the job will take to run in D-HH:MM

#SBATCH --output=/n/holylabs/LABS/joonholee_lab/Users/shufay/k-ipie/examples/19-kpt_afqmc/%x.%j.out # STDOUT
#SBATCH --error=/n/holylabs/LABS/joonholee_lab/Users/shufay/k-ipie/examples/19-kpt_afqmc/%x.%j.err # STDERR

# Email
#SBATCH --mail-user=su2254@columbia.edu
##SBATCH --mail-type=BEGIN
##SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

ulimit -s unlimited

# Prelims.
export HOME="/n/home10/shufay"
export WORKDIR="/n/holylabs/LABS/joonholee_lab/Users/shufay/k-ipie/examples/19-kpt_afqmc"
export SAVEDIR="/n/holylabs/LABS/joonholee_lab/Users/shufay/k-ipie/examples/19-kpt_afqmc"
#export SCRATCH="/n/netscratch/joonholee_lab/Lab/shufay"
#export TMPDIR=${SCRATCH}
export PYTHONPATH=
export PYTHONPATH=/n/holylabs/LABS/joonholee_lab/Users/shufay/k-ipie/

module purge
module load Anaconda2/2019.10-fasrc01
module load gcc/13.2.0-fasrc01
module load openmpi/5.0.2-fasrc01
module load cuda/12.4.1-fasrc01
export MPICC=$(which mpicc)

# Activate Anaconda work environment
source $HOME/.bashrc
conda deactivate
conda deactivate
conda activate ipie_gpu

# Print the essential SLURM job parameters
echo "SLURM job details:"
scontrol show job $SLURM_JOB_ID

# Echo
echo "HOME=${HOME}"
echo "WORKDIR=${WORKDIR}"
echo "SAVEDIR=${SAVEDIR}"
echo "SCRATCH=${SCRATCH}"
echo "TMPDIR=${TMPDIR}"
echo "PYTHONPATH=${PYTHONPATH}"
echo
echo "========================================================================"
echo

nvidia-smi

# Inputs
nu=${1}
nkx=${2}
nky=${3}
ecut=${4}
shift_bz=${5}

#mpirun -n 4 python -u run_afqmc_gpu.py
srun --mpi=pmix -n 1 --gres=gpu:1 --cpu-bind=cores --kill-on-bad-exit=1 python -u run_afqmc_gpu.py

# End of script

# Mark the time job
date
