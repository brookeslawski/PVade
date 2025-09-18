#!/bin/bash
#SBATCH --account=siliconranch
#SBATCH --time=1:00:00
##SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --qos=high
#SBATCH --partition=debug
#SBATCH --output=pproc_%j.log

module load conda
conda activate myjupyter

export OMP_NUM_THREADS=1

python /projects/pvopt/aps_dfd_2024/PVade/postprocessing/pproc_fluid_h5_to_pkl.py
