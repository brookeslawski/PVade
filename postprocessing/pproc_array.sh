#!/bin/bash
#SBATCH --account=siliconranch
##SBATCH --time=1:00:00
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --qos=high
##SBATCH --partition=debug
#SBATCH --output=pproc_%A_%a.log

module load conda
conda activate myjupyter

export OMP_NUM_THREADS=1

# Define all the casepaths
casepaths=(
    "halfwing_turbinflow_duramat_validation_10.0deg/"
    "halfwing_turbinflow_duramat_validation_-10.0deg/"
    "halfwing_turbinflow_duramat_validation_40.0deg/"
)

# Select the casepath based on SLURM_ARRAY_TASK_ID
casepath="${casepaths[$SLURM_ARRAY_TASK_ID]}"

# Run your script for each casepath
python /projects/pvopt/aps_dfd_2024/PVade/postprocessing/pproc_fluid_h5_to_pkl.py --casepath "$casepath"
