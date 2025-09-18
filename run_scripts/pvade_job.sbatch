#!/bin/bash

#SBATCH --account=siliconranch
#SBATCH --job-name=halfwing_duramat_turbinflow
#SBATCH --nodes=4
##SBATCH --nodes=2
#SBATCH --time=48:00:00
##SBATCH --time=1:00:00
#SBATCH --qos=high
##SBATCH --partition=debug
#SBATCH --mail-user=brooke.stanislawski@nrel.gov
#SBATCH --mail-type=ALL

module load conda
conda deactivate
source activate /projects/pvopt/aps_dfd_2024/conda_env/
export OMP_NUM_THREADS=1

# num of cores (-n below) = num_nodes * 96
srun -n 384 python -u /projects/pvopt/aps_dfd_2024/PVade/pvade_main.py \
     --input_file=/projects/pvopt/aps_dfd_2024/PVade/input/turbinflow_duramat_case_study_final.yaml \
     --general.output_dir=${output_dir} \
     --domain.l_char=0.17 \
     --pv_array.tracker_angle=${tracker_angle} \
     --fluid.h5_filename=${input_h5_filename}
