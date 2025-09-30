#!/bin/bash

for tracker_angle in 40.0 -40.0 10.0 -10.0
do
    # Modify input_h5_filename based on the tracker_angle value
    if [[ "${tracker_angle}" == "40.0" ]]; then
        input_h5_filename=input/pct_constrained_turb_ny154_nz38_sonic1_30s_u9.101_50Hz.h5
    elif [[ "${tracker_angle}" == "-40.0" ]]; then
        input_h5_filename=input/pct_constrained_turb_ny154_nz38_sonic1_30s_u8.127_50Hz.h5
    elif [[ "${tracker_angle}" == "-10.0" ]]; then
        input_h5_filename=input/pct_constrained_turb_ny154_nz38_sonic1_30s_u9.0_50Hz.h5
    elif [[ "${tracker_angle}" == "10.0" ]]; then
        input_h5_filename=input/pct_constrained_turb_ny154_nz38_sonic1_30s_u7.142_50Hz.h5
    else
        echo "No valid input_h5_filename for tracker_angle=${tracker_angle}"
        continue
    fi
    
    output_dir=/scratch/bstanisl/pvade/turb_inflow/output/halfwing_stiffer_turbinflow_duramat_validation_${tracker_angle}deg/ \
    output_log=${output_dir}/log.txt
    sbatch --job-name=duramatval_${tracker_angle}deg \
           --output=${output_log} \
           --export=ALL,input_h5_filename=${input_h5_filename},output_dir=${output_dir},tracker_angle=${tracker_angle} \
           pvade_job.sbatch
done
