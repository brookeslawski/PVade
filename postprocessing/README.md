# Post-Processing PVade Simulations

`PVade` outputs `.xdmf` and `.h5` solution files that can be post-processed using Python in preparation for analysis. It is recommended that the `.xdmf` files be viewed in ParaView to check that the simulation ran as expected, but for quantitative analysis, the following steps can be taken:

This example post-processes 4 simulations with arrays in Slurm.

NOTE: You will need to modify file paths and output directory names for your specific cases.

From the parent solution folder on Kestrel, run
    ```bash
    sbatch --array=0-2 pproc_array.sh
    ```
This command runs `PVade/postprocessing/pproc_fluid_h5_to_pkl.py` on each simulation folder specified by `casepaths`. This script spatially interpolates the flow to a rectilinear mesh and saves the output as a `.pkl` file containing the time-series data for the fluid.

To compare the simulation output to the experimental data from the DuraMAT field campaign for a single case, open up `PVade/postprocessing/loads_validation.ipynb` on Kestrel's JupyterLab and run.