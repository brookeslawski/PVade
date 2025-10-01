import h5py
import yaml
import pickle
import numpy as np
from scipy.interpolate import griddata
import argparse
import os

# Add argument parsing for casepath
parser = argparse.ArgumentParser(description="Process fluid .h5 data to pickle format.")
parser.add_argument("--casepath", required=True, help="Path to the case directory.")
args = parser.parse_args()
casepath = args.casepath

parentdir = '/scratch/bstanisl/pvade/turb_inflow/'
# casepath = 'halfwing_turbinflow_duramat_validation_neg40deg/'

# output_dir=parentdir+'output/'+casepath
output_dir = os.path.join(parentdir, 'output', casepath)

# read input parameters
with open(output_dir+'input_params.yaml', 'r') as file:
    params = yaml.safe_load(file)

print('tracker_angle = {} m/s'.format(params['pv_array']['tracker_angle']), flush=True)
dt = params['solver']['dt']

save_pkl_name = f'{casepath[:-1]}_angle_{params['pv_array']['tracker_angle']}.pkl'

fname = output_dir + 'solution/solution_fluid.h5'

with h5py.File(fname, "r") as f:
    print("Reading from " + fname, flush=True)
    
    # Mesh coordinates  ===============
    coords = f["Mesh/fluid_mesh.xdmf/geometry"][:]
    print("Coords shape:", coords.shape, flush=True)  # Should be (538, 3) or (538, 2)
    
    # Velocity ===============
    velocity_group = f["Function/velocity "]
    
    # Sort dataset keys numerically
    keys = sorted(velocity_group.keys(), key=lambda k: float(k.replace("_", ".")))
    
    # Load all time steps into list
    data_list = [velocity_group[k][:] for k in keys]
    rawdata = np.stack(data_list)  # Shape: (n_timesteps, n_points, 3)

print("Combined velocity shape:", rawdata.shape, flush=True)  # e.g. (100, 5000, 3)

# compute approximate nx, ny, nz
# assuming same domain shape           
N = coords.shape[0]
Lx = params['domain']['x_max']-params['domain']['x_min']
Lz = params['domain']['z_max']-params['domain']['z_min']
Ly = params['domain']['y_max']-params['domain']['y_min']

volume = Lx * Ly * Lz
s = (N / volume) ** (1/3)

nx = round(s * Lx)
ny = round(s * Ly)
nz = round(s * Lz)

print(f"Estimated grid points: nx={nx}, ny={ny}, nz={nz}", flush=True)
print(f"Check: nx * ny * nz = {nx * ny * nz}", flush=True)

# Interpolate to regular grid
nt, npoints, ncomp = rawdata.shape

# Create regular grid to interpolate onto
xi = np.linspace(coords[:, 0].min(), coords[:, 0].max(), nx)
yi = np.linspace(coords[:, 1].min(), coords[:, 1].max(), ny)
# zi = np.linspace(coords[:, 2].min(), coords[:, 2].max(), nz)
mask = coords[:, 2] > 0.0
log_spaced = np.logspace(np.log10(coords[:, 2][mask].min()), np.log10(coords[:, 2].max()), nz-1)
zi = np.concatenate(([0.0], log_spaced))
X, Y, Z = np.meshgrid(xi, yi, zi, indexing='ij')

# Interpolate to regular grid
alldata = {}

alldata['u'] = np.full((nt, nx, ny, nz), np.nan, dtype=float) #np.empty((nt, nx, ny, nz))
alldata['v'] = np.full((nt, nx, ny, nz), np.nan, dtype=float) #np.empty((nt, nx, ny, nz))
alldata['w'] = np.full((nt, nx, ny, nz), np.nan, dtype=float) #np.empty((nt, nx, ny, nz))
# alldata['T'] = np.empty((nt, nx, ny))

# Interpolate
# for t in np.arange(100, 104): #range(nt):
for t in np.arange(100, nt): #range(nt):
    if t % 100 == 0:
        print('nt = {}'.format(t), flush=True)
    alldata['u'][t, :, :, :] = griddata(coords, rawdata[t, :, 0], (X, Y, Z), method='nearest')
    alldata['v'][t, :, :, :] = griddata(coords, rawdata[t, :, 1], (X, Y, Z), method='nearest')
    alldata['w'][t, :, :, :] = griddata(coords, rawdata[t, :, 2], (X, Y, Z), method='nearest')

del rawdata

# Save X,Y,Z
alldata['X'] = X
alldata['Y'] = Y
alldata['Z'] = Z

# Saving interpolated data to pickle file
with open(parentdir+'interp_data/timeseries_data/'+save_pkl_name, 'wb') as f:
    pickle.dump(alldata, f)
