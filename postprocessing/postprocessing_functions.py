#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# functions needed to post-process pvade outputs and compare against experimental data from DuraMAT
import numpy as np
import pandas as pd
import yaml
import pickle
from datetime import timedelta

# spectra
from scipy.signal import welch
from numpy import hanning
import math

from scipy.spatial.transform import Rotation as R
from scipy.interpolate import interp1d


def read_pvade_csv(pvade_output_dir):
    """
    Reads in lift and drag values from pvade simulation outputs
    """

    tmp = pd.read_csv(pvade_output_dir+'solution/lift_and_drag.csv')
    tmp.rename(columns={'#Time': 'Time'}, inplace=True)
    tmp = tmp.set_index('Time')
    loadsdata = tmp
    
    # read in accelerations
    tmp2 = pd.read_csv(pvade_output_dir+'solution/accel_pos.csv')
    #handling runs with updated outputs
    if 'deformation' in tmp2.columns[0]:
        loadsdata['accel-x'] = tmp2['x-acceleration'].values
        loadsdata['accel-y'] = tmp2['y-acceleration'].values
        loadsdata['accel-z'] = tmp2['z-acceleration'].values
        # loadsdata['accel-magnitude'] = loadsdata['accel-x']**2 + loadsdata['accel-y']**2 + loadsdata['accel-z']**2
        loadsdata['def-x'] = tmp2['#x-deformation'].values
        loadsdata['def-y'] = tmp2['y-deformation'].values
        loadsdata['def-z'] = tmp2['z-deformation'].values
        # loadsdata['def-magnitude'] = loadsdata['def-x']**2 + loadsdata['def-y']**2 + loadsdata['def-z']**2
    else: # output is mislabeled, correct here
        loadsdata['def-x'] = tmp2['#x-pos'].values
        loadsdata['def-y'] = tmp2['y-pos'].values
        loadsdata['def-z'] = tmp2['z-pos'].values
        # loadsdata['accel-magnitude'] = loadsdata['accel-x']**2 + loadsdata['accel-y']**2 + loadsdata['accel-z']**2
    
    # clean up
    del tmp
    del tmp2

    return loadsdata

def read_params(pvade_output_dir):
    params = {}
    with open(pvade_output_dir+'input_params.yaml', 'r') as file:
        params_yaml = yaml.safe_load(file)

    # print('tracker_angle = {} m/s'.format(params_yaml['pv_array']['tracker_angle']))
    params['tracker_angle'] = params_yaml['pv_array']['tracker_angle']
    params['dt_sim'] = params_yaml['solver']['dt']
    params['dt_xdmf'] = params_yaml['solver']['save_xdmf_interval']
    params['panel_span'] = params_yaml['pv_array']['panel_span']
    params['panel_chord'] = params_yaml['pv_array']['panel_chord']
    params['panel_height'] = params_yaml['pv_array']['elevation']
    params['tf'] = params_yaml['solver']['t_final']
    params['z_max'] = params_yaml['domain']['z_max']

    return params

def read_pvade_output_pkl(pkl_path):
    readdata = {}
    coords = {}
    
    try:
        with open(pkl_path, 'rb') as f:
            # try:
            readrawdata = pickle.load(f)

            # print(rawdata[winddir][case].keys())
            readdata['u'] = readrawdata['u']
            readdata['v'] = readrawdata['v']
            readdata['w'] = readrawdata['w']
            print('vel data nt, nx, ny, nz = ', np.shape(readdata['u']))
    except Exception as e:
        print(f"Error loading pickle file: {e}")

    # i think coords are only different for each tilt angle
    if 'X' in readrawdata.keys():
        tmpX = readrawdata['X']
        print('coord nx, ny, nz = ', np.shape(tmpX))
        del tmpX
    
        coords['X'] = readrawdata['X']
        coords['Y'] = readrawdata['Y']
        coords['Z'] = readrawdata['Z']
                            
    del readrawdata

    return readdata, coords

def calc_magnitude(vector):
    """
    Calculates velocity magnitude of a velocity vector.

    vector must contain 'u', 'v', 'w' components
    """
    
    return (vector['u (m/s)']**2 + vector['v (m/s)']**2 + vector['w (m/s)']**2)**0.5

def calc_force_coefficient(fx, vel_mag, sfc_area):
    """
    Calculates lift or drag coefficient of force vector.

    fx = fx for drag
    fx = fz for lift
    """

    return fx/(0.5*1*vel_mag**2*sfc_area)

def read_exp_csv_data(exp_data_fn):
    # read csv from DuraMAT data
    raw_exp_data = pd.read_csv(exp_data_fn, index_col='Time')
    raw_exp_data.index = pd.to_datetime(raw_exp_data.index)
    t0_sonic1 = raw_exp_data.index[0]
    
    exp_data = raw_exp_data
    
    # construct time index
    tmp = (exp_data.index[1]-exp_data.index[0])
    dt_sonic1 = round(tmp.total_seconds(), 3)
    tf_sonic1 = len(exp_data) * dt_sonic1 # final time [s]
    t_sonic1 = np.arange(0.0, tf_sonic1, dt_sonic1)
    
    exp_data['u (m/s)'] = -1.0*exp_data['u (m/s)'] # to make it positive from the west
    exp_data['Time'] = t_sonic1
    exp_data = exp_data.set_index('Time')

    return t0_sonic1, tf_sonic1, exp_data

def cut_exp_signal(tilt, t0_sonic1, tf_sim, exp_data):
    """
    Trim experimental signal to the period simulated
    """
    if tilt == -40.0:
        tstart = 250.0
    elif tilt == 40.0:
        tstart = 0.0
    elif tilt == -10.0:
        tstart = 280.0
    elif tilt == 10.0:
        tstart = 100.0
    print(f'start time of {tilt} case exp data = ', t0_sonic1 + timedelta(seconds=tstart))
    
    exp_data = exp_data[exp_data.index >= tstart]
    print(exp_data.index[0])

    exp_data = exp_data[exp_data.index <= tf_sim+tstart]
    # cases_df.loc[tilt, 'Start Time'] = exp_data.index[0]
    exp_data.index = exp_data.index - tstart
    exp_data = exp_data[exp_data.index > 0.0]
    return exp_data

# def add_exp_calc_quantities(exp_data, sfc_area):
#     exp_data['Drag Coefficient'] = calc_force_coefficient(exp_data['Drag force (kN)']*1000, exp_data['3D wind speed (m/s)'], sfc_area)
#     exp_data['Lift Coefficient'] = calc_force_coefficient(exp_data['Lift force (kN)']*1000, exp_data['3D wind speed (m/s)'], sfc_area)    

def convert_sim_accel_coords_to_exp(tilt, sim_data):
    tracker_angle = -tilt
    
    # real data
    pvade_accel = np.full((np.shape(sim_data)[0], 3), np.nan)
    pvade_accel[:, 0] = sim_data['accel-x']
    pvade_accel[:, 1] = sim_data['accel-y']
    pvade_accel[:, 2] = sim_data['accel-z']
    
    # Make a rotation matrix about the y-axis
    Ry = R.from_euler("y", tracker_angle, degrees=True).as_matrix()
    
    # Apply the rotation matrix to our simulation data
    # Now it should match the experimental setup with the mapping:
    # our x' = their y
    # our y' = their z
    # our z' = their x
    rotated_accel = np.dot(pvade_accel, Ry.T)

    print('x, y, z = ', pvade_accel)
    print('x\', y\', z\' = ', rotated_accel)

    return rotated_accel

def interpolate_to_exp_freq(new_time_index, sim_data):
    interp_sim_data = pd.DataFrame(index=new_time_index)
    for col in sim_data.columns:
        interpolator = interp1d(sim_data.index, sim_data[col].values, kind='linear')  # Use 'linear' or 'cubic' as needed
        interp_sim_data[col] = interpolator(new_time_index)
    return interp_sim_data

def merge_dataframes(exp_data, sim_data, interp_sim_data):
    alldata = {}
    alldata['exp'] = exp_data.copy()
    alldata['sim'] = sim_data.copy()
    alldata['sim_interp'] = interp_sim_data.copy()
    
    # adjustments
    alldata['exp'] = alldata['exp'].rename(columns={'3D wind speed (m/s)':'vel_mag (m/s)'})

    # convert from N to kN for drag and lift forces
    for comp in ['fx','fy','fz']:
        alldata['sim'][comp+'_0'] = alldata['sim'][comp+'_0']/1000
        alldata['sim_interp'][comp+'_0'] = alldata['sim_interp'][comp+'_0']/1000
    
    for key in ['sim','sim_interp']:
        alldata[key] = alldata[key].rename(columns={'fx_0':'Drag force (kN)','fy_0':'Lateral force (kN)','fz_0':'Lift force (kN)',
                                                   'fx_nd_calc':'Drag Coefficient','fz_nd_calc':'Lift Coefficient', 
                                                    'accel-x-prime':'AccNE Y (g)', 'accel-z-prime':'AccNE X (g)'})
        # alldata['sim'] = alldata['sim'].rename(columns={'fx_0':'Drag force (kN)','fy_0':'Lateral force (kN)','fz_0':'Lift force (kN)'})
        # alldata['sim_interp'] = alldata['sim_interp'].rename(columns={'fx_0':'Drag force (kN)','fy_0':'Lateral force (kN)','fz_0':'Lift force (kN)'})
    return alldata

# Smooth high frequency region
def runningMeanFast(x, N):
    """
    Calculates the running mean of an array x over a window size N.

    Returns:
        np.ndarray: Smoothed signal
    """
    return np.convolve(x, np.ones(N)/N, mode='same') 

def compute_spectra(u, U_mean, height, fs, overlap, smoothing=False, normalize_freq=True):
    """
    Computes the normalized energy spectrum of a velocity time series `u`,
    smooths the high-frequency tail, and returns the normalized frequency
    and smoothed normalized power spectral density.

    Parameters:
        u (array-like): Time series of velocity fluctuations (1D)
        U_mean (float): Mean flow velocity (for normalization)
        height (float): Reference height (for normalization)
        fs (float): Sampling frequency [Hz]
        overlap (int): Number of samples to overlap in Welch’s method

    Returns:
        nf_U_corr (np.ndarray): Normalized frequency array
        nPxxf_mod_U_corr (list of np.ndarray): Concatenated original and smoothed
            normalized power spectral density
    """

    # Length of the input time series (used for FFT size and window)
    nblock = len(u)

    # Use a Hamming window for spectral estimation
    win = np.hamming(math.floor(nblock/10))

    # Convert u to a pandas Series for easier handling (e.g., dropna, stats)
    U_corr = pd.Series(u)
    # do I need to detrend this? - probably not bc of "detrend = constant" in welch function call

    # Standard deviation of the fluctuations (used to normalize PSD)
    u_std = U_corr.std()

    # Compute power spectral density using Welch's method
    f_U_corr, Pxxf_U_corr = welch(U_corr.dropna(), fs, window=win, noverlap=overlap, nfft=nblock, detrend='constant', return_onesided=True) # detrend constant removes mean (zero-mean segments)
    
    # Normalize frequency: non-dimensional frequency = f * z / U_mean
    if normalize_freq:
        nf_U_corr = f_U_corr*height/abs(U_mean)
    else:
        nf_U_corr = f_U_corr

    # Normalize power: dimensionless spectral density
    nPxxf_U_corr = (f_U_corr*Pxxf_U_corr)/u_std**2

    # Identify index where normalized frequency > 0.3 (start smoothing here)
    index_highfreq_U_corr = list(np.where([abs(nf_U_corr)>0.3]))

    # Extract the high-frequency tail to smooth
    nPxxf_smooth_U_corr = nPxxf_U_corr[index_highfreq_U_corr[0][0]:len(nPxxf_U_corr)]

    # Apply running mean to smooth the high-frequency region
    avg_window_size = int(nblock/6)
    nPxxf_smooth_U_corr = runningMeanFast(nPxxf_smooth_U_corr,avg_window_size) # 200)

    # Concatenate the low-frequency part (unsmoothed) with the smoothed tail
    # nPxxf_mod_U_corr = [nPxxf_U_corr[0:index_highfreq_U_corr[0][0]-1],nPxxf_smooth_U_corr]
    nPxxf_mod_U_corr = nPxxf_U_corr # without smoothing

    # return nf_U_corr, nPxxf_mod_U_corr
    return nf_U_corr, nPxxf_mod_U_corr

def compute_TI(u,v,w):
    """
    Compute turbulent intensity
    """
    umag = (u**2 + v**2 + w**2)**0.5
    
    Iu = np.std(u, axis=0) / np.mean(umag, axis=0)
    Iv = np.std(v, axis=0) / np.mean(umag, axis=0)
    Iw = np.std(w, axis=0) / np.mean(umag, axis=0)

    return Iu, Iv, Iw