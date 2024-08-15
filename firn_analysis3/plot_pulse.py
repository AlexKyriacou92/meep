import numpy as np
import math
import h5py
import sys
from sys import argv, exit
import configparser
from mpi4py import MPI
import meep as mp
import os
from matplotlib import pyplot as pl
from scipy.signal import butter, lfilter
from utils_CFM import get_spec_from_file, get_true_rx, cut_arr, get_pulse_from_file, do_IR
from utils_CFM import butterBandpassFilter, findNearest, butterBandpass
from scipy.signal import decimate
nArgs = len(argv)
if nArgs == 2:
    fname_config = argv[1]
else:
    print('Input error, enter the config file for the analysis')
    print('For example: $ python', argv[0], 'config_analysis.txt')
    exit()

'''
def get_profile_from_file(fname_hdf):
    with h5py.File(fname_hdf, 'r') as hdf_in:
'''


config = configparser.ConfigParser()
config.read(fname_config)

input_config = config['DATAFILES']
label_config = config['LABELS']
rx_config = config['RX']
fname_list = []
for key in input_config.keys():
    fname_k = input_config[key]
    fname_list.append(fname_k)
label_list = []
for key in label_config.keys():
    label_k = label_config[key]
    label_list.append(label_k)

with h5py.File(fname_list[0],'r') as hdf_in:
    for key in hdf_in.keys():
        print(key)

x_rx = float(rx_config['X_RX'])
z_rx = float(rx_config['Z_RX'])
nSims = len(fname_list)

rx_pulse_list = []

IR_data = np.genfromtxt('AIR_LDPA_RNOG.txt',skip_header=4)
IR_freq = IR_data[:,0]/1e9
IR_spec = IR_data[:,1]

with h5py.File(fname_list[0],'r') as input_hdf0:
    sourceDepth = float(input_hdf0.attrs['sourceDepth'])
x_rx_actual, z_rx_actual = get_true_rx(fname_list[0], x_rx, z_rx)

rx_pulse0, tspace = get_pulse_from_file(fname_list[0], x_rx, z_rx,pol_mode='zpol')

fontsize=20
labelsize=16

fig = pl.figure(figsize=(10,6),dpi=150)
ax = fig.add_subplot(111)
title_str = 'Simulated Trace\n$z_{tx} =$ ' + str(sourceDepth) + ' m, '
title_str += '$x_{rx} = $ ' + str(x_rx) + ' m, $z_{rx} = $ ' + str(z_rx) + ' m'
ax.set_title(title_str, fontsize=fontsize)

t_delta_minus = 800
t_delta_plus = 1400

gain_dB = 60.
gain_power = 10**(gain_dB/10.)
gain_v = np.sqrt(gain_power)
import util_nuRadioMC
travel_times, amp_list, path_lengths, solution_types = util_nuRadioMC.get_ray_points(0, sourceDepth, x_rx, z_rx, 'analytic')

rx_spec_list = []
for i in range(nSims):
    fname_i = fname_list[i]
    label_i = label_list[i]
    with h5py.File(fname_i, 'r') as hdf_in:
        sourceDepth = float(hdf_in.attrs['sourceDepth'])

    rx_pulse, tspace = get_pulse_from_file(fname_hdf=fname_i, x=x_rx, z=z_rx, pol_mode='zpol')
    dt = tspace[1] - tspace[0]
    print('dt=',dt, 'sample frequency=', 1/dt)

    if i == 0:
        ax.plot(tspace, rx_pulse*1000, label=label_i)
        ii_max = np.argmax(abs(rx_pulse))
        t_max = tspace[ii_max]
    else:
        ax.plot(tspace, rx_pulse*1000, label=label_i,alpha=0.65)
    rx_spectrum = np.fft.rfft(rx_pulse0)
    rx_spec_list.append(rx_spectrum)
    rx_pulse_list.append(rx_pulse0)
    dt2 = tspace[1]-tspace[0]
    freq_space = np.fft.rfftfreq(len(rx_pulse),dt2)
ax.grid()

nRays = len(travel_times)
colors = ['orange', 'r', 'r']
line_types = ['--', '-.', ':']
t_low = t_max-t_delta_minus
t_high = t_max+t_delta_plus
print(t_low, t_high)
if nRays > 0:
    for i in range(nRays):
        print(solution_types[i],travel_times[i], amp_list[i])
        amp_ratio = amp_list[i]/amp_list[-1]
        print(amp_ratio)
        ax.axvline(travel_times[i] + 100, linestyle=line_types[i],color = colors[i],label=solution_types[i])
ax.set_xlim(t_low, t_high)

if nRays > 1:
    if t_high < travel_times[1]:
        ax.set_xlim(t_low, travel_times[1]+500)
ax.tick_params(axis='both', labelsize=labelsize)
ax.set_xlabel('Time [ns]',fontsize=fontsize)
ax.set_ylabel('E field amplitude [mV]',fontsize=fontsize)
ax.legend(fontsize=labelsize, loc='upper right')
fig.savefig('rx_pulse.png')
pl.show()

fig = pl.figure(figsize=(10,6),dpi=150)
ax = fig.add_subplot(111)
title_str = 'Simulated Trace Abs \n$z_{tx} =$ ' + str(sourceDepth) + ' m, '
title_str += '$x_{rx} = $ ' + str(x_rx) + ' m, $z_{rx} = $ ' + str(z_rx) + ' m'
ax.set_title(title_str, fontsize=fontsize)
for i in range(nSims):
    ax.plot(tspace, abs(rx_pulse_list[i])*1e3,label=label_list[i])
ax.grid()
ax.tick_params(axis='both', labelsize=labelsize)
ax.set_xlabel('Time [ns]',fontsize=fontsize)
ax.set_ylabel('E field amplitude [mV]',fontsize=fontsize)
ax.set_xlim(t_low, t_high)

ax.legend(fontsize=labelsize, loc='upper right')
pl.show()

fig = pl.figure(figsize=(10,6),dpi=150)
ax = fig.add_subplot(111)
title_str = 'Simulated Trace Spectrum \n$z_{tx} =$ ' + str(sourceDepth) + ' m, '
title_str += '$x_{rx} = $ ' + str(x_rx) + ' m, $z_{rx} = $ ' + str(z_rx) + ' m'
ax.set_title(title_str, fontsize=fontsize)
for i in range(nSims):
    ax.plot(freq_space, abs(rx_spec_list[i]),label=label_list[i])
ax.grid()
ax.tick_params(axis='both', labelsize=labelsize)
ax.set_xlabel('Frequency [GHz]',fontsize=fontsize)
ax.set_ylabel('E field amplitude [mV/Hz]',fontsize=fontsize)
ax.legend(fontsize=labelsize, loc='upper right')
fig.savefig('rx_spectrum.png')
pl.show()