import meep as mp
import numpy as np
import math
import h5py
import matplotlib.pyplot as pl
import sys
import cmasher as cmr

#fname_in = 'mpi_tests/sim_testing_mpi.h5'
fname_in = sys.argv[1]

hdf_in = h5py.File(fname_in, 'r')
tspace = np.array(hdf_in.get('tspace'))
Ez = np.array(hdf_in.get('Ez'))
Er = np.array(hdf_in.get('Er'))

epsilon_r = np.array(hdf_in.get('epsilon_r'))

rxPulses = np.array(hdf_in.get('rxPulses'))
rxList = np.array(hdf_in.get('rxList'))
print(rxPulses.shape)
print(rxList.shape)
print(rxList)

rx_x_ii = 100
#rx_z_ii = -20

nRx = len(rxList)
dr_arr = np.zeros(nRx)

rx_pulses = []
rx_depths = []
for i in range(nRx):
    rx_i = rxList[i]
    rx_x = rx_i[0]
    rx_z = rx_i[1]
    if rx_x == 100 and rx_z >= 50 and rx_z <= 100:
        rx_pulses.append(rxPulses[i])
        rx_depths.append(rx_z)
    #dr_arr[i] = np.sqrt((rx_x_ii-rx_x)**2 + (rx_z_ii - rx_z)**2)

#ii_select = np.argmin(dr_arr)
#print('ii_select', ii_select)
#print(rx_x_ii, rx_z_ii, rxList[ii_select])

fig = pl.figure(figsize=(10, 5), dpi=120)
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
for i in range(len(rx_pulses)):
    rx_pulse_ii = rx_pulses[i]
    #rx_pulse_ii = rxPulses[ii_select]
    nSamples = len(rx_pulse_ii)
    dt = tspace[1] - tspace[0]
    rx_spectrum_ii = np.fft.rfft(rx_pulse_ii)
    #rx_spectrum_ii = np.fft.fftshift(rx_spectrum_ii)
    freq_space = np.fft.rfftfreq(nSamples, dt)
    #freq_space = np.fft.fftshift(freq_space)

    ax1.plot(tspace, rx_pulse_ii.real, label=str(rx_depths[i]))
    ax2.plot(freq_space, abs(rx_spectrum_ii)**2, label=str(rx_depths[i]))

ax1.grid()
ax1.legend()
ax1.set_ylabel('Ez [V]')
ax1.set_xlabel('Time [ns]')
ax2.grid()
ax2.legend()
pl.show()

iceDepth = float(hdf_in.attrs['iceDepth'])
airHeight = float(hdf_in.attrs['airHeight'])
iceRange = float(hdf_in.attrs['iceRange'])
boreholeRadius = float(hdf_in.attrs['boreholeRadius'])
sourceDepth = float(hdf_in.attrs['sourceDepth'])
Z_tot = -iceDepth + airHeight
Z_cent = Z_tot/2

for i in range(nRx):
    print('check if array not empty', rxList[i], i, np.any(rxPulses[i] != 0))

fig, ax = pl.subplots()
ax.set_aspect('auto')
print(iceDepth, airHeight)
pmesh = pl.imshow(10*np.log10(abs(Ez)), vmin=-100, vmax=-50, aspect='auto', cmap='hot',extent=[0, iceRange, -iceDepth-Z_cent, -Z_cent+airHeight])
pl.axis('on')
#ax.set_ylim(-iceDepth, 10)
ax.scatter(boreholeRadius, -sourceDepth+1, c='k')
cbar = fig.colorbar(pmesh)
cbar.set_label('Amplitude [dBu]')
#plot_title = "dipole_"+name+"_d_"+str(depth)
#pl.title(plot_title)
ax.set_xlabel("radial direction (m)")
ax.set_ylabel("depth (m)")
pl.show()

fig, ax = pl.subplots()
ax.set_aspect('auto')
print(iceDepth, airHeight)
pmesh = pl.imshow(10*np.log10(abs(Er)), vmin=-100, vmax=-50, aspect='auto', cmap='hot',extent=[0, iceRange, -iceDepth-Z_cent, -Z_cent+airHeight])
pl.axis('on')
#ax.set_ylim(-iceDepth, 10)
ax.scatter(boreholeRadius, -sourceDepth+1, c='k')
cbar = fig.colorbar(pmesh)
cbar.set_label('Amplitude [dBu]')
#plot_title = "dipole_"+name+"_d_"+str(depth)
#pl.title(plot_title)
ax.set_xlabel("radial direction (m)")
ax.set_ylabel("depth (m)")
pl.show()

fig, ax = pl.subplots()
ax.set_aspect('auto')
print(iceDepth, airHeight)
pmesh = pl.imshow(epsilon_r, aspect='auto', cmap='viridis',extent=[0, iceRange, -iceDepth-Z_cent, airHeight-Z_cent])
pl.axis('on')
#ax.set_ylim(-iceDepth-Z_cent, 10)
ax.scatter(boreholeRadius, -sourceDepth+1, c='k')
cbar = fig.colorbar(pmesh)
cbar.set_label('Permittivity')
#plot_title = "dipole_"+name+"_d_"+str(depth)
#pl.title(plot_title)
ax.set_xlabel("radial direction (m)")
ax.set_ylabel("depth (m)")
pl.show()