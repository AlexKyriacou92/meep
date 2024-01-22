import meep as mp
import numpy as np
import math
import h5py
import matplotlib.pyplot as pl
import sys
import cmasher as cmr

fname_in = sys.argv[1]

hdf_in = h5py.File(fname_in, 'r')
tspace = np.array(hdf_in.get('tspace'))
Ez = np.array(hdf_in.get('Ez'))
Er = np.array(hdf_in.get('Er'))
epsilon_r = np.array(hdf_in.get('epsilon_r'))

rxPulses = np.array(hdf_in.get('rxPulses'))
rxList = np.array(hdf_in.get('rxList'))

rx_x_ii = 45
rx_z_ii = -30

nRx = len(rxList)
dr_arr = np.zeros(nRx)
for i in range(nRx):
    rx_i = rxList[i]
    rx_x = rx_i[0]
    rx_z = rx_i[1]

    dr_arr[i] = np.sqrt((rx_x_ii-rx_x)**2 + (rx_z_ii - rx_z)**2)

ii_select = np.argmin(dr_arr)

print(ii_select)
print(rx_x_ii, rx_z_ii, rxList[ii_select])

rx_pulse_ii = rxPulses[ii_select]

fig = pl.figure(figsize=(8,5),dpi=120)
ax = fig.add_subplot(111)
ax.plot(tspace, rx_pulse_ii.real, label='Ez')
ax.grid()
ax.legend()
pl.show()

iceDepth = float(hdf_in.attrs['iceDepth'])
airHeight = float(hdf_in.attrs['airHeight'])
iceRange = float(hdf_in.attrs['iceRange'])
boreholeRadius = float(hdf_in.attrs['boreholeRadius'])
sourceDepth = float(hdf_in.attrs['sourceDepth'])
Z_tot = -iceDepth + airHeight
Z_cent = Z_tot/2

for i in range(nRx):
    print(i, np.any(rxPulses[i] != 0))

fig, ax = pl.subplots()
ax.set_aspect('auto')
print(iceDepth, airHeight)
pmesh = pl.imshow(10*np.log10(abs(Er)), vmin=-40, vmax=-10,aspect='auto', cmap='hot',extent=[0, iceRange, -iceDepth-Z_cent, -Z_cent+airHeight])
pl.axis('on')
ax.set_ylim(-iceDepth-Z_cent, 10)
ax.scatter(boreholeRadius, -sourceDepth+1, c='k')
cbar = fig.colorbar(pmesh)
cbar.set_label('Amplitude [dBu]')
#plot_title = "dipole_"+name+"_d_"+str(depth)
#pl.title(plot_title)
ax.set_xlabel("radial direction (m)")
ax.set_ylabel("depth (m)")
pl.show()

