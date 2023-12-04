import meep as mp
import numpy as np
import math
import h5py
import matplotlib.pyplot as pl
import sys
import cmasher as cmr

fname_in = sys.argv[1]

input_hdf = h5py.File(fname_in, 'r')
Z_ice = float(input_hdf.attrs['Z_ice'])
H_air = float(input_hdf.attrs['H_air'])
R_ice = float(input_hdf.attrs['R_ice'])
r_bh = float(input_hdf.attrs['r_bh'])
z_tx = float(input_hdf.attrs['z_tx'])
#freq_cw = float(input_hdf.attrs['freq_cw'])
r_tx = float(input_hdf.attrs['r_tx'])
#pad = float(input_hdf.attrs['pad'])

Ez = np.array(input_hdf.get('Ez'))
Er = np.array(input_hdf.get('Er'))

E_sq = abs(Ez)**2 + abs(Er)**2
epsilon_r = np.array(input_hdf.get('epsilon_r'))
input_hdf.close()

Z_tot = -Z_ice + H_air
Z_cent = Z_tot/2

fig, ax = pl.subplots()
ax.set_aspect('auto')
print(Z_ice, H_air)
pl.imshow(abs(E_sq), vmin=1e-6, vmax=1e-5, aspect='auto', cmap='hot',extent=[0, R_ice, -Z_ice-Z_cent, -Z_cent+H_air])
pl.axis('on')
ax.set_ylim(-Z_ice-Z_cent, 10)

#plot_title = "dipole_"+name+"_d_"+str(depth)
#pl.title(plot_title)
ax.set_xlabel("radial direction (m)")
ax.set_ylabel("depth (m)")
pl.colorbar()
pl.show()

fig, ax = pl.subplots()
ax.set_aspect('auto')
pl.imshow(np.sqrt(epsilon_r), aspect='auto', cmap=cmr.arctic, vmin=1.0, vmax=1.8, extent=[0, R_ice, -Z_ice-Z_cent, -Z_cent+H_air])
pl.axis('on')
#plot_title = "dipole_"+name+"_d_"+str(depth)
#pl.title(plot_title)
ax.set_xlabel("radial direction (m)")
ax.set_ylabel("depth (m)")
cbar = pl.colorbar()
cbar.set_label('Refractive Index n')
ax.set_ylim(-Z_ice-Z_cent, 10)

pl.show()