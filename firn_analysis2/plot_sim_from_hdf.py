import meep as mp
import numpy as np
import math
import h5py
import matplotlib.pyplot as pl
import sys
import cmasher as cmr

from data import meep_field

fname_hdf = sys.argv[1]

meep_field_in = meep_field()
meep_field_in.setup_from_hdf(fname_hdf=fname_hdf)

Ez = meep_field_in.Ez
Er = meep_field_in.Er
Eabs = meep_field_in.get_Eabs()
ref_index = meep_field_in.get_refractive()

Z_ice = meep_field_in.iceDepth
H_air = meep_field_in.airHeight
R_ice = meep_field_in.iceRange
r_bh = meep_field_in.boreholeRadius
z_tx = meep_field_in.sourceDepth
r_tx = meep_field_in.sourceRange

Z_tot = -Z_ice + H_air
Z_cent = Z_tot/2

fig, ax = pl.subplots()
ax.set_aspect('auto')
print(Z_ice, H_air)
pmesh = pl.imshow(10*np.log10(abs(Ez)), vmin=-60, vmax=-10,aspect='auto', cmap='hot',extent=[0, R_ice, -Z_ice-Z_cent, -Z_cent+H_air])
pl.axis('on')
ax.set_ylim(-Z_ice-Z_cent, 10)
ax.scatter(r_bh, -z_tx+1, c='k')
cbar = fig.colorbar(pmesh)
cbar.set_label('Amplitude [dBu]')
#plot_title = "dipole_"+name+"_d_"+str(depth)
#pl.title(plot_title)
ax.set_xlabel("radial direction (m)")
ax.set_ylabel("depth (m)")
pl.show()

fig, ax = pl.subplots()
ax.set_aspect('auto')
print(Z_ice, H_air)
pmesh = pl.imshow(10*np.log10(abs(Er)),vmin=-60, vmax=-10, aspect='auto', cmap='hot',extent=[0, R_ice, -Z_ice-Z_cent, -Z_cent+H_air])
pl.axis('on')
ax.set_ylim(-Z_ice-Z_cent, 10)
ax.scatter(r_bh, -z_tx+1, c='k')
cbar = fig.colorbar(pmesh)
cbar.set_label('Amplitude [dBu]')
#plot_title = "dipole_"+name+"_d_"+str(depth)
#pl.title(plot_title)
ax.set_xlabel("radial direction (m)")
ax.set_ylabel("depth (m)")
pl.show()

fig, ax = pl.subplots()
ax.set_aspect('auto')
print(Z_ice, H_air)
pmesh = pl.imshow(10*np.log10(Eabs), vmin=-60, vmax=-10, aspect='auto', cmap='hot',extent=[0, R_ice, -Z_ice-Z_cent, -Z_cent+H_air])
pl.axis('on')
ax.set_ylim(-Z_ice-Z_cent, 10)
ax.scatter(r_bh, -z_tx+1, c='k')
cbar = fig.colorbar(pmesh)
cbar.set_label('Amplitude [dBu]')
#plot_title = "dipole_"+name+"_d_"+str(depth)
#pl.title(plot_title)
ax.set_xlabel("radial direction (m)")
ax.set_ylabel("depth (m)")
pl.savefig('example_Eabs.png')

pl.show()

fig, ax = pl.subplots()
ax.set_aspect('auto')
pmesh = pl.imshow(ref_index, aspect='auto', cmap=cmr.arctic, vmin=1.0, vmax=1.8, extent=[0, R_ice, -Z_ice-Z_cent, -Z_cent+H_air])
pl.axis('on')
cbar = fig.colorbar(pmesh)
cbar.set_label('Refractive Index')
#plot_title = "dipole_"+name+"_d_"+str(depth)
#pl.title(plot_title)
ax.set_xlabel("radial direction (m)")
ax.set_ylabel("depth (m)")

ax.set_ylim(-Z_ice-Z_cent, 10)
pl.savefig('example_epsilon.png')

pl.show()
