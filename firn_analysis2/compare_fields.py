import meep as mp
import numpy as np
import math
import h5py
import matplotlib.pyplot as pl
import sys
import cmasher as cmr

from data import meep_field

fname_hdf1 = sys.argv[1]
fname_hdf2 = sys.argv[2]

meep_field1 = meep_field()
meep_field2 = meep_field()

meep_field1.setup_from_hdf(fname_hdf=fname_hdf1)
meep_field2.setup_from_hdf(fname_hdf=fname_hdf2)

Eabs_1 = meep_field1.get_Eabs()
Eabs_2 = meep_field2.get_Eabs()

E_ratio = Eabs_2/Eabs_1

Z_ice = meep_field1.iceDepth
H_air = meep_field1.airHeight
R_ice = meep_field1.iceRange
r_bh = meep_field1.boreholeRadius
z_tx = meep_field1.sourceDepth
r_tx = meep_field1.sourceRange

Z_tot = -Z_ice + H_air
Z_cent = Z_tot/2

fig, ax = pl.subplots()
ax.set_aspect('auto')
print(Z_ice, H_air)
pmesh = pl.imshow(10*np.log10(E_ratio), aspect='auto', cmap='hot',extent=[0, R_ice, -Z_ice-Z_cent, -Z_cent+H_air])
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