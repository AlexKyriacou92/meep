import meep as mp
import numpy as np
import math
import h5py

fname_out_prefix = 'southpole_dipole_func_'

import matplotlib.pyplot as pl
resolution = 12.5	#how many pixels per meter
n_air = 1.0

#NOTE -> Z > 0 is above, z < 0 is below
def southpole(R):
    z = R[2]
    if z > 0:
        return mp.Medium(index=n_air)
    else:
        z_depth = -1*z
        A = 1.78
        B = 0.43
        C = 0.0132 #1/m
        return mp.Medium(index=A-B*math.exp(-C*z_depth))
#Frequency
freq_cw = 100.0 #MHz
wavelength = 1300/freq_cw #m.MHz or m / us
freq_meep = 1/wavelength
amp_tx = 1.0

#Borehole:
r_bh = 0.15 # 10 cm
r_cent = r_bh/2.

#Source Depth
z_tx = 20.0
r_tx = r_bh/2.

#Add Padding
pad = 2.0

#Ice Geometry
Z_ice = 50.0 # Ice Height
H_air = 10.0 # Air Height
R_ice = 50.0 # Ice Radius #Cylindrical Coordinates!
R_tot = R_ice  + pad
Z_tot_geometry = Z_ice + H_air
Z_tot = Z_tot_geometry + 2*pad

Z_cent = (H_air - Z_ice)/2 # Center of 'Block
R_cent = R_ice/2

n_ice = 1.78
v_ice = 1/n_ice
t_start = 2 * R_tot / v_ice

Ydim = mp.inf
dimensions = mp.CYLINDRICAL #Define Cylindrical Symmetry
cell = mp.Vector3(R_ice, Ydim, Z_tot)
pml_layers = [mp.PML(pad)]

geometry_dipole = [
    mp.Block(center=mp.Vector3(0, Ydim, H_air/2),
             size=mp.Vector3(r_bh, mp.inf, Z_tot),
             material=mp.Medium(index=n_air)),
    mp.Block(center=mp.Vector3(0, Ydim, -Z_cent/2),
             size=mp.Vector3(R_ice, mp.inf, Z_tot),
             material=southpole)
]


# create the source
sources_dipole = []
sources_dipole.append(mp.Source(mp.ContinuousSource(frequency=freq_meep),
            amplitude=amp_tx, component=mp.Ez,
            center=mp.Vector3(r_tx,0,-z_tx),
            size=mp.Vector3(0,0,0)))

sim_dipole = mp.Simulation(force_complex_fields=True,
                cell_size=cell,
                dimensions=mp.CYLINDRICAL,
                boundary_layers=pml_layers,
                geometry=geometry_dipole,
                sources=sources_dipole,
                resolution=resolution)
sim_dipole.run(until=t_start)

fname_out = fname_out_prefix + 'z_tx_' + str(z_tx) + 'm_freq=' + str(freq_cw) + 'MHz_out.h5'
output_hdf = h5py.File(fname_out, 'w')
output_hdf.attrs['Z_ice'] = Z_ice
output_hdf.attrs['H_air'] = H_air
output_hdf.attrs['R_ice'] = R_ice
output_hdf.attrs['r_bh'] = r_bh
output_hdf.attrs['z_tx'] = z_tx
output_hdf.attrs['freq_cw'] = freq_cw
output_hdf.attrs['r_tx'] = r_tx
output_hdf.attrs['pad'] = pad

output_hdf.create_dataset('Ez', data=sim_dipole.get_array(center=mp.Vector3(), size=cell, component=mp.Ez))
output_hdf.create_dataset('Er', data=sim_dipole.get_array(center=mp.Vector3(), size=cell, component=mp.Er))
output_hdf.create_dataset('epsilon_r', data=sim_dipole.get_array(center=mp.Vector3(), size=cell, component=mp.Dielectric))
output_hdf.close()