import meep as mp
import numpy as np
import math
import h5py
import cmasher as cmr

def get_data(fname_txt):
    n_data = np.genfromtxt(fname_txt)
    zprof = n_data[:,0]
    nprof = n_data[:,1]
    return nprof, zprof

def findNearest(x_arr, x):
    dx = abs(x_arr - x)
    ii = np.argmin(dx)
    return ii
n_air = 1.0
fname_out_prefix = 'southpole_dipole_data_'
fname_data = 'spice2019_indOfRef_core1_5cm.txt'
nprof_sp, zprof_sp = get_data(fname_data)


#Frequency
freq_cw = 100.0 #MHz
wavelength = 300/freq_cw #m.MHz or m / us
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
R_tot = R_ice + pad
Z_tot_geometry = Z_ice + H_air
Z_tot = Z_tot_geometry + 2*pad

Z_cent = (H_air - Z_ice)/2 # Center of 'Block
print('Z_cent', Z_cent)
R_cent = R_ice/2

n_ice = 1.78
v_ice = 1/n_ice
t_start = 2 * R_tot / v_ice


Th = Z_ice + H_air +2*pad    # height of grid

def refractive_index_data(R, n_data=nprof_sp, z_data = zprof_sp):
    z = R[2]# + Th/2 - H_air

    z_depth = -1*z
    ii_z = findNearest(z_data, z_depth)
    ni = n_data[ii_z]
    n_medium = ni
    return mp.Medium(index=n_medium)
import matplotlib.pyplot as pl

nDepths0 = 100
z_space = np.linspace(0, Z_ice, nDepths0)
n_space = np.ones(nDepths0)
for i in range(len(z_space)):
    z_i = z_space[i]
    r_i = [0, 0, -1*z_i]
    n_space[i] = refractive_index_data(r_i).epsilon_diag.z
fig = pl.figure(figsize=(8,5),dpi=120)
ax = fig.add_subplot(111)
ax.plot(z_space, n_space)
ax.grid()
pl.show()
resolution = 12.5	#how many pixels per meter
n_air = 1.0

#NOTE -> Z > 0 is above, z < 0 is below

Ydim = mp.inf
dimensions = mp.CYLINDRICAL #Define Cylindrical Symmetry
cell = mp.Vector3(R_ice, Ydim, Z_tot)
pml_layers = [mp.PML(pad)]

'''
geometry_dipole = [
    mp.Block(center=mp.Vector3(0, Ydim, Z_cent),
             size=mp.Vector3(r_bh, mp.inf, Z_tot),
             material=mp.Medium(index=n_air)),
    mp.Block(center=mp.Vector3(0, Ydim, Z_cent),
             size=mp.Vector3(R_ice, mp.inf, Z_tot),
             material=refractive_index_data)
]
'''

geometry_dipole = [
    mp.Block(center=mp.Vector3(0, Ydim, -Z_ice/2),
             size=mp.Vector3(r_bh, mp.inf, H_air),
             material=mp.Medium(index=1.0)),
    mp.Block(center=mp.Vector3(0, Ydim, H_air/2),
             size=mp.Vector3(R_ice, mp.inf, H_air),
             material=mp.Medium(index=1.0)),
    mp.Block(center=mp.Vector3(0, Ydim, -Z_ice/2),
             size=mp.Vector3(R_ice, mp.inf, Z_ice),
             material=refractive_index_data)
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

#print('Cell Size:',sim_dipole.cell_size)
#print('Material Function', sim_dipole.material_function)
#print('Epsilon_Function', sim_dipole.epsilon_func)
#print('')

print(sim_dipole.get_epsilon_grid(xtics=2*np.ones(len(z_space)), ytics=np.zeros(len(z_space)), ztics=z_space))
print(sim_dipole.initialize_field())
fig, ax = pl.subplots()
ax.set_aspect('auto')
pl.imshow(sim_dipole.get_array(center=mp.Vector3(), size=cell, component=mp.Dielectric), aspect='auto', cmap=cmr.arctic,extent=[0, R_ice, -Z_ice, H_air])
pl.axis('on')
#plot_title = "dipole_"+name+"_d_"+str(depth)
#pl.title(plot_title)
ax.set_xlabel("radial direction (m)")
ax.set_ylabel("depth (m)")
pl.colorbar()
pl.show()

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