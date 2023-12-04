import meep as mp
import numpy as np
import math
import h5py
import cmasher as cmr
import sys
import configparser


if len(sys.argv) == 2:
    fname_config = sys.argv[1]
    fname_nprofile = sys.argv[2]
else:
    fname_config = 'config_in.txt'
    fname_nprofile = 'spice2019_indOfRef_core1_5cm.txt'

config = configparser.ConfigParser()
config.read(fname_config)

geometry = config['GEOMETRY']
transmitter = config['TRANSMITTER']
settings = config['SETTINGS']

def get_data(fname_txt):
    n_data = np.genfromtxt(fname_txt)
    zprof = n_data[:,0]
    nprof = n_data[:,1]
    return nprof, zprof

def findNearest(x_arr, x):
    dx = abs(x_arr - x)
    ii = np.argmin(dx)
    return ii
fname_out_prefix = 'southpole_dipole_data_'
fname_data = 'spice2019_indOfRef_core1_5cm.txt'
nprof_sp, zprof_sp = get_data(fname_data)

nair = 1.0      # index of air
Z_ice = float(geometry['Z_ice'])  # depth of ice
H_air = float(geometry['H_air'])     # height of air
R_domain = float(geometry['R_domain']) #radius of domain
r_bh = float(geometry['r_bh']) #radius of borehole
pad = float(geometry['pad'])

z_tx = float(transmitter['z_tx']) #depth of diple below ice
r_tx = float(transmitter['r_tx']) + r_bh/2
#Frequency
freq_cw = float(transmitter['freq_cw']) #MHz
c_meep = 300.0 # m/us
wavelength = c_meep/freq_cw #m.MHz or m / us
freq_meep = 1/wavelength #Meep Units
amp_tx = float(transmitter['amp_tx'])
R_tot = R_domain + pad

R_cent = r_bh/2 + R_tot/2

Z_tot = Z_ice + H_air + 2*pad
H_aircent = H_air/2 # Central Height of Air
Z_icecent = Z_ice/2 # Central Depth of Ice
r_cent = r_bh/2
R_ice = R_tot-r_bh #size of Ice Block

nice = 1.78
resolution = 12.5	#how many pixels per meter
mpp = 1/resolution    # meters per pixel
column_size = R_domain/mpp	# number of pixels in the column
vice = 1/nice   	# phase velocity in ice
t_start = 2*R_tot/vice # approximate time to reach steady state

print('r_tx=', r_tx)
print('H_aircent', H_aircent)
print('Z_icecent', Z_icecent)
def nProfile_func(R):
    z = R[2]
    A = 1.78
    B = 0.43
    C = 0.0132 #1/m
    #return mp.Medium(index=A-B*math.exp(-C*(z + Z_tot/2 - H_air)))
    return mp.Medium(index= A - B * math.exp(-C * z))

def nProfile_data(R, zprof_data=zprof_sp, nprof_data=nprof_sp):
    z = R[2]
    ii_z = findNearest(zprof_data, z)
    n_z = nprof_data[ii_z]
    return mp.Medium(index=n_z)
##*********************************************************************##

##**********************Simulation Setup*******************************##
dimensions = mp.CYLINDRICAL

pml_layers = [mp.PML(pad)]
z_tx = 20.0 # + Z_tot/2 - H_air
cell = mp.Vector3(2*R_tot, mp.inf, Z_tot)

geometry_dipole = [
    mp.Block(center=mp.Vector3(r_cent, 0, 0),
             size=mp.Vector3(r_bh, mp.inf, Z_tot),
             material=mp.Medium(index=nair)),
    mp.Block(center=mp.Vector3(R_cent, 0, H_aircent),
             size=mp.Vector3(R_ice, mp.inf, H_air),
             material=mp.Medium(index=nair)),
    mp.Block(center=mp.Vector3(R_cent, 0, Z_icecent),
             size=mp.Vector3(R_ice, mp.inf, Z_ice),
             material=nProfile_data)
]


# create the source
sources_dipole = []
sources_dipole.append(mp.Source(mp.ContinuousSource(frequency=freq_meep),
            component=mp.Ez,
            center=mp.Vector3(r_cent/2,0,z_tx),
            size=mp.Vector3(0,0,0)))

# create simulation
sim_dipole = mp.Simulation(force_complex_fields=True,
                cell_size=cell,
                dimensions=mp.CYLINDRICAL,
                boundary_layers=pml_layers,
                geometry=geometry_dipole,
                sources=sources_dipole,
                resolution=resolution)

# this runs the simulation until we have reached the steady state
sim_dipole.run(until=t_start)
##*********************************************************************##

##********************** E-Field Qunatities *******************************##


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