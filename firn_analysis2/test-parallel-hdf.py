import meep as mp
import numpy as np
import math
import h5py
import cmasher as cmr
import sys

from mpi4py import MPI
comm = MPI.COMM_WORLD


nair = 1.0      # index of air
Z_ice = 50     # depth of ice
H_air = 15     # height of air

R = 50 #radius of domain
r_bh = 0.15 #radius of borehole
pad = 2
z_tx_in = 20.0
z_tx = z_tx_in #depth of diple below ice

#Frequency
freq_cw = 150.0 #MHz
c_meep = 300.0 # m/us
wavelength = c_meep/freq_cw # a = 1 m

freq_meep = 1/wavelength #Meep Units, f [1/a]
print('freq_meep', freq_meep, 'wavelength', wavelength, 'freq_cw', freq_cw)

bandwidth = 60.0
band_meep = bandwidth
band_meep = (c_meep/bandwidth)**-1.


amp_tx = 1.0
R_tot = R + pad

R_cent = r_bh/2 + R_tot/2

Z_tot = Z_ice + H_air + 2*pad
H_aircent = H_air/2 # Central Height of Air
Z_icecent = Z_ice/2 # Central Depth of Ice
r_cent = r_bh/2
R_ice = R_tot-r_bh #size of Ice Block
r_tx = r_cent

nice = 1.78
#resolution = 12.5	#how many pixels per meter
resolution = 12.5
mpp = 1/resolution    # meters per pixel
print('m per pixel',mpp)
column_size = R/mpp	# number of pixels in the column
#vice = c_meep/nice   	# phase velocity in ice
vice = 1/nice   	# phase velocity in ice
t_start = 2*R_tot/vice # Time is in length units

print('t_start = ', t_start)

print('r_tx=', r_tx)
print('H_aircent', H_aircent)
print('Z_icecent', Z_icecent)

#================================================================
#  Make Refractive Index Profile
#=================================================================
def nProfile_func(R):
    z = R[2]
    A = 1.78
    B = 0.43
    C = 0.0132 #1/m
    #return mp.Medium(index=A-B*math.exp(-C*(z + Z_tot/2 - H_air)))
    return mp.Medium(index= A - B * math.exp(-C * z))

#=================================================================
# Get RX Functions
#================================================================
rxList = []

x_ranges = [2., 10., 20., 30., 45.]
z_depths = [-40., -30., -20., -10., -2., 2.]

nRanges = len(x_ranges)
nDepths = len(z_depths)
for i in range(nRanges):
    for j in range(nDepths):
        pt_ij = mp.Vector3(x_ranges[i], mp.inf, z_depths[j])
        rxList.append(pt_ij)
nRx = len(rxList)

c_mns = 0.3 #Speed of Light in m / ns

dt_ns = 0.5 # ns
dt_m = dt_ns * c_mns # Meep Units a / c
Courant = 0.5

dt_C = Courant * mpp # The simulation time step [a/c] -> dt = C dr , C : Courant Factor, C = 0.5 (default)
# For accurate sims, C <= n_max / sqrt(nDimensions)

nSteps = int(t_start/dt_C) + 10
print(nSteps)
pulse_rx_arr = np.zeros((nRx, nSteps),dtype='complex')
tspace = np.linspace(0, t_start / c_mns, nSteps)

def get_amp_at_t2(sim):
    nRx = len(rxList)
    factor = dt_m / dt_C
    tstep = sim.timestep()
    ii_step = int(float(tstep) / factor) - 1 #TODO: Check if this starts as 0 or 1
    for i in range(nRx):
        rx_pt = rxList[i]
        amp_at_pt = sim.get_field_point(c=mp.Ez, pt=rx_pt)
        pulse_rx_arr[i, ii_step] = amp_at_pt
#=================================================================
# Make Filename
#================================================================

#================================================================
#               Setup Simulation Geometry
#================================================================

dimensions = mp.CYLINDRICAL

pml_layers = [mp.PML(pad)]
z_tx = 20.0 # + Z_tot/2 - H_air
cell = mp.Vector3(2*R_tot, mp.inf, Z_tot) #is this too large?

geometry_dipole = [
    mp.Block(center=mp.Vector3(r_cent, 0, 0),
             size=mp.Vector3(r_bh, mp.inf, Z_tot),
             material=mp.Medium(index=nair)),
    mp.Block(center=mp.Vector3(R_cent, 0, H_aircent),
             size=mp.Vector3(R_ice, mp.inf, H_air),
             material=mp.Medium(index=nair)),
    mp.Block(center=mp.Vector3(R_cent, 0, Z_icecent),
             size=mp.Vector3(R_ice, mp.inf, Z_ice),
             material=nProfile_func)
]

# create the source
sources_dipole = []
t_begin = 20.0

source1 = mp.Source(mp.GaussianSource(frequency=freq_meep, fwidth=band_meep, start_time = t_begin),
                    component=mp.Ez,
                    center=mp.Vector3(r_cent/2,0,z_tx),
                    size=mp.Vector3(0,0,0))

sources_dipole.append(source1)

# create simulation
sim_dipole = mp.Simulation(force_complex_fields=True,
                cell_size=cell,
                dimensions=mp.CYLINDRICAL,
                boundary_layers=pml_layers,
                geometry=geometry_dipole,
                sources=sources_dipole,
                resolution=resolution,
                Courant = Courant)

sim_dipole.init_sim()
sim_dipole.use_output_directory('mpi_tests')
sim_dipole.run(mp.at_every(dt_C, get_amp_at_t2),until=t_start)
fname_prefix = 'sim_testing_mpi_'

fname_out = fname_prefix + 'z_tx_' + str(z_tx) + 'm_freq=' + str(freq_cw) + 'MHz_out.h5'
#output_hdf = h5py.File(fname_out, 'w')

with h5py.File(fname_out, 'w') as output_hdf:
    output_hdf.attrs['Z_ice'] = Z_ice
    output_hdf.attrs['H_air'] = H_air
    output_hdf.attrs['R_ice'] = R_ice
    output_hdf.attrs['r_bh'] = r_bh
    output_hdf.attrs['z_tx'] = z_tx
    output_hdf.attrs['freq_cw'] = freq_cw
    output_hdf.attrs['r_tx'] = r_tx
    output_hdf.attrs['pad'] = pad

    rxList_out = []
    for i in range(nRx):
        rx_i = rxList[i]
        rxList_out.append([rx_i.x, rx_i.z])
    output_hdf.create_dataset('rxList', data=rxList_out)
    output_hdf.create_dataset('rxPulses', data=pulse_rx_arr)
    output_hdf.create_dataset('tspace', data=tspace)
    output_hdf.create_dataset('Ez', data=sim_dipole.get_array(center=mp.Vector3(), size=cell, component=mp.Ez))
    output_hdf.create_dataset('Er', data=sim_dipole.get_array(center=mp.Vector3(), size=cell, component=mp.Er))
    output_hdf.create_dataset('epsilon_r', data=sim_dipole.get_array(center=mp.Vector3(), size=cell, component=mp.Dielectric))
    output_hdf.close()