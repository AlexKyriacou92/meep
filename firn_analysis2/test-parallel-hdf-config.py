import numpy as np
import math
import h5py
import cmasher as cmr
import sys
import configparser
from mpi4py import MPI
import meep as mp

comm = MPI.COMM_WORLD

if len(sys.argv) == 2:
    fname_config = sys.argv[1]
else:
    fname_config = 'config_in.txt'

config = configparser.ConfigParser()
config.read(fname_config)

geometry = config['GEOMETRY']
transmitter = config['TRANSMITTER']
settings = config['SETTINGS']
receiver = config['RECEIVER']
ref_index = config['REFRACTIVE INDEX']

fname_prefix = settings['prefix']
fname_data = ref_index['data_file']
fname_rx_list = receiver['rx_list']

rx_list = np.genfromtxt(fname_rx_list)

def get_data(fname_txt):
    n_data = np.genfromtxt(fname_txt)
    zprof = n_data[:,0]
    nprof = n_data[:,1]
    return nprof, zprof

def findNearest(x_arr, x):
    dx = abs(x_arr - x)
    ii = np.argmin(dx)
    return ii

nprof_sp, zprof_sp = get_data(fname_data)

nair = 1.0      # index of air
iceDepth = float(geometry['iceDepth'])  # depth of ice
airHeight = float(geometry['airHeight'])     # height of air
iceRange = float(geometry['iceRange']) #radius of domain
boreholeRadius = float(geometry['boreholeRadius']) #radius of borehole
pad = float(geometry['pad'])

sourceDepth = float(transmitter['sourceDepth']) #depth of diple below ice
sourceRange = float(transmitter['sourceRange']) + boreholeRadius/2

print('sourceDepth', sourceDepth)
#Frequency
frequency = float(transmitter['frequency']) #MHz
bandwidth = float(transmitter['bandwidth'])
c_meep = 300.0 # m/us
wavelength = c_meep/frequency #m.MHz or m / us
freq_meep = 1/wavelength #Meep Units

wavelength_band = c_meep/bandwidth
band_meep = 1/wavelength_band

sourceAmp = float(transmitter['sourceAmp'])
R_tot = iceRange + pad

R_cent = boreholeRadius/2 + R_tot/2

Z_tot = iceDepth + airHeight + 2*pad
H_aircent = airHeight/2 # Central Height of Air
Z_icecent = iceDepth/2 # Central Depth of Ice
r_cent = boreholeRadius/2
iceRange_wo_bh = R_tot-boreholeRadius #size of Ice Block

nice = 1.78
resolution = 12.5	#how many pixels per meter
mpp = 1/resolution    # meters per pixel
column_size = iceRange/mpp	# number of pixels in the column
vice = 1/nice   	# phase velocity in ice
t_start = 2*R_tot/vice # approximate time to reach steady state

print('r_tx=', sourceRange)
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
cell = mp.Vector3(2*R_tot, mp.inf, Z_tot)

geometry_dipole = [
    mp.Block(center=mp.Vector3(r_cent, 0, Z_icecent),
             size=mp.Vector3(boreholeRadius, mp.inf, iceDepth),
             material=mp.Medium(index=nair)),
    mp.Block(center=mp.Vector3(R_cent, 0, H_aircent),
             size=mp.Vector3(iceRange_wo_bh, mp.inf, airHeight),
             material=mp.Medium(index=nair)),
    mp.Block(center=mp.Vector3(R_cent, 0, Z_icecent),
             size=mp.Vector3(iceRange_wo_bh, mp.inf, iceDepth),
             material=nProfile_data)
]


# create the source
sources_dipole = []
t_begin = 20.0


source1 = mp.Source(mp.GaussianSource(frequency=freq_meep, fwidth=band_meep, start_time = t_begin),
                    component=mp.Ez,
                    center=mp.Vector3(sourceRange, 0, sourceDepth),
                    size=mp.Vector3(0,0,0))
'''
source1 = mp.Source(mp.ContinuousSource(frequency=freq_meep),component=mp.Ez,center=mp.Vector3(sourceRange, 0, sourceDepth),
                    size=mp.Vector3(0,0,0))
'''

sources_dipole.append(source1)
# create simulation
sim_dipole = mp.Simulation(force_complex_fields=True,
                cell_size=cell,
                dimensions=mp.CYLINDRICAL,
                boundary_layers=pml_layers,
                geometry=geometry_dipole,
                sources=sources_dipole,
                resolution=resolution)

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
nRx = len(rx_list)
for i in range(nRx):
    rx_x = rx_list[i][0]
    rx_z = rx_list[i][1]
    pt_ij = mp.Vector3(rx_x, mp.inf, rx_z)
    rxList.append(pt_ij)
print(rxList)
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
        '''
        if rx_pt.z == sourceDepth and rx_pt.x == 20:
            print('field at z = ', sourceDepth, 'm, E(z) =', amp_at_pt)
        if rx_pt.z == -sourceDepth and rx_pt.x == 20:
            print('field at z = ', -sourceDepth, 'm, E(z) =', amp_at_pt)
        '''
        pulse_rx_arr[i, ii_step] = amp_at_pt

path2sim = settings['path2output']
sim_dipole.init_sim()
sim_dipole.use_output_directory(path2sim)
sim_dipole.run(mp.at_every(dt_C, get_amp_at_t2),until=t_start)

fname_out = path2sim + '/' + fname_prefix + 'z_tx_' + str(sourceDepth) + 'm_freq=' + str(frequency) + 'MHz_out.h5'

for i in range(nRx):
    print('check if all elements are zero', np.all(pulse_rx_arr[i] == 0), pulse_rx_arr[i])

def add_data_to_hdf(hdf_in, label, dataset):
    '''
    if label in hdf_in.keys():
        hdf_in[label] = dataset
    else:
        hdf_in.create_dataset(label, data=dataset)
    '''
    #Check if label exists
    check_bool = label in hdf_in.keys()
    if check_bool == False:
        hdf_in.create_dataset(label, data=dataset)

with h5py.File(fname_out, 'a', driver='mpio', comm=MPI.COMM_WORLD) as output_hdf:
    output_hdf.attrs['iceDepth'] = iceDepth
    output_hdf.attrs['airHeight'] = airHeight
    output_hdf.attrs['iceRange'] = iceRange
    output_hdf.attrs['boreholeRadius'] = boreholeRadius
    output_hdf.attrs['sourceDepth'] = sourceDepth
    output_hdf.attrs['frequency'] = frequency
    output_hdf.attrs['sourceRange'] = sourceRange
    output_hdf.attrs['pad'] = pad

    rxList_out = []
    for i in range(nRx):
        rx_i = rxList[i]
        rxList_out.append([rx_i.x, rx_i.z])
    rx_label = 'rxList'
    pulse_label = 'rxPulses'
    tspace_label = 'tspace'
    Ez_label = 'Ez'
    Er_label = 'Er'
    eps_label = 'epsilon_r'
    Ez_data = sim_dipole.get_array(center=mp.Vector3(), size=cell, component=mp.Ez)
    Er_data = sim_dipole.get_array(center=mp.Vector3(), size=cell, component=mp.Er)
    Eps_data = sim_dipole.get_array(center=mp.Vector3(), size=cell, component=mp.Dielectric)

    add_data_to_hdf(output_hdf, rx_label, rxList_out)
    add_data_to_hdf(output_hdf, pulse_label, pulse_rx_arr)
    add_data_to_hdf(output_hdf, Ez_label, Ez_data)
    add_data_to_hdf(output_hdf, Er_label, Er_data)
    add_data_to_hdf(output_hdf, tspace_label, tspace)
    add_data_to_hdf(output_hdf, eps_label, Eps_data)
    output_hdf.close()