import datetime

import numpy as np
import math
import h5py
import sys
import configparser
from mpi4py import MPI
import meep as mp
import os
import time
from util import get_data, findNearest, nProfile_func
import Askaryan_Signal
from Askaryan_Signal import create_pulse, TeV

comm = MPI.COMM_WORLD

if len(sys.argv) == 2:
    fname_config = sys.argv[1]
else:
    fname_config = 'config_in.txt'

print('Initiating Simulation of a Askaryan pulse generated from a point source inside a modelled ice-sheet (range-independent)')
config_file = os.path.join(os.path.dirname(__file__), fname_config)
config = configparser.ConfigParser()
config.read(config_file)
now = datetime.datetime.now()
print('Simulation Started at: ', now)
print('')
#Input Parameters
print('Input Parameters from config file', fname_config)

geometry = config['GEOMETRY']
print('Geometry:')
for key in geometry.keys():
    print(key, geometry[key])
source = config['SOURCE']
print('source:')
for key in source.keys():
    print(key, source[key])
settings = config['SETTINGS']
print('settings:')
for key in settings.keys():
    print(key, settings[key])
receiver = config['RECEIVER']
print('receiver:')
for key in receiver.keys():
    print(key, receiver[key])
ref_index = config['REFRACTIVE INDEX']
print('ref index')
for key in ref_index.keys():
    print(key, ref_index[key])

fname_prefix = settings['prefix']
fname_data = ref_index['data_file']
fname_rx_list = receiver['rx_list']

rx_list = np.genfromtxt(fname_rx_list)

nprof_sp, zprof_sp = get_data(fname_data)

nair = 1.0      # index of air
iceDepth = float(geometry['iceDepth'])  # depth of ice
airHeight = float(geometry['airHeight'])     # height of air
iceRange = float(geometry['iceRange']) #radius of domain
boreholeRadius = float(geometry['boreholeRadius']) #radius of borehole
pad = float(geometry['pad'])
dpml = pad/2

sourceDepth = float(source['sourceDepth']) #depth of diple below ice
sourceRange = float(source['sourceRange']) + boreholeRadius/2

print('sourceDepth', sourceDepth)

a_scale = 1 # a = 1m

c_mMHz = 300.0 # m/us or m MHz
c_mGHz = 0.3 # m/ns or m GHz
c_meep = 1.0 #Speed of Light is set to 1 -> Scale Invariance of Maxwell's Equations

#RESOLUTION
nice = 1.78
resolution = float(geometry['resolution'])	#how many pixels per meter
mpp = 1/resolution    # meters per pixel, also dx
column_size = iceRange/mpp	# number of pixels in the column
vice = 1/nice   	# phase velocity in ice

t_start_ns = float(source['startTime']) #ns TODO: Make Start Time Controllable
t_start_meep = t_start_ns*c_mGHz
print('Start Time in [m]', t_start_meep)

S_courant = 0.5 #Courant Factor S -> dt = S dx / c or S mpp /c
dt_meep = S_courant * mpp

dt_ns = dt_meep/c_mGHz
dt_us = dt_meep/c_mMHz
print('dt_m = ', dt_meep, 'm')
print('dt_ns = ', dt_ns, 'ns')
t_end_meep = 2*nice*iceRange # Enough 'time' for the signal to traverse the simulation domain twice if n = n_ice
t_end_ns = t_end_meep/c_mGHz

t_space_meep = np.arange(0, t_end_meep, dt_meep)
t_space_ns = t_space_meep * c_mGHz
nSteps = int(t_end_meep / dt_meep)
print('nSteps=',nSteps)

fsamp_MHz = 1/dt_us
wavel_samp = c_mMHz/fsamp_MHz # THIS IS THE SAME AS DT_M
fsamp_meep = 1/wavel_samp # 1 / DT_M
print('sample frequency', fsamp_MHz, 'MHz')

print('Size of Time Space =', len(t_space_meep), 'nSteps t_end_meep/mpp =', nSteps)
#Redefine Geometry for Meep
R_tot = iceRange + pad + dpml# Total Geometry Radius (cylindrical coordinate system) including the padded region

R_cent = boreholeRadius/2 + R_tot/2 # Center of the Geometry (here center means half of the radius)

Z_tot = iceDepth + airHeight + 2*pad #Total Geometry Height: Including Ice and Air and the padded region
H_aircent = airHeight/2 # Central Height of Air #
Z_icecent = iceDepth/2 # Central Depth of Ice
r_cent = boreholeRadius/2 # Centre of Breohole
iceRange_wo_bh = R_tot-boreholeRadius #size of Ice Block without the borehole


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

#Define Askaryan Signal
sources_dipole = []
#Create Source with Askaryan Emission
Esh_eV = float(source['showerEnergy'])
Esh_TeV = 1e18/TeV # Energy of Neutrino-Induced Shower 10^18 eV  TODO: Make Config Controlled
dtheta_nu = float(source['dthetaV']) # Offset Angle between v and c TODO: Make Config Controlled
R_alpha = float(source['attenuationEquivalent']) #Distance used to define 'pre-attenuation' -> filters out high frequencies
t_start_us = t_start_ns/1e3

#Define Askaryan Pulse at Source
pulse_in, tspace_in_ns = create_pulse(Esh=Esh_TeV, dtheta_v=dtheta_nu,R_alpha=R_alpha, t_min=-t_start_us, t_max=500e-3-t_start_us,z_alpha=sourceDepth)
print('nSteps=', nSteps)
print('len(pulse_in) = ', len(pulse_in), 'len(tspace_in)', len(tspace_in_ns))
tspace_meep = tspace_in_ns * c_mGHz
print('len(tspace_meep)', len(tspace_meep))
def pulse_meep(t, pulse_in=pulse_in, t_space_in=tspace_in_ns):
    tspace_meep = t_space_in * c_mGHz # Meep Time Domain - units of distance
    if t < max(tspace_meep):
        ii = findNearest(tspace_meep, t)
        return pulse_in[ii]
    else:
        return 0
print('len(tspace_meep) 2', len(tspace_meep))
source1 = mp.Source(mp.CustomSource(src_func = pulse_meep),
                    component=mp.Ez,
                    center=mp.Vector3(sourceRange, 0, sourceDepth))

sources_dipole.append(source1)
# create simulation
sim_dipole = mp.Simulation(force_complex_fields=True,
                cell_size=cell,
                dimensions=mp.CYLINDRICAL,
                boundary_layers=pml_layers,
                geometry=geometry_dipole,
                sources=sources_dipole,
                resolution=resolution)
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

pulse_z_rx_arr = np.zeros((nRx, nSteps),dtype='complex')
pulse_r_rx_arr = np.zeros((nRx, nSteps),dtype='complex')
tspace_actual = np.zeros((nRx, nSteps),dtype='float')
dt_C = S_courant*mpp
def get_amp_at_t2(sim):
    nRx = len(rxList)
    factor = dt_meep / dt_C
    tstep = sim.timestep()
    time_meep = sim.meep_time()
    ii_step = int(float(tstep) / factor) - 1
    for i in range(nRx):
        rx_pt = rxList[i]
        amp_z_at_pt = sim.get_field_point(c=mp.Ez, pt=rx_pt)
        amp_r_at_pt = sim.get_field_point(c=mp.Er, pt=rx_pt)

        pulse_z_rx_arr[i, ii_step] = amp_z_at_pt
        pulse_r_rx_arr[i, ii_step] = amp_r_at_pt
        tspace_actual[i, ii_step] = time_meep
path2sim = settings['path2output']
tstart_initial = time.time()
sim_dipole.init_sim()
tend_initial = time.time()
now = datetime.datetime.now()
duration = tend_initial-tstart_initial
print('Simulation Initialization Complete, simulation run starting at: ', now)
print('Duration: ', datetime.timedelta(seconds=duration))
print('')
sim_dipole.use_output_directory(path2sim)

tstart_run = time.time()
sim_dipole.run(mp.at_every(dt_C, get_amp_at_t2),until=t_end_meep)
print('')
tend_run = time.time()
now = datetime.datetime.now()
duration = tend_run - tstart_run
print('Simulation Run Complete at', now)
print('Duration: ', datetime.timedelta(seconds=duration))

fname_out = path2sim + '/' + fname_prefix + '_z_tx_' + str(sourceDepth) + '_dtheta=' + str(dtheta_nu) + '_askaryan.h5'
for i in range(nRx):
    print('check if all elements [r] are zero', np.all(pulse_r_rx_arr[i] == 0), pulse_r_rx_arr[i])
    print('check if all elements [z] are zero', np.all(pulse_z_rx_arr[i] == 0), pulse_z_rx_arr[i])

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
    output_hdf.attrs['freq_samp'] = fsamp_MHz/1e3
    output_hdf.attrs['dt'] = dt_ns
    output_hdf.attrs['sourceRange'] = sourceRange
    output_hdf.attrs['pad'] = pad
    output_hdf.attrs['dpml'] = dpml

    rxList_out = []
    for i in range(nRx):
        rx_i = rxList[i]
        rxList_out.append([rx_i.x, rx_i.z])
    rx_label = 'rxList'
    pulse_r_label = 'rxPulses_r'
    pulse_z_label = 'rxPulses_z'

    tspace_label = 'tspace'
    tspace_meep_label = 'tspace_meep'

    zProfile_label = 'zProfile'
    nProfile_label = 'nProfile'
    txPulse_label = 'txPulse'

    #add_data_to_hdf(output_hdf, txPulse_label, pulse_out)
    add_data_to_hdf(output_hdf, txPulse_label, pulse_in)
    add_data_to_hdf(output_hdf, zProfile_label, zprof_sp)
    add_data_to_hdf(output_hdf, nProfile_label, nprof_sp)
    add_data_to_hdf(output_hdf, rx_label, rxList_out)
    add_data_to_hdf(output_hdf, pulse_r_label, pulse_r_rx_arr)
    add_data_to_hdf(output_hdf, pulse_z_label, pulse_z_rx_arr)

    add_data_to_hdf(output_hdf, tspace_label, t_space_ns)
    add_data_to_hdf(output_hdf, tspace_meep_label, t_space_meep)
    add_data_to_hdf(output_hdf, 'tspace_actual', tspace_actual)
    add_data_to_hdf(output_hdf, 'tspace_tx', tspace_in_ns)

now = datetime.datetime.now()
print('Simulation Complete at', now)