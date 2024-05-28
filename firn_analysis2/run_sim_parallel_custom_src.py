import numpy as np
import math
import h5py
import sys
import configparser
from mpi4py import MPI
import meep as mp
import os
from util import get_data, findNearest
import util

comm = MPI.COMM_WORLD

if len(sys.argv) == 2:
    fname_config = sys.argv[1]
else:
    fname_config = 'config_in.txt'

print('Initiating Simulation of a Gaussian pulse generated from a point source inside a modelled ice-sheet (range-independent)')

config_file = os.path.join(os.path.dirname(__file__), fname_config)
config = configparser.ConfigParser()
config.read(config_file)

#Input Parameters
print('Input Parameters from config file', fname_config)

geometry = config['GEOMETRY']
print('Geometry:')
for key in geometry.keys():
    print(key, geometry[key])
transmitter = config['TRANSMITTER']
print('Transmitter:')
for key in transmitter.keys():
    print(key, transmitter[key])
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

#UNITS OF LENGTH ARE IN METERS
'''
Meep: c = 1
Length scale a = 1m
Time = a/c = 1m
Frequency = c/a (or 'inverse meters')
'''

fname_prefix = settings['prefix']
fname_data = ref_index['data_file']
fname_rx_list = receiver['rx_list']

rx_list = np.genfromtxt(fname_rx_list) #Receiver List

nprof_sp, zprof_sp = get_data(fname_data) #Refractive Index imported from file

nair = 1.0      # index of air
iceDepth = float(geometry['iceDepth'])  # depth of ice
airHeight = float(geometry['airHeight'])     # height of air
iceRange = float(geometry['iceRange']) #radius of domain
boreholeRadius = float(geometry['boreholeRadius']) #radius of borehole
pad = float(geometry['pad'])
dpml = pad/2

sourceDepth = float(transmitter['sourceDepth']) #depth of diple below ice
sourceRange = float(transmitter['sourceRange']) + boreholeRadius/2

print('sourceDepth', sourceDepth)
#Frequency
freq_MHz = float(transmitter['frequency']) # MHz
band_MHz = float(transmitter['bandwidth']) # MHz
freq_GHz = freq_MHz/1e3
band_GHz = band_MHz/1e3
a_scale = 1 # a = 1m

c_mMHz = 300.0 # m/us or m MHz
c_mGHz = 0.3 # m/ns or m GHz
c_meep = 1.0 #Speed of Light is set to 1 -> Scale Invariance of Maxwell's Equations
wavelength_1m = c_mMHz/freq_MHz # Wavelegnth in metres [m]
wavelength_band_1m = c_mMHz/band_MHz #Wavelength in metres [m]
freq_meep = c_meep/wavelength_1m # Freq in Inverse Meters [1/m]
band_meep = c_meep/wavelength_band_1m # Bandwidth in Inverse Metere [1/m]
sourceAmp = float(transmitter['sourceAmp'])

#RESOLUTION
nice = 1.78
resolution = 12.5	#how many pixels per meter
mpp = 1/resolution    # meters per pixel, also dx
column_size = iceRange/mpp	# number of pixels in the column
vice = 1/nice   	# phase velocity in ice

t_start_ns = 20.0 #ns
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
t_space_ns = t_space_meep/c_mGHz
nSteps = int(t_end_meep / dt_meep)

fsamp_MHz = 1/dt_us
wavel_samp = c_mMHz/fsamp_MHz # THIS IS THE SAME AS DT_M
fsamp_meep = 1/wavel_samp # 1 / DT_M
print('sample frequency', fsamp_MHz)

print('Size of Time Space =', len(t_space_meep), 'nSteps t_end_meep/mpp =', nSteps)

print(freq_meep-band_meep/2, freq_meep+band_meep/2, fsamp_meep)
#Redefine Geometry for Meep
R_tot = iceRange + pad + dpml# Total Geometry Radius (cylindrical coordinate system) including the padded region

R_cent = boreholeRadius/2 + R_tot/2 # Center of the Geometry (here center means half of the radius)

Z_tot = iceDepth + airHeight + 2*pad #Total Geometry Height: Including Ice and Air and the padded region
H_aircent = airHeight/2 # Central Height of Air #
Z_icecent = iceDepth/2 # Central Depth of Ice
r_cent = boreholeRadius/2 # Centre of Breohole
iceRange_wo_bh = R_tot-boreholeRadius #size of Ice Block without the borehole

#DEFINE THE REFRACTIVE INDEX

#Define Conducitivity (Attenuation)
c_ms = 3e8 #Speed of Light SI units [m/s]
eps_r_ice = nice**2
epsilon0 = 8.85e-12 #Vacuum Permittivity SI units [F/m]
cond_in_SM = 4.7e-6 # [S/m] Based on the Attenuation Coefficient (assuming A(f) = 1) Source: Aguilar et al. Radiofrequency ice dielectric measurements at Summit Station, Greenland
cond_in_meep = (1/c_ms)*cond_in_SM/(eps_r_ice*epsilon0)
def nProfile_data(R, zprof_data=zprof_sp, nprof_data=nprof_sp):
    z = R[2]
    ii_z = findNearest(zprof_data, z)
    n_z = nprof_data[ii_z]
    #return mp.Medium(index=n_z, D_conductivity=cond_in_meep)
    return mp.Medium(index=n_z)

def band_limited_pulse(freq_meep=freq_meep, band_meep=band_meep, t_space_meep=t_space_meep, t_start_meep = t_start_meep):
    #This defines the time dependent E-field pulse generated at the transmitter [V/m]
    # If I want to approximate Askayan emission, I need a physical description of the pulse amplitude
    fmin = freq_meep-band_meep/2
    fmax = freq_meep + band_meep / 2
    amp = sourceAmp

    nSamples = len(t_space_meep)
    impulse = np.zeros(nSamples)
    ii0 = util.findNearest(t_space_meep, t_start_meep)
    impulse[ii0] = amp
    pulse_out = util.butterBandpassFilter(impulse, fmin, fmax, fsamp_meep) + 0j
    return pulse_out
pulse_out = band_limited_pulse(freq_meep, band_meep, t_space_meep, t_start_meep)

def band_limited_pulse_meep(t, pulse_out=pulse_out, t_space_meep=t_space_meep): #DEFINED USING MEEP UNITS-> freq = 1/wavel
    ii = findNearest(t_space_meep, t)
    return pulse_out[ii]

#TODO: Add Noise Sources
#TODO: Add Attenuation

pulse_example = []
for t in t_space_meep:
    pulse_example.append(band_limited_pulse_meep(t))
print('pulse type:', type(pulse_example[0]))

##**********************Simulation Setup*******************************##
dimensions = mp.CYLINDRICAL

pml_layers = [mp.PML(dpml)] #TODO: Check if this is along the whole geometry
cell = mp.Vector3(R_tot, mp.inf, Z_tot)
#TODO: Should it be rather : cell = mp.Vector3(R_tot, mp.inf, Z_tot) ?
print('Define Geometry')
geometry_dipole = [
    mp.Block(center=mp.Vector3(R_cent, 0, H_aircent), #AIR
             size=mp.Vector3(iceRange, mp.inf, airHeight),
             material=mp.Medium(index=nair)),
    mp.Block(center=mp.Vector3(R_cent, 0, Z_icecent), #ICE
             size=mp.Vector3(iceRange, mp.inf, iceDepth),
             material=nProfile_data)
]

sources_dipole = []
'''
source1 = mp.Source(mp.GaussianSource(frequency=freq_meep, fwidth=band_meep, start_time = t_start_meep),
                    component=mp.Ez,
                    center=mp.Vector3(sourceRange, 0, sourceDepth),
                    size=mp.Vector3(0,0,0))
'''
source1 = mp.Source(mp.CustomSource(src_func = band_limited_pulse_meep),
                    component=mp.Ez,
                    center=mp.Vector3(sourceRange, 0, sourceDepth))
sources_dipole.append(source1)

sources_dipole.append(source1)
# create simulation
sim_dipole = mp.Simulation(force_complex_fields=True,
                cell_size=cell,
                dimensions=mp.CYLINDRICAL,
                boundary_layers=pml_layers,
                geometry=geometry_dipole,
                sources=sources_dipole,
                resolution=resolution)

rxList = []
nRx = len(rx_list)
for i in range(nRx):
    rx_x = rx_list[i][0]
    rx_z = rx_list[i][1]
    pt_ij = mp.Vector3(rx_x, mp.inf, rx_z)
    rxList.append(pt_ij)
print(rxList)

dt_C = S_courant * mpp
pulse_rx_arr = np.zeros((nRx, nSteps),dtype='complex')

def get_amp_at_t2(sim):
    nRx = len(rxList)
    factor = dt_meep / dt_C
    tstep = sim.timestep()
    ii_step = int(float(tstep) / factor) - 1 #TODO: Check if this starts as 0 or 1
    for i in range(nRx):
        rx_pt = rxList[i]

        amp_at_pt = sim.get_field_point(c=mp.Ez, pt=rx_pt)
        pulse_rx_arr[i, ii_step] = amp_at_pt

path2sim = settings['path2output']
print('Initialize Simulation')
sim_dipole.init_sim()
print('Defining Output Directory')
sim_dipole.use_output_directory(path2sim)
print('Running Simulation \n')
sim_dipole.run(mp.at_every(dt_C, get_amp_at_t2),until=t_end_meep)

fname_out = path2sim + '/' + fname_prefix + 'z_tx_' + str(sourceDepth) + 'm_freq=' + str(freq_MHz) + 'MHz_out.h5'

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
    output_hdf.attrs['frequency'] = freq_GHz
    output_hdf.attrs['bandwidth'] = band_GHz
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
    pulse_label = 'rxPulses'
    tspace_label = 'tspace'
    tspace_meep_label = 'tspace_meep'

    zProfile_label = 'zProfile'
    nProfile_label = 'nProfile'
    txPulse_label = 'txPulse'

    add_data_to_hdf(output_hdf, txPulse_label, pulse_out)
    add_data_to_hdf(output_hdf, zProfile_label, zprof_sp)
    add_data_to_hdf(output_hdf, nProfile_label, nprof_sp)
    add_data_to_hdf(output_hdf, rx_label, rxList_out)
    add_data_to_hdf(output_hdf, pulse_label, pulse_rx_arr)
    add_data_to_hdf(output_hdf, tspace_label, t_space_ns)
    add_data_to_hdf(output_hdf, tspace_meep_label, t_space_meep)

print('Simulation Complete')