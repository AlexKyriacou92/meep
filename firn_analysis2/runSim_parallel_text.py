import numpy as np
import math
import h5py
import sys
import configparser
from mpi4py import MPI
import meep as mp
import os
from util import get_data, findNearest, add_data_to_hdf
import util

if len(sys.argv) == 2:
    fname_config_in = sys.argv[1]
else:
    fname_config_in = 'config_in.txt'

def cylindrical_meep(fname_config):
    comm = MPI.COMM_WORLD

    config_file = os.path.join(os.path.dirname(__file__), fname_config)
    config = configparser.ConfigParser()
    config.read(config_file)

    # Read In Simulations Settings
    geometry = config['GEOMETRY']
    transmitter = config['TRANSMITTER']
    settings = config['SETTINGS']
    receiver = config['RECEIVER']
    ref_index = config['REFRACTIVE INDEX']
    fname_prefix = settings['prefix']
    fname_data = ref_index['data_file']
    fname_rx_list = receiver['rx_list']

    #Receiver Position Array
    rx_list = np.genfromtxt(fname_rx_list)

    #Refractive index Profile
    nprof_sp, zprof_sp = get_data(fname_data)
    nair = 1.0  # index of air

    # Simulation Geometry
    iceDepth = float(geometry['iceDepth'])  # depth of ice
    airHeight = float(geometry['airHeight'])  # height of air
    iceRange = float(geometry['iceRange'])  # radius of domain
    boreholeRadius = float(geometry['boreholeRadius'])  # radius of borehole
    pad = float(geometry['pad']) # Width of PML Layer

    # Source/Transmitter Positon
    sourceDepth = float(transmitter['sourceDepth'])  # depth of dipole below ice
    sourceRange = float(transmitter['sourceRange']) + boreholeRadius / 2

    print('sourceDepth', sourceDepth)
    # Frequency
    frequency = float(transmitter['frequency'])  # MHz
    bandwidth = float(transmitter['bandwidth'])
    c_meep = 300.0  # m/us
    wavelength = c_meep / frequency  # m.MHz or m / us
    freq_meep = 1 / wavelength  # Meep Units

    wavelength_band = c_meep / bandwidth
    band_meep = 1 / wavelength_band

    sourceAmp = float(transmitter['sourceAmp'])
    R_tot = iceRange + pad

    R_cent = boreholeRadius / 2 + R_tot / 2

    Z_tot = iceDepth + airHeight + 2 * pad
    H_aircent = airHeight / 2  # Central Height of Air
    Z_icecent = iceDepth / 2  # Central Depth of Ice
    r_cent = boreholeRadius / 2
    iceRange_wo_bh = R_tot - boreholeRadius  # size of Ice Block

    nice = 1.78
    resolution = 12.5  # how many pixels per meter
    mpp = 1 / resolution  # meters per pixel
    column_size = iceRange / mpp  # number of pixels in the column
    vice = 1 / nice  # phase velocity in ice
    t_start = 2 * R_tot / vice  # approximate time to reach steady state

    print('r_tx=', sourceRange)
    print('H_aircent', H_aircent)
    print('Z_icecent', Z_icecent)

    def nProfile_data(R, zprof_data=zprof_sp, nprof_data=nprof_sp):
        z = R[2]
        ii_z = findNearest(zprof_data, z)
        n_z = nprof_data[ii_z]
        return mp.Medium(index=n_z)

    ##*********************************************************************##

    ##**********************Simulation Setup*******************************##
    sim_dimensions = mp.CYLINDRICAL  # Cylindrical Coordinate Systen
    pml_layers = [mp.PML(pad)]  # Add Absorbing Layer 'Padding'
    cell = mp.Vector3(2 * R_tot, mp.inf, Z_tot)  # Cell: Simulation Geometry

    # SetUp Simulation Geometry [Borehole, Air, Ice]
    borehole = mp.Block(center=mp.Vector3(r_cent, 0, Z_icecent),
                        size=mp.Vector3(boreholeRadius, mp.inf, iceDepth),
                        material=mp.Medium(index=nair))
    air = mp.Block(center=mp.Vector3(R_cent, 0, H_aircent),
                   size=mp.Vector3(iceRange_wo_bh, mp.inf, airHeight),
                   material=mp.Medium(index=nair))
    ice = mp.Block(center=mp.Vector3(R_cent, 0, Z_icecent),
                   size=mp.Vector3(iceRange_wo_bh, mp.inf, iceDepth),
                   material=nProfile_data)

    geometry_dipole = [borehole, air, ice]

    # create the source
    sources_dipole = []  # List of RF Sources
    t_begin = 20.0  # Time to Start the Pulse

    # RF Source -> Gaussian Pulse, Polarization (Ez), centre position, size
    source1 = mp.Source(mp.GaussianSource(frequency=freq_meep, fwidth=band_meep, start_time=t_begin),
                        component=mp.Ez,
                        center=mp.Vector3(sourceRange, 0, sourceDepth),
                        size=mp.Vector3(0, 0, 0))
    sources_dipole.append(source1)

    # Create Simulation
    sim_dipole = mp.Simulation(force_complex_fields=True,
                               cell_size=cell,
                               dimensions=sim_dimensions,
                               boundary_layers=pml_layers,
                               geometry=geometry_dipole,
                               sources=sources_dipole,
                               resolution=resolution)

    # SetUp HDF Parallel
    # filename
    path2sim = settings['path2output']
    fname_out = path2sim + '/' + fname_prefix + 'z_tx_' + str(sourceDepth) + 'm_freq=' + str(frequency) + 'MHz_out.h5'
    output_hdf = h5py.File(fname_out, 'a', driver='mpio', comm=comm)

    # Save Features
    output_hdf.attrs['iceDepth'] = iceDepth
    output_hdf.attrs['airHeight'] = airHeight
    output_hdf.attrs['iceRange'] = iceRange
    output_hdf.attrs['boreholeRadius'] = boreholeRadius
    output_hdf.attrs['sourceDepth'] = sourceDepth
    output_hdf.attrs['frequency'] = frequency
    output_hdf.attrs['sourceRange'] = sourceRange
    output_hdf.attrs['pad'] = pad

    # =================================================================
    # Get RX Functions
    # ================================================================
    rxList = []
    nRx = len(rx_list)
    for i in range(nRx):
        rx_x = rx_list[i][0]
        rx_z = rx_list[i][1]
        pt_ij = mp.Vector3(rx_x, mp.inf, rx_z)
        rxList.append(pt_ij)

    rxList_out = []
    for i in range(nRx):
        rx_i = rxList[i]
        rxList_out.append([rx_i.x, rx_i.z])
    rx_label = 'rxList'
    add_data_to_hdf(output_hdf, rx_label, rxList_out)
    # ================================================================

    c_mns = 0.3  # Speed of Light in m / ns
    dt_ns = 0.5  # ns
    dt_m = dt_ns * c_mns  # Meep Units a / c
    Courant = 0.5

    dt_C = Courant * mpp  # The simulation time step [a/c] -> dt = C dr , C : Courant Factor, C = 0.5 (default)
    # For accurate sims, C <= n_max / sqrt(nDimensions)

    nSteps = int(t_start / dt_C) + 10
    print('nSteps =', nSteps)

    #Create DataSet to Save the Pulses at RX
    rxPulses = util.add_dataset(output_hdf, 'rxPulses', dimensions=(nRx, nSteps),dtype='complex')
    tspace = np.linspace(0, t_start / c_mns, nSteps)

    eps_label = 'epsilon_r'
    tspace_label = 'tspace'
    #Ez_label = 'Ez'
    #Er_label = 'Er'
    rx_label = 'rxList'
    add_data_to_hdf(output_hdf, tspace_label, tspace)
    add_data_to_hdf(output_hdf, rx_label, rxList_out)

    def save_amp_at_t(sim):
        factor = dt_m / dt_C
        tstep = sim.timestep()
        ii_step = int(float(tstep) / factor) - 1
        for i in range(nRx):
            rx_pt = rxList[i]
            amp_at_pt = sim.get_field_point(c=mp.Ez, pt=rx_pt)
            rxPulses[i, ii_step] = amp_at_pt

    sim_dipole.init_sim()
    #Eps_data = sim_dipole.get_array(center=mp.Vector3(), size=cell, component=mp.Dielectric)
    #add_data_to_hdf(output_hdf, eps_label, Eps_data)

    sim_dipole.use_output_directory(path2sim)
    sim_dipole.run(mp.at_every(dt_C, save_amp_at_t), until=t_start)
    output_hdf.close()
    #TODO: Do I save Eps_data and Ez_data or not?

if __name__ == '__main__':
    print('Begin Simulation')
    cylindrical_meep(fname_config=fname_config_in)
    print('End Simulation')