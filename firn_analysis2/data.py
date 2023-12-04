import meep as mp
import numpy as np
import math
import h5py
import cmasher as cmr
import sys
import configparser
import argparse
def make_nFunc_meep(R, nFunc):
    z = R[2]
    return mp.Medium(index=nFunc(z))
def get_profile_from_data(fname_data):
    n_data = np.genfromtxt(fname_data)
    zprof = n_data[:,0]
    nprof = n_data[:,1]
    return nprof, zprof

def findNearest(x_arr, x):
    dx = abs(x_arr - x)
    ii = np.argmin(dx)
    return ii

class meep_field: #Stores a Monochromatic Field to Field -> read/write
    def __init__(self, iceDepth=30, iceRange=50, airHeight=10, frequency=100, sourceDepth=10, sourceAmp = 1.0, sourceRange=0, boreholeRadius=0, resolution = 12.5, pad = 2.0):
        self.iceDepth = iceDepth
        self.iceRange = iceRange
        self.airHeight = airHeight
        self.frequency = frequency
        self.sourceDepth = sourceDepth
        self.sourceRange = sourceRange
        self.boreholeRadius = boreholeRadius
        self.pad = pad
        self.sourceAmp = sourceAmp
        self.resolution = resolution
    def setup(self, nFunc=None, nVec=None, zVec=None, nVal=1.78):
        #Frequency
        freq_cw = self.frequency #MHz
        c_meep = 300.0 # m/us
        wavelength = c_meep/freq_cw #m.MHz or m / us
        freq_meep = 1/wavelength #Meep Units
        amp_tx = self.sourceAmp
        R_tot = self.iceRange + self.pad

        R_cent = self.boreholeRadius/2 + R_tot/2

        Z_tot = self.iceDepth + self.airHeight + 2*self.pad
        H_aircent = self.airHeight/2 # Central Height of Air
        Z_icecent = self.iceDepth/2 # Central Depth of Ice
        r_cent = self.boreholeRadius/2
        R_ice = R_tot - self.boreholeRadius #size of Ice Block
        r_tx = r_cent

        nice = 1.78
        resolution = self.resolution #how many pixels per meter
        mpp = 1/resolution    # meters per pixel
        column_size = R_tot/mpp	# number of pixels in the column
        vice = 1/nice   	# phase velocity in ice
        self.t_start = 2*R_tot/vice # approximate time to reach steady state
        nair = 1.78

        dimensions = mp.CYLINDRICAL
        pml_layers = [mp.PML(self.pad)]
        cell = mp.Vector3(2*R_tot, mp.inf, Z_tot)


        borehole_block = mp.Block(center=mp.Vector3(r_cent, 0, 0),
                                  size=mp.Vector3(self.boreholeRadius, mp.inf, Z_tot),
                                  material=mp.Medium(index=nair))
        air_block = mp.Block(center=mp.Vector3(R_cent, 0, H_aircent),
                            size=mp.Vector3(R_ice, mp.inf, self.airHeight),
                            material=mp.Medium(index=nair))
        if nFunc != None and nVec == None and zVec == None:
            def nFunc_meep(R, nFunc_in=nFunc):
                z = R[2]
                return mp.Medium(index=nFunc_in(z))

            ice_block = mp.Block(center=mp.Vector3(R_cent, 0, Z_icecent),
                                 size=mp.Vector3(R_ice, mp.inf, self.iceDepth),
                                 material=nFunc_meep)
        elif nVec != None and zVec != None and nFunc == None:
            def nProfile_data(R, zprof_data=zVec, nprof_data=nVec):
                z = R[2]
                ii_z = findNearest(zprof_data, z)
                n_z = nprof_data[ii_z]
                return mp.Medium(index=n_z)
            ice_block = mp.Block(center=p.Vector3(R_cent, 0, Z_icecent),
                                 size=mp.Vector3(R_ice, mp.inf, self.iceDepth),
                                 material=nProfile_data)
        else:
            ice_block = mp.Block(center=p.Vector3(R_cent, 0, Z_icecent),
                                 size=mp.Vector3(R_ice, mp.inf, self.iceDepth),
                                 material=nVal)
        geometry_dipole = [borehole_block, air_block, ice_block]
        source = mp.Source(mp.ContinuousSource(frequency=freq_meep),
                           component=mp.Ez,
                           center=mp.Vector3(r_cent/2,0,self.sourceDepth),
                           size=mp.Vector3(0,0,0))
        sources_dipole = [source]
        sim_dipole = mp.Simulation(force_complex_fields=True,
                                   cell_size=cell,
                                   dimensions=mp.CYLINDRICAL,
                                   boundary_layers=pml_layers,
                                   geometry=geometry_dipole,
                                   sources=sources_dipole,
                                   resolution=resolution)
        self.simulation = sim_dipole
        return self.simulation

    def setup_from_config(self, fname_config, nFunc=None, nVec=None, zVec=None, nVal = 1.78):
        config = configparser.ConfigParser()
        config.read(fname_config)
        geometry = config['GEOMETRY']
        transmitter = config['TRANSMITTER']
        
        self.iceDepth = float(geometry['iceDepth'])
        self.iceRange = float(geometry['iceRange'])
        self.airHeight = float(geometry['airHeight'])
        self.frequency = float(transmitter['frequency'])
        self.sourceDepth = float(transmitter['sourceDepth'])
        self.sourceRange = float(transmitter['sourceRange'])
        self.boreholeRadius = float(geometry['boreholeRadius'])
        self.pad = float(geometry['pad'])
        self.sourceAmp = float(transmitter['sourceAmp'])
        self.resolution = float(geometry['resolution'])

        #Frequency
        freq_cw = self.frequency #MHz
        c_meep = 300.0 # m/us
        wavelength = c_meep/freq_cw #m.MHz or m / us
        freq_meep = 1/wavelength #Meep Units
        amp_tx = self.sourceAmp
        R_tot = self.iceRange + self.pad

        R_cent = self.boreholeRadius/2 + R_tot/2

        Z_tot = self.iceDepth + self.airHeight + 2*self.pad
        H_aircent = self.airHeight/2 # Central Height of Air
        Z_icecent = self.iceDepth/2 # Central Depth of Ice
        r_cent = self.boreholeRadius/2
        R_ice = R_tot - self.boreholeRadius #size of Ice Block
        r_tx = r_cent

        nice = 1.78
        resolution = self.resolution #how many pixels per meter
        mpp = 1/resolution    # meters per pixel
        column_size = R_tot/mpp	# number of pixels in the column
        vice = 1/nice   	# phase velocity in ice
        self.t_start = 2*R_tot/vice # approximate time to reach steady state
        nair = 1.0

        dimensions = mp.CYLINDRICAL
        pml_layers = [mp.PML(self.pad)]
        cell = mp.Vector3(2*R_tot, mp.inf, Z_tot)


        borehole_block = mp.Block(center=mp.Vector3(r_cent, 0, 0),
                                  size=mp.Vector3(self.boreholeRadius, mp.inf, Z_tot),
                                  material=mp.Medium(index=nair))
        air_block = mp.Block(center=mp.Vector3(R_cent, 0, H_aircent),
                            size=mp.Vector3(R_ice, mp.inf, self.airHeight),
                            material=mp.Medium(index=nair))
        if type(nFunc) != type(None) and type(nVec) == type(None) and type(zVec) == type(None):
            def nFunc_meep(R, nFunc_in=nFunc):
                z = R[2]
                return mp.Medium(index=nFunc_in(z))

            ice_block = mp.Block(center=mp.Vector3(R_cent, 0, Z_icecent),
                                 size=mp.Vector3(R_ice, mp.inf, self.iceDepth),
                                 material=nFunc_meep)
        elif type(nVec) != type(None) and type(zVec) != type(None) and type(nFunc) == type(None):
            def nProfile_data(R, zprof_data=zVec, nprof_data=nVec):
                z = R[2]
                ii_z = findNearest(zprof_data, z)
                n_z = nprof_data[ii_z]
                return mp.Medium(index=n_z)
            ice_block = mp.Block(center=mp.Vector3(R_cent, 0, Z_icecent),
                                 size=mp.Vector3(R_ice, mp.inf, self.iceDepth),
                                 material=nProfile_data)
        else:
            ice_block = mp.Block(center=mp.Vector3(R_cent, 0, Z_icecent),
                                 size=mp.Vector3(R_ice, mp.inf, self.iceDepth),
                                 material=nVal)
        geometry_dipole = [borehole_block, air_block, ice_block]
        source = mp.Source(mp.ContinuousSource(frequency=freq_meep),
                           component=mp.Ez,
                           center=mp.Vector3(r_cent/2,0,self.sourceDepth),
                           size=mp.Vector3(0,0,0))
        sources_dipole = [source]
        self.cell = cell
        sim_dipole = mp.Simulation(force_complex_fields=True,
                                   cell_size=cell,
                                   dimensions=mp.CYLINDRICAL,
                                   boundary_layers=pml_layers,
                                   geometry=geometry_dipole,
                                   sources=sources_dipole,
                                   resolution=resolution)
        self.simulation = sim_dipole
        return self.simulation
    #TODO: Setup Run From HDF File
    def setup_from_hdf(self, fname_hdf):
        with h5py.File(fname_hdf, 'r') as input_hdf:
            self.iceDepth = float(input_hdf.attrs['iceDepth'])
            self.iceRange = float(input_hdf.attrs['iceRange'])
            self.airHeight = float(input_hdf.attrs['airHeight'])
            self.frequency = float(input_hdf.attrs['frequency'])
            self.sourceDepth = float(input_hdf.attrs['sourceDepth'])
            self.sourceRange = float(input_hdf.attrs['sourceRange'])
            self.boreholeRadius = float(input_hdf.attrs['boreholeRadius'])
            self.pad = float(input_hdf.attrs['pad'])
            self.sourceAmp = float(input_hdf.attrs['sourceAmp'])
            self.resolution = float(input_hdf.attrs['resolution'])

            self.Ez = np.array(input_hdf['Ez'])
            self.Er = np.array(input_hdf['Er'])
            self.epsilon_r = np.array(input_hdf['epsilon_r'])

    def run_simulation(self, fname_out = None, label=None):
        sim_dipole = self.simulation
        sim_dipole.run(until=self.t_start)

        #Save Simulation
        output_hdf = h5py.File(fname_out, 'w')
        #Save Simulation
        if label == None:
            output_hdf.attrs['Label'] = fname_out[:-3]
        else:
            output_hdf.attrs['Label'] = label
        output_hdf.attrs['iceDepth'] = self.iceDepth
        output_hdf.attrs['airHeight'] = self.airHeight
        output_hdf.attrs['iceRange'] = self.iceRange
        output_hdf.attrs['boreholeRadius'] = self.boreholeRadius
        output_hdf.attrs['sourceDepth'] = self.sourceDepth
        output_hdf.attrs['frequency'] = self.frequency
        output_hdf.attrs['sourceRange'] = self.sourceRange
        output_hdf.attrs['pad'] = self.pad
        output_hdf.attrs['resolution'] = self.resolution
        output_hdf.attrs['sourceAmp'] = self.sourceAmp

        self.Ez = sim_dipole.get_array(center=mp.Vector3(), size=self.cell, component=mp.Ez)
        self.Er = sim_dipole.get_array(center=mp.Vector3(), size=self.cell, component=mp.Er)
        self.Eabs = np.sqrt(abs(self.Er)**2 + abs(self.Ez)**2)
        self.epsilon_r = sim_dipole.get_array(center=mp.Vector3(), size=self.cell, component=mp.Dielectric)
        self.index = np.sqrt(self.epsilon_r)
        output_hdf.create_dataset('Ez', data=self.Ez)
        output_hdf.create_dataset('Er', data=self.Er)
        output_hdf.create_dataset('epsilon_r', data=self.epsilon_r)
        output_hdf.close()

        self.simulation = sim_dipole

    def get_Ez(self):
        return self.Ez

    def get_Er(self):
        return self.Er

    def get_Eabs(self):
        Esq = abs(self.Ez)**2 + abs(self.Er)**2
        return np.sqrt(Esq)

    def get_epsilon_r(self):
        return self.epsilon_r

    def get_refractive(self):
        return np.sqrt(self.epsilon_r)