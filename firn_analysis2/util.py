import numpy as np
import math
from numpy.lib.format import open_memmap
import meep as mp

def get_data(fname_txt):
    n_data = np.genfromtxt(fname_txt)
    zprof = n_data[:,0]
    nprof = n_data[:,1]
    return nprof, zprof

def findNearest(x_arr, x):
    dx = abs(x_arr - x)
    ii = np.argmin(dx)
    return ii

def add_data_to_hdf(hdf_in, label, dataset):
    #Check if label exists
    check_bool = label in hdf_in.keys()
    if check_bool == False:
        hdf_in.create_dataset(label, data=dataset)
    return -1

def add_dataset(hdf_in, label, dimensions, dtype):
    #Check if label exists
    check_bool = label in hdf_in.keys()
    if check_bool == False:
        dset = hdf_in.create_dataset(label, dimensions, dtype=dtype)
    else:
        dset = hdf_in[label]
    return dset

# Create Memmap
def create_memmap(file, dimensions, data_type ='complex'):
    A = open_memmap(file, shape = dimensions, mode='w+', dtype = data_type)
    return A

def nProfile_func(R):
    z = R[2]
    A = 1.78
    B = 0.43
    C = 0.0132 #1/m
    #return mp.Medium(index=A-B*math.exp(-C*(z + Z_tot/2 - H_air)))
    return mp.Medium(index= A - B * math.exp(-C * z))