import numpy as np
import math
from numpy.lib.format import open_memmap
import meep as mp
from scipy.signal import butter, lfilter

def butterBandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butterBandpassFilter(data, lowcut, highcut, fs, order=3):
    b, a = butterBandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y
def get_data(fname_txt):
    n_data = np.genfromtxt(fname_txt)
    zprof = n_data[:,0]
    nprof = n_data[:,1]
    return nprof, zprof

def findNearest(x_arr, x):
    dx = abs(x_arr - x)
    ii = np.argmin(dx)
    return ii


def select_rx_ii(rxList, x_in, z_in):
    dr_list = []
    for rx in rxList:
        x_rx = float(rx[0])
        z_rx = float(rx[1])
        dr_sq = (x_rx-x_in)**2 + (z_rx - z_in)**2
        dr = np.sqrt(dr_sq)
        dr_list.append(dr)
    ii_rx = np.argmin(dr_list)
    return ii_rx

def get_actual_rx(rxList, x_in, z_in):
    ii_rx = select_rx_ii(rxList, x_in, z_in)
    rx_actual = rxList[ii_rx]
    x_rx = float(rx_actual[0])
    z_rx = float(rx_actual[1])
    return x_rx, z_rx

def get_pulse(rxPulses, rxList, x_in, z_in):
    ii_rx = select_rx_ii(rxList, x_in, z_in)
    rx_pulse = rxPulses[ii_rx]
    return rx_pulse

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