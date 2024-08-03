import numpy as np
import math
import h5py
import sys
from sys import argv, exit
import configparser
import os
from matplotlib import pyplot as pl
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

def findNearest(x, xi):
    return np.argmin(abs(x-xi))

def do_IR0(data, tspace, IR_spec, IR_freq):
    spec0 = np.fft.fft(data)
    nSamples = len(tspace)
    dt = tspace[1] - tspace[0]
    fspace0 = np.fft.fftfreq(nSamples, dt)
    spec_shift = np.fft.fftshift(spec0)
    freq_shift = np.fft.fftshift(fspace0)
    nMid = findNearest(freq_shift, 0)
    freq_shift_pos = freq_shift[nMid:]
    freq_shift_neg = -1 * np.flip(freq_shift[:nMid])

    nFreq_pos = len(freq_shift_pos)
    nFreq_neg = len(freq_shift_neg)
    IR_arr = np.zeros(nSamples)
    IR_pos = np.zeros(nFreq_pos)
    IR_neg = np.zeros(nFreq_neg)
    for i in range(nFreq_pos):
        freq_plus = freq_shift_pos[i]
        #freq_neg = freq_shift_neg[j_neg]
        if freq_plus < min(IR_freq):
            IR_pos[i] = 0
        elif freq_plus > max(IR_freq):
            IR_pos[i] = 0
        elif freq_plus <= max(IR_freq) and freq_plus >= min(IR_freq):
            k_pos = findNearest(IR_freq, freq_plus)
            IR_pos[i] = IR_spec[k_pos]
    for i in range(nFreq_neg):
        freq_neg = freq_shift_neg[i]
        j_neg = findNearest(freq_shift_neg, freq_neg)
        if freq_plus < min(IR_freq):
            IR_neg[i] = 0
        elif freq_plus > max(IR_freq):
            IR_neg[i] = 0
        elif freq_plus <= max(IR_freq) and freq_plus >= min(IR_freq):
            k_neg = findNearest(IR_freq, freq_plus)
            IR_neg[i] = IR_spec[k_neg]
    IR_arr[nMid:] = IR_pos
    IR_arr[:nMid] = np.flip(IR_neg)
    spec_shift *= IR_arr
    spec_out = np.fft.ifftshift(spec_shift)
    data_out = np.fft.ifft(spec_out)
    return data_out, spec_shift, IR_arr

def do_IR(data, tspace, IR_spec, IR_freq):
    spec0 = np.fft.fft(data)
    nSamples = len(tspace)
    dt = tspace[1] - tspace[0]
    fspace0 = np.fft.fftfreq(nSamples, dt)
    spec_shift = np.fft.fftshift(spec0)
    freq_shift = np.fft.fftshift(fspace0)
    nMid = findNearest(freq_shift, 0)
    freq_shift_pos = freq_shift[nMid:]
    freq_shift_neg = -1 * np.flip(freq_shift[:nMid])

    nFreq_pos = len(freq_shift_pos)
    nFreq_neg = len(freq_shift_neg)
    IR_arr = np.zeros(nSamples)
    IR_pos = np.zeros(nFreq_pos)
    IR_neg = np.zeros(nFreq_neg)
    fmin = min(IR_freq)
    fmax = max(IR_freq)
    IR_pos = np.interp(freq_shift_pos, IR_freq, IR_spec)
    IR_neg = np.interp(freq_shift_neg, IR_freq, IR_spec)

    IR_arr[nMid:] = IR_pos
    IR_arr[:nMid] = np.flip(IR_neg)
    spec_shift *= IR_arr
    spec_out = np.fft.ifftshift(spec_shift)
    data_out = np.fft.ifft(spec_out)
    return data_out, spec_shift, IR_arr

def get_pulse_from_file(fname_hdf, x, z, pol_mode='None'):
    with h5py.File(fname_hdf,'r') as hdf_in:
        if pol_mode == 'None':
            rxPulses = np.array(hdf_in['rxPulses'])
        elif pol_mode == 'zpol':
            rxPulses = np.array(hdf_in['rxPulses_z'])
        elif pol_mode == 'rpol':
            rxPulses = np.array(hdf_in['rxPulses_r'])
        else:
            print('error! enter polarizaiton mode: None, zpol or hpol')
            return -1
        rxList = np.array(hdf_in['rxList'])
        tspace0 = np.array(hdf_in['tspace'])
        nRx = len(rxList)

    dr_list = np.zeros(nRx)
    for i in range(nRx):
        x_rx_i = rxList[i, 0]
        z_rx_i = rxList[i, 1]
        dx = abs(x_rx_i - x)
        dz = abs(z_rx_i - z)
        dr_sq = dx ** 2 + dz ** 2
        dr = np.sqrt(dr_sq)
        dr_list[i] = dr
    ii_rx = np.argmin(dr_list)

    x_actual = rxList[ii_rx, 0]
    z_actual = rxList[ii_rx, 1]

    dt0 = tspace0[1] - tspace0[0]
    dt = dt0 * 3.728
    t_max0 = max(tspace0)
    t_max = t_max0 * 3.728
    nSamples = len(tspace0)
    tspace = np.linspace(0, t_max, nSamples)

    if (z_actual - z == 0) and (x_actual - x == 0):
        print('Match, RX found at (', x_actual,',', z_actual, ')')
    else:
        print('Warning!, mismatch between RX input and RX selected')
        print('Input RX: (', x, ',', z, '), Actual RX: (', x_actual, ',', z_actual, ')')
    rxPulse_out = rxPulses[ii_rx]
    return rxPulse_out, tspace

def get_true_rx(fname_hdf, x, z):
    with h5py.File(fname_hdf,'r') as hdf_in:
        rxList = np.array(hdf_in['rxList'])
        nRx = len(rxList)

    dr_list = np.zeros(nRx)
    for i in range(nRx):
        x_rx_i = rxList[i, 0]
        z_rx_i = rxList[i, 1]
        dx = abs(x_rx_i - x)
        dz = abs(z_rx_i - z)
        dr_sq = dx ** 2 + dz ** 2
        dr = np.sqrt(dr_sq)
        dr_list[i] = dr
    ii_rx = np.argmin(dr_list)

    x_actual = rxList[ii_rx, 0]
    z_actual = rxList[ii_rx, 1]
    return x_actual, z_actual

def cut_arr(y, x, xmin, xmax):
    imin = findNearest(x, xmin)
    imax = findNearest(x, xmax)
    return y[imin:imax]

def get_tspace_from_file(fname_hdf):
    with h5py.File(fname_hdf, 'r') as hdf_in:
        tspace = np.array(hdf_in['tspace'])
    tspace  *= 3.728
    return tspace
def get_rxList_from_file(fname_hdf):
    with h5py.File(fname_hdf, 'r') as hdf_in:
        rxList = np.array(hdf_in['rxList'])
    return rxList

def get_sourceDepth_from_file(fname_hdf):
    with h5py.File(fname_hdf, 'r') as hdf_in:
        sourceDepth = float(hdf_in.attrs['sourceDepth'])
    return sourceDepth

def get_rxPulses_from_file(fname_hdf):
    with h5py.File(fname_hdf, 'r') as hdf_in:
        rxPulses = np.array(hdf_in['rxPulses'])
    return rxPulses

def get_nProfiles_from_file(fname_hdf):
    with h5py.File(fname_hdf, 'r') as hdf_in:
        nProfile = np.array(hdf_in['nProfile'])
        zProfile = np.array(hdf_in['zProfile'])
    return nProfile, zProfile
def get_spec_from_file(fname_hdf, x, z):
    with h5py.File(fname_hdf,'r') as hdf_in:
        rxPulses = np.array(hdf_in['rxPulses'])
        rxList = np.array(hdf_in['rxList'])
        tspace = np.array(hdf_in['tspace'])
        nRx = len(rxList)

    dr_list = np.zeros(nRx)
    for i in range(nRx):
        x_rx_i = rxList[i, 0]
        z_rx_i = rxList[i, 1]
        dx = abs(x_rx_i - x)
        dz = abs(z_rx_i - z)
        dr_sq = dx ** 2 + dz ** 2
        dr = np.sqrt(dr_sq)
        dr_list[i] = dr
    ii_rx = np.argmin(dr_list)

    x_actual = rxList[ii_rx, 0]
    z_actual = rxList[ii_rx, 1]

    if (z_actual - z == 0) and (x_actual - x == 0):
        print('Match, RX found at (', x_actual, ',', z_actual, ')')
    else:
        print('Warning!, mismatch between RX input and RX selected')
        print('Input RX: (', x, ',', z, '), Actual RX: (', x_actual, ',', z_actual, ')')
    rxPulse_out = rxPulses[ii_rx]
    nSamples = len(rxPulse_out)
    dt = (tspace[1] - tspace[0]) * 3.728

    rxPulse_out = butterBandpassFilter(rxPulse_out, 0.05, 0.3, fs=1/dt)
    rxSpec_out = np.fft.rfft(rxPulse_out)
    rxSpec_out /= float(nSamples)
    fspace = np.fft.rfftfreq(nSamples, dt)
    return rxSpec_out, fspace

def butterHighpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = sig.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butterHighpassFilter(data, cutoff, fs, order=5):
    b, a = butterHighpass(cutoff, fs, order=order)
    y = sig.filtfilt(b, a, data)
    return y

def butterLowpass(highcut, fs, order=5):
    nyq = 0.5 * fs
    high = highcut / nyq
    b, a = butter(order, high, btype='low')
    return b, a


def butterLowpassFilter(data, highcut, fs, order=3):
    b, a = butterLowpass(highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y
