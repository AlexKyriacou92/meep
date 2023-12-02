# GPL v3

import numpy as np
import scipy as sci
import scipy.constants as constant
import scipy.io.wavfile as wav
import scipy.signal as sig
import scipy.interpolate as interp
from scipy.signal import butter, lfilter
from numpy import linalg as la
import csv
from numpy.lib.format import open_memmap
import h5py
from scipy.interpolate import interp1d
from scipy.signal import decimate
import scipy


I = 1.j
c_light = .29979246;  # m/ns
c0 = c_light * 1e9

pi = 3.14159265358979323846;  # radians
twoPi = 2. * pi;  # radians
z_0 = 50;  # ohms
deg = pi / 180.;  # radians
kB = 8.617343e-11;  # MeV/kelvin
kBJoulesKelvin = 1.38e-23;  # J/kelvin
rho = 1.168e-3;  # sea level density
x_0 = 36.7;  # radiation length in air
e_0 = .078;  # ionization energy

m = 1.;
ft = .3047 * m;
cm = .01 * m;
mm = .001 * m;

ns = 1.;
us = ns * 1e3;
ms = ns * 1e6;
s = ns * 1e9;

GHz = 1.;
MHz = .001 * GHz;
kHz = 1e-6 * GHz;
Hz = 1e-9 * GHz;

def power(V, start, end):
    powV=V*V
    return np.sum(powV[start:end])

def doFFT(V):
    return np.fft.fft(V)

def doIFFT(V):
    return np.fft.ifft(V)

def findNearest(array, value):
    array = np.asarray(array, dtype='complex')
    idx = (np.abs(array - value)).argmin()
    return idx#array[idx]