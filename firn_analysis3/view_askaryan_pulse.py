import numpy as np
import math
import h5py
import sys
from sys import argv, exit
import configparser

import pylab as pl
from scipy.signal import butter, lfilter
import os

from math import pi
from numpy import rad2deg, deg2rad, sin, cos, sqrt, log, exp, log10
import scipy.signal as sig

from Askaryan_Signal import create_pulse, TeV

fname_config = sys.argv[1]

config_file = os.path.join(os.path.dirname(__file__), fname_config)
config = configparser.ConfigParser()
config.read(config_file)

source = config['SOURCE']
Esh_eV = float(source['showerEnergy'])
sourceDepth = float(source['sourceDepth']) #depth of diple below ice

Esh_TeV = 1e18/TeV # Energy of Neutrino-Induced Shower 10^18 eV  TODO: Make Config Controlled
dtheta_nu = float(source['dthetaV']) # Offset Angle between v and c TODO: Make Config Controlled
R_alpha = float(source['attenuationEquivalent']) #Distance used to define 'pre-attenuation' -> filters out high frequencies
t_start_us = 50/1e3

#Define Askaryan Pulse at Source
pulse_in, tspace_in_ns = create_pulse(Esh=Esh_TeV, dtheta_v=dtheta_nu,R_alpha=R_alpha, t_min=-t_start_us, t_max=500e-3-t_start_us,z_alpha=sourceDepth)
spec_in = np.fft.rfft(pulse_in)
freq_space = np.fft.rfftfreq(len(pulse_in), tspace_in_ns[1]-tspace_in_ns[0])

ii_max = np.argmax(abs(spec_in))
freq_amp_max = freq_space[ii_max]
print('Max frequency, f_max=',freq_amp_max*1e3, 'MHz')

fig = pl.figure(figsize=(12,6),dpi=120)
ax = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
fig.suptitle(r'$|\theta_{v}-\theta_{c}| = $' + str(dtheta_nu) + ' [deg], $E_{sh}= 10^{' + str(np.log10(Esh_eV)) + '}$ eV')

ax.set_title(r'Pulse')
ax.plot(tspace_in_ns, pulse_in)
ax.set_xlabel('Time t [ns]')
ax.set_ylabel('E(t) [V/m]')
ax.grid()

ax2.set_title(r'Spectrum')
ax2.plot(freq_space, abs(spec_in))
ax2.set_xlabel('Frequency f [GHz]')
ax2.set_ylabel('Amplitude [V/m/GHz]')
ax2.grid()
pl.show()