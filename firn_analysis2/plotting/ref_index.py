import os.path

import meep as mp
import numpy as np
import math
import h5py
import matplotlib.pyplot as pl
import sys

nArgs = len(sys.argv)
fname_list = []
if nArgs == 1:
    print('error, must input txt file defining ref-index as an arg:')
    sys.exit()
elif nArgs == 2:
    fname_in = sys.argv[1]
    fname_list.append(fname_in)
elif len(sys.argv) > 2:
    for i in range(1, nArgs):
        fname_i = sys.argv[i]
        fname_list.append(fname_i)

nProfiles = len(fname_list)

def get_profile(fname_n):
    data = np.genfromtxt(fname_n)
    n_profile = data[:,1]
    z_profile = data[:,0]
    return n_profile, z_profile

fig = pl.figure(figsize=(6,10), dpi=120)
ax1 = fig.add_subplot(111)
for i in range(nProfiles):
    fname_i = fname_list[i]
    n_prof_i, z_prof_i = get_profile(fname_i)
    fname_label = os.path.basename(fname_i)
    ax1.plot(n_prof_i, z_prof_i, label=fname_label)
ax1.grid()
ax1.set_xlabel('Ref Index n(z)')
ax1.set_ylabel('Depth z [m]')
z_max = max(z_prof_i)
z_min = -10
ax1.set_ylim(z_max, z_min)
ax1.legend()
fig.savefig('n_prof_full.png')
pl.show()

fig = pl.figure(figsize=(6,10), dpi=120)
ax1 = fig.add_subplot(111)
for i in range(nProfiles):
    fname_i = fname_list[i]
    n_prof_i, z_prof_i = get_profile(fname_i)
    fname_label = os.path.basename(fname_i)
    ax1.plot(n_prof_i, z_prof_i, label=fname_label)
ax1.grid()
ax1.set_xlabel('Ref Index n(z)')
ax1.set_ylabel('Depth z [m]')
z_max = 25.
z_min = -2
ax1.set_ylim(z_max, z_min)
ax1.set_xlim(1.25, 1.6)

ax1.legend()
fig.savefig('n_prof_zoom.png')
pl.show()

if nProfiles > 1:
    fig = pl.figure(figsize=(12, 10), dpi=120)
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    n_prof_list = []
    for i in range(nProfiles):
        fname_i = fname_list[i]
        n_prof_i, z_prof_i = get_profile(fname_i)
        fname_label = os.path.basename(fname_i)
        ax1.plot(n_prof_i, z_prof_i, label=fname_label)
        n_prof_list.append(n_prof_i)
    ax1.grid()
    ax1.set_xlabel('Ref Index n(z)')
    ax1.set_ylabel('Depth z [m]')
    z_max = 25.
    z_min = -2
    ax1.set_ylim(z_max, z_min)
    ax1.set_xlim(1.25, 1.6)
    ax1.legend()

    n_prof_0 = n_prof_list[0]
    for i in range(1, nProfiles):
        n_prof_i = n_prof_list[i]
        dn_prof_i = n_prof_i - n_prof_0
        fname_i = fname_list[i]
        fname_label = os.path.basename(fname_i)
        ax2.plot(dn_prof_i, z_prof_i,label=fname_label)
    ax2.grid()
    ax2.set_xlabel(r'Ref Index Residuals $\Delta n(z)$')
    ax2.set_ylabel('Depth z [m]')
    ax2.set_ylim(z_max, z_min)
    fig.savefig('n_prof_compare.png')
    pl.show()