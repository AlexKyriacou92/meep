import numpy as np
import math
import h5py
import cmasher as cmr
import sys

from matplotlib import pyplot as pl

if len(sys.argv) == 3:
    year_in = float(sys.argv[1])
    month_in = float(sys.argv[2])
else:
    print('error, you need to input: python', sys.argv[0], 'year month')
    sys.exit()

fname_nprof_hdf = 'nProf_CFM_deep2.h5'
with h5py.File(fname_nprof_hdf,'r') as nprof_hdf:
    '''
    for key in nprof_hdf.keys():
        print(key)
    '''
    n_profile_matrix = np.array(nprof_hdf['n_profile_matrix'])
    z_profile = np.array(nprof_hdf['z_profile'])
n_profile_matrix = n_profile_matrix[1:]
nProfiles = len(n_profile_matrix)
nDepths = len(n_profile_matrix[0])

with h5py.File('CFMresults.hdf5', 'r') as CFM_hdf:
    '''
    for key in CFM_hdf.keys():
        print(key)
    '''
    density_arr = np.array(CFM_hdf['density'])
print(density_arr[:,0])

date_list = density_arr[1:,0]
#print(density_arr[0,:])

print(len(date_list), nProfiles)
year_i = year_in
month_i = month_in

date_num_i = year_i + month_i/12.

def findNearest(x_arr, x):
    index = np.argmin(abs(x_arr - x))
    return index

def create_profile(nprof, zprof, fname):
    nDepths = len(zprof)
    with open(fname, 'w') as fout:
        for i in range(nDepths):
            z_i = zprof[i]
            n_i = nprof[i]
            line = str(z_i) + '\t' + str(n_i) + '\n'
            fout.write(line)

ii_date = findNearest(date_list, date_num_i)
print(year_i, month_i, date_num_i, date_list[ii_date])

n_profile_select = n_profile_matrix[ii_date]

'''
fig = pl.figure(figsize=(8,5),dpi=120)
ax = fig.add_subplot(111)
ax.plot(z_profile, n_profile_select, label=str(year_i) + ' ' + str(month_i))
ax.grid()
ax.set_ylabel('Ref Index n')
ax.set_xlabel('Depth z [m]')
ax.legend()
pl.show()
'''

fname_out = 'nProf_CFM_' + str(year_i) + '_' + str(month_i) + '.txt'
create_profile(n_profile_select, z_profile, fname_out)