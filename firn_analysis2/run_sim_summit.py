import meep as mp
import numpy as np
import math
import h5py
import cmasher as cmr
import sys
import configparser

from data import meep_field, get_profile_from_data, findNearest
fname_in = 'config_summit.txt'
config = configparser.ConfigParser()
config.read(fname_in)
geometry = config['GEOMETRY']
iceDepth = float(geometry['iceDepth'])

nSims = 10

def get_dates(fname_in):
    with  h5py.File(fname_in, 'r') as input_hdf:
        data_matrix = np.array(input_hdf['density'])
    date_arr = data_matrix[1:,0]
    return date_arr

def ym2date(year, month):
    return year + month/12.

def date2ym(date):
    year = int(date)
    month = round(date%year * 12)
    return year, month


fname_config = 'config_ICRC_summit_km.txt'
fname_nprofile = 'nProf_CFM_deep2.h5'
fname_CFM = 'CFMresults.hdf5'

date_arr = get_dates(fname_CFM)


nprofile_hdf = h5py.File(fname_nprofile, 'r')
nprof_mat = np.array(nprofile_hdf.get('n_profile_matrix'))
zprof_mat = np.array(nprofile_hdf.get('z_profile'))
nDepths0 = len(zprof_mat)
nProfiles = len(nprof_mat)


ii_cut = findNearest(zprof_mat, 100)
zprof_cut = zprof_mat[:ii_cut]
nDepths = len(zprof_cut)
nprof_in = np.ones((nProfiles, nDepths))
nprof_mat = nprof_mat[:,:ii_cut]

nprof_in = nprof_in

start_year = 2011
end_year = int(date_arr[-1]) + 1
year_list = np.arange(start_year, end_year, 1)
year_id_list = []
print(year_list)
nYears = len(year_list)
fname_list = []
n_matrix_yr = np.ones((nYears, nDepths))
for i in range(nYears):
    jj = findNearest(date_arr, year_list[i])
    year_id_list.append(jj)
nProfiles_sim = nYears

for i in range(nProfiles_sim):
    meep_field_i = meep_field()
    ii_select = year_id_list[i]
    meep_field_i.setup_from_config(fname_config=fname_config, nVec=nprof_mat[jj], zVec=zprof_cut)