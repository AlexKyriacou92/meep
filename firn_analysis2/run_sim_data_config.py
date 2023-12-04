import meep as mp
import numpy as np
import math
import h5py
import cmasher as cmr
import sys

from data import meep_field, get_profile_from_data

fname_in = 'config_in.txt'
fname_data = 'spice2019_indOfRef_core1_5cm.txt'
nprof_sp, zprof_sp = get_profile_from_data(fname_data=fname_data)

meep_field_sp = meep_field()
meep_field_sp.setup_from_config(fname_config=fname_in, nVec=nprof_sp, zVec=zprof_sp)

fname_out = 'spice2019_indOfRef_core1_5cm_freq=' + str(meep_field_sp.frequency) + 'MHz_ztx=' + str(meep_field_sp.sourceDepth) + '.h5'
meep_field_sp.run_simulation(fname_out, 'spice_core')