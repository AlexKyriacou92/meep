import meep as mp
import numpy as np
import math
import h5py
import cmasher as cmr
import sys

from data import meep_field

fname_in = 'config_in.txt'
meep_field_sp = meep_field()

def southpole(z):
    n_surf = 1.3
    n_ice = 1.78
    k = -0.0132

    dn = n_surf-n_ice
    return n_ice + dn*np.exp(k*z)

meep_field_sp.setup_from_config(fname_config=fname_in, nFunc=southpole)
fname_out = 'southpole_func_freq=' + str(meep_field_sp.frequency) + 'MHz_ztx=' + str(meep_field_sp.sourceDepth) + '.h5'
meep_field_sp.run_simulation(fname_out=fname_out, label='southpole_func')