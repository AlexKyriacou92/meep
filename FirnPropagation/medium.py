import numpy as np
import matplotlib.pyplot as pl

def get_profile_from_txt(fname_txt):
    profile_data = np.genfromtxt(fname_txt)
    z_profile = profile_data[:,0]
    n_profile = profile_data[:,1]
    return n_profile, z_profile