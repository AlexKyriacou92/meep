import meep as mp
import numpy as np
import math
import h5py
import cmasher as cmr
import sys
import configparser

import util

if len(sys.argv) == 2:
    fname_config = sys.argv[1]
    fname_nprofile = sys.argv[2]
else:
    fname_config = 'config_in.txt'
    fname_nprofile = 'spice2019_indOfRef_core1_5cm.txt'

config = configparser.ConfigParser()
config.read(fname_config)

geometry = config['GEOMETRY']
transmitter = config['TRANSMITTER']
settings = config['SETTINGS']