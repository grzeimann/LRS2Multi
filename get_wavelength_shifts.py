#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 13:52:44 2023

@author: gregz
"""
import numpy as np
import argparse as ap
import logging
import warnings
import tables
import sys
from astropy.table import Table
sys.path.append("..") 

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from skimage.registration import phase_cross_correlation
from astropy.modeling.models import Gaussian1D
from astropy.modeling.fitting import LevMarLSQFitter

from virusraw import VIRUSRaw


def setup_logging(logname='input_utils'):
    '''Set up a logger for shuffle with a name ``input_utils``.

    Use a StreamHandler to write to stdout and set the level to DEBUG if
    verbose is set from the command line
    '''
    log = logging.getLogger('input_utils')
    if not len(log.handlers):
        fmt = '[%(levelname)s - %(asctime)s] %(message)s'
        fmt = logging.Formatter(fmt)

        level = logging.INFO

        handler = logging.StreamHandler()
        handler.setFormatter(fmt)
        handler.setLevel(level)

        log = logging.getLogger('input_utils')
        log.setLevel(logging.DEBUG)
        log.addHandler(handler)
    return log

warnings.filterwarnings("ignore")


parser = ap.ArgumentParser(add_help=True)

parser.add_argument('object_table', type=str,
                    help='''name of the output file''')

parser.add_argument('hdf5file', type=str,
                    help='''name of the hdf5 file''')


args = parser.parse_args(args=None)
args.log = setup_logging('get_object_table')

def_wave = np.linspace(3470, 5540, 1036)

sns.set_context('talk')
sns.set_style('ticks')
plt.rcParams["font.family"] = "Times New Roman"

basedir = '/work/03946/hetdex/maverick'
hdf5file = args.hdf5file
h5file = tables.open_file(hdf5file, mode='r')
h5table = h5file.root.Cals
ifuslots = list(np.unique(['%03d' % i for i in h5table.cols.ifuslot[:]]))
ifuslots = ifuslots[3:4]
T = Table.read(args.object_table, format='ascii.fixed_width_two_line')

keys = list([str(t) for t in T['Exposure']])
values = list(T['Description'])

twi_obs = [key for key, value in zip(keys, values) if value == 'skyflat']
CdA_obs = [key for key, value in zip(keys, values) if value == 'Cd-A']
Hg_obs = [key for key, value in zip(keys, values) if value == 'Hg']
Dark_obs = [key for key, value in zip(keys, values) if value == 'dark']
LDLS_obs = [key for key, value in zip(keys, values) if value == 'ldls_long']


line_list = [3610.508, 4046.565, 4358.335, 4678.149, 4799.912,
                      4916.068, 5085.822, 5460.750]

fit_waves = [np.abs(def_wave - line) <=40. for line in line_list]

CdA_list = []
for arc in CdA_obs:
    date = arc[:8]
    obs = int(arc[8:15])
    exp = int(arc[15:])
    virus = VIRUSRaw(date, obs, h5table, basepath=basedir, exposure_number=exp,
                     ifuslots=ifuslots)
    for ifuslot in ifuslots:  
        monthly_average = virus.info[ifuslot].lampspec * 1.
        current_observation = virus.info[ifuslot].orig * 1.
        current_observation[np.isnan(current_observation)] = 0.0
        shifts = np.zeros((current_observation.shape[0],))
        for fiber in np.arange(current_observation.shape[0]):
            FFT = phase_cross_correlation(current_observation[fiber, :][np.newaxis, :],
                                          monthly_average[fiber, :][np.newaxis, :], 
                                          normalization=None, upsample_factor=100)
            shifts[fiber] = FFT[0][1]
        print(shifts)