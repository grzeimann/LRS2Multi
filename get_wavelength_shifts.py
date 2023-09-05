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
import seaborn as sns
from skimage.registration import phase_cross_correlation
from astropy.time import Time
from astropy.io import fits
from multiprocessing import Pool

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

parser.add_argument('outname', type=str,
                    help='''name appended to shift output''')
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
ifuslots = ifuslots
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

thresh = 150.

arc_list = CdA_obs + Hg_obs

def get_shift(arc):
    date = arc[:8]
    obs = int(arc[8:15])
    exp = int(arc[15:])
    virus = VIRUSRaw(date, obs, h5table, basepath=basedir, exposure_number=exp,
                     ifuslots=ifuslots)
    shift_dictionary = {}
    for ifuslot in ifuslots:
        shift_dictionary[ifuslot] = np.nan * np.ones((448, len(line_list)))
    for ifuslot in ifuslots:  
        monthly_average = virus.info[ifuslot].lampspec * 1.
        current_observation = virus.info[ifuslot].orig * 1.
        current_observation[np.isnan(current_observation)] = 0.0
        shifts = np.ones((current_observation.shape[0], len(line_list))) * np.nan
        for fiber in np.arange(current_observation.shape[0]):
            fit_waves = [np.abs(virus.info[ifuslot].wavelength[fiber] - line) <= 20. for line in line_list]
            for j, waverange in enumerate(fit_waves):
                if np.nanmax(current_observation[fiber, waverange]) > thresh:
                    FFT = phase_cross_correlation(current_observation[fiber, waverange][np.newaxis, :],
                                                  monthly_average[fiber, waverange][np.newaxis, :], 
                                                  normalization=None, upsample_factor=100)
                    shifts[fiber, j] = FFT[0][1]
        shift_dictionary[ifuslot] = shifts
    timeobs = Time(virus.info[ifuslot].header['DATE'])
    hum = virus.info[ifuslot].header['HUMIDITY']
    temp = virus.info[ifuslot].header['AMBTEMP']
    virus.log.info('Shifts finished %s_%07d_exp%02d' % (date, obs, exp))
    return shift_dictionary, timeobs, hum, temp


P = Pool(16)
res = P.map(get_shift, arc_list)
P.close()
shift_dictionary = {}
for ifuslot in ifuslots:
    shift_dictionary[ifuslot] = np.nan * np.ones((len(arc_list), 448, len(line_list)))
for ifuslot in ifuslots:  
    shift_dictionary[ifuslot] = [r[0][ifuslot] for r in res]
time_list = [r[1] for r in res]
hum_list = [r[2] for r in res]
temp_list = [r[3] for r in res]
for ifuslot in ifuslots:
    name = 'wavelength_shifts_%s_%s.fits' % (ifuslot, args.outname)
    f  = fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(shift_dictionary[ifuslot]),
                       fits.ImageHDU(np.array([t.mjd for t in time_list])),
                       fits.ImageHDU(np.array([t for t in hum_list])),
                       fits.ImageHDU(np.array([t for t in temp_list]))])
    f.writeto(name, overwrite=True)
