#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 15:22:15 2022

@author: gregz
"""

from astropy.io import fits
import numpy as np
from datetime import datetime, timedelta
import glob
import os.path as op
from scipy.interpolate import interp1d
import sys

def get_script_path():
    ''' Get LRS2Multi absolute path name '''
    return op.dirname(op.realpath(sys.argv[0]))

channels = ['uv', 'orange', 'red', 'farred']
date = '20220615'
ndays = 30
folder = '/work/03946/hetdex/maverick/LRS2/CALS'
date_ = datetime(int(date[:4]), int(date[4:6]), int(date[6:]))
datel = date_ - timedelta(days=int(ndays/2))
def_name = 'cal_%s_%s.fits'
basefile = '/work/03946/hetdex/maverick/LRS2/ENG22-2-004/multi_20220421_0000012_exp01_%s.fits'

for channel in channels:
    wavelength, masterbias, trace, masterflt = ([], [], [], [])
    for i in np.arange(ndays):
        ndate = datel + timedelta(days=float(i))
        daten = '%04d%02d%02d' % (ndate.year, ndate.month, ndate.day)
        all_names = sorted(glob.glob(op.join(folder, def_name % (daten, channel))))
        if len(all_names) < 1:
            continue
        f = fits.open(all_names[0])
        x = f['xypos'].data[:, 0]
        y = f['xypos'].data[:, 1]
        wavelength.append(f['wavelength'].data)
        trace.append(f['trace'].data)
        masterbias.append(f['masterbias'].data)
        masterflt.append(f['masterFlat'].data)
        norm = f['response'].data[1]
        def_wave = f['response'].data[0]
    g = fits.open(basefile % channel)
    J = interp1d(g[7].data[0], g[7].data[1], fill_value='extrapolate',
                   bounds_error=False)
    K = interp1d(g[7].data[0], g[7].data[2], fill_value='extrapolate',
                   bounds_error=False)
    adrx = J(def_wave)
    adry = K(def_wave)
    hdulist = fits.HDUList([fits.PrimaryHDU(np.median(wavelength, axis=0)),
                            fits.ImageHDU(np.median(trace, axis=0)),
                            fits.ImageHDU(np.median(masterbias, axis=0)),
                            fits.ImageHDU(np.median(masterflt, axis=0)),
                            fits.ImageHDU(x),
                            fits.ImageHDU(y),
                            fits.ImageHDU(norm),
                            fits.ImageHDU(def_wave),
                            fits.ImageHDU(adrx),
                            fits.ImageHDU(adry)])
    names = ['wavelength', 'trace', 'masterbias', 'masterflt', 'x', 'y',
             'norm', 'def_wave', 'adrx', 'adry']
    for hdu, name in zip(hdulist, names):
        hdu.header['EXTNAME'] = name
    hdulist.writeto(op.join(get_script_path(), 'calibrations', 
                            'cal_%s.fits' % channel), overwrite=True)
    