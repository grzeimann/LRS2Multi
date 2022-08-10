#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 09:48:01 2022

@author: gregz
"""
import logging
import numpy as np
import os.path as op
import warnings
import glob
from astropy.io import fits
from fiber_utils import base_reduction, rectify
from fiber_utils import get_spectra_error, get_spectra, get_spectra_chi2

import tarfile
import sys

warnings.filterwarnings("ignore")

def get_script_path():
    ''' Get LRS2Multi absolute path name '''
    return op.dirname(op.realpath(sys.argv[0]))

class LRS2Raw:
    ''' 
    Wrapper for reduction routines for raw data 
    
    '''
    def __init__(self, basepath, date, observation_number, exposure_number=1):
        '''
        

        Parameters
        ----------
        basepath : TYPE
            DESCRIPTION.
        date : TYPE
            DESCRIPTION.
        observation_number : TYPE
            DESCRIPTION.
        exposure_number : TYPE, optional
            DESCRIPTION. The default is 1.

        Returns
        -------
        None.

        '''
        self.setup_logging()
        side_dict = {'blue': ['uv', 'orange'], 'red': ['red', 'farred']}
        channel_dict = {'uv': ['056LL', '056LU'], 
                        'orange': ['056RU', '056RL'],
                        'red': ['066LL', '066LU'],
                        'farred': ['066RU', '066RL']}
        tarfolder = op.join(basepath, date, 'lrs2', 
                                 'lrs2%07d.tar' % observation_number)
        path = op.join(basepath, date, 'lrs2', 'lrs2%07d' % observation_number,
                       'exp%02d' % exposure_number)
        expstr = 'exp%02d' % exposure_number
        ampcase = '056LL.fits'
        self.log.info('Looking for %s and %s' % (expstr, ampcase))
        if op.exists(tarfolder):
            self.log.info('Found tarfile %s' % tarfolder)
            T = tarfile.open(tarfolder, 'r')
            flag = True
            gotfile = False
            while flag:
                try:
                    a = T.next()
                    name = a.name
                except:
                    flag = False
                    continue  
                if (expstr in name) and (ampcase in name):
                    b = fits.open(T.extractfile(a))
                    flag = False
                    gotfile = True
                    T.close()
            if not gotfile:
                self.log.error('No files found for %s' % path)
                sys.exit('Cowardly exiting; please check input')
            filename = name
        else:
            tarfolder = None
            filenames = glob.glob(op.join(path, '*056LL.fits'))
            if len(filename) < 1:
                self.log.error('No files found here %s' % path)
                sys.exit('Cowardly exiting; please check input')
            filename = filenames[0]
            b = fits.open(filename)
        Target = b[0].header['OBJECT']
        if '_056' in Target:
            side = 'blue'
        if '_066' in Target:
            side = 'red'
        self.side = side
        self.info = {}
        for channel in side_dict[self.side]:
            self.info[channel] = self.ChannelInfo(channel)
            self.reduce_channel(filename, channel_dict[channel], channel)
    
    class ChannelInfo:
        
        # Create channel info
        def __init__(self, channel):
            f = fits.open(op.join(get_script_path(), 'calibrations',
                                  'cal_%s.fits' % channel))
            self.wavelength = f['wavelength'].data
            self.masterbias = f['masterbias'].data
            self.trace = f['trace'].data
            self.masterflt = f['masterflt'].data
            self.def_wave = f['def_wave'].data
            self.x = f['x'].data
            self.y = f['y'].data
            self.adrx = f['adrx'].data
            self.adry = f['adry'].data
            self.norm = f['norm'].data
    
    def setup_logging(self, logname='lrs2raw'):
        '''Set up a logger for shuffle with a name ``lrs2 advanced``.

        Use a StreamHandler to write to stdout and set the level to DEBUG if
        verbose is set from the command line
        '''
        log = logging.getLogger(logname)
        if not len(log.handlers):
            fmt = '[%(levelname)s - %(asctime)s] %(message)s'
            fmt = logging.Formatter(fmt)

            level = logging.INFO

            handler = logging.StreamHandler()
            handler.setFormatter(fmt)
            handler.setLevel(level)

            log = logging.getLogger(logname)
            log.setLevel(logging.DEBUG)
            log.addHandler(handler)
        self.log = log
        self.log.propagate = False

    
    def reduce_channel(self, filename, amp, channel, tarname=None):
        '''
        

        Parameters
        ----------
        filename : TYPE
            DESCRIPTION.
        amp : TYPE
            DESCRIPTION.
        masterbias : TYPE
            DESCRIPTION.
        tarname : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        '''
        
        filename1 = filename.replace('056LL', amp[0])
        filename2 = filename.replace('056LL', amp[1])
        array_flt1, e1, header = base_reduction(filename1, get_header=True)
        array_flt2, e2 = base_reduction(filename2)
        image = np.vstack([array_flt1, array_flt2])
        E = np.vstack([e1, e2])
        image[:] -= self.info[channel].masterbias
        spec = get_spectra(image, self.info[channel].trace)
        specerr = get_spectra_error(E, self.info[channel].trace)
        chi2 = get_spectra_chi2(self.info[channel].masterflt, image, E, 
                                self.info[channel].trace)
        badpix = chi2 > 10.
        specerr[badpix] = np.nan
        spec[badpix] = np.nan
        specrect, errrect = rectify(spec, specerr, 
                                    self.info[channel].wavelength,
                                    self.info[channel].def_wave)
        specrect[:] /= header['EXPTIME']
        errrect[:] /= header['EXPTIME']
        self.info[channel].data = specrect
        self.info[channel].datae = errrect