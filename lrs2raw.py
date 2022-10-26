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
from fiber_utils import base_reduction, rectify, get_powerlaw
from fiber_utils import get_spectra_error, get_spectra, get_spectra_chi2

import tarfile
import sys

warnings.filterwarnings("ignore")

def get_script_path():
    ''' Get LRS2Multi absolute path name '''
    return op.dirname(op.abspath(__file__))

class LRS2Raw:
    ''' 
    Wrapper for reduction routines for raw data 
    
    '''
    def __init__(self, date, observation_number, exposure_number=1,
                 side=None, from_raw=False,
                 basepath='/work/03946/hetdex/maverick'):
        '''
        

        Parameters
        ----------
        date : TYPE
            DESCRIPTION.
        observation_number : TYPE
            DESCRIPTION.
        exposure_number : TYPE, optional
            DESCRIPTION. The default is 1.
        side : TYPE, optional
            DESCRIPTION. The default is None.
        from_raw : TYPE, optional
            DESCRIPTION. The default is False.
        basepath : TYPE, optional
            DESCRIPTION. The default is '/work/03946/hetdex/maverick'.

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
        if from_raw:
            tarfolder = op.join(basepath, '%s.tar')
        else:
            tarfolder = op.join(basepath, date, 'lrs2', 
                                     'lrs2%07d.tar' % observation_number)
        path = op.join(basepath, date, 'lrs2', 'lrs2%07d' % observation_number,
                       'exp%02d' % exposure_number)
        expstr = 'exp%02d' % exposure_number
        ampcase = '056LL'
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
            filenames = glob.glob(op.join(path, '*056LL*.fits'))
            if len(filename) < 1:
                self.log.error('No files found here %s' % path)
                sys.exit('Cowardly exiting; please check input')
            filename = filenames[0]
            b = fits.open(filename)
        if side is None:
            Target = b[0].header['OBJECT']
            if '_056' in Target:
                side = 'blue'
            if '_066' in Target:
                side = 'red'
        self.side = side
        self.info = {}
        name = 'multi_%s_%07d_%s_%s.fits'
        for channel in side_dict[self.side]:
            self.info[channel] = self.ChannelInfo(channel)
            self.reduce_channel(filename, channel_dict[channel], channel, 
                                tarfolder=tarfolder)
            self.info[channel].filename = name % (date, observation_number,
                                                  expstr, channel)
            self.info[channel].date = date
            self.info[channel].observation_number = observation_number
            self.info[channel].exposure_number = exposure_number
    
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

    def reduce_channel(self, filename, amp, channel, tarfolder=None):
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
        
        # Basic reduction
        array_flt1, e1, header = base_reduction(filename1, tarfolder=tarfolder,
                                                get_header=True)
        array_flt2, e2 = base_reduction(filename2, tarfolder=tarfolder)
        image = np.vstack([array_flt1, array_flt2])
        E = np.vstack([e1, e2])
        image[:] -= self.info[channel].masterbias
        
        # Get powerlaw
        plaw = get_powerlaw(image, self.info[channel].trace)
        self.info[channel].image = image
        self.info[channel].plaw = plaw
        image[:] -= self.info[channel].plaw
        
        # Get spectra and error
        spec = get_spectra(image, self.info[channel].trace)
        specerr = get_spectra_error(E, self.info[channel].trace)
        chi2 = get_spectra_chi2(self.info[channel].masterflt, image, E, 
                                self.info[channel].trace)
        
        # Mark pixels effected by cosmics
        badpix = chi2 > 10.
        specerr[badpix] = np.nan
        spec[badpix] = np.nan
        
        # Rectify spectra
        specrect, errrect = rectify(spec, specerr, 
                                    self.info[channel].wavelength,
                                    self.info[channel].def_wave)
        factor = (6.626e-27 * (3e18 / self.info[channel].def_wave) /
                  header['EXPTIME'] / 51.4e4)
        specrect[:] *= factor
        errrect[:] *= factor
        self.info[channel].orig = spec
        self.info[channel].data = specrect
        self.info[channel].datae = errrect
        self.info[channel].header = header
        self.info[channel].channel = channel
        self.info[channel].objname = header['OBJECT']
        self.info[channel].spec_ext = np.array([self.info[channel].def_wave, 
                                                self.info[channel].def_wave*0.,
                                                self.info[channel].def_wave*0.,
                                                self.info[channel].def_wave*0.,
                                                self.info[channel].def_wave*0.,
                                                self.info[channel].norm])