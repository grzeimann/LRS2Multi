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
from astropy.io import fits
from astropy.stats import biweight_location
import tarfile

warnings.filterwarnings("ignore")

class LRS2Raw:
    ''' 
    Wrapper for reduction routines for raw data 
    
    '''
    def __init__(self, date=None, observation_number=None):
        pass
    
    
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
        
    def orient_image(self, image, amp, ampname):
        '''
        Orient the images from blue to red (left to right)
        Fibers are oriented to match configuration files
        '''
        if amp == "LU":
            image[:] = image[::-1, ::-1]
        if amp == "RL":
            image[:] = image[::-1, ::-1]
        if ampname is not None:
            if ampname == 'LR' or ampname == 'UL':
                image[:] = image[:, ::-1]
        return image
    
    
    def base_reduction(self, filename, tarname=None, get_header=False):
        if tarname is None:
            a = fits.open(filename)
        else:
            try:
                t = tarfile.open(tarname, 'r')
                a = fits.open(t.extractfile('/'.join(filename.split('/')[-4:])))
            except:
                self.log.warning('Could not open %s' % filename)
                return np.zeros((1032, 2064)), np.zeros((1032, 2064))
        image = np.array(a[0].data, dtype=float)
        # overscan sub
        overscan_length = 32 * (image.shape[1] / 1064)
        O = biweight_location(image[:, -(overscan_length-2):])
        image[:] = image - O
        # trim image
        image = image[:, :-overscan_length]
        gain = a[0].header['GAIN']
        gain = np.where(gain > 0., gain, 0.85)
        rdnoise = a[0].header['RDNOISE']
        rdnoise = np.where(rdnoise > 0., rdnoise, 3.)
        amp = (a[0].header['CCDPOS'].replace(' ', '') +
               a[0].header['CCDHALF'].replace(' ', ''))
        try:
            ampname = a[0].header['AMPNAME']
        except:
            ampname = None
        header = a[0].header
        a = self.orient_image(image, amp, ampname) * gain
        E = np.sqrt(rdnoise**2 + np.where(a > 0., a, 0.))
        if tarname is not None:
            t.close()
        if get_header:
            return a, E, header
        return a, E