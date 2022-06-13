#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 13:46:18 2022

@author: gregz
"""

from lrs2multi import LRS2Multi
import os.path as op
import numpy as np

class LRS2Object:
    ''' Wrapper for reduction routines with processed data, multi*.fits '''
    def __init__(self, filenames, detwave=None, wave_window=None):
        '''
        Class initialization

        Parameters
        ----------
        filename : str
            multi*.fits filename for reduction
        '''
        self.detwave = detwave
        self.wave_window = wave_window
        self.sides = {}
        self.norms = {}
        observations = [('_').join(op.basename(filename).split('_')[:4]) 
                        for filename in filenames]
        unique_observations = np.unique(observations)
        for observation in unique_observations:
            self.sides[observation] = []

        for filename, observation in zip(filenames, observations):
            L = LRS2Multi(filename)
            self.sides[observation].append(L)
    
    def sky_subtraction(self, xc=None, yc=None, sky_radius=5., detwave=None, 
                        wave_window=None, local=False, pca=False, 
                        func=np.nanmean, local_kernel=7., obj_radius=3.,
                        obj_sky_thresh=1.):
        ''' Subtract Sky '''
        for key in self.sides.keys():
            for L in self.sides[key]:
                if (L.channel == 'orange') or (L.channel=='red'):
                    L.detwave = self.detwave
                    L.wave_window = self.wave_window[key]
                    L.sky_subtraction(xc=xc, yc=yc, sky_radius=sky_radius, 
                                      detwave=detwave, wave_window=wave_window, 
                                      local=local, pca=pca, 
                                      func=func, local_kernel=local_kernel, 
                                      obj_radius=obj_radius,
                                      obj_sky_thresh=obj_sky_thresh)
            for i, L in enumerate(self.sides[key]):
                if (L.channel == 'uv') or (L.channel=='farred'):
                    if i == 0:
                        j = 1
                    else:
                        j = 0
                    L.adrx0 = self.sides[key][j].adrx0
                    L.adry0 = self.sides[key][j].adry0
                    avgadrx = np.mean(L.adrx)
                    avgadry = np.mean(L.adry)
                    xc = self.sides[key][j].centroid_x
                    yc = self.sides[key][j].centroid_y
                    L.sky_subtraction(xc=xc+(avgadrx-L.adrx0), yc=yc+(avgadry-L.adrx0), 
                                      sky_radius=sky_radius, 
                                      detwave=detwave, wave_window=wave_window, 
                                      local=local, pca=pca, 
                                      func=func, local_kernel=local_kernel, 
                                      obj_radius=obj_radius,
                                      obj_sky_thresh=obj_sky_thresh)
    def extract_spectrum(self, xc=None, yc=None, detwave=None, 
                         wave_window=None, use_aperture=True, radius=2.5,
                         model=None):
        ''' Extract Spectrum '''
        for key in self.sides.keys():
            for L in self.sides[key]:
                if (L.channel == 'orange') or (L.channel=='red'):
                    L.extract_spectrum(xc=xc, yc=yc, detwave=detwave, 
                                       wave_window=wave_window, 
                                       use_aperture=use_aperture, 
                                       radius=radius,
                                       model=model)
            for i, L in enumerate(self.sides[key]):
                if (L.channel == 'uv') or (L.channel=='farred'):
                    if i == 0:
                        j = 1
                    else:
                        j = 0
                    L.extract_spectrum(xc=self.sides[key][j].centroid_x, 
                                       yc=self.sides[key][j].centroid_y, 
                                       detwave=detwave, 
                                       wave_window=wave_window, 
                                       use_aperture=use_aperture, 
                                       radius=radius,
                                       model=model)
    def calculate_norm(self, detwave=None, wave_window=None, func=np.nansum):
        if detwave is None:
            detwave = self.detwave
        if wave_window is None:
            wave_window = self.wave_window
        for key in self.sides.keys():
            for L in self.sides[key]:
                if (L.channel == 'orange') or (L.channel=='red'):
                    L.calculate_norm(detwave=detwave, wave_window=wave_window, 
                                     func=func)
                    self.norm[key] = L.norm
        self.avgnorm = np.nanmean(list(self.norm.values()))
        
    def normalize(self, detwave=None, wave_window=None, func=np.nansum):
        self.calculate_norm(detwave=detwave, wave_window=wave_window, 
                            func=func)
        for key in self.sides.keys():
            for L in self.sides[key]:
                L.log.info('%s: %0.2f' % (L.filename, 
                                          self.avgnorm / self.norm[key]))
                L.normalize(self.avgnorm / self.norm[key])
                
