#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 13:46:18 2022

@author: gregz
"""

from lrs2multi import LRS2Multi
import os.path as op
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.convolution import convolve, Gaussian1DKernel, interpolate_replace_nans
import astropy.units as u
from astropy.nddata import NDData, StdDevUncertainty
from specutils import Spectrum1D, SpectralRegion
from astropy.table import Table

class LRS2Object:
    ''' Wrapper for reduction routines with processed data, multi*.fits '''
    def __init__(self, filenames, detwave=6563., wave_window=10.,
                 red_detect_channel='red', blue_detect_channel='orange'):
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
        self.red_detect_channel = red_detect_channel
        self.blue_detect_channel = blue_detect_channel
        blue_dict = {'orange': 'uv', 'uv': 'orange'}
        red_dict = {'red': 'farred', 'farred': 'red'}
        self.red_other_channel = red_dict[red_detect_channel]
        self.blue_other_channel = blue_dict[blue_detect_channel]

        observations = [('_').join(op.basename(filename).split('_')[:4]) 
                        for filename in filenames]
        unique_observations = np.unique(observations)
        for observation in unique_observations:
            self.sides[observation] = []

        for filename, observation in zip(filenames, observations):
            L = LRS2Multi(filename)
            L.detwave = detwave
            L.wave_window = wave_window
            self.sides[observation].append(L)
            try:
                millum = L.header['MILLUM'] / 1e4
                throughp = L.header['THROUGHP']
            except:
                millum = 51.4
                throughp = 1.0
            L.log.info('%s: %s with %0.2fs, %0.2fcm2, %0.2f' % (op.basename(L.filename)[:-5],
                                               L.header['OBJECT'], 
                                               L.header['EXPTIME'],
                                               millum, throughp))
    
    def setup_plotting(self):
        N = len(list(self.sides.keys()))
        remove = False
        if N % 2 == 1:
            remove = True
        nrows = int(np.ceil(N / 2.))
        fig, ax = plt.subplots(nrows, 2, figsize=((2.*7.4, nrows*3.5)),
                               sharex=True, sharey=True,
                               gridspec_kw={'wspace':0.01, 'hspace':0.15})
        ax = ax.ravel()
        i = 0
        for key in self.sides.keys():
            for L in self.sides[key]:
                L.ax = ax[i]
            i += 1
        if remove:
            ax[-1].remove()
        self.fig = fig
    
    def subtract_sky(self, xc=None, yc=None, sky_radius=5., detwave=None, 
                        wave_window=None, local=False, pca=False, 
                        func=np.nanmean, local_kernel=7., obj_radius=3.,
                        obj_sky_thresh=1.):
        ''' Subtract Sky '''
        if detwave is None:
            detwave = self.detwave
        if wave_window is None:
            wave_window = self.wave_window
        for key in self.sides.keys():
            for L in self.sides[key]:
                if (L.channel == self.blue_detect_channel) or (L.channel==self.red_detect_channel):
                    L.sky_subtraction(xc=xc, yc=yc, sky_radius=sky_radius, 
                                      detwave=detwave, wave_window=wave_window, 
                                      local=local, pca=pca, 
                                      func=func, local_kernel=local_kernel, 
                                      obj_radius=obj_radius,
                                      obj_sky_thresh=obj_sky_thresh)
            for i, L in enumerate(self.sides[key]):
                if (L.channel == self.blue_other_channel) or (L.channel==self.red_other_channel):
                    if i == 0:
                        j = 1
                    else:
                        j = 0
                    L.adrx0 = self.sides[key][j].adrx0
                    L.adry0 = self.sides[key][j].adry0
                    L.centroid_x = self.sides[key][j].centroid_x
                    L.centroid_y = self.sides[key][j].centroid_y
                    avgadrx = np.mean(L.adrx)
                    avgadry = np.mean(L.adry)
                    L.sky_subtraction(xc=self.sides[key][j].centroid_x+(avgadrx-L.adrx0), 
                                      yc=self.sides[key][j].centroid_y+(avgadry-L.adrx0), 
                                      sky_radius=sky_radius, 
                                      detwave=detwave, wave_window=wave_window, 
                                      local=local, pca=pca, 
                                      func=func, local_kernel=local_kernel, 
                                      obj_radius=obj_radius,
                                      obj_sky_thresh=obj_sky_thresh)
    
    def set_manual_extraction(self, xc, yc, detwave=None, 
                              wave_window=None):
        for key in self.sides.keys():
            for L in self.sides[key]:
                L.manual_extraction(xc=xc, yc=yc, detwave=detwave,
                                    wave_window=wave_window)
                
    def extract_spectrum(self, xc=None, yc=None, detwave=None, 
                         wave_window=None, use_aperture=True, radius=2.5,
                         model=None, func=np.nanmean, attr='skysub'):
        ''' Extract Spectrum '''
        for key in self.sides.keys():
            for L in self.sides[key]:
                if (L.channel == self.blue_detect_channel) or (L.channel==self.red_detect_channel):
                    L.extract_spectrum(xc=xc, yc=yc, detwave=detwave, 
                                       wave_window=wave_window, 
                                       use_aperture=use_aperture, 
                                       radius=radius,
                                       model=model,
                                       func=func, attr=attr)
            for i, L in enumerate(self.sides[key]):
                if (L.channel == self.blue_other_channel) or (L.channel==self.red_other_channel):
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
                                       model=model,
                                       func=func, attr=attr)
    def calculate_norm(self, detwave=None, wave_window=None, func=np.nansum):
        for key in self.sides.keys():
            for L in self.sides[key]:
                if (L.channel == self.blue_detect_channel) or (L.channel==self.red_detect_channel):
                    L.calculate_norm(detwave=detwave, wave_window=wave_window, 
                                     func=func)
                    self.norms[key] = L.norm
        self.avgnorm = np.nanmean(list(self.norms.values()))
        
    def normalize(self, detwave=None, wave_window=None, func=np.nansum):
        self.calculate_norm(detwave=detwave, wave_window=wave_window, 
                            func=func)
        for key in self.sides.keys():
            for L in self.sides[key]:
                if (L.channel == self.blue_detect_channel) or (L.channel==self.red_detect_channel):
                    L.log.info('%s: %0.2f' % (op.basename(L.filename), 
                                              self.avgnorm / self.norms[key]))
                L.normalize(self.avgnorm / self.norms[key])

    def rectify(self, newwave):
        for key in self.sides.keys():
            for L in self.sides[key]:
                L.rectify(newwave)

    def get_astrometry(self):
        for key in self.sides.keys():
            for L in self.sides[key]:
                L.get_astrometry()

    def make_cube(self, newwave, redkernel=1.8, bluekernel=0.1):
        for key in self.sides.keys():
            for L in self.sides[key]:
                if (L.channel == 'red') or (L.channel == 'farred'):
                    kernel = redkernel
                if (L.channel == 'orange') or (L.channel == 'uv'):
                    kernel = bluekernel
                L.log.info('Making cube for %s' % (op.basename(L.filename)))
                L.make_cube(newwave, kernel=kernel)

    def plot_spectrum(self):
        for key in self.sides.keys():
            for L in self.sides[key]:
                plt.plot(L.spec1D.spectral_axis, 
                         L.spec1D.flux, color='steelblue', lw=0.5)
                plt.plot(L.spec1Dsky.spectral_axis, 
                         L.spec1Dsky.flux, color='firebrick', lw=0.5)
        if hasattr(self, 'spec1D'):
            plt.plot(self.spec1D.spectral_axis.value, self.spec1D.flux.value, 
                     'k-', lw=0.5)
        
    def calculate_sn(self, detwave=None, wave_window=None):
        self.SN = {}
        for key in self.sides.keys():
            for L in self.sides[key]:
                if (L.channel == self.blue_detect_channel) or (L.channel==self.red_detect_channel):
                    self.SN[key] = L.calculate_sn(detwave, wave_window)
                    L.log.info('SN for %s: %0.2f' % (op.basename(L.filename),
                               self.SN[key]))

    def combine_spectra(self):
        specs = []
        variances = []
        weights = []
        for key in self.sides.keys():
            for L in self.sides[key]:
                wave = L.spec1D.spectral_axis.value
                specs.append(L.spec1D.flux.value)
                weights.append(self.SN[key] * np.isfinite(specs[-1]))
                variances.append(1./L.spec1D.uncertainty.array)
        specs, weights, variances = [np.array(x) for x in 
                                     [specs, weights, variances]]
        weights[weights < np.nanmax(weights, axis=0) * 0.2] = np.nan
        spec = np.nansum(specs * weights, axis=0) / np.nansum(weights, axis=0)
        error = np.sqrt(np.nansum(variances * weights, axis=0)) / np.nansum(weights, axis=0) 
        spec[spec == 0.] = np.nan
        nansel = np.isnan(spec)
        error[nansel] = np.nan
        flam_unit = (u.erg / u.cm**2 / u.s / u.AA)
        nd = NDData(spec, unit=flam_unit, mask=np.isnan(spec),
                    uncertainty=StdDevUncertainty(error))
        self.spec1D = Spectrum1D(spectral_axis=wave*u.AA, 
                                 flux=nd.data*nd.unit, uncertainty=nd.uncertainty,
                                 mask=nd.mask)
    def combine_cubes(self):
        for key in self.sides.keys():
            for L in self.sides[key]:
                continue
    
    def smooth_resolution(self, redkernel=2.25, bluekernel=0.1):
        for key in self.sides.keys():
            for L in self.sides[key]:
                if (L.channel == 'red') or (L.channel == 'farred'):
                    kernel = redkernel
                if (L.channel == 'orange') or (L.channel == 'uv'):
                    kernel = bluekernel
                L.smooth_resolution(kernel)
    
    def write_combined_spectrum(self, outname=None):
        keys = list(self.sides.keys())
        L = self.sides[keys[0]][0]
        if outname is None:
            outname = L.header['QOBJECT'] + '_combined_spectrum.fits'
        names = ['wavelength', 'f_lam', 'e_lam']
        A = np.array([self.spec1D.spectral_axis.value, 
                      self.spec1D.flux.value,
                      self.spec1D.uncertainty.array])
        T = Table([self.spec1D.spectral_axis.value, 
                   self.spec1D.flux.value,
                   self.spec1D.uncertainty.array], 
                   names=names)
        T.write('%s_combined_spectrum.dat' % L.header['QOBJECT'], format='ascii.fixed_width_two_line')
        f1 = fits.PrimaryHDU(A)
        he = L.header
        for key in he.keys():
            if key in f1.header:
                continue
            if 'SEC' in key:
                continue
            if ('BSCALE' in key) or ('BZERO' in key):
                continue
            try:
                f1.header[key] = he[key]
            except:
                continue
        
        f1.header['DWAVE'] = self.spec1D.spectral_axis.value[1] - self.spec1D.spectral_axis.value[0]
        f1.header['WAVE0'] = self.spec1D.spectral_axis.value[0]
        f1.header['WAVESOL'] = 'WAVE0 + DWAVE * linspace(0, NAXIS1)'
        f1.header['WAVEUNIT'] = 'A'
        f1.header['FLUXUNIT'] = 'ergs/s/cm2/A'
        f1.header['CRVAL1'] = self.spec1D.spectral_axis.value[0]
        f1.header['CRPIX1'] = 1
        f1.header['CDELT1'] = self.spec1D.spectral_axis.value[1] - self.spec1D.spectral_axis.value[0]
        for i, name in enumerate(names):
            f1.header['ROW%i' % (i+1)] = name
        f1.writeto(outname, overwrite=True)