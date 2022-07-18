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
import astropy.units as u
from astropy.nddata import NDData, StdDevUncertainty
from specutils import Spectrum1D
from astropy.table import Table


class LRS2Object:
    ''' Wrapper for reduction routines with processed data, multi*.fits 
    
    Examples
    --------
    
    
        
    '''
    def __init__(self, filenames, detwave=6563., wave_window=10.,
                 red_detect_channel='red', blue_detect_channel='orange'):
        '''
        Class initialization

        Parameters
        ----------
        filenames : list
            Path of multi*.fits filename for reduction
        detwave : float
            The wavelength for the detection algorithm. The same wavelength is
            used for both B and R observation so a common wavelength is 
            required.
        wave_window : float
            The +/- Angstrom window for running detections
        red_detect_channel : string
            Should be 'red' or 'farred'
        blue_detect_channel : string
            Should be 'uv' or 'orange'
            
        Examples
        --------
        >>> folder = PROGRAM_FOLDER
        >>> object = OBJECT_NAME
        >>> filenames = get_filenames_for_program(folder, object)
        >>> LRS2 = LRS2Multi(filenames)
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
            L.log.info('%s: %s with %0.2fs, %0.2fcm2, %0.2f' %
                       (op.basename(L.filename)[:-5], L.header['OBJECT'], 
                        L.header['EXPTIME'], millum, throughp))
    
    def setup_plotting(self, forall=False):
        '''
        This will set up the figure and axes for making image plots

        Parameters
        ----------
        forall : boolean, optional
            If True, all channels will set up for plotting rather than just 
            the detection channels. The default is False.

        '''
        N = len(list(self.sides.keys()))
        remove = False
        if N % 2 == 1:
            if not forall:
                remove = True
        if forall:
            nrows = int(N)
        else:
            nrows = int(np.ceil(N / 2.))
        fig, ax = plt.subplots(nrows, 2, figsize=((2.*7.4, nrows*3.5)),
                               sharex=True, sharey=True,
                               gridspec_kw={'wspace':0.01, 'hspace':0.15})
        ax = ax.ravel()
        i = 0
        for key in self.sides.keys():
            for L in self.sides[key]:
                L.ax = ax[i]
                if forall:
                    i += 1
            if not forall:
                i += 1
        if remove:
            ax[-1].remove()
        self.fig = fig
    
    def subtract_sky(self, xc=None, yc=None, sky_radius=5., detwave=None, 
                        wave_window=None, local=False, pca=False, 
                        correct_ftf_from_skylines=False,
                        func=np.nanmean, local_kernel=7., obj_radius=3.,
                        obj_sky_thresh=1., ncomp=25, bins=25, peakthresh=7.):
        '''
        

        Parameters
        ----------
        xc : TYPE, optional
            DESCRIPTION. The default is None.
        yc : TYPE, optional
            DESCRIPTION. The default is None.
        sky_radius : TYPE, optional
            DESCRIPTION. The default is 5..
        detwave : TYPE, optional
            DESCRIPTION. The default is None.
        wave_window : TYPE, optional
            DESCRIPTION. The default is None.
        local : TYPE, optional
            DESCRIPTION. The default is False.
        pca : TYPE, optional
            DESCRIPTION. The default is False.
        correct_ftf_from_skylines : TYPE, optional
            DESCRIPTION. The default is False.
        func : TYPE, optional
            DESCRIPTION. The default is np.nanmean.
        local_kernel : TYPE, optional
            DESCRIPTION. The default is 7..
        obj_radius : TYPE, optional
            DESCRIPTION. The default is 3..
        obj_sky_thresh : TYPE, optional
            DESCRIPTION. The default is 1..
        ncomp : TYPE, optional
            DESCRIPTION. The default is 25.
        bins : TYPE, optional
            DESCRIPTION. The default is 25.
        peakthresh : TYPE, optional
            DESCRIPTION. The default is 7..

        Returns
        -------
        None.

        '''
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
                                      obj_sky_thresh=obj_sky_thresh,
                                      ncomp=ncomp, bins=bins, 
                                      peakthresh=peakthresh,
                                      correct_ftf_from_skylines=correct_ftf_from_skylines)
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
                                      obj_sky_thresh=obj_sky_thresh,
                                      ncomp=ncomp, bins=bins,
                                      peakthresh=peakthresh,
                                      correct_ftf_from_skylines=correct_ftf_from_skylines)
    
    def set_manual_extraction(self, xc, yc, detwave=None, 
                              wave_window=None):
        '''
        

        Parameters
        ----------
        xc : TYPE
            DESCRIPTION.
        yc : TYPE
            DESCRIPTION.
        detwave : TYPE, optional
            DESCRIPTION. The default is None.
        wave_window : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        '''
        cnt = 0
        for key in self.sides.keys():
            for L in self.sides[key]:
                L.manual_extraction(xc=xc[cnt], yc=yc[cnt], detwave=detwave,
                                    wave_window=wave_window)
            cnt += 1
    
    def set_pca_wave_mask(self, lines, redshift, window=5.):
        for key in self.sides.keys():
            for L in self.sides[key]:
                L.set_pca_wave_mask(lines=lines, redshift=redshift,
                                    window=window)

    def extract_spectrum(self, xc=None, yc=None, detwave=None, 
                         wave_window=None, use_aperture=True, radius=2.5,
                         model=None, func=np.nanmean, attr='skysub'):
        '''
        

        Parameters
        ----------
        xc : TYPE, optional
            DESCRIPTION. The default is None.
        yc : TYPE, optional
            DESCRIPTION. The default is None.
        detwave : TYPE, optional
            DESCRIPTION. The default is None.
        wave_window : TYPE, optional
            DESCRIPTION. The default is None.
        use_aperture : TYPE, optional
            DESCRIPTION. The default is True.
        radius : TYPE, optional
            DESCRIPTION. The default is 2.5.
        model : TYPE, optional
            DESCRIPTION. The default is None.
        func : TYPE, optional
            DESCRIPTION. The default is np.nanmean.
        attr : TYPE, optional
            DESCRIPTION. The default is 'skysub'.

        Returns
        -------
        None.

        '''
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
        '''
        

        Parameters
        ----------
        detwave : TYPE, optional
            DESCRIPTION. The default is None.
        wave_window : TYPE, optional
            DESCRIPTION. The default is None.
        func : TYPE, optional
            DESCRIPTION. The default is np.nansum.

        Returns
        -------
        None.

        '''
        for key in self.sides.keys():
            for L in self.sides[key]:
                if (L.channel == self.blue_detect_channel) or (L.channel==self.red_detect_channel):
                    L.calculate_norm(detwave=detwave, wave_window=wave_window, 
                                     func=func)
                    self.norms[key] = L.norm
        self.avgnorm = np.nanmean(list(self.norms.values()))
        
    def normalize(self, detwave=None, wave_window=None, func=np.nansum):
        '''
        

        Parameters
        ----------
        detwave : TYPE, optional
            DESCRIPTION. The default is None.
        wave_window : TYPE, optional
            DESCRIPTION. The default is None.
        func : TYPE, optional
            DESCRIPTION. The default is np.nansum.

        Returns
        -------
        None.

        '''
        self.calculate_norm(detwave=detwave, wave_window=wave_window, 
                            func=func)
        for key in self.sides.keys():
            for L in self.sides[key]:
                if (L.channel == self.blue_detect_channel) or (L.channel==self.red_detect_channel):
                    L.log.info('%s: %0.2f' % (op.basename(L.filename), 
                                              self.norms[key] / self.avgnorm))
                L.normalize(self.avgnorm / self.norms[key])

    def rectify(self, newwave):
        '''
        

        Parameters
        ----------
        newwave : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        for key in self.sides.keys():
            for L in self.sides[key]:
                L.rectify(newwave)

    def get_astrometry(self):
        '''
        

        Returns
        -------
        None.

        '''
        for key in self.sides.keys():
            for L in self.sides[key]:
                L.get_astrometry()

    def make_cube(self, newwave, redkernel=1.8, bluekernel=0.1,
                  scale=0.4, ran=[-7., 7., -7., 7.], radius=0.7, kernel=0.1):
        '''
        

        Parameters
        ----------
        newwave : TYPE
            DESCRIPTION.
        redkernel : TYPE, optional
            DESCRIPTION. The default is 1.8.
        bluekernel : TYPE, optional
            DESCRIPTION. The default is 0.1.

        Returns
        -------
        None.

        '''
        for key in self.sides.keys():
            for L in self.sides[key]:
                if (L.channel == 'red') or (L.channel == 'farred'):
                    kernel = redkernel
                if (L.channel == 'orange') or (L.channel == 'uv'):
                    kernel = bluekernel
                L.log.info('Making cube for %s' % (op.basename(L.filename)))
                L.make_cube(newwave, kernel=kernel,
                            scale=scale, ran=ran, radius=radius)

    def plot_spectrum(self, ax):
        '''
        

        Parameters
        ----------
        ax : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        for key in self.sides.keys():
            for L in self.sides[key]:
                ax.plot(L.spec1D.spectral_axis, 
                         L.spec1D.flux*1e17, color='steelblue', lw=0.5)
                ax.plot(L.spec1Dsky.spectral_axis, 
                         L.spec1Dsky.flux*1e17, color='firebrick', lw=0.5)
        if hasattr(self, 'spec1D'):
            ax.plot(self.spec1D.spectral_axis.value, self.spec1D.flux.value*1e17, 
                     'k-', lw=0.5)
        
    def calculate_sn(self, detwave=None, wave_window=None):
        '''
        

        Parameters
        ----------
        detwave : TYPE, optional
            DESCRIPTION. The default is None.
        wave_window : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        '''
        self.SN = {}
        for key in self.sides.keys():
            for L in self.sides[key]:
                if (L.channel == self.blue_detect_channel) or (L.channel==self.red_detect_channel):
                    self.SN[key] = L.calculate_sn(detwave, wave_window)
                    L.log.info('SN for %s: %0.2f' % (op.basename(L.filename),
                               self.SN[key]))
    def get_single_spectrum_for_side(self, L_dict):
        '''
        

        Returns
        -------
        None.

        '''
        w, y, z = ([], [], [])
        for L in L_dict:
            wave = L.spec1D.spectral_axis.value
            y.append(L.spec1D.flux.value)
            z.append(L.spec1D.uncertainty.array)
            if L.side == 'blue':
                l1 = 4635.
                l2 = 4645.
                wsel = (wave >= l1) * (wave <= l2)
            else:
                l1 = 8275.
                l2 = 8400.
                wsel = (wave >= l1) * (wave <= l2)
            if (L.channel == 'farred') or (L.channel == 'orange'):
                _w = (wave - l1) / (l2 - l1)
                w.append(_w)
            else:
                _w = (l2 - wave) / (l2 - l1)
                w.append(_w)
        sSp = np.nanmean(y, axis=0)
        sSp[wsel] = y[0][wsel] * w[0][wsel] + y[1][wsel] * w[1][wsel]
        esSp = np.nanmean(z, axis=0)
        esSp[wsel] = np.sqrt((z[0][wsel] * w[0][wsel])**2 +
                             (y[1][wsel] * w[1][wsel])**2)
        return sSp, esSp
    
    def combine_spectra(self):
        '''
        

        Returns
        -------
        None.

        '''
        specs = []
        variances = []
        weights = []
        for key in self.sides.keys():
            for L in self.sides[key]:
                wave = L.spec1D.spectral_axis.value
            sSp, esSp = self.get_single_spectrum_for_side(self.sides[key])
            specs.append(sSp)
            weights.append(self.SN[key] * np.isfinite(specs[-1]))
            variances.append(esSp)
        specs, weights, variances = [np.array(x) for x in 
                                     [specs, weights, variances]]
        weights[weights < np.nanmax(weights, axis=0) * 0.2] = np.nan
        weights = weights / np.nansum(weights, axis=0)[np.newaxis, :]
        spec = np.nansum(specs * weights, axis=0)
        error = np.sqrt(np.nansum(variances * weights**2, axis=0))
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
        '''
        

        Returns
        -------
        None.

        '''
        specs = []
        variances = []
        weights = []
        for key in self.sides.keys():
            for L in self.sides[key]:
                wave = L.spec3D.spectral_axis.value
                specs.append(L.spec3D.flux.value)
                weights.append(self.SN[key] * np.isfinite(specs[-1]))
                variances.append(1./L.spec3D.uncertainty.array)
        L.log.info('Making combined cube')
        specs, weights, variances = [np.array(x) for x in 
                                     [specs, weights, variances]]
        weights[weights < np.nanmax(weights, axis=0) * 0.2] = np.nan
        weights = weights / np.nansum(weights, axis=0)[np.newaxis, :]
        spec = np.nansum(specs * weights, axis=0)
        error = np.sqrt(np.nansum(variances * weights**2, axis=0))
        spec[spec == 0.] = np.nan
        nansel = np.isnan(spec)
        error[nansel] = np.nan
        flam_unit = (u.erg / u.cm**2 / u.s / u.AA)
        nd = NDData(spec, unit=flam_unit, mask=np.isnan(spec),
                    uncertainty=StdDevUncertainty(error))
        self.spec3D = Spectrum1D(spectral_axis=wave*u.AA, 
                                 flux=nd.data*nd.unit, uncertainty=nd.uncertainty,
                                 mask=nd.mask)
    
    def smooth_resolution(self, redkernel=2.25, bluekernel=0.1):
        '''
        

        Parameters
        ----------
        redkernel : TYPE, optional
            DESCRIPTION. The default is 2.25.
        bluekernel : TYPE, optional
            DESCRIPTION. The default is 0.1.

        Returns
        -------
        None.

        '''
        for key in self.sides.keys():
            for L in self.sides[key]:
                if (L.channel == 'red') or (L.channel == 'farred'):
                    kernel = redkernel
                if (L.channel == 'orange') or (L.channel == 'uv'):
                    kernel = bluekernel
                L.smooth_resolution(kernel)
    
    def write_combined_spectrum(self, outname=None):
        '''
        

        Parameters
        ----------
        outname : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        '''
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
        T.write(outname.replace('fits', 'dat'), format='ascii.fixed_width_two_line')
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
        
    def write_cube(self, outname=None):
        '''
        Write data cube to fits file
        
        Parameters
        ----------
        wave : 1d numpy array
            Wavelength for data cube
        xgrid : 2d numpy array
            x-coordinates for data cube
        ygrid : 2d numpy array
            y-coordinates for data cube
        Dcube : 3d numpy array
            Data cube, corrected for ADR
        outname : str
            Name of the outputted fits file
        he : object
            hdu header object to carry original header information
        '''
        keys = list(self.sides.keys())
        L = self.sides[keys[-1]][0]
        wave = self.spec3D.spectral_axis.value
        data = np.moveaxis(self.spec3D.flux.value, -1, 0)
        hdu = fits.PrimaryHDU(np.array(data, dtype='float32'))
        hdu.header['CRVAL1'] = L.skycoord.ra.deg
        hdu.header['CRVAL2'] = L.skycoord.dec.deg
        hdu.header['CRVAL3'] = wave[0]
        hdu.header['CRPIX1'] = int(L.xgrid.shape[0]/2.)
        hdu.header['CRPIX2'] = int(L.xgrid.shape[1]/2.)
        hdu.header['CRPIX3'] = 1
        hdu.header['CTYPE1'] = 'RA---TAN'
        hdu.header['CTYPE2'] = 'DEC--TAN'
        hdu.header['CTYPE3'] = 'WAVE'
        hdu.header['CUNIT1'] = 'deg'
        hdu.header['CUNIT2'] = 'deg'
        hdu.header['CUNIT3'] = 'Angstrom'
        hdu.header['SPECSYS'] = 'TOPOCENT'
        hdu.header['CDELT1'] = (L.xgrid[0, 0] - L.xgrid[0, 1]) / 3600.
        hdu.header['CDELT2'] = (L.ygrid[1, 0] - L.ygrid[0, 0]) / 3600.
        hdu.header['CDELT3'] = (wave[1] - wave[0])
        for key in L.header.keys():
            if key in hdu.header:
                continue
            if ('CCDSEC' in key) or ('DATASEC' in key):
                continue
            if ('BSCALE' in key) or ('BZERO' in key):
                continue
            hdu.header[key] = L.header[key]
        if outname is None:
            outname = L.header['QOBJECT'] + '_combined_cube.fits'
        hdu.writeto(outname, overwrite=True)