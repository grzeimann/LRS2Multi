#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 11:31:38 2022

@author: gregz
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 16 10:08:18 2021

@author: gregz
"""

import logging
import numpy as np
import os.path as op
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.modeling.models import Gaussian1D, Gaussian2D
from astropy.convolution import convolve, Gaussian1DKernel, Gaussian2DKernel
from astropy.modeling.fitting import LevMarLSQFitter
from scipy.interpolate import interp1d, interp2d, griddata


class LRS2Multi:
    ''' Wrapper for reduction routines with processed data, multi*.fits '''
    def __init__(self, filename):
        '''
        Class initialization

        Parameters
        ----------
        filename : str
            multi*.fits filename for reduction
        '''   
        f = fits.open(filename)
        channel = op.basename(filename).split('_')[-1][:-5]
        objname = f[0].header['OBJECT']
        wave = f[6].data[0]
        norm = f[6].data[-1]
        data = f[0].data
        data[f[3].data==0.] = np.nan
        datae = f[3].data
        datae[f[3].data==0.] = np.nan
        uvmask = np.abs(wave-3736.0) < 1.6
        data[:, uvmask] = np.nan
        datae[:, uvmask] = np.nan
        for i in np.arange(data.shape[0]):
            sel = np.isnan(data[i])
            if sel.sum() > 200:
                continue
            data[i] = np.interp(wave, wave[~sel], data[i, ~sel], left=np.nan, 
                             right=0.0)
            datae[i] = np.interp(wave, wave[~sel], datae[i, ~sel], left=np.nan, 
                             right=0.0)
            datae[i, sel] = datae[i, sel]*1.5

        if channel == 'orange':
            data[:140] = data[:140] / 1.025
            data[140:] = data[140:] / 0.975
        J = interp1d(f[7].data[0], f[7].data[1], fill_value='extrapolate',
                       bounds_error=False)
        K = interp1d(f[7].data[0], f[7].data[2], fill_value='extrapolate',
                       bounds_error=False)
        self.data = data
        self.error = datae
        self.x = f[5].data[:, 0] * 1.
        self.y = f[5].data[:, 1] * 1.
        self.wave = wave * 1.
        self.norm = norm * 1.
        self.adrx = J(wave)
        self.adry = K(wave)
        self.adrx0 = None
        self.adry0 = None
        self.skysub = self.data * 1.
        self.sky = 0. * self.data
        self.ra = f[5].data[:, 4] * 1.
        self.dec = f[5].data[:, 5] * 1.
        self.setup_logging()
        self.objname = objname
        self.header = f[0].header
        self.detwave = wave[int(len(wave)/2)]
        self.wave_window = 5.
        self.set_big_grid()
        self.filename = filename
        self.channel = channel
        self.spec_ext = f[6].data
        
    def setup_logging(self, logname='lrs2multi'):
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
        
    def collapse_wave(self, detwave=None, wave_window=None, quick_skysub=True):
        # Collapse spectrum 
        if detwave is None:
            detwave = self.detwave
        if wave_window is None:
            wave_window = self.wave_window
        wsel = np.abs(self.wave-detwave) < wave_window
        Y = np.nanmean(self.data[:, wsel], axis=1)
        if quick_skysub:
            Y = Y - np.nanpercentile(Y, 0.15)   
        return Y
    
    def make_image(self, detwave=None, wave_window=None, quick_skysub=True):
        # Collapse spectrum 
        if detwave is None:
            detwave = self.detwave
        if wave_window is None:
            wave_window = self.wave_window
        wsel = np.abs(self.wave-detwave) < wave_window
        Y = np.nanmean(self.data[:, wsel], axis=1)
        if quick_skysub:
            Y = Y - np.nanpercentile(Y, 0.15)
        y = np.arange(-4, 4.4, 0.4)
        x = np.arange(-7, 7.4, 0.4)
        xgrid, ygrid = np.meshgrid(x, y)
        P = np.zeros((len(self.x), 2))
        P[:, 0] = self.x
        P[:, 1] = self.y
        image = griddata(P, Y, (xgrid, ygrid))
        return image
    
    def plot_image(self, detwave=None, wave_window=None, quick_skysub=True):
        # Collapse spectrum 
        if detwave is None:
            detwave = self.detwave
        if wave_window is None:
            wave_window = self.wave_window
        Y = self.collapse_wave(detwave, wave_window, quick_skysub=False)
        y = Y / np.nanpercentile(Y, 15)
        vmax = np.nanpercentile(y, 99)
        vmin = np.nanpercentile(y, 1)
        xc, yc = self.find_centroid(detwave, wave_window)
        plt.figure(figsize=(7.4, 3))
        cax = plt.scatter(self.x, self.y, c=y, cmap=plt.get_cmap('coolwarm'),
                          vmin=vmin, vmax=vmax, marker='h', s=220)
        plt.scatter(xc, yc, marker='x', color='k', s=100)
        plt.colorbar(cax)
        plt.axis([-7., 7., -4., 4.])
        plt.show()

    def find_centroid(self, detwave=None, wave_window=None, quick_skysub=True,
                      radius=4, func=np.nanmean):
        # Collapse spectrum 
        if detwave is None:
            detwave = self.detwave
        if wave_window is None:
            wave_window = self.wave_window
        wsel = np.abs(self.wave-detwave) < wave_window
        Y = func(self.data[:, wsel], axis=1)
        if quick_skysub:
            Y = Y - np.nanpercentile(Y, 0.15)
        # Get initial model
        x0, y0 = (self.x[np.nanargmax(Y)], 
                  self.y[np.nanargmax(Y)])
        GM = Gaussian2D(amplitude=np.nanmax(Y)*1e17, x_mean=x0, 
                        y_mean=y0)
        d = np.sqrt((self.x-x0)**2 + (self.y-y0)**2)
        dsel = (d < radius) * (np.isfinite(Y))
        fitter = LevMarLSQFitter()
        fit = fitter(GM, self.x[dsel], self.y[dsel], Y[dsel])
        d = np.sqrt((fit.x_mean.value-x0)**2 + (fit.y_mean.value-y0)**2)
        self.log.info('Centroid: %0.2f %0.2f' % (fit.x_mean.value, fit.y_mean.value))
        if d < 1.5:
            xc, yc = (fit.x_mean.value, fit.y_mean.value)
        else:
            xc, yc = (x0, y0)
        return xc, yc
    
    def model_source(self, detwave=None, wave_window=None, quick_skysub=True,
                     func=np.nanmean):
        # Collapse spectrum 
        if detwave is None:
            detwave = self.detwave
        if wave_window is None:
            wave_window = self.wave_window
        wsel = np.abs(self.wave-detwave) < wave_window
        Y = func(self.data[:, wsel], axis=1)
        if quick_skysub:
            Y = Y - np.nanpercentile(Y, 0.15)
        # Get initial model
        x0, y0 = (self.x[np.nanargmax(Y)], 
                  self.y[np.nanargmax(Y)])
        GM = Gaussian2D(amplitude=np.nanmax(Y)*1e17, x_mean=x0, 
                        y_mean=y0)
        fitter = LevMarLSQFitter()
        fit = fitter(GM, self.x, self.y, Y)
        return fit

    def set_big_grid(self):
        # Big grid for total flux
        x , y = (self.x, self.y)
        nx = np.hstack([x, x[:-22], x[22:], x+12.39+0.59/2., x-12.39-0.59/2.,
                        x[:-22]-12.39-0.59/2., x[:-22]+12.39+0.59/2.,
                        x[22:]-12.39-0.59/2., x[22:]+12.39+0.59/2.])
        ny = np.hstack([y, y[:-22]+7.08, y[22:]-7.08,  y-0.59,  y-0.59,
                        y[:-22]+7.08-0.59, y[:-22]+7.08-0.59,
                        y[22:]-7.08-0.59, y[22:]-7.08-0.59])
        self.largex = nx
        self.largey = ny
    
    def sky_subtraction(self, use_radius=True, radius=5, percentile=15, 
                        detwave=None, wave_window=None, 
                        smooth=False, pca=False):
        if detwave is None:
            detwave = self.detwave
        if wave_window is None:
            wave_window = self.wave_window
        if use_radius:
            xc, yc = self.find_centroid(detwave=detwave, wave_window=wave_window, 
                                        quick_skysub=True)
            fiber_sel = np.sqrt((self.x - xc)**2 + (self.y - yc)**2) > radius
        else:
            Y = self.collapse_wave(detwave=detwave, wave_window=wave_window, 
                                   quick_skysub=False)
            fiber_sel = Y < np.nanpercentile(Y, percentile)
        sky = np.nanmedian(self.data[fiber_sel], axis=0)
        sky = sky[np.newaxis, :] * np.ones((280,))[:, np.newaxis]
        self.sky = sky
        self.skysub = self.data - self.sky
    
    def extract_spectrum(self, xc=None, yc=None, detwave=None, 
                         wave_window=None, use_aperture=True, radius=2.5,
                         model=None):
        if detwave is None:
            detwave = self.detwave
        if wave_window is None:
            wave_window = self.wave_window
        if (xc is None) or (yc is None):
            xc, yc = self.find_centroid(detwave=detwave, 
                                        wave_window=wave_window, 
                                        quick_skysub=True)
            self.adrx0 = self.adrx[np.argmin(np.abs(self.wave-detwave))]
            self.adry0 = self.adry[np.argmin(np.abs(self.wave-detwave))]
        spectrum = self.wave * np.nan
        spectrum_error = spectrum * 1.
        cor = spectrum * 1.
        if self.adrx0 is None:
            self.log.warning('Can not extract spectrum until adrx0 is set')
            return None
        if (use_aperture==False) * (model is None):
            self.log.warning("You must provide a model if aperture extraction isn't used")
            return None
        for i in np.arange(len(self.wave)):
            offx = self.adrx[i] - self.adrx0
            offy = self.adry[i] - self.adry0
            x = xc + offx
            y = yc + offy
            d = np.sqrt((self.x - x)**2 + (self.y - y)**2)
            rsel = d < radius
            if use_aperture:
                spectrum[i] = np.nansum(self.skysub[rsel, i], axis=0)
                spectrum_error[i] = np.sqrt(np.nansum(
                                         self.error[rsel, i]**2, axis=0))
            else:
                W = model(self.x - offx, self.y - offy)
                WT = model(self.largex - offx, self.largey - offy)
                cor[i] = W[rsel].sum() / WT.sum()
                W = W / W[rsel].sum()
                spectrum[i] = (np.nansum(W[rsel] * self.skysub[rsel, i]) /
                                  np.nansum(W[rsel]**2))
                spectrum_error[i] = (np.sqrt(np.nansum((
                                     self.error[rsel, i])**2 * W[rsel])) / 
                                        np.nansum(W[rsel]**2))
                spectrum[i] /= cor[i]
                spectrum_error[i] /= cor[i]
        self.spectrum = spectrum
        self.spectrum_error = spectrum_error
        return cor
    
    def calculate_norm(self, wavemid=6700, wave_window=100, func=np.nanmean):
        wsel = np.abs(self.wave - wavemid) < wave_window
        norm = func(self.spectrum[wsel])
        self.norm = norm
    
    def write_spectrum(self):
        outname = self.filename.replace('multi', 'spectrum')
        self.spec_ext[1] = self.spectrum
        f1 = fits.PrimaryHDU(self.spec_ext)
        he = self.header
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
        names = ['wavelength', 'F_lambda', 'Sky_lambda', 'e_F_lambda',
                 'e_Sky_lambda', 'response']
        f1.header['DWAVE'] = self.wave[1] - self.wave[0]
        f1.header['WAVE0'] = self.wave[0]
        f1.header['WAVESOL'] = 'WAVE0 + DWAVE * linspace(0, NAXIS1)'
        f1.header['WAVEUNIT'] = 'A'
        f1.header['FLUXUNIT'] = 'ergs/s/cm2/A'
        for i, name in enumerate(names):
            f1.header['ROW%i' % (i+1)] = name
        f1.writeto(outname, overwrite=True)