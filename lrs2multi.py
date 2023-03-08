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
import warnings
from astropy.io import fits
from astropy.modeling.models import Gaussian2D, Polynomial2D, Polynomial1D
from astropy.convolution import convolve, Gaussian1DKernel, Gaussian2DKernel
from astropy.convolution import interpolate_replace_nans

from astropy.modeling.fitting import LevMarLSQFitter
from scipy.interpolate import interp1d, griddata
from scipy.signal import medfilt2d
from sklearn.decomposition import PCA
from specutils import Spectrum1D
from astrometry import Astrometry
import astropy.units as u
from astropy.nddata import NDData, StdDevUncertainty
from specutils.manipulation import FluxConservingResampler, convolution_smooth
from astropy.coordinates import SkyCoord, EarthLocation
from astropy.time import Time


warnings.filterwarnings("ignore")
                      
class LRS2Multi:
    ''' 
    Wrapper for reduction routines with processed data, multi*.fits 
    
    '''
    def __init__(self, filename, lrs2raw_object=None, detwave=None, 
                 wave_window=None, ignore_mask=False):
        '''
        Class initialization

        Parameters
        ----------
        filename : str
            multi*.fits filename for reduction
        '''
        self.setup_logging()
        if lrs2raw_object is None:
            self.read_file(filename, ignore_mask=ignore_mask)
        else:
            self.get_info_from_lrs2raw_object(lrs2raw_object)
        if detwave is None:
            self.detwave = self.wave[int(len(self.wave)/2)]
        else:
            self.detwave = detwave
        self.adrx0 = self.adrx[np.argmin(np.abs(self.wave-self.detwave))]
        self.adry0 = self.adry[np.argmin(np.abs(self.wave-self.detwave))]
        if wave_window is None:
            self.wave_window = 5.
        else:
            self.wave_window = wave_window

        if ((self.channel == 'orange') or (self.channel == 'uv')):
            self.side = 'blue'
        else: 
            self.side = 'red'
        lrs2_dict = {'blue':[-50.,-150.], 'red':[49.42, -150.14]}
        self.lrs2_coords = lrs2_dict[self.side]
        try:
            self.get_barycor()
        except:
            self.barycor = 0.0
        self.set_big_grid()
        self.wave = self.wave * (1. + self.barycor / 2.99892e8)
        self.fill_bad_fibers()
        self.manual = False
        
    def read_file(self, filename, ignore_mask=False):
        '''
        

        Parameters
        ----------
        filename : TYPE
            DESCRIPTION.
        ignore_mask : TYPE, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        None.

        '''
        f = fits.open(filename)
        channel = op.basename(filename).split('_')[-1][:-5]
        objname = f[0].header['OBJECT']
        wave = f[6].data[0]
        norm = f[6].data[-1]
        data = f[0].data
        if not ignore_mask:
            data[f[3].data==0.] = np.nan
        datae = f[3].data
        if not ignore_mask:
            datae[f[3].data==0.] = np.nan
        uvmask = np.abs(wave-3736.0) < 1.6
        if uvmask.sum() > 0:
           data[52:115, uvmask] = np.nan
           datae[52:115, uvmask] = np.nan
           C = data * 1.
           C = convolve(data, Gaussian2DKernel(1.5))
           data[52:115, uvmask] = C[52:115, uvmask]
        for i in np.arange(data.shape[0]):
            sel = np.isnan(data[i])
            for i in np.arange(1, 3):
                sel[i:] += sel[:-i]
                sel[:-i] += sel[i:]
            data[i][sel] = np.nan
            datae[i, sel] = np.nan
        badfibers = np.isnan(data).sum(axis=1) > 150.
        data[badfibers] = np.nan
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
        self.normcurve = norm * 1.
        self.adrx = J(wave)
        self.adry = K(wave)
        self.skysub = self.data * 1.
        self.sky =  self.data * 0.
        self.ra = f[5].data[:, 4] * 1.
        self.dec = f[5].data[:, 5] * 1.
        self.objname = objname
        self.header = f[0].header
        self.filename = filename
        self.channel = channel
        self.spec_ext = f[6].data
        
    def get_info_from_lrs2raw_object(self, lrs2raw_object):
        '''
        

        Parameters
        ----------
        lrs2raw_object : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        
        self.data = lrs2raw_object.data
        self.error = lrs2raw_object.datae
        self.x = lrs2raw_object.x
        self.y = lrs2raw_object.y
        self.wave = lrs2raw_object.def_wave
        self.normcurve = lrs2raw_object.norm
        self.adrx = lrs2raw_object.adrx
        self.adry = lrs2raw_object.adry
        
        self.skysub = self.data * 1.
        self.sky =  self.data * 0.
        self.ra = lrs2raw_object.x
        self.dec = lrs2raw_object.y
        self.objname = lrs2raw_object.objname
        self.header = lrs2raw_object.header
        self.filename = lrs2raw_object.filename
        self.channel = lrs2raw_object.channel
        self.spec_ext = lrs2raw_object.spec_ext

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
        self.log.propagate = False
        
    def get_barycor(self):
        tm = self.header['EXPTIME']
        tk = Time(self.header['DATE']) + tm / 2. * u.second
        posk = '%s %s' % (self.header['QRA'], self.header['QDEC'])
        sc = SkyCoord(posk, unit=(u.hourangle, u.deg), obstime=tk)
        loc = EarthLocation.of_site('McDonald Observatory')
        vcorr = sc.radial_velocity_correction(kind='barycentric',
                                              location=loc)
        self.barycor = vcorr.value
        
    def fill_bad_fibers(self):
        D = np.sqrt((self.x[np.newaxis,:] - self.x[:, np.newaxis])**2 + 
                    (self.y[np.newaxis,:] - self.y[:, np.newaxis])**2)
        for i in np.arange(self.data.shape[1]):
            bad = np.where(np.isnan(self.data[:, i]))[0]
            for j in bad:
                neigh = D[j, :] < 0.7
                self.data[j, i] = np.nanmean(self.data[neigh, i])
                self.error[j, i] = np.nanmean(self.error[neigh, i])
        
    def collapse_wave(self, detwave=None, wave_window=None, quick_skysub=True,
                      func=np.nanmean, attr='data'):
        # Collapse spectrum 
        if detwave is None:
            detwave = self.detwave
        if wave_window is None:
            wave_window = self.wave_window
        wsel = np.abs(self.wave-detwave) < wave_window
        Y = func(getattr(self, attr)[:, wsel], axis=1)
        if quick_skysub:
            Y = Y - np.nanpercentile(Y, 0.15)   
        return Y
    
    def make_image(self, detwave=None, wave_window=None, quick_skysub=True,
                   attr='data'):
        # Collapse spectrum 
        if detwave is None:
            detwave = self.detwave
        if wave_window is None:
            wave_window = self.wave_window
        wsel = np.abs(self.wave-detwave) < wave_window
        Y = np.nanmean(getattr(self, attr)[:, wsel], axis=1)
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
    
    def plot_image(self, detwave=None, wave_window=None, quick_skysub=True,
                   func=np.nanmean, radius=4., attr='data',
                   sky_radius=5., sky_image=False,
                   sky_annulus=False, inner_sky_radius=3., outer_sky_radius=4.):
        # Collapse spectrum 
        if detwave is None:
            detwave = self.detwave
        if wave_window is None:
            wave_window = self.wave_window
        Y = self.collapse_wave(detwave, wave_window, quick_skysub=False,
                               func=func, attr=attr)
        if (sky_image * (hasattr(self, 'skypix'))):
            Y = np.nanmean(self.data[:, self.skypix], axis=1)
            Y = Y / np.nanmean(Y) * 1e-17
        D = np.sqrt((self.x[np.newaxis,:] - self.x[:, np.newaxis])**2 + 
                    (self.y[np.newaxis,:] - self.y[:, np.newaxis])**2)
        G = np.exp(-0.5 * D**2 / 0.8**2)
        G = G / np.nansum(G, axis=1)[:, np.newaxis]
        Y = np.nansum(Y[:, np.newaxis] * G, axis=1)
        y = Y * 1e17
        vmax = np.nanpercentile(y, 99)
        vmin = np.nanpercentile(y, 1)
        if self.manual:
            xc = self.centroid_x
            yc = self.centroid_y
        else:
            xc, yc = self.find_centroid(detwave, wave_window, func=func, 
                                        radius=radius)
        cax = self.ax.scatter(self.x, self.y, c=y, cmap=plt.get_cmap('coolwarm'),
                          vmin=vmin, vmax=vmax, marker='h', s=250)
        self.ax.scatter(xc, yc, marker='x', color='k', s=100)
        if hasattr(self, 'sky_sel'):
            self.ax.scatter(self.x[self.sky_sel], self.y[self.sky_sel], 
                            marker='.', color='red')
        name = ' '.join(op.basename(self.filename)[:-5].split('_')[1:])
        self.ax.text(0, 4, '%s' % name, fontsize=16, ha='center',
                     va='bottom')
        if hasattr(self, 'sn'):
            self.ax.text(0, 3.0, 'SN = %0.1f' % self.sn, fontsize=16, ha='center',
                         va='bottom', color='w')
        t = np.linspace(0, 2.*np.pi, 366)
        xp = radius * np.cos(t) + xc
        yp = radius * np.sin(t) + yc
        self.ax.plot(xp, yp, 'k--', lw=2)
        if sky_annulus:
            xp = inner_sky_radius * np.cos(t) + xc
            yp = inner_sky_radius * np.sin(t) + yc
            self.ax.plot(xp, yp, 'r--', lw=2)
            xp = outer_sky_radius * np.cos(t) + xc
            yp = outer_sky_radius * np.sin(t) + yc
            self.ax.plot(xp, yp, 'r--', lw=2)
        else:
            xp = sky_radius * np.cos(t) + xc
            yp = sky_radius * np.sin(t) + yc
        self.ax.plot(xp, yp, 'r--', lw=2)
        self.ax.tick_params(axis='both', which='both', direction='in')
        self.ax.tick_params(axis='y', which='both', left=True, right=True)
        self.ax.tick_params(axis='x', which='both', bottom=True, top=True)
        self.ax.tick_params(axis='both', which='major', length=8, width=2)
        self.ax.tick_params(axis='both', which='minor', length=5, width=1)
        self.ax.minorticks_on()
        plt.sca(self.ax)
        plt.colorbar(cax)
        plt.axis([-7., 7., -3.99, 3.99])

    def find_centroid(self, detwave=None, wave_window=None, quick_skysub=True,
                      radius=4, func=np.nanmean, attr='data',
                      use_percentile_sky=False, percentile=25, 
                      sky_radius=2.5):
        # Collapse spectrum 
        if detwave is None:
            detwave = self.detwave
        if wave_window is None:
            wave_window = self.wave_window
        Y = self.collapse_wave(detwave, wave_window, quick_skysub=quick_skysub,
                               func=func, attr=attr)
        if use_percentile_sky:
            sel = Y < np.nanpercentile(Y, percentile)
            xc = np.nanmedian(self.x[sel])
            yc = np.nanmedian(self.y[sel])
            self.sky_sel = np.sqrt((self.x - xc)**2 + (self.y - yc)**2) < sky_radius 
        D = np.sqrt((self.x[np.newaxis,:] - self.x[:, np.newaxis])**2 + 
                    (self.y[np.newaxis,:] - self.y[:, np.newaxis])**2)
        G = np.exp(-0.5 * D**2 / 0.8**2)
        G = G / np.nansum(G, axis=1)[:, np.newaxis]
        Y = np.nansum(Y[:, np.newaxis] * G, axis=1)
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
        if d < 1.5:
            xc, yc = (fit.x_mean.value, fit.y_mean.value)
        else:
            xc, yc = (x0, y0)
        self.log.info('%s Centroid: %0.2f %0.2f' % (op.basename(self.filename), xc, yc))
        self.centroid_x = xc
        self.centroid_y = yc
        self.adrx0 = self.adrx[np.argmin(np.abs(self.wave-detwave))]
        self.adry0 = self.adry[np.argmin(np.abs(self.wave-detwave))]
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
        
    def get_continuum(self, y, sel, bins=25):
        yz = y * 1.
        yz[sel] = np.nan
        x = np.array(np.arange(len(y)), dtype=float)
        xc = np.array([np.nanmean(xi) for xi in np.array_split(x, bins)])
        yc = np.array([np.nanmedian(xi) for xi in np.array_split(yz, bins)])
        sel = np.isfinite(yc)
        if sel.sum() > bins/2.:
            I = interp1d(xc[sel], yc[sel], kind='linear', bounds_error=False,
                         fill_value='extrapolate')
            return I(x)
        else:
            return 0. * y

    def pca_fit(self, H, data, sel):
        sel = sel * np.isfinite(data)
        sol = np.linalg.lstsq(H.T[sel], data[sel])[0]
        res = np.dot(H.T, sol)
        return res
    
    def get_peaks_above_thresh(self, sky, thresh=7., neigh=3):
        mask = sky > thresh * np.nanmedian(sky)
        for i in np.arange(1, neigh):
            mask[i:] += mask[:-i]
            mask[:-i] += mask[i:]
        return mask
    
    def expand_mask(self, mask, pix=6):
        for i in np.arange(1, 6):
            mask[i:] += mask[:-i]
            mask[:-i] += mask[i:]
        return mask
    
    def manual_extraction(self, xc, yc, detwave=None, wave_window=None):
        if detwave is None:
            detwave = self.detwave
        if wave_window is None:
            wave_window = self.wave_window
        self.centroid_x = xc
        self.centroid_y = yc
        self.adrx0 = self.adrx[np.argmin(np.abs(self.wave-detwave))]
        self.adry0 = self.adry[np.argmin(np.abs(self.wave-detwave))]
        self.manual = True
        
    def set_sky_mask(self, xc, yc, radius=2.):
        D = np.sqrt((self.x - xc)**2 + (self.y - yc)**2)
        self.skymask = D > radius
    
    def set_pca_wave_mask(self, lines, redshift, window=5.):
        self.pca_wave_mask = np.zeros(self.wave.shape, dtype=bool)
        for line in lines:
            self.pca_wave_mask += np.abs(self.wave - line*(1+redshift)) < window
    
    def sky_subtraction(self, xc=None, yc=None, sky_radius=5., detwave=None, 
                        wave_window=None, local=False, pca=False, 
                        correct_ftf_from_skylines=False,
                        func=np.nanmean, local_kernel=7., obj_radius=3.,
                        obj_sky_thresh=1., ncomp=25, bins=25,
                        peakthresh=7., pca_iter=1, percentile=25,
                        use_percentile_sky=False, polymodel=False,
                        polyorder=4, sky_annulus=False, inner_sky_radius=2.5,
                        outer_sky_radius=5., line_by_line=False):
        if detwave is None:
            detwave = self.detwave
        if wave_window is None:
            wave_window = self.wave_window
        if self.manual:
            xc = self.centroid_x
            yc = self.centroid_y
        if ((xc is None) and (yc is None)):
            xc, yc = self.find_centroid(detwave=detwave, wave_window=wave_window, 
                                        quick_skysub=True, radius=obj_radius,
                                        func=func, percentile=percentile,
                                        use_percentile_sky=use_percentile_sky,
                                        sky_radius=sky_radius)
        if use_percentile_sky:
            sky_sel = self.sky_sel
        else:
            sky_sel = np.sqrt((self.x - xc)**2 + (self.y - yc)**2) > sky_radius
        if sky_annulus:
            isky_sel = np.sqrt((self.x - xc)**2 + (self.y - yc)**2) > inner_sky_radius
            osky_sel = np.sqrt((self.x - xc)**2 + (self.y - yc)**2) < outer_sky_radius
            sky_sel = isky_sel * osky_sel
        if hasattr(self, 'skymask'):
            sky_sel = sky_sel * self.skymask
        self.skyfiber_sel = sky_sel
        self.fiber_sky = np.nanmedian(self.data[sky_sel], axis=0)
        sky = self.fiber_sky[np.newaxis, :] * np.ones((280,))[:, np.newaxis]
        self.sky = sky
        if polymodel:
            fitter = LevMarLSQFitter()
            for i in np.arange(len(self.wave)):
                Y = self.data[:, i]
                offx = self.adrx[i] - self.adrx0
                offy = self.adry[i] - self.adry0
                x = xc + offx
                y = yc + offy
                d = np.sqrt((self.x - x)**2 + (self.y - y)**2)
                rsel = d > obj_radius
                rsel = rsel * (Y != 0.) * np.isfinite(Y)
                P = Polynomial2D(polyorder, c0_0=np.nanmedian(Y[rsel]))
                if rsel.sum() > (polyorder*5):
                    fit = fitter(P, self.x[rsel], self.y[rsel], Y[rsel])
                    mod = fit(self.x, self.y)
                    self.sky[:, i] = mod
            self.log.info('%s Finished Polynomial Subtraction' %(op.basename(self.filename)))
        
        if line_by_line:
            fitter = LevMarLSQFitter()
            uy = np.unique(self.y)
            
            for i in np.arange(len(self.wave)):
                Y = self.data[:, i]
                offx = self.adrx[i] - self.adrx0
                offy = self.adry[i] - self.adry0
                x = xc + offx
                y = yc + offy
                d = np.sqrt((self.x - x)**2 + (self.y - y)**2)
                rsel = d > obj_radius
                rsel = rsel * (Y != 0.) * np.isfinite(Y)
                P = Polynomial1D(polyorder)
                for uyi in uy:
                    rowsel = self.y == uyi
                    msel = rsel*rowsel
                    if msel.sum() > (polyorder + 3):
                        fit = fitter(P, self.x[msel], Y[msel])
                        mod = fit(self.x[rowsel])
                        self.sky[rowsel, i] = mod
            self.log.info('%s Finished Line by Line Subtraction' %(op.basename(self.filename)))
        self.skysub = self.data - self.sky
        self.pca_sky = self.skysub * np.nan
        self.local_sky = self.skysub * np.nan
        self.cont_model = self.skysub * np.nan
        obj_sel = np.sqrt((self.x - xc)**2 + (self.y - yc)**2) < obj_radius
        self.fiber_obj = np.nanmean(self.skysub[obj_sel], axis=0)
        ratio = self.fiber_obj / self.fiber_sky 
        ignore_waves = ratio > obj_sky_thresh
        ignore_waves = self.expand_mask(ignore_waves)
        if hasattr(self, 'pca_wave_mask'):
            ignore_waves += self.pca_wave_mask
            
        # Only pick sky pixels 7 > the average of the sky continuum
        skypix_alone = self.get_peaks_above_thresh(self.fiber_sky, 
                                              thresh=peakthresh)
        skypix = (skypix_alone * (~ignore_waves)) 
        self.skypix = skypix
        if correct_ftf_from_skylines:
            if skypix.sum() > 50.:
                Y = np.nanmean(self.data[:, skypix], axis=1)
                Y = Y / np.nanmean(Y)
                D = self.data / Y[:, np.newaxis]
                self.fiber_sky = np.nanmedian(D[sky_sel], axis=0)
                sky = self.fiber_sky[np.newaxis, :] * np.ones((280,))[:, np.newaxis]
                self.sky = sky
                self.skysub = D - self.sky
        if local:
            back = self.skysub * 1.
            back[obj_sel] = np.nan
            back[:, ignore_waves] = np.nan
            G = Gaussian2DKernel(local_kernel)
            new_back = interpolate_replace_nans(back, G, **{'boundary':'wrap'})
            new_back[np.isnan(new_back)] = 0.0
            kernel = self.round_up_to_odd(local_kernel)
            new_back = medfilt2d(new_back, kernel)
            self.local_sky = new_back
            self.skysub[:] = self.skysub - new_back
        if pca:
            # Subtract continuum for PCA fitting
            cont_sub = self.skysub * np.nan
            self.pca_sky = self.skysub * 0.
            for k in np.arange(pca_iter):
                for i in np.arange(cont_sub.shape[0]):
                    if np.isfinite(self.skysub[i]).sum() > 100:
                        self.cont_model[i] = self.get_continuum(self.skysub[i], 
                                                                ignore_waves, 
                                                                bins=bins)
                        cont_sub[i] = (self.skysub[i] - self.cont_model[i])
    
                       
                # Fit PCA Model
                yK = cont_sub[self.skyfiber_sel] / self.normcurve
                yK[np.isnan(yK)] = 0.0
                pca = PCA(n_components=ncomp).fit(yK)
                Hk = pca.components_
                
    
                self.pca_sky_temp = np.zeros(self.skysub.shape)
                # Fit residuals for PCA eigenvalues and subtract model
                for i in np.arange(self.skysub.shape[0]):
                    if np.isfinite(self.skysub[i]).sum() > 100:
                        yp = self.skysub[i] - self.get_continuum(self.skysub[i], 
                                                           ignore_waves, bins=bins)
                        yp[np.isnan(yp)] = 0.0
                    else: 
                        continue
                    res = self.pca_fit(Hk, yp / self.normcurve, skypix)
                    ycopy = 0. * yp
                    ycopy[:] = res * self.normcurve
                    ycopy[~skypix_alone] = 0.0
                    self.pca_sky_temp[i] = ycopy
                self.pca_sky += self.pca_sky_temp
                self.skysub = self.skysub - self.pca_sky_temp
            
            
    def extract_spectrum(self, xc=None, yc=None, detwave=None, 
                         wave_window=None, use_aperture=True, radius=2.5,
                         model=None, func=np.nanmean, attr='skysub'):
        if detwave is None:
            detwave = self.detwave
        if wave_window is None:
            wave_window = self.wave_window
        if self.manual:
            xc = self.centroid_x
            yc = self.centroid_y        
        if ((xc is None) and (yc is None)):
            xc, yc = self.find_centroid(detwave=detwave, 
                                        wave_window=wave_window, 
                                        quick_skysub=True,
                                        func=func, attr=attr)
        spectrum = self.wave * np.nan
        spectrum_error = spectrum * 1.
        skyspectrum = spectrum * 1.
        cor = spectrum * 1.
        
        hexarea = 3 * np.sqrt(3.)*(0.59 / (2*np.sqrt(3.)) * 2.)**2 / 2.
        circarea = np.pi * radius**2
        
        for i in np.arange(len(self.wave)):
            offx = self.adrx[i] - self.adrx0
            offy = self.adry[i] - self.adry0
            x = xc + offx
            y = yc + offy
            d = np.sqrt((self.x - x)**2 + (self.y - y)**2)
            rsel = d < radius
            apcor = circarea / (rsel.sum() * hexarea)
            cor = rsel.sum()
            if use_aperture:
                spectrum[i] = np.nansum(self.skysub[rsel, i], axis=0) * apcor
                spectrum_error[i] = np.sqrt(np.nansum(
                                         self.error[rsel, i]**2, axis=0)) * apcor
                skyspectrum[i] = np.nansum(self.sky[rsel, i], axis=0) * apcor
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
                spectrum[i] = (np.nansum(W[rsel] * self.sky[rsel, i]) /
                                  np.nansum(W[rsel]**2))
                spectrum[i] /= cor[i]
                spectrum_error[i] /= cor[i]
        self.spectrum = spectrum
        self.spectrum_error = spectrum_error
        self.skyspectrum = skyspectrum
        flam_unit = (u.erg / u.cm**2 / u.s / u.AA)
        nd = NDData(self.spectrum, unit=flam_unit, mask=np.isnan(self.spectrum),
                    uncertainty=StdDevUncertainty(self.spectrum_error))
        self.spec1D = Spectrum1D(spectral_axis=self.wave*u.AA, 
                                 flux=nd.data*nd.unit, uncertainty=nd.uncertainty,
                                 mask=nd.mask)
        nd = NDData(self.skyspectrum, unit=flam_unit, mask=np.isnan(self.spectrum),
                    uncertainty=StdDevUncertainty(self.spectrum_error))
        self.spec1Dsky = Spectrum1D(spectral_axis=self.wave*u.AA, 
                                 flux=nd.data*nd.unit, uncertainty=nd.uncertainty,
                                 mask=nd.mask)
        
    def rectify(self, newwave):
        new_disp_grid = newwave * u.AA
        fluxc_resample = FluxConservingResampler()
        self.spec1D = fluxc_resample(self.spec1D, new_disp_grid)
        self.spec1Dsky = fluxc_resample(self.spec1Dsky, new_disp_grid)

    def calculate_sn(self, detwave=None, wave_window=None):
        if detwave is None:
            detwave = self.detwave
        if wave_window is None:
            wave_window = self.wave_window
        wsel = np.abs(self.spec1D.spectral_axis.value - self.detwave) < self.wave_window
        signal = np.nansum(self.spec1D.flux.value[wsel])
        noise = np.sqrt(np.nansum(1./self.spec1D.uncertainty.array[wsel]))
        self.sn = signal / noise
        return signal / noise
    
    def calculate_norm(self, detwave=None, wave_window=None, func=np.nanmean):
        if detwave is None:
            detwave = self.detwave
        if wave_window is None:
            wave_window = self.wave_window
        wsel = np.abs(self.spec1D.spectral_axis.value - detwave) < wave_window
        norm = func(self.spec1D.flux.value[wsel])
        self.norm = norm
        
    def normalize(self, norm):
        self.spec1D *= norm
        self.spec1Dsky *= norm
        self.skysub *= norm
        self.error *= norm
        self.spectrum *= norm
        self.spectrum_error *= norm

    def smooth_resolution(self, kernel):
        G = Gaussian1DKernel(kernel)
        self.spec1D = convolution_smooth(self.spec1D, G)
        self.spec1Dsky = convolution_smooth(self.spec1Dsky, G)
    
    def round_up_to_odd(self, f):
        return int(np.ceil(f) // 2 * 2 + 1)
    
    def get_ADR_RAdec(self, xoff, yoff, astrometry_object):
            '''
            Use an astrometry object to convert delta_x and delta_y into
            delta_ra and delta_dec
            
            Parameters
            ----------
            astrometry_object : Astrometry Class
                Contains the astrometry to convert x and y to RA and Dec
            '''
            tRA, tDec = astrometry_object.tp.wcs_pix2world(xoff, yoff, 1)
            ADRra = ((tRA - astrometry_object.ra0) * 3600. *
                          np.cos(np.deg2rad(astrometry_object.dec0)))
            ADRdec = (tDec - astrometry_object.dec0) * 3600.
            return ADRra, ADRdec
    
    def get_astrometry_external(self, R, D, rot):
        xc, yc = (0., 0.)
        s_str = self.header['QRA'] + " " + self.header['QDEC']
        L = s_str
        self.skycoord = SkyCoord(L, unit=(u.hourangle, u.deg))
        A = Astrometry(R, D, 0., 0., 0.)
        A.tp = A.setup_TP(R, D, rot, A.x0,  A.y0)
        lrs2ra, lrs2dec = A.tp.wcs_pix2world(self.lrs2_coords[1] - self.y, 
                                             self.lrs2_coords[0] + self.x, 1)
        self.ra = lrs2ra
        self.dec = lrs2dec
        self.delta_ra = (np.cos(np.deg2rad(self.skycoord.dec.deg)) * -1 *
                         (self.ra - self.skycoord.ra.deg) * 3600.)
        self.delta_dec = (self.dec - self.skycoord.dec.deg) * 3600.
        self.raoff, self.decoff = self.get_ADR_RAdec(self.adrx+xc-self.adrx0, 
                                                     self.adry+yc-self.adry0, A)

    def get_astrometry(self):
        xc, yc = (0., 0.)
        s_str = self.header['QRA'] + " " + self.header['QDEC']
        L = s_str
        self.skycoord = SkyCoord(L, unit=(u.hourangle, u.deg))
        A = Astrometry(self.skycoord.ra.deg, self.skycoord.dec.deg, 
                       self.header['PARANGLE'],
                       xc, yc, x_scale=1., y_scale=1., kind='lrs2')
        self.raoff, self.decoff = self.get_ADR_RAdec(self.adrx+xc-self.adrx0, 
                                                     self.adry+yc-self.adry0, A)
        ra, dec = A.tp.wcs_pix2world(self.x-self.centroid_x, 
                                     self.y-self.centroid_y, 1)
        self.ra = ra
        self.dec = dec
        self.delta_ra = (np.cos(np.deg2rad(self.skycoord.dec.deg)) *
                         (ra - self.skycoord.ra.deg) * 3600.)
        self.delta_dec = (dec - self.skycoord.dec.deg) * 3600.
    
    def write_skysub(self):
        H = fits.HDUList([fits.PrimaryHDU(self.skysub), 
                          fits.ImageHDU(self.local_sky),
                          fits.ImageHDU(self.pca_sky)])
        outname = self.filename.replace('multi', 'skysub')
        H.writeto(outname, overwrite=True)
    
    def write_spectrum(self):
        outname = op.basename(self.filename.replace('multi', 'spectrum'))
        self.spec_ext[1] = self.spectrum
        self.spec_ext[2] = self.spectrum_error
        self.spec_ext[3] = self.skyspectrum
        self.spec_ext[4] = self.spectrum_error
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
        f1.header['CRVAL1'] = self.wave[0]
        f1.header['CRPIX1'] = 1
        f1.header['CDELT1'] = self.wave[1] - self.wave[0]
        for i, name in enumerate(names):
            f1.header['ROW%i' % (i+1)] = name
        f1.writeto(outname, overwrite=True)
        
    def write_cube(self, Dcube, wave, outname):
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
        hdu = fits.PrimaryHDU(np.array(Dcube, dtype='float32'))
        hdu.header['CRVAL1'] = self.skycoord.ra.deg
        hdu.header['CRVAL2'] = self.skycoord.dec.deg
        hdu.header['CRVAL3'] = wave[0]
        hdu.header['CRPIX1'] = int(self.xgrid.shape[0]/2.)
        hdu.header['CRPIX2'] = int(self.xgrid.shape[1]/2.)
        hdu.header['CRPIX3'] = 1
        hdu.header['CTYPE1'] = 'RA---TAN'
        hdu.header['CTYPE2'] = 'DEC--TAN'
        hdu.header['CTYPE3'] = 'WAVE'
        hdu.header['CUNIT1'] = 'deg'
        hdu.header['CUNIT2'] = 'deg'
        hdu.header['CUNIT3'] = 'Angstrom'
        hdu.header['SPECSYS'] = 'TOPOCENT'
        hdu.header['CDELT1'] = (self.xgrid[0, 0] - self.xgrid[0, 1]) / 3600.
        hdu.header['CDELT2'] = (self.ygrid[1, 0] - self.ygrid[0, 0]) / 3600.
        hdu.header['CDELT3'] = (wave[1] - wave[0])
        for key in self.header.keys():
            if key in hdu.header:
                continue
            if ('CCDSEC' in key) or ('DATASEC' in key):
                continue
            if ('BSCALE' in key) or ('BZERO' in key):
                continue
            hdu.header[key] = self.header[key]
        hdu.writeto(outname, overwrite=True)

    def interpolate_spectrum(self, wave, new_wave, spectrum, kernel):
        new_spec = np.interp(new_wave, wave, spectrum, left=np.nan, right=np.nan)
        new_spec = convolve(new_spec, Gaussian1DKernel(kernel), 
                            preserve_nan=True, boundary='extend')
        return new_spec

    def make_cube(self, newwave, scale=0.4, ran=[-7., 7., -7., 7.],
                  
                  
                  radius=0.7, kernel=0.1):
        '''
        Make data cube for a given ifuslot
        
        Parameters
        ----------
        xloc : 1d numpy array
            fiber positions in the x-direction [ifu frame]
        yloc : 1d numpy array
            fiber positions in the y-direction [ifu frame]
        data : 2d numpy array
            fiber spectra [N fibers x M wavelength]
        Dx : 1d numpy array
            Atmospheric refraction in x-direction as a function of wavelength
        Dy : 1d numpy array
            Atmospheric refraction in y-direction as a function of wavelength
        ftf : 1d numpy array
            Average fiber to fiber normalization (1 value for each fiber)
        scale : float
            Spatial pixel scale in arcseconds
        seeing_fac : float
        
        Returns
        -------
        Dcube : 3d numpy array
            Data cube, corrected for ADR
        xgrid : 2d numpy array
            x-coordinates for data cube
        ygrid : 2d numpy array
            y-coordinates for data cube
        '''
        a, b = self.skysub.shape
        N1 = int((ran[1] - ran[0]) / scale)+1
        N2 = int((ran[3] - ran[2]) / scale)+1
        xgrid, ygrid = np.meshgrid(np.linspace(ran[0], ran[1], N1),
                                   np.linspace(ran[2], ran[3], N2))
        Dcube = np.zeros((b,) + xgrid.shape)
        Ecube = np.zeros((b,) + xgrid.shape)

        S = np.zeros((self.skysub.shape[0], 2))
        D = np.sqrt((self.delta_ra - self.delta_ra[:, np.newaxis])**2 +
                    (self.delta_dec - self.delta_dec[:, np.newaxis])**2)
        
        bottom = 3 * np.sqrt(3.)*(0.59 / (2*np.sqrt(3.)) * 2.)**2 / 2.
        area = scale**2 / bottom
        good = np.isnan(self.skysub).sum(axis=1) < 150
        W = np.zeros(D.shape, dtype=bool)
        W[D < radius] = True
        for k in np.arange(b):
            S[:, 0] = self.delta_ra - self.raoff[k]
            S[:, 1] = self.delta_dec - self.decoff[k]
            sel = (self.skysub[:, k] / np.nansum(self.skysub[:, k] * W, axis=1)) <= 0.4
            sel *= np.isfinite(self.skysub[:, k]) * good * (self.error[:, k] > 0.)
            if np.sum(sel) > 15:
                Dcube[k, :, :] = (griddata(S[sel], self.skysub[sel, k],
                                           (xgrid, ygrid), method='linear') * area)
                Ecube[k, :, :] = (griddata(S[sel], self.error[sel, k],
                                           (xgrid, ygrid), method='linear') * area)
        
        NewCUBE = np.zeros((len(newwave), xgrid.shape[0], xgrid.shape[1]))
        NewCUBEe = np.zeros((len(newwave), xgrid.shape[0], xgrid.shape[1]))
        for i in np.arange(xgrid.shape[0]):
            for j in np.arange(xgrid.shape[1]):
                NewCUBE[:, i, j] = self.interpolate_spectrum(self.wave, 
                                                             newwave, 
                                                             Dcube[:, i, j], 
                                                             kernel)
                NewCUBEe[:, i, j] = self.interpolate_spectrum(self.wave, 
                                                              newwave, 
                                                              Ecube[:, i, j], 
                                                              kernel) 
        self.cube = NewCUBE
        self.error_cube = NewCUBEe
        flam_unit = (u.erg / u.cm**2 / u.s / u.AA)
        nd = NDData(np.moveaxis(self.cube, 0, -1), unit=flam_unit, 
                    mask=np.isnan(np.moveaxis(self.cube, 0, -1)),
                    uncertainty=StdDevUncertainty(np.moveaxis(self.error_cube, 0, -1)))
        self.spec3D = Spectrum1D(spectral_axis=newwave*u.AA, 
                                 flux=nd.data*nd.unit, uncertainty=nd.uncertainty,
                                 mask=nd.mask)
        self.xgrid = xgrid
        self.ygrid = ygrid