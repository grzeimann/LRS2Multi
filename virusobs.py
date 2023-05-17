#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 17 12:35:16 2023

@author: gregz
"""


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 09:48:01 2022

@author: gregz
"""
import numpy as np
import os.path as op
import warnings
from astropy.io import fits
from astrometry import Astrometry
from astropy.coordinates import SkyCoord
from astropy.convolution import convolve, Gaussian1DKernel
from astropy.stats import biweight_midvariance
import astropy.units as u
from fiber_utils import get_fiber_to_fiber, find_peaks


warnings.filterwarnings("ignore")

def get_script_path():
    ''' Get VIRUSRaw absolute path name '''
    return op.dirname(op.abspath(__file__))

class VIRUSObs:
    
    def __init__(self, sciRaw_list, arcRaw_list=None, DarkRaw_list=None,
                 twiRaw_list=None, LDLSRaw_list=None, dither_index=[0, 1, 2]):
        '''
        

        Parameters
        ----------
        sciRaw_list : TYPE
            DESCRIPTION.
        arcRaw_list : TYPE, optional
            DESCRIPTION. The default is None.
        DarkRaw_list : TYPE, optional
            DESCRIPTION. The default is None.
        twiRaw_list : TYPE, optional
            DESCRIPTION. The default is None.
        LDLSRaw_list : TYPE, optional
            DESCRIPTION. The default is None.
        dither_index : TYPE, optional
            DESCRIPTION. The default is [0, 1, 2].

        Returns
        -------
        None.

        '''
        self.sciRaw_list = sciRaw_list
        self.arcRaw_list = arcRaw_list
        self.twiRaw_list = twiRaw_list
        self.LDLSRaw_list = LDLSRaw_list
        self.DarkRaw_list = DarkRaw_list
        ########################################################################
        # Correct wavelength default to the lamps taken that night
        ########################################################################
        if not self.twiRaw_list:
            self.get_dark_correction(self.twiRaw_list)
        if not self.LDLSRaw_list:
            self.get_dark_correction(self.LDLSRaw_list)
        self.get_dark_correction(self.sciRaw_list)
        ########################################################################
        # Get Fiber to Fiber (i.e., relative normalization) from twilight frames
        ########################################################################
        self.get_ftf_correction()
        
        ########################################################################
        # Correct wavelength default to the lamps taken that night
        ########################################################################
        self.get_wave_correction()
        
        ########################################################################
        # Subtract Sky
        ########################################################################
        self.get_sky_subtraction()
        
        ########################################################################
        # Subtract Sky
        ########################################################################
        self.get_astrometry(dither_index=dither_index)
    
    def get_ftf_correction(self, low_thresh=0.5):
        '''
        

        Parameters
        ----------
        low_thresh : TYPE, optional
            DESCRIPTION. The default is 0.5.

        Returns
        -------
        None.

        '''
        if not self.twiRaw_list:
            print('No Twi Exposures in twiRaw_list')
            return None
        if not self.LDLS_list:
            print('No Twi Exposures in twiRaw_list')
            return None
        channels = ['virus']
        for channel in channels:
            twidata = 0 * self.twiRaw_list[0].info[channel].data
            twidatae = 0 * self.twiRaw_list[0].info[channel].datae
            wave = self.twiRaw_list[0].info[channel].def_wave
            for twi in self.twiRaw_list:
                twidata[:] += twi.info[channel].data*1e17
                twidatae[:] += (twi.info[channel].data*1e17)**2
            twidatae = np.sqrt(twidatae)
            ftf, mask = get_fiber_to_fiber(twidata, twidatae, wave)
            medvals = np.nanmedian(ftf, axis=1)
            medval = np.nanmedian(medvals)
            mask = medvals < low_thresh * medval
            y = []
            for ldls in self.LDLS_list:
                y.append(ldls.info[channel].data / ftf)
            avg = np.nanmedian(y, axis=0)
            ldlsftf, smask = get_fiber_to_fiber(avg, twidatae, wave)
            avg = avg / ldlsftf
            avgspec = np.nanmedian(avg, axis=0)
            div = avg / avgspec[np.newaxis, :]
            ftf *= div
            for sciRaw in self.sciRaw_list:
                sciRaw.info[channel].ftf = ftf
                sciRaw.info[channel].data /= ftf
                sciRaw.info[channel].datae /= ftf
                sciRaw.info[channel].data[mask] = np.nan
                sciRaw.info[channel].datae[mask] = np.nan
            
    def get_wave_correction(self):
        '''
        

        Returns
        -------
        None.

        '''
        line_list = {}
        line_list['virus'] = [3610.508, 4046.565, 4358.335, 4678.149, 4799.912,
                      4916.068, 5085.822, 5460.750]
        channels = ['virus']
        for channel in channels:
            def_wave = self.arcRaw_list[0].info[channel].def_wave
            arc_spectra = 0. * self.arcRaw_list[0].info[channel].data
            for arc in self.arcRaw_list:
                arc_spectra[:] += arc.info[channel].data*1e17
            lines = np.array(line_list[channel])
            std = np.sqrt(biweight_midvariance(arc_spectra, ignore_nan=True))
            matches = np.ones((arc_spectra.shape[0], len(lines))) * np.nan
            for i, spec in enumerate(arc_spectra):
                xloc, h = find_peaks(spec, thresh=std*15.)
                if len(xloc) == 0.:
                    continue
                waves = np.interp(xloc, np.arange(len(def_wave)), def_wave)
                for j, line in enumerate(lines):
                    ind = np.argmin(np.abs(waves - line))
                    dist = np.abs(waves[ind] - line)
                    if dist < 5.:
                        matches[i, j] = waves[ind]
            model = matches * 0.
            G = Gaussian1DKernel(4.)
            for j in np.arange(len(lines)):       
                model[:, j] = convolve(matches[:, j] - lines[j], G, boundary='wrap')
            wave_correction = 0. * arc_spectra
            for i, spec in enumerate(arc_spectra):
                sel = np.isfinite(model[i])
                wave_correction[i] = np.polyval(np.polyfit(lines[sel], model[i][sel], 2), 
                                                def_wave)
                for science in self.sciRaw_list:
                    science.info[channel].data[i] = np.interp(def_wave,
                                                              def_wave-wave_correction[i],
                                                              science.info[channel].data[i])
                    science.info[channel].datae[i] = np.interp(def_wave,
                                                               def_wave-wave_correction[i],
                                                               science.info[channel].datae[i])
            for science in self.sciRaw_list:
                science.info[channel].wave_correction = wave_correction
        return None

    def get_dark_correction(self, frame_list):
        '''
        

        Parameters
        ----------
        exptime : TYPE, optional
            DESCRIPTION. The default is 360..

        Returns
        -------
        None.

        '''
        if not self.darkRaw_list:
            print('No Dark Exposures in darkRaw_list')
            return None
        channels = ['virus']
        for channel in channels:
            zp = []
            for v in self.darkRaw_list:
                zp.append(v.info[channel].data)
            avg = np.nanmedian(zp, axis=0)
            nchunks = 14
            nfib, ncols = v.info['virus'].data.shape
            Z = np.zeros((nfib, nchunks))
            xi = [np.mean(x) for x in np.array_split(np.arange(ncols), nchunks)]
            x = np.arange(ncols)
            i = 0
            for chunk in np.array_split(avg, nchunks, axis=1):
                Z[:, i] = np.nanmedian(chunk, axis=1)
                i += 1
            image = avg * 0.
            for ind in np.arange(Z.shape[0]):
                p0 = np.polyfit(xi, Z[ind], 4)
                model = np.polyval(p0, x)
                image[ind] = model
            mult1 = (self.darkRaw_list[0].info[channel].area *
                     self.darkRaw_list[0].info[channel].transparency *
                     self.darkRaw_list[0].info[channel].exptime)
            for science in frame_list:
                mult2 = (science.info[channel].area *
                         science.info[channel].transparency *
                         science.info[channel].exptime)
                science.info[channel].dark = image * mult1 / mult2
                science.info[channel].data[:] -= science.info[channel].dark

    def get_sky_subtraction(self, sky=None, skyfibers=None):
        '''
        

        Parameters
        ----------
        sky : TYPE, optional
            DESCRIPTION. The default is None.
        skyfibers : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        '''
        channels = ['virus']
        for channel in channels:
            for science in self.sciRaw_list:
                if sky is not None:
                    science.info[channel].skysub = science.info[channel].data - sky[np.newaxis, :]
                    continue
                medsky = np.nanmedian(science.info[channel].data, axis=0)
                science.info[channel].sky = medsky
                science.info[channel].skysub = science.info[channel].data - medsky[np.newaxis, :]

    def get_astrometry(self, dither_index=None,
                       fplane_file='/work/03730/gregz/maverick/fplaneall.txt'):
        '''
        
    
        Parameters
        ----------
        dither_index : TYPE, optional
            DESCRIPTION. The default is None.
        fplane_file : TYPE, optional
            DESCRIPTION. The default is '/work/03730/gregz/maverick/fplaneall.txt'.
    
        Returns
        -------
        None.
    
        '''
        channels = ['virus']
        for channel in channels:
            for cnt, science in enumerate(self.sciRaw_list):
                xc, yc = (0., 0.)
                s_str = (science.info[channel].header['QRA'] + " " +
                         science.info[channel].header['QDEC'])
                L = s_str
                science.info[channel].skycoord = SkyCoord(L, unit=(u.hourangle, u.deg))
                A = Astrometry(science.info[channel].skycoord.ra.deg, 
                               science.info[channel].skycoord.dec.deg, 
                               science.info[channel].header['PARANGLE'],
                               xc, yc, fplane_file=fplane_file)
                x = science.info[channel].x
                y = science.info[channel].y
                if dither_index is not None:
                    x += science.dither_pattern[dither_index[cnt], 0]
                    y += science.dither_pattern[dither_index[cnt], 1]
                ra, dec = A.get_ifupos_ra_dec(science.ifuslot, x, y)
                science.info[channel].ra = ra
                science.info[channel].dec = dec
                
    def get_ADR_RAdec(self, xoff, yoff, astrometry_object):
        '''
        
    
        Parameters
        ----------
        xoff : TYPE
            DESCRIPTION.
        yoff : TYPE
            DESCRIPTION.
        astrometry_object : TYPE
            DESCRIPTION.
    
        Returns
        -------
        ADRra : TYPE
            DESCRIPTION.
        ADRdec : TYPE
            DESCRIPTION.
    
        '''
        tRA, tDec = astrometry_object.tp.wcs_pix2world(xoff, yoff, 1)
        ADRra = ((tRA - astrometry_object.ra0) * 3600. *
                      np.cos(np.deg2rad(astrometry_object.dec0)))
        ADRdec = (tDec - astrometry_object.dec0) * 3600.
        return ADRra, ADRdec
            
    def get_spectrum(self, ra, dec, radius=3, ring_radius=2):
        '''
        
    
        Parameters
        ----------
        ra : TYPE
            DESCRIPTION.
        dec : TYPE
            DESCRIPTION.
        radius : TYPE, optional
            DESCRIPTION. The default is 3.
        ring_radius : TYPE, optional
            DESCRIPTION. The default is 2.
    
        Returns
        -------
        TYPE
            DESCRIPTION.
    
        '''
        channels = ['virus']
        spec_list, dlist, back_list = ([], [], [])
        for channel in channels:
            for cnt, science in enumerate(self.sciRaw_list):
                dra = (np.cos(np.deg2rad(dec)) * (science.info[channel].ra - ra) * 3600.)
                ddec = (science.info[channel].dec - dec) * 3600
                sel = np.sqrt(dra**2 + ddec**2) < radius
                backsel = (np.sqrt(dra**2 + ddec**2) > radius) * (np.sqrt(dra**2 + ddec**2) < radius+ring_radius)
                for sp in science.info[channel].skysub[sel]:
                    spec_list.append(sp)
                for sp in science.info[channel].skysub[backsel]:
                    back_list.append(sp)
        return spec_list, dlist, np.nansum(np.array(spec_list) - np.nanmedian(back_list), axis=0)
            