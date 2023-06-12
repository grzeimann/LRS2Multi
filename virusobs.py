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
from astropy.stats import biweight_location as biweight
import astropy.units as u
from fiber_utils import get_fiber_to_fiber, find_peaks
from scipy.interpolate import griddata


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
        self.def_wave = np.linspace(3470, 5540, 1036)
        self.sciRaw_list = sciRaw_list
        self.arcRaw_list = arcRaw_list
        self.twiRaw_list = twiRaw_list
        self.LDLSRaw_list = LDLSRaw_list
        self.DarkRaw_list = DarkRaw_list
        self.ifuslots = self.sciRaw_list[0].ifuslots
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
        self.get_delta_ra_dec()
        self.get_ADR_RAdec()
        
            
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
        self.sciRaw_list[0].log.info('Getting Fiber to Fiber Correction')
        T = np.zeros((448 * len(self.ifuslots), 1036))
        E = T * 0.
        for i, ifuslot in enumerate(self.ifuslots):
            li = i * 448
            hi = (i + 1) * 448
            twidata = 0 * self.twiRaw_list[0].info[ifuslot].data
            twidatae = 0 * self.twiRaw_list[0].info[ifuslot].datae
            wave = self.twiRaw_list[0].info[ifuslot].def_wave
            for twi in self.twiRaw_list:
                twidata[:] += twi.info[ifuslot].data*1e17
                twidatae[:] += (twi.info[ifuslot].data*1e17)**2
            twidatae = np.sqrt(twidatae)
            
            T[li:hi] = twidata
            E[li:hi] = twidatae
    
        ftf, mask = get_fiber_to_fiber(T, E, wave)
        medvals = np.nanmedian(ftf, axis=1)
        mask = medvals < low_thresh
        for i, ifuslot in enumerate(self.ifuslots):
            li = i * 448
            hi = (i + 1) * 448
            if not self.LDLSRaw_list:
                print('No LDLS Exposures in ldlsRaw_list')
            else:
                y = []
                for ldls in self.LDLSRaw_list:
                    y.append(ldls.info[ifuslot].data / ftf[li:hi])
                avg = np.nanmedian(y, axis=0)
                ldlsftf, smask = get_fiber_to_fiber(avg, E[li:hi], wave)
                avg = avg / ldlsftf
                avgspec = np.nanmedian(avg, axis=0)
                div = avg / avgspec[np.newaxis, :]
                ftf[li:hi] *= div
            for sciRaw in self.sciRaw_list:
                sciRaw.info[ifuslot].ftf = ftf[li:hi]
                sciRaw.info[ifuslot].data /= ftf[li:hi]
                sciRaw.info[ifuslot].datae /= ftf[li:hi]
                sciRaw.info[ifuslot].data[mask[li:hi]] = np.nan
                sciRaw.info[ifuslot].datae[mask[li:hi]] = np.nan
            
    def get_wave_correction(self):
        '''
        

        Returns
        -------
        None.

        '''
        if not self.arcRaw_list:
            print('No arc Exposures in arcRaw_list')
            return None
        self.sciRaw_list[0].log.info('Getting Wavelength Correction')
        line_list = [3610.508, 4046.565, 4358.335, 4678.149, 4799.912,
                      4916.068, 5085.822, 5460.750]
        for ifuslot in self.ifuslots:
            def_wave = self.arcRaw_list[0].info[ifuslot].def_wave
            arc_spectra = 0. * self.arcRaw_list[0].info[ifuslot].data
            for arc in self.arcRaw_list:
                arc_spectra[:] += arc.info[ifuslot].data*1e17
            lines = np.array(line_list)
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
                    science.info[ifuslot].data[i] = np.interp(def_wave,
                                                              def_wave-wave_correction[i],
                                                              science.info[ifuslot].data[i])
                    science.info[ifuslot].datae[i] = np.interp(def_wave,
                                                               def_wave-wave_correction[i],
                                                               science.info[ifuslot].datae[i])
            for science in self.sciRaw_list:
                science.info[ifuslot].wave_correction = wave_correction
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
        if not self.DarkRaw_list:
            print('No Dark Exposures in darkRaw_list')
            return None
        self.sciRaw_list[0].log.info('Subtracting Dark')
        for ifuslot in self.ifuslots:
            zp = []
            for v in self.DarkRaw_list:
                zp.append(v.info[ifuslot].data)
            avg = np.nanmedian(zp, axis=0)
            nchunks = 14
            nfib, ncols = v.info[ifuslot].data.shape
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
            mult1 = (self.DarkRaw_list[0].info[ifuslot].area *
                     self.DarkRaw_list[0].info[ifuslot].transparency *
                     self.DarkRaw_list[0].info[ifuslot].exptime)
            for science in frame_list:
                mult2 = (science.info[ifuslot].area *
                         science.info[ifuslot].transparency *
                         science.info[ifuslot].exptime)
                science.info[ifuslot].dark = image * mult1 / mult2
                science.info[ifuslot].data[:] -= science.info[ifuslot].dark

    def get_sky_subtraction(self, skylist=None, skyfibers=None):
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
        self.sciRaw_list[0].log.info('Subtracting Sky')
        if skylist is None:
            for science in self.sciRaw_list:
                fibers = np.zeros((448*len(self.ifuslots), 1036))
                for i, ifuslot in enumerate(self.ifuslots):
                    li = i * 448
                    hi = (i + 1) * 448
                    fibers[li:hi] = science.info[ifuslot].data
                medsky = biweight(fibers, axis=0, ignore_nan=True)
                for i, ifuslot in enumerate(self.ifuslots):
                    science.info[ifuslot].sky = medsky
                    science.info[ifuslot].skysub = science.info[ifuslot].data - medsky[np.newaxis, :]
        else:
            for sky, science in zip(skylist, self.sciRaw_list):
                for i, ifuslot in enumerate(self.ifuslots):
                    science.info[ifuslot].sky = sky
                    science.info[ifuslot].skysub = science.info[ifuslot].data - sky[np.newaxis, :]

        

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
        self.sciRaw_list[0].log.info('Getting Astrometry')
        for ifuslot in self.ifuslots:
            for cnt, science in enumerate(self.sciRaw_list):
                xc, yc = (0., 0.)
                s_str = (science.info[ifuslot].header['QRA'] + " " +
                         science.info[ifuslot].header['QDEC'])
                L = s_str
                science.info[ifuslot].skycoord = SkyCoord(L, unit=(u.hourangle, u.deg))
                A = Astrometry(science.info[ifuslot].skycoord.ra.deg, 
                               science.info[ifuslot].skycoord.dec.deg, 
                               science.info[ifuslot].header['PARANGLE'],
                               xc, yc, fplane_file=fplane_file)
                science.info[ifuslot].astrometry = A
                x = science.info[ifuslot].x
                y = science.info[ifuslot].y
                if dither_index is not None:
                    x += science.dither_pattern[dither_index[cnt], 0]
                    y += science.dither_pattern[dither_index[cnt], 1]
                ra, dec = A.get_ifupos_ra_dec(science.info[ifuslot].ifuslot, x, y)
                science.info[ifuslot].ra = ra
                science.info[ifuslot].dec = dec
    
    def get_delta_ra_dec(self):
        '''


        Returns
        -------
        None.

        '''
        mra = np.mean(np.hstack([science.info[ifuslot].ra 
                                 for science in self.sciRaw_list 
                                 for ifuslot in self.ifuslots]))
        mdec = np.mean(np.hstack([science.info[ifuslot].dec 
                                  for science in self.sciRaw_list 
                                  for ifuslot in self.ifuslots]))
        for ifuslot in self.ifuslots:
            for cnt, science in enumerate(self.sciRaw_list):
                dra = (np.cos(np.deg2rad(mdec)) * (science.info[ifuslot].ra - mra) * 3600.)
                ddec = (science.info[ifuslot].dec - mdec) * 3600
                science.info[ifuslot].ra_center = mra
                science.info[ifuslot].dec_center = mdec
                science.info[ifuslot].dra = dra
                science.info[ifuslot].ddec = ddec
                
    def get_ADR_RAdec(self):
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
        for ifuslot in self.ifuslots:
            for cnt, science in enumerate(self.sciRaw_list):
                xoff = science.adrx
                yoff = science.adry
                tRA, tDec = science.info[ifuslot].astrometry.tp.wcs_pix2world(xoff, yoff, 1)
                ADRra = ((tRA - science.info[ifuslot].astrometry.ra0) * 3600. *
                         np.cos(np.deg2rad(science.info[ifuslot].astrometry.dec0)))
                ADRdec = (tDec - science.info[ifuslot].astrometry.dec0) * 3600.
                science.info[ifuslot].adrra = ADRra
                science.info[ifuslot].adrdec = ADRdec
            
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
        spec_list, back_list = ([], [])
        
        for ifuslot in self.ifuslots:
            for cnt, science in enumerate(self.sciRaw_list):
                dra = (np.cos(np.deg2rad(dec)) * (science.info[ifuslot].ra - ra) * 3600.)
                ddec = (science.info[ifuslot].dec - dec) * 3600
                sel = np.sqrt(dra**2 + ddec**2) < radius
                backsel = (np.sqrt(dra**2 + ddec**2) > radius) * (np.sqrt(dra**2 + ddec**2) < radius+ring_radius)
                for sp in science.info[ifuslot].skysub[sel]:
                    spec_list.append(sp)
                for sp in science.info[ifuslot].skysub[backsel]:
                    back_list.append(sp)
        self.fiber_aperture_list = spec_list
        self.fiber_back_list = back_list
        aperture_correction = (len(self.fiber_aperture_list) * np.pi * 0.75**2 /
                              (np.pi * radius**2))
        self.extracted_spectrum = (np.nansum(np.array(self.fiber_aperture_list) -
                                            np.nanmedian(self.fiber_back_list),
                                            axis=0) / aperture_correction)
            
    def make_cube(self, scale=0.7, ran=[-50, 50, -50, 50]):
        '''
        

        Parameters
        ----------
        scale : TYPE, optional
            DESCRIPTION. The default is 0.7.
        ran : TYPE, optional
            DESCRIPTION. The default is [-50, 50, -50, 50].
        seeing_fac : TYPE, optional
            DESCRIPTION. The default is 1.8.
        radius : TYPE, optional
            DESCRIPTION. The default is 1.5.

        Returns
        -------
        TYPE
            DESCRIPTION.
        TYPE
            DESCRIPTION.
        TYPE
            DESCRIPTION.
        TYPE
            DESCRIPTION.

        '''
        data, error = ([], [])
        for ifuslot in self.ifuslots:
            for cnt, science in enumerate(self.sciRaw_list):
                data.append(science.info[ifuslot].skysub)
                error.append(science.info[ifuslot].datae)
        data, error = [np.vstack(x) for x in [data, error]]
        a, b = data.shape
        N1 = int((ran[1] - ran[0]) / scale) + 1
        N2 = int((ran[3] - ran[2]) / scale) + 1
        xgrid, ygrid = np.meshgrid(np.linspace(ran[0], ran[1], N1),
                                   np.linspace(ran[2], ran[3], N2))
        Dcube = np.zeros((b,)+xgrid.shape)
        Ecube = Dcube * 0.
        area = np.pi * 0.75**2
        for k in np.arange(b):
            S = np.zeros((data.shape[0], 2))
            for ifuslot in self.ifuslots:
                for cnt, science in enumerate(self.sciRaw_list):
                    li = cnt * 448
                    hi = (cnt + 1) * 448
                    S[li:hi, 0] = science.info[ifuslot].dra - science.info[ifuslot].adrra[k]
                    S[li:hi, 1] = science.info[ifuslot].ddec - science.info[ifuslot].adrdec[k]
            sel = np.isfinite(data[:, k])
            if np.any(sel):
                grid_z = griddata(S[sel], data[sel, k],
                                  (xgrid, ygrid), method='linear')
                Dcube[k, :, :] =  grid_z * scale**2 / area
                grid_z = griddata(S[sel], error[sel, k],
                                  (xgrid, ygrid), method='linear')
                Ecube[k, :, :] = grid_z * scale**2 / area
        self.cube = Dcube
        self.error_cube = Ecube
        self.xgrid = xgrid
        self.ygrid = ygrid


    def write_cube(self, outname):
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
        wave = self.def_wave
        for ifuslot in self.ifuslots:
            for cnt, science in enumerate(self.sciRaw_list):
                he = science.info[ifuslot].header
                racenter = science.info[ifuslot].ra_center
                deccenter = science.info[ifuslot].dec_center
        hdu = fits.PrimaryHDU(np.array(self.cube, dtype='float32'))
        hdu.header['CRVAL1'] = racenter
        hdu.header['CRVAL2'] = deccenter
        hdu.header['CRVAL3'] = wave[0]
        hdu.header['CRPIX1'] = self.xgrid.shape[1] / 2.
        hdu.header['CRPIX2'] = self.xgrid.shape[0] / 2.
        hdu.header['CRPIX3'] = 1
        hdu.header['CTYPE1'] = 'pixel'
        hdu.header['CTYPE2'] = 'pixel'
        hdu.header['CTYPE3'] = 'pixel'
        hdu.header['CDELT1'] = (self.xgrid[0, 1] - self.xgrid[0, 0])/3600.
        hdu.header['CDELT2'] = (self.ygrid[1, 0] - self.ygrid[0, 0])/3600.
        hdu.header['CDELT3'] = wave[1] - wave[0]
        for key in he.keys():
            if key in hdu.header:
                continue
            if ('CCDSEC' in key) or ('DATASEC' in key):
                continue
            if ('BSCALE' in key) or ('BZERO' in key):
                continue
            hdu.header[key] = he[key]
        hdu.writeto(outname, overwrite=True)