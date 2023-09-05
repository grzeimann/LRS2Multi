#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 09:48:01 2022

@author: gregz
"""
import logging
import numpy as np
import os.path as op
import os
import warnings
import glob
from astropy.io import fits
from astropy.table import Table
from astrometry import Astrometry
from astropy.coordinates import SkyCoord
import astropy.units as u
from fiber_utils import base_reduction, rectify, get_powerlaw
from fiber_utils import get_spectra_error, get_spectra, get_spectra_chi2
from datetime import datetime
import tarfile
import sys
import tables
from scipy.signal import medfilt



warnings.filterwarnings("ignore")

def get_script_path():
    ''' Get VIRUSRaw absolute path name '''
    return op.dirname(op.abspath(__file__))

class VIRUSRaw:
    ''' 
    Wrapper for reduction routines for raw data 
    
    '''
    def __init__(self, date, observation_number, h5table, 
                 exposure_number=1,
                 ifuslots=['052'], from_archive=False,
                 basepath='/work/03946/hetdex/maverick',
                 lightmode=True):
        '''
        

        Parameters
        ----------
        date : TYPE
            DESCRIPTION.
        observation_number : TYPE
            DESCRIPTION.
        hdf5file : TYPE
            h5 calibration table
        exposure_number : float, optional
            DESCRIPTION. The default is 1.
        ifuslots : list, optional
            DESCRIPTION. The default is ['052'].
        from_archive : TYPE, optional
            DESCRIPTION. The default is False.
        basepath : TYPE, optional
            DESCRIPTION. The default is '/work/03946/hetdex/maverick'.

        Returns
        -------
        None.

        '''
        
        # Default dither pattern for 3 exposures
        dither_pattern = np.array([[0., 0.], [1.215, -0.70], [1.215, 0.70]])

        # Rectified wavelength
        def_wave = np.linspace(3470., 5540., 1036)
        self.amporder = ['RU', 'RL', 'LL', 'LU']
        # ADR model
        wADR = [3500., 4000., 4500., 5000., 5500.]
        ADRx = [-0.74, -0.4, -0.08, 0.08, 0.20]
        ADRx = np.polyval(np.polyfit(wADR, ADRx, 3), def_wave)
        self.adrx = ADRx
        self.adry = ADRx * 0.
        self.dither_pattern = dither_pattern
        
        self.setup_logging()
        self.date = date
        self.observation = observation_number
        self.exposure_number = exposure_number
        tarfolder = op.join(basepath, date, 'virus', 
                                 'virus%07d.tar' % observation_number)
        if from_archive:
            tarfolder = op.join(basepath, '%s.tar')
        else:
            tarfolder = op.join(basepath, date, 'virus', 
                                     'virus%07d.tar' % observation_number)
        path = op.join(basepath, date, 'virus', 'lrs2%07d' % observation_number,
                       'exp%02d' % exposure_number)
        expstr = 'exp%02d' % exposure_number
        ampcase = '%sLL' % ifuslots[0]
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
                    flag = False
                    gotfile = True
                    T.close()
            if not gotfile:
                self.log.error('No files found for %s' % path)
                sys.exit('Cowardly exiting; please check input')
            filename = name
        else:
            tarfolder = None
            filenames = glob.glob(op.join(path, '*%sLL*.fits' % ifuslots[0]))
            if len(filenames) < 1:
                self.log.error('No files found here %s' % path)
                sys.exit('Cowardly exiting; please check input')
            filename = filenames[0]
        self.ifuslots = ifuslots
        self.info = {}
        name = 'multi_%s_%07d_%s_%s.fits'
        cnt = 0
        for ifuslot in self.ifuslots:
            fname = filename.replace('%sLL' % ifuslots[0], 
                                     '%s%s' % (ifuslot, 'LL'))
            self.log.info('Loading calibration information')
            
            # Basic reduction
            array_flt1, e1, header = base_reduction(fname, tarfolder=tarfolder,
                                                    get_header=True)
            self.info[ifuslot] = self.ChannelInfo(ifuslot, h5table, header,
                                                  amp_list=self.amporder)
            self.log.info('Masking pixels')
            self.mask_from_ldls(ifuslot)
            self.log.info('Reducing ifuslot')
            self.reduce_channel(fname, ifuslot, tarfolder=tarfolder, 
                                amp_list=self.amporder)
            self.info[ifuslot].filename = name % (date, observation_number,
                                                  expstr, ifuslot)
            self.info[ifuslot].date = date
            self.info[ifuslot].observation_number = observation_number
            self.info[ifuslot].exposure_number = exposure_number
            if cnt == 0:
                area, transparency, iq, M = self.get_mirror_illumination_guider(ifuslot)
                self.log.info('Transparency, Area, Exptime: %0.2f, %0.2f, %0.1f' %
                              (transparency, area / 51.4e4, self.info[ifuslot].exptime))
                cnt += 1

            self.info[ifuslot].area = area
            self.info[ifuslot].transparency = transparency
            self.info[ifuslot].iq = iq
            self.info[ifuslot].guider_info = M
            self.info[ifuslot].data[:] *= 51.4e4 / area / transparency 
            self.info[ifuslot].data[:] /= self.info[ifuslot].response[np.newaxis, :] 
            self.info[ifuslot].datae[:] *= 51.4e4 / area / transparency
            self.info[ifuslot].datae[:] /= self.info[ifuslot].response[np.newaxis, :]
            self.info[ifuslot].header['MILLUM'] = area
            self.info[ifuslot].header['THROUGHP'] = transparency
            if lightmode:
                del (self.info[ifuslot].masterflt, 
                     self.info[ifuslot].mastersci, 
                     self.info[ifuslot].masterbias,
                     self.info[ifuslot].image,
                     self.info[ifuslot].plaw,
                     self.info[ifuslot].badpixels)
                    
    
    class ChannelInfo:
        # Create channel info
        def __init__(self, ifuslot, h5table, header,
                     amp_list=['RU', 'RL', 'LL', 'LU']):
            '''
            

            Parameters
            ----------
            ifuslot : TYPE
                DESCRIPTION.
            h5table : TYPE
                DESCRIPTION.
            amp_list : TYPE, optional
                DESCRIPTION. The default is ['RU', 'RL', 'LL', 'LU'].

            Returns
            -------
            None.

            '''
            # get h5 file and info
            ifuslots = ['%03d' % i for i in h5table.cols.ifuslot[:]]
            contids = [i.decode("utf-8") % i for i in h5table.cols.contid[:]]
            specids = [i.decode("utf-8") for i in h5table.cols.specid[:]]
            ifuids = [i.decode("utf-8") for i in h5table.cols.ifuid[:]]
            amps = [x.decode("utf-8") for x in h5table.cols.amp[:]]
            inds = []
            for amp in amp_list:
                cnt = 0
                for ifusl, specid, ifuid, contid, ampi in zip(
                        ifuslots, specids, ifuids, contids, amps):
                    if ((ifusl == ifuslot) and (amp == ampi)):
                        if ((header['SPECID'] == int(specid)) and
                            (header['IFUID'] == ifuid) and
                            (header['CONTID'] == contid)):
                            inds.append(cnt)
                    cnt += 1
            # ifupos, wavelength, masterbias, trace, masterflt
            self.trace_flag = [False, False, False, False]
            self.wavelength_flag = [False, False, False, False]
            for attr in ['wavelength', 'masterbias', 'trace', 'masterflt',
                         'ifupos', 'mastersci', 'lampspec']:
                image_list = []
                cnt = 0
                for ind in inds:
                    image_list.append(getattr(h5table.cols, attr)[ind])
                    if attr == 'wavelength':
                        if ((np.min(image_list[-1]) < 3460) or
                            (np.max(image_list[-1]) > 5550)):
                            self.wavelength_flag[cnt] = True
                            image_list[-1][:] = np.linspace(3500, 5500, 1032)
                    if attr == 'trace':
                        if ((np.min(image_list[-1]) < 2) or
                            (np.max(image_list[-1]) > 1030)):
                            self.trace_flag[cnt] = True
                            image_list[-1][:] = 512 * np.ones((1032,))
                        image_list[-1] = image_list[-1] + cnt * 1032
                        cnt += 1
                image = np.vstack(image_list)
                setattr(self, attr, image)
            self.def_wave = np.linspace(3470, 5540, 1036)
            self.x = self.ifupos[:, 0]
            self.y = self.ifupos[:, 1]
            DIRNAME = get_script_path()
            T = Table.read(op.join(DIRNAME, 'calibrations/virus_throughput.txt'),
                                   format='ascii.fixed_width_two_line')
            throughput = np.array(T['throughput'])
            T = Table.read(op.join(DIRNAME, 'calibrations/virus_normalization.txt'),
                                   format='ascii.fixed_width_two_line')
            normalization = np.array(T['normalization'])
            self.response = throughput / normalization
    
    def setup_logging(self, logname='virusraw'):
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
        
    def mask_from_ldls(self, ifuslot, niter=3, filter_length=11,
                       sigma=5, badcolumnthresh=300):
        '''
        

        Parameters
        ----------
        niter : TYPE, optional
            DESCRIPTION. The default is 3.
        filter_length : TYPE, optional
            DESCRIPTION. The default is 11.
        sigma : TYPE, optional
            DESCRIPTION. The default is 5.
        badcolumnthresh : TYPE, optional
            DESCRIPTION. The default is 300.

        Returns
        -------
        None.

        '''
        image = self.info[ifuslot].masterflt
        bad = np.zeros(image.shape, dtype=bool)
        
        for ind in np.arange(image.shape[0]):
            y = image[ind] * 1.
            for i in np.arange(niter):
                m = medfilt(y, filter_length)
                dev = (image[ind] - m) / np.sqrt(np.where(m<25, 25, m))
                flag = np.abs(dev) > sigma
                y[flag] = m[flag]
            bad[ind] = flag
        for i in np.arange(4):
            li = i * 1032
            hi = (i + 1) * 1032
            badcolumn = bad[li:hi].sum(axis=0) > badcolumnthresh
            bad[li:hi][:, badcolumn] = True
            if self.info[ifuslot].trace_flag[i]:
                bad[li:hi] = True
            if self.info[ifuslot].wavelength_flag[i]:
                bad[li:hi] = True
        self.info[ifuslot].badpixels = bad

    def reduce_channel(self, filename, ifuslot,
                       tarfolder=None,
                       amp_list=['RU', 'RL', 'LL', 'LU']):
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
        
        filename1 = filename.replace('%sLL' % ifuslot, 
                                     '%s%s' % (ifuslot, amp_list[0]))
        filename2 = filename.replace('%sLL' % ifuslot, 
                                     '%s%s' % (ifuslot, amp_list[1]))
        filename3 = filename.replace('%sLL' % ifuslot, 
                                     '%s%s' % (ifuslot, amp_list[2]))
        filename4 = filename.replace('%sLL' % ifuslot, 
                                     '%s%s' % (ifuslot, amp_list[3]))
        
        # Basic reduction
        array_flt1, e1, header = base_reduction(filename1, tarfolder=tarfolder,
                                                get_header=True)
        self.log.info('Base reduction done for %s' % filename1)
        if header['EXPTIME'] < 0:
            header['EXPTIME'] = header['REXPTIME'] + 7
        self.info[ifuslot].exptime = header['EXPTIME']
        array_flt2, e2 = base_reduction(filename2, tarfolder=tarfolder)
        self.log.info('Base reduction done for %s' % filename2)
        array_flt3, e3 = base_reduction(filename3, tarfolder=tarfolder)
        self.log.info('Base reduction done for %s' % filename3)
        array_flt4, e4 = base_reduction(filename4, tarfolder=tarfolder)
        self.log.info('Base reduction done for %s' % filename4)

        
        image = np.vstack([array_flt1, array_flt2, array_flt3, array_flt4])
        E = np.vstack([e1, e2, e3, e4])
        image[:] -= self.info[ifuslot].masterbias
        
        # Get powerlaw
        self.log.info('Getting Powerlaw for %s' % ifuslot)
        plaw = get_powerlaw(image, self.info[ifuslot].trace)
        self.info[ifuslot].image = image
        self.info[ifuslot].plaw = plaw
        image[:] -= self.info[ifuslot].plaw
        
        image[self.info[ifuslot].badpixels] = np.nan
        
        # Get spectra and error
        self.log.info('Getting Spectra for %s' % ifuslot)
        spec = get_spectra(image, self.info[ifuslot].trace)
        specerr = get_spectra_error(E, self.info[ifuslot].trace)
        chi2 = get_spectra_chi2(self.info[ifuslot].masterflt, image, E, 
                                self.info[ifuslot].trace)
        self.info[ifuslot].chi2 = chi2

        # Mark pixels effected by cosmics
        badpix = chi2 > 10.
        specerr[badpix] = np.nan
        spec[badpix] = np.nan
        
        # Rectify spectra
        specrect, errrect = rectify(spec, specerr, 
                                    self.info[ifuslot].wavelength,
                                    self.info[ifuslot].def_wave)
        factor = (6.626e-27 * (3e18 / self.info[ifuslot].def_wave) /
                  header['EXPTIME'] / 51.4e4)
        specrect[:] *= factor
        errrect[:] *= factor
        self.info[ifuslot].orig = spec
        self.info[ifuslot].data = specrect
        self.info[ifuslot].datae = errrect
        self.info[ifuslot].header = header
        self.info[ifuslot].ifuslot = ifuslot
        self.info[ifuslot].objname = header['OBJECT']
        self.info[ifuslot].spec_ext = np.array([self.info[ifuslot].def_wave, 
                                                self.info[ifuslot].def_wave*0.,
                                                self.info[ifuslot].def_wave*0.,
                                                self.info[ifuslot].def_wave*0.,
                                                self.info[ifuslot].def_wave*0.,
                                                self.info[ifuslot].response])
        
    def get_mirror_illumination_throughput(self, fn=None, default=51.4e4, default_t=1.,
                                           default_iq=1.8):
        ''' Use hetillum from illum_lib to calculate mirror illumination (cm^2) '''
        try:
            F = fits.open(fn)
            names = ['RHO_STRT', 'THE_STRT', 'PHI_STRT', 'X_STRT', 'Y_STRT']
            r, t, p, x, y = [F[0].header[name] for name in names]
            mirror_illum = float(os.popen('/work/03730/gregz/maverick/illum_lib/hetillum -p'
                                 ' -x "[%0.4f,%0.4f,%0.4f]" "[%0.4f,%0.4f]" 256' %
                                          (x, y, p, 0.042, 0.014)).read().split('\n')[0])
            area = mirror_illum * default
            if (F[0].header['TRANSPAR'] < 0.1) or (F[0].header['TRANSPAR'] > 1.05):
                transpar = default_t
            else:
                transpar = F[0].header['TRANSPAR'] * 1.
            if (F[0].header['IQ'] < 0.8) or (F[0].header['IQ'] > 4.0):
                iq = default_iq
            else:
                iq = F[0].header['IQ']
        except:
            area = default
            transpar = default_t
            iq = default_iq
        return area, transpar, iq


    def get_mirror_illumination_guider(self, ifuslot, default=51.4e4, 
                                       default_t=1., default_iq=1.8,
                                       path='/work/03946/hetdex/maverick'):
        try:
            M = []
            path = op.join(path, self.date)
            DT = self.info[ifuslot].header['DATE']
            exptime = self.info[ifuslot].header['EXPTIME']
            y, m, d, h, mi, s = [int(x) for x in [DT[:4], DT[5:7], DT[8:10], DT[11:13],
                                 DT[14:16], DT[17:19]]]
            d0 = datetime(y, m, d, h, mi, s)
            tarfolders = op.join(path, 'gc*', '*.tar')
            tarfolders = sorted(glob.glob(tarfolders))
            if len(tarfolders) == 0:
                area = 51.4e4
                return default, default_t, default_iq, np.array([])
            for tarfolder in tarfolders:
                T = tarfile.open(tarfolder, 'r')

                init_list = sorted([name for name in T.getnames()
                                    if name[-5:] == '.fits'])
                final_list = []
                
                for t in init_list:
                    DT = op.basename(t).split('_')[0]
                    y, m, d, h, mi, s = [int(x) for x in [DT[:4], DT[4:6], DT[6:8],
                                         DT[9:11], DT[11:13], DT[13:15]]]
                    d = datetime(y, m, d, h, mi, s)
                    p = (d - d0).seconds

                    if (p > -10.) * (p < exptime+10.):
                        final_list.append(t)
                for fn in final_list:
                    fobj = T.extractfile(T.getmember(fn))
                    M.append(self.get_mirror_illumination_throughput(fobj))
            M = np.array(M)
            sel = M[:, 2] != 1.8
            if sel.sum() > 0.:
                transpar = np.median(M[sel, 1])
                area = np.median(M[sel, 0])
                iq = np.median(M[sel, 2])
            else:
                area = 51.4e4
                transpar = 1.
                iq = 1.8
            return area, transpar, iq, M
        except: 
            return default, default_t, default_iq, np.array([])