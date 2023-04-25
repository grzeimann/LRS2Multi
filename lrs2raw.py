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
from fiber_utils import base_reduction, rectify, get_powerlaw
from fiber_utils import get_spectra_error, get_spectra, get_spectra_chi2
from datetime import datetime
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
        self.date = date
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
            if len(filenames) < 1:
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
        cnt = 0
        for channel in side_dict[self.side]:
            self.info[channel] = self.ChannelInfo(channel)
            self.reduce_channel(filename, channel_dict[channel], channel, 
                                tarfolder=tarfolder)
            self.info[channel].filename = name % (date, observation_number,
                                                  expstr, channel)
            self.info[channel].date = date
            self.info[channel].observation_number = observation_number
            self.info[channel].exposure_number = exposure_number
            if cnt == 0:
                area, transparency, iq = self.get_mirror_illumination_guider()
            self.info[channel].area = area
            self.info[channel].transparency = transparency
            self.info[channel].iq = iq
            cnt += 1
    
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
        
    def get_mirror_illumination_throughput(self, fn=None, default=51.4e4, default_t=1.,
                                           default_iq=1.8):
        ''' Use hetillum from illum_lib to calculate mirror illumination (cm^2) '''
        try:
            F = fits.open(fn)
            names = ['RHO_STRT', 'THE_STRT', 'PHI_STRT', 'X_STRT', 'Y_STRT']
            r, t, p, x, y = [F[0].header[name] for name in names]
            mirror_illum = float(os.popen('/home1/00156/drory/illum_lib/hetillum -p'
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
            self.log.info('Using default mirror illumination value')
            area = default
            transpar = default_t
            iq = default_iq
        return area, transpar, iq


    def get_mirror_illumination_guider(self, default=51.4e4, default_t=1.,
                                       default_iq=1.8,
                                       path='/work/03946/hetdex/maverick'):
        try:
            M = []
            path = op.join(path, self.date)
            key = list(self.info.keys())[0]
            print(key)
            DT = self.info[key].header['DATE-OBS']
            exptime = self.info[key].header['EXPTIME']
            y, m, d, h, mi, s = [int(x) for x in [DT[:4], DT[4:6], DT[6:8], DT[9:11],
                                 DT[11:13], DT[13:15]]]
            d0 = datetime(y, m, d, h, mi, s)
            tarfolders = op.join(path, 'gc*', '*.tar')
            tarfolders = sorted(glob.glob(tarfolders))
            print(tarfolders)
            if len(tarfolders) == 0:
                area = 51.4e4
                self.log.info('No guide camera tarfolders found')
                return default, default_t, default_iq
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
            return area, transpar, iq
        except: 
            self.log.info('Using default mirror illumination: %0.2f m^2' % (default/1e4))
            return default, default_t, default_iq