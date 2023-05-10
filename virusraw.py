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
from fiber_utils import base_reduction, rectify, get_powerlaw
from fiber_utils import get_spectra_error, get_spectra, get_spectra_chi2
from datetime import datetime
import tarfile
import sys
import tables
warnings.filterwarnings("ignore")

def get_script_path():
    ''' Get VIRUSRaw absolute path name '''
    return op.dirname(op.abspath(__file__))

class VIRUSRaw:
    ''' 
    Wrapper for reduction routines for raw data 
    
    '''
    def __init__(self, date, observation_number, hdf5file, 
                 exposure_number=1,
                 ifuslot='052', from_archive=False,
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

        # ADR model
        wADR = [3500., 4000., 4500., 5000., 5500.]
        ADRx = [-0.74, -0.4, -0.08, 0.08, 0.20]
        ADRx = np.polyval(np.polyfit(wADR, ADRx, 3), def_wave)
        self.adrx = ADRx
        self.adry = ADRx * 0.
        self.dither_pattern = dither_pattern
        
        self.setup_logging()
        self.date = date
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
        ampcase = '%sLL' % ifuslot
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
            filenames = glob.glob(op.join(path, '*%sLL*.fits' % ifuslot))
            if len(filenames) < 1:
                self.log.error('No files found here %s' % path)
                sys.exit('Cowardly exiting; please check input')
            filename = filenames[0]
        self.ifuslot = ifuslot
        self.info = {}
        name = 'multi_%s_%07d_%s_%s.fits'
        cnt = 0
        channel = 'virus'
        self.info[channel] = self.ChannelInfo(ifuslot, hdf5file)
        self.reduce_channel(filename, ifuslot, 
                            tarfolder=tarfolder)
        self.info[channel].filename = name % (date, observation_number,
                                              expstr, channel)
        self.info[channel].date = date
        self.info[channel].observation_number = observation_number
        self.info[channel].exposure_number = exposure_number
        if cnt == 0:
            area, transparency, iq = self.get_mirror_illumination_guider(channel)
        self.info[channel].area = area
        self.info[channel].transparency = transparency
        self.info[channel].iq = iq
        self.info[channel].data[:] *= 51.4e4 / area / transparency 
        self.info[channel].data[:] /= self.info[channel].response[np.newaxis, :] 
        self.info[channel].datae[:] *= 51.4e4 / area / transparency
        self.info[channel].datae[:] /= self.info[channel].response[np.newaxis, :]
        self.info[channel].header['MILLUM'] = area
        self.info[channel].header['THROUGHP'] = transparency
    
    class ChannelInfo:
        
        # Create channel info
        def __init__(self, ifuslot, hdf5file, 
                     amp_list=['LL', 'LU', 'RL', 'RU']):
            # get h5 file and info
            h5file = tables.open_file(hdf5file, mode='r')
            h5table = h5file.root.Cals
            ifuslots = ['%03d' % i for i in h5table.cols.ifuslot[:]]
            amps = [x.decode("utf-8") for x in h5table.cols.amp[:]]
            
            inds = []
            for amp in amp_list:
                cnt = 0
                for ifusl, ampi in zip(ifuslots, amps):
                    if ifusl == ifuslot and amp == ampi:
                        inds.append(cnt)
                    cnt += 1
            # ifupos, wavelength, masterbias, trace, masterflt
            for attr in ['wavelength', 'masterbias', 'trace', 'masterflt',
                         'ifupos', 'mastertwi']:
                image_list = []
                cnt = 0
                for ind in inds:
                    image_list.append(getattr(h5table.cols, attr)[ind])
                    if attr == 'trace':
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
            self.response = throughput * normalization
    
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

    def reduce_channel(self, filename, ifuslot, channel='virus',
                       tarfolder=None,
                       amp_list=['LL', 'LU', 'RL', 'RU']):
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
        if header['EXPTIME'] < 0:
            header['EXPTIME'] = header['REXPTIME'] + 7
        array_flt2, e2 = base_reduction(filename2, tarfolder=tarfolder)
        array_flt3, e3 = base_reduction(filename3, tarfolder=tarfolder)
        array_flt4, e4 = base_reduction(filename4, tarfolder=tarfolder)

        image = np.vstack([array_flt1, array_flt2, array_flt3, array_flt4])
        E = np.vstack([e1, e2, e3, e4])
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
                                                self.info[channel].response])
        
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
            area = default
            transpar = default_t
            iq = default_iq
        return area, transpar, iq


    def get_mirror_illumination_guider(self, channel, default=51.4e4, 
                                       default_t=1., default_iq=1.8,
                                       path='/work/03946/hetdex/maverick'):
        try:
            M = []
            path = op.join(path, self.date)
            DT = self.info[channel].header['DATE']
            exptime = self.info[channel].header['EXPTIME']
            y, m, d, h, mi, s = [int(x) for x in [DT[:4], DT[5:7], DT[8:10], DT[11:13],
                                 DT[14:16], DT[17:19]]]
            d0 = datetime(y, m, d, h, mi, s)
            tarfolders = op.join(path, 'gc*', '*.tar')
            tarfolders = sorted(glob.glob(tarfolders))
            if len(tarfolders) == 0:
                area = 51.4e4
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
            return default, default_t, default_iq