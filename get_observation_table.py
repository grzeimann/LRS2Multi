#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 09:44:08 2023

@author: gregz
"""

import glob
import os.path as op
import numpy as np
import tarfile
from astropy.io import fits
import argparse as ap
import logging
import warnings
import datetime
import sys
from astropy.table import Table

def setup_logging(logname='input_utils'):
    '''Set up a logger for shuffle with a name ``input_utils``.

    Use a StreamHandler to write to stdout and set the level to DEBUG if
    verbose is set from the command line
    '''
    log = logging.getLogger('input_utils')
    if not len(log.handlers):
        fmt = '[%(levelname)s - %(asctime)s] %(message)s'
        fmt = logging.Formatter(fmt)

        level = logging.INFO

        handler = logging.StreamHandler()
        handler.setFormatter(fmt)
        handler.setLevel(level)

        log = logging.getLogger('input_utils')
        log.setLevel(logging.DEBUG)
        log.addHandler(handler)
    return log

warnings.filterwarnings("ignore")


parser = ap.ArgumentParser(add_help=True)

parser.add_argument('output_file', type=str,
                    help='''name of the output file''')

parser.add_argument("-sd", "--start_date",
                    help='''Start Date, e.g., 20170321, YYYYMMDD''',
                    type=str, default=None)

parser.add_argument("-ed", "--end_date",
                    help='''Start Date, e.g., 20170326, YYYYMMDD''',
                    type=str, default=None)

args = parser.parse_args(args=None)
args.log = setup_logging('get_object_table')

dateatt = ['start_date', 'end_date']
if args.start_date is None:
    args.log.error('You must include two of the following: '
                   '"start_date", "end_date", or "date_length"')
    sys.exit(1)
if args.end_date is None:
    args.log.error('You must include two of the following: '
                   '"start_date", "end_date", or "date_length"')
    sys.exit(1)
dates = {}
for da in dateatt:
    dates[da] = datetime.datetime(int(getattr(args, da)[:4]),
                                  int(getattr(args, da)[4:6]),
                                  int(getattr(args, da)[6:]))
    
args.daterange = [datetime.date.fromordinal(i)
                  for i in range(dates[dateatt[0]].toordinal(),
                                 dates[dateatt[1]].toordinal())]
args.daterange = ['%04d%02d%02d' % (daten.year, daten.month, daten.day)
                  for daten in args.daterange]

def get_objects(dates, instrument='lrs2', rootdir='/work/03946/hetdex/maverick',
                targets=None):
    tarfolders = []
    for date in dates:
        tarnames = sorted(glob.glob(op.join(rootdir, date, instrument, '%s0000*.tar' % instrument)))
        for t in tarnames:
            tarfolders.append(t)
    objectdict = {}
    for tarfolder in tarfolders:
        date = tarfolder.split('/')[-3]
        obsnum = int(tarfolder[-11:-4])
        NEXP = 1
        T = tarfile.open(tarfolder, 'r')
        try:
            if targets is not None:
                flag = True
                while flag:
                    a = T.next()
                    name = a.name
                    if name[-5:] == '.fits':
                        b = fits.open(T.extractfile(a))
                        targ = b[0].header['OBJECT']
                        flag = False
                    else:
                        flag = True
                if targ not in targets:
                    continue
            names_list = T.getnames()
            names_list = [name for name in names_list if name[-5:] == '.fits']
            exposures = np.unique([name.split('/')[1] for name in names_list])
            ifuslots = np.unique([op.basename(name).split('_')[1][:3] for name in names_list])
            b = fits.open(T.extractfile(T.getmember(names_list[0])))
            Target = b[0].header['OBJECT']
            
            NEXP = len(exposures)
            for i in np.arange(NEXP):
                objectdict['%s_%07d_%02d' % (date, obsnum, i+1)] = Target
                args.log.info('%s_%07d_%02d: %s' % (date, obsnum, i+1, Target))
        except:
            objectdict['%s_%07d_%02d' % (date, obsnum, NEXP)] = ''
            continue
        T.close()
    return objectdict, ifuslots


########################################################################
# Get the observations for a list of nights
########################################################################
basedir = '/work/03946/hetdex/maverick'
dates = args.daterange  # Needs to be a list
objectdict, ifuslots = get_objects(dates, instrument='virus',rootdir=basedir)
    
########################################################################
# We will use the object dictionary from the previous cell for file IDs
########################################################################
keys = list(objectdict.keys())
values = list(objectdict.values())

T = Table([keys, values], names=['Exposure', 'Description'],
          dtype=['str', 'str'])
T.write(args.output_file, format='ascii.fixed_width_two_line',
        overwrite=True)