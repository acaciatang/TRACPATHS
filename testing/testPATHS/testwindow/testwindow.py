#!/usr/bin/env python3

"""Takes in Video containing BEETags, outputs potential tags, identified tags, final estimated tag positions, and if requested video for checking."""

__appname__ = 'TRACPATHS.py'
__author__ = 'Acacia Tang (tst116@ic.ac.uk)'
__version__ = '0.0.1'

#imports
import sys
import os

import pandas as pd

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import time
import subprocess

import TRACPATHS.PATHS as P

def main(argv):
    outname = argv[1]
    testvar = argv[2]
    f = open(outname+'_testwindow'+testvar+'_time.csv', 'w')
    f.write('file, 1/hz, seconds')
    f.close()

    raw = pd.read_csv(outname+'_raw.csv')
    noID = pd.read_csv(outname+'_noID.csv')
    
    raw.sort_values(by="frame", inplace=True, ignore_index=True)
    noID.sort_values(by="frame", inplace=True, ignore_index=True)

    print(outname)
    dist1 = 30
    time1 = 30
    dist2 = 100
    time2 = 100

    #split data by ID, find gaps
    if testvar != '50000':
        raw = raw[5039 - int(testvar) < raw["frame"]]
        raw = raw[raw["frame"] <= 5039 + int(testvar)]
        noID = noID[5039 - int(testvar) < noID["frame"]]
        noID = noID[noID["frame"] <= 5039 + int(testvar)]

    t0 = time.process_time()
    wrangled = P.wrangle(raw, noID, outname, dist1, time1, dist2, time2, False)
    f = open(outname+'_testwindow'+testvar+'_time.csv', 'a')
    f.write(','.join([outname, testvar, str(time.process_time()-t0)]))
    f.close()
    
    wrangled.to_csv(path_or_buf = outname + "_testwindow" + '_' + testvar + ".csv", na_rep = "NA", index = False)
    return 0

if __name__ == "__main__": 
    """Makes sure the "main" function is called from command line"""  
    status = main(sys.argv)
    sys.exit(status)
