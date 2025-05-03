"""Takes in Video containing BEETags, outputs potential tags, identified tags, final estimated tag positions, and if requested video for checking."""

__appname__ = 'TRACPATHS.py'
__author__ = 'Acacia Tang (tst116@ic.ac.uk)'
__version__ = '0.0.1'

#imports
import sys
import os
import pandas as pd
import copy
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import time

import TRACPATHS.PATHS as P

def main(argv):
    print(argv)
    outname = argv[1]
    testvar = argv[2]
    f = open(outname+'_testhz'+testvar+'_time.csv', 'w')
    f.write('file, 1/hz, seconds\n')
    f.close()

    wholeraw = pd.read_csv(outname+'_raw.csv')
    wholenoID = pd.read_csv(outname+'_noID.csv')

    space = int(testvar)
    dist1 = 30*space
    time1 = 30
    dist2 = 100*space
    time2 = 100

    raw = copy.deepcopy(wholeraw[wholeraw["frame"]%space == 0])
    noID = copy.deepcopy(wholenoID[wholenoID["frame"]%space == 0])

    raw = raw[5040 - space - 500*space < raw["frame"]]
    raw = raw[raw["frame"] <= 5040 - space + 500*space]
    noID = noID[5040 - space - 500*space < noID["frame"]]
    noID = noID[noID["frame"] <= 5040 - space + 500*space]

    raw['FRAME'] = raw['frame']
    raw['frame'] = raw['FRAME']/space
    noID['FRAME'] = noID['frame']
    noID['frame'] = noID['FRAME']/space

    raw.reset_index(inplace = True, drop = True)
    noID.reset_index(inplace = True, drop = True)

    raw.drop('FRAME', axis=1, inplace=True)
    noID.drop('FRAME', axis=1, inplace=True)

    t0 = time.process_time()
    wrangled = P.wrangle(raw, noID, outname, dist1, time1, dist2, time2, False)
    f = open(outname+'_testhz'+testvar+'_time.csv', 'a')
    f.write(','.join([outname, testvar, str(time.process_time()-t0)])+'\n')
    f.close()
    
    wrangled.to_csv(path_or_buf = outname + "_testhz" + '_' + testvar + ".csv", na_rep = "NA", index = False)
    print('Done')
    return 0

if __name__ == "__main__": 
    """Makes sure the "main" function is called from command line"""  
    status = main(sys.argv)
    sys.exit(status)
