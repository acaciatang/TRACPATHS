
#imports
import sys
import glob
import os
import cv2
import pandas as pd
import time

import TRACPATHS.TRAC as T

def main(argv):
    """ Main entry point of the program """
    timeFile = open('testi_time.csv', 'w')
    timeFile.write('file, i, seconds')
    timeFile.close()
    for frame in glob.glob('/Volumes/MResData/MRes/methodspaper/Data/frames/5040/*.png'):
        img = cv2.imread(os.path.join(frame))
        outname = os.path.join('/Volumes/MResData/MRes/methodspaper/reun_summer2024/Results/testi', os.path.splitext(os.path.basename(frame))[0])
        alltaglists = pd.read_csv('/Volumes/MResData/MRes/methodspaper/reun_summer2024/data/TagList.csv', header = 0, index_col = 0)
        index = os.path.splitext(os.path.basename(frame))[0]
        index = index[:-6]
        taglist = (alltaglists.loc[index]).dropna().astype(int)
        models = [T.drawmodel(id) for id in taglist]
        for i in range(10):
            t0 = time.process_time()
            frameData, cannotRead, bkgd = T.processimg(img, taglist, models, f=0, red = True, i=i, minSize=500, maxSize=2500)
            timeFile = open('testi_time.csv', 'a')
            timeFile.write('\n'+','.join([outname, str(i), str(time.process_time()-t0)]))
            timeFile.close()

            cv2.imwrite(outname + '_' + str(i) + "_testi.png", bkgd)

            output = frameData.rename(columns={0:'frame', 1:'ID', 2:'centroidX', 3:'centroidY', 4:'frontX', 5:'frontY', 6:'1cm', 7:'score', 8:'check'})
            output.to_csv(path_or_buf = outname + '_' + str(i) + "_testi_raw.csv", na_rep = "NA", index = False)
            #output2 = cannotRead.rename(columns={0:'frame', 1:'ID', 2:'centroidX', 3:'centroidY', 4:'frontX', 5:'frontY', 6:'1cm', 7:'score', 8:'check'})
            #output2.to_csv(path_or_buf = outname + '_' + str(i) + "_testi_noID.csv", na_rep = "NA", index = False)

    return 0

if __name__ == "__main__": 
    """Makes sure the "main" function is called from command line"""  
    status = main(sys.argv)
    sys.exit(status)