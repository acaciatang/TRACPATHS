#imports
import sys
import subprocess
import os
import cv2
import pandas as pd
import time
import glob

import TRACPATHS.TRAC as T

def main(argv):
    """ Main entry point of the program """
    timeFile = open('testsize_time.csv', 'w')
    timeFile.write('file, scalePercent, seconds')
    timeFile.close()
    for frame in glob.glob('/Volumes/MResData/MRes/methodspaper/Data/frames/5040/*.png'):
        img = cv2.imread(frame)
        outname = os.path.join('/Volumes/MResData/MRes/methodspaper/reun_summer2024/Results/testsize', os.path.splitext(os.path.basename(frame))[0])
        alltaglists = pd.read_csv('/Volumes/MResData/MRes/methodspaper/reun_summer2024/data/TagList.csv', header = 0, index_col = 0)
        index = os.path.splitext(os.path.basename(frame))[0]
        index = index[:-6]
        taglist = (alltaglists.loc[index]).dropna().astype(int)
        models = [T.drawmodel(id) for id in taglist]

        for i in range(10):
            scale_percent = int(i+1)/10
            width = int(img.shape[1] * scale_percent)
            height = int(img.shape[0] * scale_percent)
            dim = (width, height)
            
            # resize image
            resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
            t0 = time.process_time()
            frameData, cannotRead, bkgd = T.processimg(resized, taglist, models, f=0, red = True, i=4, minSize = 500*scale_percent, maxSize = 2500*scale_percent)
            timeFile = open('testsize_time.csv', 'a')
            timeFile.write('\n'+','.join([outname, str(scale_percent), str(time.process_time()-t0)]))
            timeFile.close()

            cv2.imwrite(outname + '_' + str(i) + "_testsize.png", bkgd)

            output['centroidX'] = output['centroidX']/scale_percent
            output['centroidY'] = output['centroidY']/scale_percent
            output = frameData.rename(columns={0:'frame', 1:'ID', 2:'centroidX', 3:'centroidY', 4:'frontX', 5:'frontY', 6:'1cm', 7:'score', 8:'check'})
            output.to_csv(path_or_buf = outname + '_' + str(i) + "_testi_raw.csv", na_rep = "NA", index = False)
            #output2 = cannotRead.rename(columns={0:'frame', 1:'ID', 2:'centroidX', 3:'centroidY', 4:'frontX', 5:'frontY', 6:'1cm', 7:'score', 8:'check'})
            #output2.to_csv(path_or_buf = outname + '_' + str(i) + "_testi_noID.csv", na_rep = "NA", index = False)

    return 0

def main(argv):
    """ Main entry point of the program """
    f = open('testsize_time.csv', 'w')
    f.write('file, scalePercent, seconds')
    f.close()
    for dir, sub, files in subprocess.os.walk('/rds/general/user/tst116/home/BPlusPlus/frames'):
        for f in files:
            img = cv2.imread(subprocess.os.path.join(dir, f))
            outname = os.path.splitext(os.path.basename(f))[0]
            alltaglists = pd.read_csv("TagList.csv", header = 0, index_col = 0)
            index = os.path.splitext(os.path.basename(f))[0]
            index = index[:-6]
            taglist = (alltaglists.loc[index]).dropna().astype(int)
            models = [T.drawmodel(id) for id in taglist]

            scale_percent = int(argv[1])/10
            width = int(img.shape[1] * scale_percent)
            height = int(img.shape[0] * scale_percent)
            dim = (width, height)
            
            # resize image
            resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

            t0 = time.process_time()
            frameData, cannotRead, bkgd = T.processimg(img=resized, outname=outname, taglist=taglist, models=models, f=0, red = True, i=4, MAX = 2500*scale_percent, MIN = 500*scale_percent)
            print (time.process_time()-t0, "seconds process time")
            f = open('testsize_time.csv', 'a')
            f.write(','.join([outname, argv[1], time.process_time()-t0]))
            f.close()
            cv2.imwrite(outname + '_' + argv[1] + "_testsize.png", bkgd)

            output = frameData.rename(columns={0:'frame', 1:'ID', 2:'centroidX', 3:'centroidY', 4:'frontX', 5:'frontY', 6:'1cm', 7:'score', 8:'check'})
            output.to_csv(path_or_buf = outname + '_' + argv[1] + "_testsize_raw.csv", na_rep = "NA", index = False)
            output2 = cannotRead.rename(columns={0:'frame', 1:'ID', 2:'centroidX', 3:'centroidY', 4:'frontX', 5:'frontY', 6:'1cm', 7:'score', 8:'check'})
            output2.to_csv(path_or_buf = outname + '_' + argv[1] + "_testsize_noID.csv", na_rep = "NA", index = False)

    return 0

if __name__ == "__main__": 
    """Makes sure the "main" function is called from command line"""  
    status = main(sys.argv)
    sys.exit(status)