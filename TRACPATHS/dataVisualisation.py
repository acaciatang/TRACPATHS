#!/usr/bin/env python3

"""Takes in Video containing BEETags, outputs potential tags, identified tags, final estimated tag positions, and if requested video for checking."""

__appname__ = 'TRACPATHS.py'
__author__ = 'Acacia Tang (tst116@ic.ac.uk)'
__version__ = '0.0.1'

#imports
import sys
import os
import argparse
import cv2
import numpy as np
import pandas as pd
import matplotlib.colors as mcolors

#####################
#parse options
def parseOptions():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--filename', type=str, default=os.path.dirname(__file__)+'/../Example/test.mp4', help='Path to video.')
    parser.add_argument('-c', '--csvname', type=str, default='test.csv', help='Path to results from PATHS.')
    parser.add_argument('-o', '--outname', type=str, default=None, help='What to name outputs and where to put them.')
    opt = parser.parse_args()
    return opt

##########################
#draw tracks
def getCoor(csvname, id):
    """Reads csvfile and fills in missing data within a certain time range for specified ID."""
    all = pd.read_csv(csvname)
    subset = all.loc[all["ID"] == id, ["frame", "centroidX", "centroidY"]] # read in frame, centroid coordinates
    
    return subset.reset_index(drop=True)

def getallCoor(csvname):
    """Fills in coordinates for all IDs."""
    all = pd.read_csv(csvname)
    IDs = pd.DataFrame(all["ID"].value_counts())
    IDs.reset_index(level=0, inplace=True)
    IDs.columns = ["ID", "freq"]
    IDs = IDs.loc[IDs["freq"] > 1, ["ID", "freq"]]
    allCoors = [getCoor(csvname, i) for i in IDs["ID"]]
    allCoors = [i for i in allCoors if i.empty == False]
    
    return allCoors

def chooseColour(i):
    """Choose unique colour for each ID."""
    fullList = mcolors.CSS4_COLORS
    names = ['lightcoral', 'silver', 'royalblue', 'pink', 'plum', 'orangered', 'navy', 'lightgreen', 'purple', 'mediumvioletred', 'tomato', 'maroon','slateblue', 'red', 'saddlebrown', 'sandybrown', 'peru', 'palegreen', 'burlywood', 'goldenrod', 'lime', 'darkkhaki', 'orange', 'yellow', 'yellowgreen', 'olivedrab', 'green', 'darkgreen', 'darkseagreen', 'turquoise', 'teal', 'aqua', 'steelblue', 'dodgerblue', 'blue', 'deepskyblue', 'blueviolet', 'magenta', 'deeppink', 'crimson']
    colour = [c for c in fullList[names[i]]]
    r = int(colour[1] + colour[2], 16)
    g = int(colour[3] + colour[4], 16)
    b = int(colour[5] + colour[6], 16)
    return (b, g, r) #opencv uses BGR for some god forsaken reason

def drawLines(allCoors, FRAME, frameNum):
    """ Draws all lines that should be there for one frame """
    frame = cv2.cvtColor(FRAME[:,:,2],cv2.COLOR_GRAY2RGB)
    
    #frame = FRAME
    for i in range(len(allCoors)): #for each ID
        isClosed = False #don't want polygon
        # choose colour 
        color = chooseColour(i)
        # Line thickness of 2 px 
        thickness = 3

        df = allCoors[i]
        df = df[df['frame'] <= frameNum]
        if df.empty == True: # nothing to plot in this frame because there is nothing
            drew = frame
            continue
        if frameNum != df['frame'].max(): # nothing to plot in this frame because tracks have stopped before this frame
            drew = frame
            continue
        test = frameNum - int(df[df['frame'] == frameNum].index.values[0]) #fix me in Wrangled: should only have one entry per ID per frame!
        df['gap'] = [int(df['frame'][i] - i) != test for i in df.index]
        toPlot = df.loc[df['gap'] == False, ["centroidX", "centroidY"]]

        pts = np.array(toPlot)
        pts = pts.reshape((-1, 1, 2)) 
        
        drew = cv2.polylines(frame, np.int32([pts]), isClosed, color, thickness)
    return drew

def drawtracks(filename, csvname, outname):
    """ Main entry point of the program """
    allCoors = getallCoor(csvname)
    
    cap = cv2.VideoCapture(filename)
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(outname + "_tracks.mp4", fourcc, 20.0, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    i=0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
                break
        frame = drawLines(allCoors, frame, i)
        cv2.putText(frame,str(i),(100,100), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,0),2,cv2.LINE_AA)
            
        # write the flipped frame
        out.write(frame)
        print("Wrote frame " + str(i))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        i+=1 

    # Release everything if job is finished
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    return 0

def drawcircles(filename, csvname, outname):
    all = pd.read_csv(csvname)
    all['1cm'] = all['1cm'].replace('X', None)
    radius = int(all['1cm'].astype(float).mean())
    cap = cv2.VideoCapture(filename)
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(outname + "_circles.mp4", fourcc, 20.0, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    i=0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
                break
        subset = all.loc[all["frame"] == i]
        frame = cv2.cvtColor(frame[:, :, 2], cv2.COLOR_GRAY2BGR)
        frame = cv2.putText(frame,str(i),(100,100), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,0),2,cv2.LINE_AA)
        for j in range(len(subset.index)):
            cv2.circle(frame, (int(float(subset.iloc[j]["centroidX"])), int(float(subset.iloc[j]["centroidY"]))), radius, (255, 0, 0), 2)
            cv2.putText(frame,str(int(subset.iloc[j]["ID"])),(int(subset.iloc[j]["frontX"]),int(subset.iloc[j]["frontY"])), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,0,0),2,cv2.LINE_AA)
            cv2.putText(frame,str(int(subset.iloc[j]["ID"])),(int(subset.iloc[j]["centroidX"]),int(subset.iloc[j]["centroidY"])), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,0),2,cv2.LINE_AA)
            
        # write the flipped frame
        out.write(frame)
        print("Wrote frame " + str(i))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        i+=1 

    # Release everything if job is finished
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    return 0

######################
#main entry point
def main(argv):
    """ Main entry point of the program."""
    opt = parseOptions()
    if vars(opt)['outname'] is None:
        vars(opt)['outname'] = os.path.splitext(os.path.basename(vars(opt)['filename']))[0]
    drawcircles(**vars(opt))
    drawtracks(**vars(opt))
    return 0

if __name__ == "__main__": 
    """Makes sure the "main" function is called from command line"""  
    status = main(sys.argv)
    sys.exit(status)