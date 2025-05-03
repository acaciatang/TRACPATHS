#!/usr/bin/env python3

"""Takes in Video containing BEETags, outputs potential tags, identified tags, final estimated tag positions, and if requested video for checking."""

__appname__ = 'TRAC.py'
__author__ = 'Acacia Tang (tst116@ic.ac.uk)'
__version__ = '0.0.1'

#imports
import sys
import os
import argparse
import cv2
import numpy as np
import math
import statistics
import pandas as pd
import copy
import itertools
from TRACPATHS import basicFx

#####################
#parse options
def parseOptions():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--filename', type=str, default=os.path.dirname(__file__)+'/../Example/test.mp4', help='Path to video.')
    parser.add_argument('-t', '--taglist', type=str, default=os.path.dirname(__file__)+'/../Example/TagList.csv', help='Path to tag list.')
    parser.add_argument('-w', '--write', type=bool, default=True, help='Whether to write output video.')
    parser.add_argument('-o', '--outname', type=str, default=None, help='What to name outputs and where to put them.')
    parser.add_argument('-r', '--red', type=bool, default=True, help='Whether to only use the red channel in analysis.')
    parser.add_argument('-m','--minSize', type=int, default=500, help='Minimum size of white region to be considered.')
    parser.add_argument('-M','--maxSize', type=int, default=2500, help='Maximum size of white region to be considered')
    parser.add_argument('-i', '--i', type=int, default=4, help='How far around corners to search.')

    opt = parser.parse_args()
    return opt

#####################
#reading functions
def findthres(cl1, thres, minSize, maxSize):
    """Given a specific threshold, convert image to black and white using global thresholding and find white blobs of appropriate size."""
    img = copy.deepcopy(cl1)
    img[img > thres] = 255
    img[img < 255] = 0

    contours = cv2.findContours(img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)[0]
    contours = [blob for blob in contours if cv2.contourArea(blob) > 0 and maxSize > cv2.contourArea(cv2.convexHull(blob)) > minSize]
    lefts = [tuple(c[c[:,:,0].argmin()][0]) for c in contours]
    leftcolours = [img[l[1], l[0]+1] for l in lefts]
    white = [contours[i] for i in range(len(contours)) if (leftcolours[i] == 255)]
    return white

def findtags(img, red, minSize, maxSize):
    """Processes a single frame and returns list of ROIs, and background with ROIs drawn on (in red)."""
    #separate out red channel
    if red:
        R = copy.deepcopy(img[:, :, 2])
    else:
        R = copy.deepcopy(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    
    print("Equalising")
    # create a CLAHE object
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl1 = clahe.apply(R)
    bkgd = cv2.cvtColor(cl1,cv2.COLOR_GRAY2RGB)
    cv2.imwrite('test_equalised.png', cl1)

    print("Finging blobs")
    #thresholding
    white = [findthres(cl1, t, minSize, maxSize) for t in [95, 143, 223]] #111, 175
    white = list(itertools.chain.from_iterable(white))

    print("Redrawing to make convex")
    #redraw contours to make convex
    redraw = [cv2.convexHull(blob) for blob in white]

    print("Drawing rectangles")
    #draw rectangles around the points (tilt)    
    rects = [cv2.minAreaRect(blob) for blob in redraw if min(cv2.boundingRect(blob))> 0]
    ROIs = list()
    process = list()

    print("Removing overlaps and determining whether to process further")
    #remove non-tags method 1
    i = 0
    for rect in rects:
        rows,cols = R.shape
        pts1 = np.float32([[rect[0][0]-rect[1][0]/2, rect[0][1]-rect[1][1]/2],[rect[0][0]+rect[1][0]/2, rect[0][1]+rect[1][1]/2], [rect[0][0]-rect[1][0]/2, rect[0][1]+rect[1][1]/2]])
        pts2 = np.float32([[0,0],[12,12],[0,12]])
        M = cv2.getAffineTransform(pts1,pts2)
        dst = cv2.warpAffine(R,M,(cols,rows))
        tag = dst[0:12, 0:12]
        big = cv2.resize(tag, dsize=(tag.shape[0]*12, tag.shape[1]*12))
        compress = np.full((12, 12), 255)
        for x in range(12):
            for y in range(12):
                compress[x,y] = np.median(big[(tag.shape[1]*y):(tag.shape[1]*(y+1)-1), (tag.shape[0]*x):(tag.shape[0]*(x+1)-1)])
        if np.max(compress)-np.min(compress[int(12/4):int(12-12/4), int(12/4):int(12-12/4)]) > 50 or np.min(compress[int(12/4):int(12-12/4), int(12/4):int(12-12/4)]) < np.max(compress)/5:
            if min(rect[1]) > 20:
                process.append(True)
                box = cv2.boxPoints(rect)
                ROIs.append(box)
                box = np.intp(box)
            else:
                process.append(False)
                box = cv2.boxPoints(rect)
                ROIs.append(box)
                box = np.intp(box)
        
        i = i+1

    print("Removing more overlaps")
    # remove overlapping ROIs
    checkOverlap = pd.DataFrame([cv2.boundingRect(blob) for blob in ROIs])
    checkOverlap['centroidX'] = checkOverlap[0]+(checkOverlap[2]/2)
    checkOverlap['centroidY'] = checkOverlap[1]+(checkOverlap[3]/2)
    checkOverlap['process'] = process
    checkOverlap = checkOverlap.sort_values("centroidX", axis=0, ascending=True, inplace=False, kind='quicksort', na_position='last', ignore_index=True)
    removeIndex = list()
    for i in range(1, len(checkOverlap.index)):
        checkAgainst = [j for j in checkOverlap.index if j != i]
        for j in checkAgainst:
            if checkOverlap.loc[i, 0] >= checkOverlap.loc[j, 0] and checkOverlap.loc[i, 1] >= checkOverlap.loc[j, 1]:
                if checkOverlap.loc[i, 0] + checkOverlap.loc[i, 2] <= checkOverlap.loc[j, 0] + checkOverlap.loc[j, 2] and checkOverlap.loc[i, 1] + checkOverlap.loc[i, 3] <= checkOverlap.loc[j, 1] + checkOverlap.loc[j, 3]:
                    removeIndex.append(i)
                elif math.sqrt((checkOverlap.loc[i, 'centroidX'] - checkOverlap.loc[j, 'centroidX'])**2 + (checkOverlap.loc[i, 'centroidY'] - checkOverlap.loc[j, 'centroidY'])**2) < 10:
                    if abs(checkOverlap.loc[i, 2]/checkOverlap.loc[i, 3]-1) > abs(checkOverlap.loc[j, 2]/checkOverlap.loc[j, 3]-1):
                        removeIndex.append(i)
                    

    checkOverlap = checkOverlap.drop(set(removeIndex))

    print("Removing ROIs containing circles")
    #check for shape
    Coordinates = list()
    
    for i in checkOverlap.index:
        print(i)
        pts = tuple(checkOverlap.loc[i])
        if min(pts) < 0:
            continue
        cropped = copy.deepcopy(img[(pts[1]-2):(pts[1]+pts[3]+2), (pts[0]-2):(pts[0]+pts[2]+2)])
        if 0 in cropped.shape:
            continue
        #equalise
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cl = cropped
        cl[:, :, 0] = clahe.apply(cropped[:, :, 0])
        cl[:, :, 1] = clahe.apply(cropped[:, :, 1])
        cl[:, :, 2] = clahe.apply(cropped[:, :, 2])

        #look for circles
        gray = cv2.cvtColor(cl, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)
        rows = gray.shape[0]
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, rows / 8, param1=50, param2=45, minRadius=0, maxRadius=0)
        if circles is None:
            Coordinates.append((max(0, pts[0]-5), max(0, pts[1]-5), min(img.shape[1], pts[2]+5), min(img.shape[0], pts[3]+5), pts[4], pts[5], pts[6]))
    return [Coordinates, bkgd]

#####################
#sorting ROIs
def scoretag(vertexes, cropped, TagList, models):
    """Returns number of mismatches, direction, and ID of best fit given an ROI and a TagList."""
    #get tag from corners
    edges = [math.sqrt((vertexes[p-1][0]-vertexes[p][0])**2 + (vertexes[p-1][1]-vertexes[p][1])**2) for p in range(len(vertexes))]
    edge = math.floor(min(edges))

    if edge < 2:
        return [999, 999, 999, 999]
    
    OneCM = edge/0.3
    rows,cols = cv2.cvtColor(cropped,cv2.COLOR_BGR2GRAY).shape
    pts1 = np.float32([vertexes[0],vertexes[2],vertexes[3]])
    pts2 = np.float32([[0,0],[edge,edge],[0,edge]])
    M = cv2.getAffineTransform(pts1,pts2)
    dst = cv2.warpAffine(cv2.cvtColor(cropped,cv2.COLOR_BGR2GRAY),M,(cols,rows))
    tag = dst[0:edge, 0:edge]   

    #draw tag
    bwtag = cv2.threshold(tag,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]

    if bwtag is None:
        return [999, 999, 999, 999]

    TAG = cv2.resize(bwtag, (6, 6))

    #ID tag
    configs = [TAG, np.rot90(TAG, k=1, axes = (0, 1)), np.rot90(TAG, k=2, axes = (0, 1)), np.rot90(TAG, k=3)]
    difference = []
    direction = []
    
    for m in models:
        diff = [np.sum(abs(m - config))/255 for config in configs]
        difference.append(min(diff))
        direction.append(diff.index(min(diff)))

    return [min(difference), direction[difference.index(min(difference))], TagList[difference.index(min(difference))], OneCM]

def drawmodel(id):
    """Given an id for a BEETag, draws BEETag as a matrix of integers."""
    binary = '%012d' %int(bin(int(id))[2:len(bin(int(id)))])
    tag = np.array([int(bit) for bit in binary])
    tag = np.reshape(tag, (4,3), 'f')
    model = np.ones((6, 6))
    model[1:5, 4] = 0
    model[1:5, 1:4] = tag
    model[1, 4] = sum(model[1:5, 1])%2
    model[2, 4] = sum(model[1:5, 2])%2
    model[3, 4] = sum(model[1:5, 3])%2
    model[4, 4] = np.sum(model[1:5, 1:4])%2

    model = np.rot90(model)*255
    return model.astype(int)

def sortroi(pts, bkgd, img, TagList, models, i):
    """Takes in a ROI and outputs a row that can be combined into a pandas dataframe."""
    if not pts[6]:
        print("Did not process")
        centroidX = round(pts[4])
        centroidY = round(pts[5])
        cv2.rectangle(bkgd,(pts[0],pts[1]),(pts[0]+pts[2], pts[1]+pts[3]),(0,255,255),2)
        cv2.circle(bkgd,(centroidX,centroidY), 5, (0,127,255), -1)
        return [bkgd, ['X', centroidX, centroidY, pts[0]+(pts[2]/2), pts[1]+(pts[3]/2), 'X', 'X', None]]
    
    cropped = img[(pts[1]):(pts[1]+pts[3]), (pts[0]):(pts[0]+pts[2])]

    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

    #convert to black and white with Otsu's thresholding
    bw = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]

    open = cv2.morphologyEx(bw, cv2.MORPH_OPEN, (7,7))
    contours = cv2.findContours(open,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)[0]
    if len(contours) == 0:
        print("Did not process")
        centroidX = round(pts[4])
        centroidY = round(pts[5])
        cv2.rectangle(bkgd,(pts[0],pts[1]),(pts[0]+pts[2], pts[1]+pts[3]),(0,255,255),2)
        cv2.circle(bkgd,(centroidX,centroidY), 5, (0,127,255), -1)
        return [bkgd, ['X', centroidX, centroidY, pts[0]+(pts[2]/2), pts[1]+(pts[3]/2), 'X', 'X', None]]
    areas = np.array([cv2.contourArea(blob) for blob in contours])
    largest = contours[np.where(areas == areas.max())[0][0]]    
    #transform polygon (polygon should just be the tag)
    epsilon = 0.01*cv2.arcLength(largest,True)
    approx = cv2.approxPolyDP(largest,epsilon,True)

    #polygon = cv2.drawContours(cropped, [approx], -1, (0,255,0), 1, cv2.LINE_AA)
    vertexes = basicFx.extremepoints(approx)
    vertexes2 = basicFx.closesttosides(approx)
    
    print("Test1")
    # get all possible positions of each of the four courners as a list of list of tuples
    vertexcombos = []
    for v in vertexes:
        xlist = list(range(max(0, v[0]-i), min(cropped.shape[1], v[0]+i+1)))
        ylist = list(range(max(0, v[1]-i), min(cropped.shape[0], v[1]+i+1)))
        combos = list(itertools.product(*[xlist, ylist]))
        vertexcombos.append(combos)
    
    results = [999, 999, 999, 999]
    for j in range(4):
        choices = vertexcombos[j]
        for c in choices:
            test = copy.deepcopy(vertexes)
            test[j] = c
            newtry = scoretag(test, cropped, TagList, models)
            if newtry[0] < results[0]:
                results = newtry
                vertexes = test

    if results[0] < 2:
        OneCM = results[3]
        centroidX = round(statistics.mean([vertexes[0][0], vertexes[1][0], vertexes[2][0], vertexes[3][0]]) + pts[0])
        centroidY = round(statistics.mean([vertexes[0][1], vertexes[1][1], vertexes[2][1], vertexes[3][1]]) + pts[1])
        frontX = round(statistics.mean([vertexes[results[1]][0], vertexes[(results[1]+1)%4][0]]) + pts[0])
        frontY = round(statistics.mean([vertexes[results[1]][1], vertexes[(results[1]+1)%4][1]]) + pts[1])
        cv2.rectangle(bkgd,(pts[0],pts[1]),(pts[0]+pts[2], pts[1]+pts[3]),(0,255,0),3)
        cv2.putText(bkgd,str(results[2]),(pts[0],pts[1]), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,0),2,cv2.LINE_AA)
        cv2.circle(bkgd,(centroidX,centroidY), 5, (255,255,0), -1)
        cv2.circle(bkgd,(frontX,frontY), 5, (255,255,0), -1)
        return [bkgd, [results[2], centroidX, centroidY, frontX, frontY, OneCM, results[0], None]] 

    print("Test2")
    # get all possible positions of each of the four courners as a list of list of tuples
    vertexcombos = []
    for v in vertexes2:
        xlist = list(range(max(0, v[0]-i), min(cropped.shape[1], v[0]+i+1)))
        ylist = list(range(max(0, v[1]-i), min(cropped.shape[0], v[1]+i+1)))
        combos = list(itertools.product(*[xlist, ylist]))
        vertexcombos.append(combos)
    
    results = [999, 999, 999, 999]
    for j in range(4):
        choices = vertexcombos[j]
        for c in choices:
            test = copy.deepcopy(vertexes2)
            test[j] = c
            newtry = scoretag(test, cropped, TagList, models)
            if newtry[0] < results[0]:
                results = newtry
                vertexes2 = test

    if results[0] < 2:
        OneCM = results[3]
        centroidX = round(statistics.mean([vertexes2[0][0], vertexes2[1][0], vertexes2[2][0], vertexes2[3][0]]) + pts[0])
        centroidY = round(statistics.mean([vertexes2[0][1], vertexes2[1][1], vertexes2[2][1], vertexes2[3][1]]) + pts[1])
        frontX = round(statistics.mean([vertexes2[results[1]][0], vertexes2[(results[1]+1)%4][0]]) + pts[0])
        frontY = round(statistics.mean([vertexes2[results[1]][1], vertexes2[(results[1]+1)%4][1]]) + pts[1])
        cv2.rectangle(bkgd,(pts[0],pts[1]),(pts[0]+pts[2], pts[1]+pts[3]),(0,255,0),3)
        cv2.putText(bkgd,str(results[2]),(pts[0],pts[1]), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,0),2,cv2.LINE_AA)
        cv2.circle(bkgd,(centroidX,centroidY), 5, (255,255,0), -1)
        cv2.circle(bkgd,(frontX,frontY), 5, (255,255,0), -1)
        return [bkgd, [results[2], centroidX, centroidY, frontX, frontY, OneCM, results[0], None]] 
    
    #cannot read
    centroidX = round(statistics.mean([vertexes[0][0], vertexes[1][0], vertexes[2][0], vertexes[3][0]]) + pts[0])
    centroidY = round(statistics.mean([vertexes[0][1], vertexes[1][1], vertexes[2][1], vertexes[3][1]]) + pts[1])
    frontX = round(statistics.mean([vertexes[0][0], vertexes[1][0]]) + pts[0])
    frontY = round(statistics.mean([vertexes[0][0], vertexes[1][0]]) + pts[1])
    cv2.rectangle(bkgd,(pts[0],pts[1]),(pts[0]+pts[2], pts[1]+pts[3]),(0,255,255),2)
    cv2.circle(bkgd,(centroidX,centroidY), 5, (0,127,255), -1)
    cv2.circle(bkgd,(frontX,frontY), 5, (0,127,255), -1)
    return [bkgd, ['X', centroidX, centroidY, frontX, frontY, 'X', 'X', None]]

def processimg(img, TagList, models, f, red, i, minSize, maxSize):
    """Processes one frame, returning two csv files of data on potential tags and read tags."""
    frameData = pd.DataFrame()
    cannotRead = pd.DataFrame()
    print('Finding tags...')
    Coordinates, bkgd = findtags(img, red, minSize, maxSize)
    print('Reading tags...')
    a = 0
    while a < len(Coordinates):
        print(a)
        bkgd, row = sortroi(Coordinates[a], bkgd, img, TagList, models, i)
        if row[0] != 'not tag':
            completerow = pd.DataFrame([f] + row).transpose()
            if row[0] != 'X':
                frameData = pd.concat([frameData,completerow], axis = 0, ignore_index=True)
                a = a+1
            else:
                cannotRead = pd.concat([cannotRead,completerow], axis = 0, ignore_index=True)
                a = a+1
        else:
            a = a+1
    frameData = frameData.rename(columns={0:'frame', 1:'ID', 2:'centroidX', 3:'centroidY', 4:'frontX', 5:'frontY', 6:'1cm', 7:'score', 8:'check'})
    cannotRead = cannotRead.rename(columns={0:'frame', 1:'ID', 2:'centroidX', 3:'centroidY', 4:'frontX', 5:'frontY', 6:'1cm', 7:'score', 8:'check'})
    if len(frameData.index) > 1:
        if len(set(frameData.ID)) < len(frameData.ID): #if there is more than one match for an ID
            doubled = set(frameData.ID[frameData.duplicated(subset='ID')])
            for d in doubled:
                problem = frameData[frameData.ID == d]
                keep = problem[problem.score == min(problem.score)]
                where = problem.loc[keep.index[0]]
                problem = problem.drop(keep.index[0])
                frameData = frameData.drop(problem.index)
                
                test1 = problem.index[problem.centroidX == where.centroidX]
                test2 = problem.index[problem.centroidY == where.centroidY]
                dontAdd = [i for i in problem.index if i in test1 and i in test2]
                problem = problem.drop(dontAdd)
                frameData = pd.concat([frameData, keep], axis=0, ignore_index=True)
                cannotRead = pd.concat([cannotRead, problem], axis=0, ignore_index=True)
    return(frameData, cannotRead, bkgd)

def findparameters(img, TagList, models, minSize, maxSize):
    """Finds correct parameters before processing the whole video."""
    frameData1 = processimg(img, TagList, models, True, minSize, maxSize)[0]
    frameData2 = processimg(img, TagList, models, False, minSize, maxSize)[0]
    if frameData1.shape[0] > frameData2.shape[0]:
        return True
    else:
        return False

def sortrois(filename, outname, taglist, red, write, i, minSize, maxSize):
    """Sorts all ROIs in a video."""
    name = os.path.basename(filename).split(".")[0]
    allTagLists = pd.read_csv(taglist, header = 0, index_col = 0)
    index = os.path.splitext(os.path.basename(filename))[0]
    index = [i for i in allTagLists.index if i in name]
    TagList = (allTagLists.loc[index]).dropna().astype(int).values.tolist()[0]
    models = [drawmodel(id) for id in TagList]
    
    wrangled = pd.DataFrame()
    noID = pd.DataFrame()
    
    f = 0
    cap = cv2.VideoCapture(filename)
    
    if write == True:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(outname + "_raw.mp4", fourcc, 20.0, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    
    if not cap.isOpened():
        cap.open()
    while cap.isOpened():
        print("Processing frame " + str(f))
        ret, img = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        
        cv2.imwrite("processing.png", img)
            
    # Define the codec and create VideoWriter object
        frameData, cannotRead, bkgd = processimg(cv2.imread("processing.png"), TagList, models, f, red, i, minSize, maxSize)
        
        #remove all potential tags that overlap with an identified tag
        
        wrangled = pd.concat([wrangled, frameData], axis=0, ignore_index=True)
        noID = pd.concat([noID, cannotRead], axis=0, ignore_index=True)
        
        if write == True:
            cv2.putText(bkgd,str(f),(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,0),2,cv2.LINE_AA)
            out.write(bkgd)
        print("Finished frame " + str(f))
        f = f+1
    if write == True:
        out.release()
    cap.release()
    cv2.destroyAllWindows()
    output = wrangled.rename(columns={0:'frame', 1:'ID', 2:'centroidX', 3:'centroidY', 4:'frontX', 5:'frontY', 6:'1cm', 7:'score', 8:'check'})
    output.to_csv(path_or_buf = outname + "_raw.csv", na_rep = "NA", index = False)
    output2 = noID.rename(columns={0:'frame', 1:'ID', 2:'centroidX', 3:'centroidY', 4:'frontX', 5:'frontY', 6:'1cm', 7:'score', 8:'check'})
    output2.to_csv(path_or_buf = outname + "_noID.csv", na_rep = "NA", index = False)
    return 0

######################
#main entry point
def main(argv):
    """ Main entry point of the program """
    #check that options given make sense
    opt = parseOptions()
    if vars(opt)['outname'] is None:
        vars(opt)['outname'] = os.path.basename(vars(opt)['filename']).split('.')[0]
    #run code
    sortrois(**vars(opt))
    return 0

if __name__ == "__main__": 
    """Makes sure the "main" function is called from command line"""  
    status = main(sys.argv)
    sys.exit(status)