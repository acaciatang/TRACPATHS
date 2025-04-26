#!/usr/bin/env python3

"""Basic functions for TRACPATHS."""

__appname__ = 'basicFx.py'
__author__ = 'Acacia Tang (tst116@ic.ac.uk)'
__version__ = '0.0.1'

#imports
import math
import numpy as np
import copy
import cv2

#####################
#basic functions
def caldis(pt, corner):
    """Calculates the distance between a point and a corner."""
    dis = math.sqrt((pt[0][0]-corner[0])**2 + (pt[0][1]-corner[1])**2)
    return dis

def caldis2(row1, row2):
    """calculates distance between two entries of tags"""
    return math.sqrt((row1['centroidX']-row2['centroidX'])**2 + (row1['centroidY']-row2['centroidY'])**2)

def extremepoints(contour):
    """Returns predicted corners of ROI: based on minimum/maxinum x and y values."""
    Xs = contour[:, 0][:,0]
    Ys = contour[:, 0][:,1]

    minX = min(Xs)
    maxX = max(Xs)
    minY = min(Ys)
    maxY = max(Ys)

    dis1 = [caldis(pt, [minX, minY]) for pt in contour]
    dis2 = [caldis(pt, [maxX, minY]) for pt in contour]
    dis3 = [caldis(pt, [maxX, maxY]) for pt in contour]
    dis4 = [caldis(pt, [minX, maxY]) for pt in contour]
    
    p1 = contour[dis1.index(min(dis1))][0]
    p2 = contour[dis2.index(min(dis2))][0]
    p3 = contour[dis3.index(min(dis3))][0]
    p4 = contour[dis4.index(min(dis4))][0]

    return [p1, p2, p3, p4]

def closesttosides(contour):
    """Returns predicted corners of ROI: points closest to the four corners of a defined ROI"""
    Xs = contour[:, 0][:,0]
    Ys = contour[:, 0][:,1]

    p1 = contour[np.argmin(Xs)][0]
    p2 = contour[np.argmin(Ys)][0]
    p3 = contour[np.argmax(Xs)][0]
    p4 = contour[np.argmax(Ys)][0]

    return [p1, p2, p3, p4]

def lrtb(contour):  
    """Returns predicted corners of ROI: leftmost, topmost, rightmost, and bottommost points"""
    leftmost = contour[contour[:,:,0].argmin()][0]
    rightmost = contour[contour[:,:,0].argmax()][0]
    topmost = contour[contour[:,:,1].argmin()][0]
    bottommost = contour[contour[:,:,1].argmax()][0]
    
    return [leftmost, topmost, rightmost, bottommost]

def convert(vertexes, x, y):
    """Converts coordinates in cropped ROIs back to original coordinates."""
    for i in range(len(vertexes)):
        vertexes[i][0][0] = vertexes[i][0][0] + x
        vertexes[i][0][1] = vertexes[i][0][1] + y

    return vertexes

