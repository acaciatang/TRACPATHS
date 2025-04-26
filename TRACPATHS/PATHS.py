#!/usr/bin/env python3

"""Takes in Video containing BEETags, outputs potential tags, identified tags, final estimated tag positions, and if requested video for checking."""

__appname__ = 'TRACPATHS.py'
__author__ = 'Acacia Tang (tst116@ic.ac.uk)'
__version__ = '0.0.1'

#imports
import os
import sys
import math
import pandas as pd
import copy
import basicFx
import argparse
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

#####################
#parse options
def parseOptions():
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--rawname', type=str, default='test_raw.csv', help='Path to video.')
    parser.add_argument('-n', '--noidname', type=str, default='test_noID.csv', help='Path to tag list.')
    parser.add_argument('-o', '--outname', type=str, default=None, help='What to name outputs and where to put them.')
    parser.add_argument('-j', '--nojump', type=bool, default=False, help='Whether to run algorithm to remove jumps in a path.')
    parser.add_argument('-d1','--dist1', type=int, default=30, help='Maximum distance to be considered part of a path.')
    parser.add_argument('-t1','--time1', type=int, default=30, help='Maximum frames passed to be considered part of a path.')
    parser.add_argument('-d2', '--dist2', type=int, default=100, help='Maximum distance to extrapolate linearly.')
    parser.add_argument('-t2', '--time2', type=int, default=100, help='Maximum frames to extrapolate linearly over.')

    opt = parser.parse_args()
    return opt

######################
#wrangling
def removeduplicates(raw, noID):
    sus = raw[raw.duplicated(subset=['ID', 'frame'], keep=False)]
    sus.iloc[:]['ID'] = 'X'
    noDuplicates = raw.drop(sus.index)
    noID = pd.concat([noID, sus], axis = 0, ignore_index=True)
    return noDuplicates, noID

def removejumps(oneID, dist1):
    removeme = pd.DataFrame()
    oneID = oneID.sort_values("frame", axis=0, ascending=True, inplace=False, kind='quicksort', na_position='last', ignore_index=True)
    phydis = [basicFx.caldis2(oneID.loc[oneID.index[r-1]], oneID.loc[oneID.index[r]]) for r in range(1, oneID.shape[0])]
    timedis = [oneID.frame[r] - oneID.frame[r-1] for r in range(1, oneID.shape[0])]
    if len(phydis) > 1:
        sus = [oneID.index[i+1] for i in range(len(phydis)-2) if phydis[i] > dist1 and phydis[i]/timedis[i] > dist1]
        if len(sus)%2 == 0:
            pairs = [(sus[i], sus[i+1]) for i in range(0, len(sus)-1, 2)]
            removeindex = [i for p in pairs for i in range(p[0], p[1])]
        else:
            pairs1 = [(sus[i], sus[i+1]) for i in range(1, len(sus)-1, 2)]
            removeindex1 = [i for p in pairs1 for i in range(p[0], p[1])]
            removeindex1.append(sus[0]-1)
            pairs2 = [(sus[i], sus[i+1]) for i in range(0, len(sus)-2, 2)]
            removeindex2 = [i for p in pairs2 for i in range(p[0], p[1])]
            removeindex2.append(sus[-1])
            if len(removeindex1) < len(removeindex2):
                removeindex = removeindex1
            else:
                removeindex = removeindex2
    else:
        removeindex = list()
    removeme = oneID.loc[removeindex]
    removeme.ID = 'X'
    input = oneID.drop(removeme.index)

    if len(input.index) < max(input.frame)/100:
        removeme = pd.concat([removeme, input], axis=0, ignore_index=True)
        input = pd.DataFrame()
        return input, removeme

    return input, removeme

def relabel(id, oneID, noID, dist1, time1):
    """Promotes suitable potential tags to identified tags."""
    #trial 1: from top
    addme = pd.DataFrame()
    h = min(oneID.frame)
    t = max(oneID.frame)
    trial = pd.DataFrame()
    model = oneID[oneID["frame"] == h]
    f1 = h+1
    while f1 < max(noID.frame):
        if f1 not in list(oneID.frame):
            candidates = noID[noID["frame"] == f1]
            if len(candidates.index) != 0:
                distances = [basicFx.caldis2(model, candidates.loc[i]) for i in candidates.index]
                if min(distances) < dist1 and f1-h < time1:
                    model = pd.concat([copy.deepcopy(candidates.loc[candidates.index[distances.index(min(distances))]]),(pd.Series([f1-h]))])
                    if model["ID"] != id:
                        model["1cm"] = "X"
                        model["score"] = "X"
                    model.iloc[1] = id
                    trial = pd.concat([trial, pd.DataFrame(model).transpose()], axis=0, ignore_index=True)
                    h = model['frame']
                    f1 = f1+1
                else:
                    if f1-h > time1:
                        nextones = [i for i in oneID.frame if i > f1]
                        if len(nextones) == 0:
                            break
                        h = nextones[0]
                        model = oneID[oneID["frame"] == h]
                        f1 = h+1
                    else:
                        f1 = f1+1
            else:
                f1 = f1+1
        else:
            h = f1
            model = oneID[oneID["frame"] == h]
            f1 = h+1
    #trial 2: from bottom
    h = min(oneID.frame)
    t = max(oneID.frame)
    model = oneID[oneID["frame"] == t]
    f2 = t-1
    while f2 > 0:
        if f2 not in list(oneID.frame):
            candidates = noID[noID["frame"] == f2]
            if len(candidates.index) != 0:
                distances = [basicFx.caldis2(model, candidates.loc[i]) for i in candidates.index]
                if min(distances) < dist1 and t-f2 < time1:
                    model = pd.concat([copy.deepcopy(candidates.loc[candidates.index[distances.index(min(distances))]]),(pd.Series([abs(t-f2)]))])
                    if model["ID"] != id:
                        model["1cm"] = "X"
                        model["score"] = "X"
                    model.iloc[1] = id
                    trial = pd.concat([trial, pd.DataFrame(model).transpose()], axis=0, ignore_index=True)
                    t = model['frame']
                    f2 = f2-1
                else:
                    if t-f2 > time1:
                        nextones = [i for i in oneID.frame if i < f2]
                        if len(nextones) == 0:
                            break
                        t = nextones[-1]
                        if t > f2:
                            break
                        model = oneID[oneID["frame"] == t]
                        f2 = t-1
                    else:
                        f2 = f2-1
            else:
                f2 = f2-1
        else:
            t = f2
            model = oneID[oneID["frame"] == t]
            f2 = t-1

    #combine the two: keep one that is closer to a found tag if different
    if len(trial) == 0:
        return trial
    trial.columns = pd.Index(['frame','ID','centroidX','centroidY','frontX','frontY','1cm','score','check','dist'],dtype='object')
    if len(trial.index) > 0:
        frames = set(trial.frame)
        for f in frames:
            subset = trial[trial['frame'] == f]
            #print(subset)
            if len(subset.index) == 1:
                addme = pd.concat([addme, pd.DataFrame(subset.iloc[0][0:7]).transpose()], axis=0, ignore_index=True)
            elif len(set(subset.centroidX)) == 1 and len(set(subset.centroidY)) == 1:
                addme = pd.concat([addme, pd.DataFrame(subset.iloc[0][0:7]).transpose()], axis=0, ignore_index=True)
            elif len(set(subset['dist'])) == 1:
                absdiff = [abs(f-frame) for frame in oneID.frame]
                diff = [f-frame for frame in oneID.frame]
                test = diff[absdiff.index(min(absdiff))]
                if test < 0:
                    addme = pd.concat([addme, pd.DataFrame(subset.loc[max(subset.index)][0:7]).transpose()], axis=0, ignore_index=True)
                else:
                    addme = pd.concat([addme, pd.DataFrame(subset.loc[min(subset.index)][0:7]).transpose()], axis=0, ignore_index=True)
            else:
                addme = pd.concat([addme, pd.DataFrame(subset.loc[pd.to_numeric(subset['dist']).idxmin()][0:7]).transpose()], axis=0, ignore_index=True)

    oneID = pd.concat([oneID, addme], axis=0, ignore_index=True)
    return oneID.sort_values("frame", axis=0, ascending=True, inplace=False, kind='quicksort', na_position='last', ignore_index=True)
    
def addmissing(oneID, dist2, time2):
    """Fill in gaps."""
    missing = pd.DataFrame()
    for i in range(oneID.shape[0]-1): #for each row except the last
        if 1 < oneID["frame"][i+1] - oneID["frame"][i] < dist2 and math.sqrt((oneID["centroidX"][i+1] - oneID["centroidX"][i])**2 + (oneID["centroidY"][i+1] - oneID["centroidY"][i])**2) < time2: #threshold by time and distance
            addframe = pd.DataFrame(list(range(int(oneID["frame"][i] + 1), int(oneID["frame"][i+1]))))
            addX = pd.DataFrame([round(oneID["centroidX"][i] + (oneID["centroidX"][i+1]-oneID["centroidX"][i]) *(j+1)/len(addframe)) for j in range(len(addframe))])
            addY = pd.DataFrame([round(oneID["centroidY"][i] + (oneID["centroidY"][i+1]-oneID["centroidY"][i]) *(j+1)/len(addframe)) for j in range(len(addframe))])
            addX2 = pd.DataFrame([round(oneID["frontX"][i] + (oneID["frontX"][i+1]-oneID["frontX"][i]) *(j+1)/len(addframe)) for j in range(len(addframe))])
            addY2 = pd.DataFrame([round(oneID["frontY"][i] + (oneID["frontY"][i+1]-oneID["frontY"][i]) *(j+1)/len(addframe)) for j in range(len(addframe))])
            addme = pd.concat([addframe, addX, addY, addX2, addY2], axis = 1)
            missing = pd.concat([missing, addme], axis=0, ignore_index=True)

    if missing.shape != (0, 0):
        missing.columns = ["frame", "centroidX", "centroidY", "frontX", "frontY"] # 0:'frame', 1:'ID', 2:'centroidX', 3:'centroidY', 4:'dir', 5:'1cm', 6:'score'  
        missing['ID'] = oneID.ID[0]
        missing['1cm'] = 'X'
        missing['score'] = 'X'

        oneID = pd.concat([oneID, missing], axis=0, ignore_index=True)
        oneID = oneID.sort_values("frame", axis=0, ascending=True, inplace=False, kind='quicksort', na_position='last', ignore_index=True)

    return oneID

def wrangle(raw, noID, outname, dist1, time1, dist2, time2, nojump):
    """Read in csv files that contain information on potential and read tags, outputting final positions of tags."""
    IDs = set(raw.ID)
    wrangled = pd.DataFrame()
    noDuplicates, noID = removeduplicates(raw, noID)
    print("Relabelling and adding missing tags")
    IDs = set(raw.ID)
    for id in IDs:
        print(id)
        oneIDInput = noDuplicates[noDuplicates["ID"] == id]
        if len(oneIDInput) == 0:
            continue
        oneID = relabel(id, oneIDInput, noID, dist1, time1)
        if len(oneID) == 0:
            continue
        patched = addmissing(oneID, dist2, time2)
        if nojump:
            patched = removejumps(patched, dist1)[0]
        wrangled = pd.concat([wrangled, patched], axis=0, ignore_index=True)
    
    wrangled = wrangled.sort_values(by=['frame'])
    wrangled.to_csv(path_or_buf = outname + ".csv", na_rep = "NA", index = False)
    return wrangled

######################
#main entry point
def main(argv):
    """ Main entry point of the program """
    opt = parseOptions()
    if vars(opt)['outname'] is None:
        vars(opt)['outname'] = os.path.basename(vars(opt)['rawname']).split('_')[0]
    raw = pd.read_csv(vars(opt)['rawname'])
    noID = pd.read_csv(vars(opt)['noidname'])

    delattr(opt, 'rawname')
    delattr(opt, 'noidname')

    wrangle(raw = raw, noID = noID, **vars(opt))

    return 0

if __name__ == "__main__": 
    """Makes sure the "main" function is called from command line"""  
    status = main(sys.argv)
    sys.exit(status)