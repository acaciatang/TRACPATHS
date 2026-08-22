import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
import argparse
import os
from collections import OrderedDict
import sys

#####################
#parse options
def parseOptions():
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--rawname', type=str, default='Results/test_raw.csv', help='Path to video.')
    parser.add_argument('-n', '--noidname', type=str, default='Results/test_noID.csv', help='Path to tag list.')
    parser.add_argument('-o', '--outname', type=str, default='Results/test', help='What to name outputs and where to put them.')
    parser.add_argument('-d1','--dist1', type=int, default=30, help='Maximum distance to be considered part of a path.')
    parser.add_argument('-t1','--time1', type=int, default=30, help='Maximum frames passed to be considered part of a path.')
    parser.add_argument('-d2', '--dist2', type=int, default=100, help='Maximum distance to extrapolate linearly.')
    parser.add_argument('-t2', '--time2', type=int, default=100, help='Maximum frames to extrapolate linearly over.')

    opt = parser.parse_args()
    return opt

######################
def removeduplicates(raw, noID):
    sus = raw[raw.duplicated(subset=['ID', 'frame'], keep=False)]
    sus['ID'] = 'X'
    noDuplicates = raw.drop_duplicates(subset=['ID', 'frame'], keep=False, ignore_index=True)
    noID = pd.concat([noID, sus], axis = 0, ignore_index=True)
    return noDuplicates, noID

def oneDir(thisRaw, noDup, allPot, dist1, time1, dist2, time2, f, outDir):
    allNext = pd.concat([noDup[noDup["frame"] == f], allPot[allPot["frame"] == f]], ignore_index=True, copy=True)
    #List and rank proposals: distance, favour self
    thisX = np.array(thisRaw['centroidX'])
    thisY = np.array(thisRaw['centroidY'])
    holdX = np.array([allNext['centroidX']])
    holdY = np.array([allNext['centroidY']])

    nextX = np.swapaxes(holdX,0,1)
    nextY = np.swapaxes(holdY,0,1)
    distM = np.sqrt((thisX - nextX)**2 + (thisY - nextY)**2) #columns is this frame
    

    proposals = OrderedDict([(int(j), [[allNext.index[i], float(distM[i,j])] for i in range(len(allNext.index)) if float(distM[i,j]) < dist1]) for j in range(len(thisRaw.index))])
    

    #Gale-Shapley
    acceptors = {i:None for i in allNext.index}
    allDone = np.array([False for i in proposals])
    while not allDone.all():
        for proposer in proposals:
            proposerList = proposals[proposer]
            if len(proposerList) == 0:
                allDone[proposer] = True
                continue
            current = proposerList[0]
            if acceptors[current[0]] is None:
                acceptors[current[0]] = [proposer, current[1]]
                allDone[proposer] = True
            elif  acceptors[current[0]][1] < current[1]:
                acceptors[current[0]] = [proposer, current[1]]
                allDone[acceptors[current[0]][0]] = False
                allDone[proposer] = True
            proposals[proposer].remove(current)
            del current
    
    del allDone
    
    #Update data
    ##extended
    for i in acceptors:
        if acceptors[i] is not None:
            if allNext['ID'][i] != thisRaw['ID'][acceptors[i][0]]:
                allNext.loc[i, "ID"] = thisRaw['ID'][acceptors[i][0]]
                allNext['frontX'] = 'X'
                allNext['frontY'] = 'X'
                allNext['1cm'] = 'X'

    nextRaws = allNext.iloc[[i for i in allNext.index if acceptors[i] is not None]]
    
    ##left unmatched, not too old
    noneFound = thisRaw[f-thisRaw['frame'] < time1]
    for i in noneFound.index:
        if noneFound['ID'][i] in nextRaws['ID'].values:
            noneFound.drop(i, axis=0, inplace=True)

    holdNext = pd.concat([nextRaws, noneFound], ignore_index = True)

    ##new paths
    newFinds = noDup[noDup["frame"] == f]
    for i in newFinds.index:
        if newFinds['ID'][i] in holdNext['ID'].values:
            newFinds.drop(i, axis=0, inplace=True)
    
    outDir = pd.concat([outDir,thisRaw], ignore_index=True)
    
    del thisRaw
    thisRaw = pd.concat([holdNext, newFinds], ignore_index=True)
    
    del holdNext, newFinds, noneFound, acceptors, proposals
    return outDir


def wrangle(raw, noID, dist1, time1, dist2, time2, outname):
    #Demote duplicate occurance
    noDup, allPot = removeduplicates(raw, noID)
    
    allPot['ID'] = 'X'
    allPot['frontX'] = 'X'
    allPot['frontY'] = 'X'

    #out files
    outForward = pd.DataFrame()
    outBackward = pd.DataFrame()


    #forward
    thisRaw = noDup[noDup["frame"] == min(noDup['frame'])]

    for f in range(int(min(noDup['frame'])+1),int(max(max(noDup['frame']), max(allPot['frame'])))):
        outFoward = oneDir(thisRaw, noDup, allPot, dist1, time1, dist2, time2, f, outForward)


    #backward
    thisRaw = pd.concat([noDup[noDup["frame"] == max(noDup["frame"])],pd.DataFrame()], ignore_index=True)

    for f in reversed(range(int(min(noDup['frame'])+1), int(max(max(noDup['frame']), max(allPot['frame']))))):
        outBackward = oneDir(thisRaw, noDup, allPot, dist1, time1, dist2, time2, f, outBackward)

    #wrangle
    combine = pd.concat([outFoward, outBackward], ignore_index=True)
    combine.drop_duplicates(inplace=True)

    multiples = combine[combine.duplicated(subset=['ID', 'frame'], keep=False)]
    addFound = multiples[multiples['frontX'] != 'X']
        
    combine.drop_duplicates(subset=['ID', 'frame'],keep=False,inplace=True)
    toInterpolate = pd.concat([addFound, combine], ignore_index=True)

    out = pd.DataFrame()
    for i in set(toInterpolate['ID']):
        subset = toInterpolate[toInterpolate['ID'] == i]
        if len(subset.index) > 1:
            subset.sort_values('frame', inplace=True, ignore_index=True)
            all = range(int(min(subset.frame)),int(max(subset.frame)+1))

            subset['diffX'] = subset.centroidX.diff()
            subset['diffY'] = subset.centroidY.diff()
            subset['diffT'] = subset.frame.diff()
            subset.loc[subset.index[0], "diffX"] = subset['diffX'][subset.index[1]]
            subset.loc[subset.index[0], "diffY"] = subset['diffY'][subset.index[1]]
            subset.loc[subset.index[0], "diffT"] = subset['diffT'][subset.index[1]]
            subset['diffD'] = (subset['diffX']**2+subset['diffX']**2)**0.5
            subset['inter'] = np.logical_and(subset['diffT'] < time2, subset['diffD'] < dist2)
            subset.drop(['diffX', 'diffY', 'diffT', 'diffD'], axis=1, inplace=True)

            subOut = subset.set_index("frame").reindex(all)
            test = subOut.inter.astype(float).interpolate(method='nearest',limit_direction='backward').astype(bool)
            subOut['frame'] = subOut.index
            subOut.ID = i
            subOut.centroidX = subOut.centroidX.interpolate()
            subOut.centroidY = subOut.centroidY.interpolate()
            subOut.drop(['inter'], axis=1, inplace=True)
            
            out = pd.concat([out, subOut[test]])
        else:
            out = pd.concat([out, subset])
    
    out.to_csv(outname+'.csv', index=False)
    return out


######################
#main entry point
def main(argv):
    """ Main entry point of the program """
    #Prep
    opt = parseOptions()
    if vars(opt)['outname'] is None:
        vars(opt)['outname'] = os.path.basename(vars(opt)['rawname']).split('_')[0]
    raw = pd.read_csv(vars(opt)['rawname'])
    noID = pd.read_csv(vars(opt)['noidname'])
    outname = vars(opt)['outname']

    dist1 = vars(opt)['dist1']
    dist2 = vars(opt)['dist2']
    time1 = vars(opt)['time1']
    time2 = vars(opt)['time2']

    wrangle(raw, noID, dist1, time1, dist2, time2, outname)

    return 0
        
if __name__ == "__main__": 
    """Makes sure the "main" function is called from command line"""  
    status = main(sys.argv)
    sys.exit(status)