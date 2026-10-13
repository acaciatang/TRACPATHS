import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
import argparse
import os
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

def chooseOne(subset):
    out = None
    return out

def oneDir(thisRaw, noDup, allPot, dist1, time1, scores, f, outDir):
    class proposer:
        def __init__(self, INDEX):
            self.index = INDEX
            self.ID = thisRaw.ID[self.index]

            self.proposalIDs = [i for i in range(len(allNext.index)) if float(distM[i,INDEX]) < dist1]
            if len(self.proposalIDs) == 0:
                self.done = True
            else:
                self.done = False
            self.dist = [float(distM[i,INDEX]) for i in self.proposalIDs]
            combined_lists = list(zip(self.dist, self.proposalIDs))
            sorted_lists = sorted(combined_lists, key=lambda x: x[0])
            self.sortedProposals = [item[1] for item in sorted_lists]
            
            self.matchID = [allNext.ID[i] for i in self.proposalIDs]
            
            if self.ID == 'X':
                self.score = 0
            else:
                self.score = scores[self.ID]
                if self.ID in self.matchID:
                    samePos = self.matchID.index(self.ID)
                    best = self.sortedProposals.pop(samePos)
                    self.sortedProposals.insert(0,best)

    class acceptor:
        def __init__(self, INDEX):
            self.match = None
            self.index = INDEX
            self.ID = allNext.ID[self.index]

    outDir = pd.concat([outDir,thisRaw], ignore_index=True)
    allNext = pd.concat([noDup[noDup["frame"] == f], allPot[allPot["frame"] == f]], ignore_index=True, copy=True)
    if len(allNext.index) == 0:
        return outDir, thisRaw

    #List and rank proposals: distance, favour self
    thisX = np.array(thisRaw['centroidX'])
    thisY = np.array(thisRaw['centroidY'])
    holdX = np.array([allNext['centroidX']])
    holdY = np.array([allNext['centroidY']])

    nextX = np.swapaxes(holdX,0,1)
    nextY = np.swapaxes(holdY,0,1)
    distM = np.sqrt((thisX - nextX)**2 + (thisY - nextY)**2) #columns is this frame
    
    
    #Gale-Shapley
    proposers = [proposer(i) for i in thisRaw.index]
    acceptors = [acceptor(i) for i in allNext.index]
    while not np.array([p.done for p in proposers]).all():
        for p in proposers:
            if len(p.sortedProposals) > 0:
                if not p.done:
                    current = acceptors[p.sortedProposals[0]]
                    if current.match is None:
                        current.match = p
                        p.done = True
                    elif current.match.score < p.score:
                        current.match.done = False
                        current.match = p
                        p.done = True
                    p.sortedProposals.remove(current.index)
                    del current
            else:
                p.done = True
    
    ##Update data
    for a in acceptors:
        if a.match is not None:
            if a.ID != a.match.ID:
                allNext.loc[a.index, "ID"] = a.match.ID
                allNext.loc[a.index, "frontX"] = 'X'
                allNext.loc[a.index, "frontY"] = 'X'
                allNext.loc[a.index, "1cm"] = 'X'

    matched = allNext.iloc[[i for i in allNext.index if acceptors[i].match is not None]]
    foundtags = noDup[noDup["frame"] == f]
    nextRaws = pd.concat([matched, foundtags], ignore_index=False)
    nextRaws.drop_duplicates(subset=['ID', 'frame'], keep='first', inplace=True) # will keep match over TRAC result

    ##left unmatched, not too old
    noneFound = thisRaw[f-thisRaw['frame'] < time1]
    for i in noneFound.index:
        if noneFound['ID'][i] in nextRaws['ID'].values:
            noneFound.drop(i, axis=0, inplace=True)
    
    del thisRaw
    nextRaws = pd.concat([nextRaws, noneFound], ignore_index = True)
    nextRaws.drop_duplicates(subset=['ID', 'frame'], keep='first', inplace=True) # will keep match over TRAC result
    
    del noneFound, acceptors, proposers
    return outDir, nextRaws

def wrangle(raw, noID, dist1, time1, dist2, time2, outname):
    #Demote duplicate occurance
    raw.drop(labels='check', axis=1, inplace=True)
    noID.drop(labels='check', axis=1, inplace=True)

    noDup, allPot = removeduplicates(raw, noID)
    
    allPot['ID'] = 'X'
    allPot['frontX'] = 'X'
    allPot['frontY'] = 'X'

    #out files
    outForward = pd.DataFrame()
    outBackward = pd.DataFrame()

    #forward
    thisRaw = noDup[noDup["frame"] == min(noDup['frame'])]
    scores = noDup.apply(pd.Series.value_counts).ID.dropna()

    for f in range(int(min(noDup['frame'])+1),int(max(max(noDup['frame']), max(allPot['frame']))+1)):
        #print(f)
        outForward, thisRaw = oneDir(thisRaw, noDup, allPot, dist1, time1, scores, f, outForward)

    outForward['dir'] = 'forward'
    
    #backward
    thisRaw = pd.concat([noDup[noDup["frame"] == max(noDup["frame"])],pd.DataFrame()], ignore_index=True)

    for f in reversed(range(int(min(noDup['frame'])), int(max(max(noDup['frame']), max(allPot['frame']))))):
        #print(f)
        outBackward, thisRaw = oneDir(thisRaw, noDup, allPot, dist1, time1, scores, f, outBackward)
    
    outBackward['dir'] = 'backward'

    #choose one path if there are two
    combine = pd.concat([outForward, outBackward], ignore_index=True)
    combine.drop_duplicates(subset=['ID', 'frame', 'centroidX', 'centroidY'], inplace=True)

    multiples = combine[combine.duplicated(subset=['ID', 'frame'], keep=False)]
    combine.drop_duplicates(subset=['ID', 'frame'], keep=False, inplace=True)

    ids = set(multiples.ID)
    for id in ids:
        oneID = multiples[multiples['ID'] == id]
        forward = oneID[oneID['dir'] == 'forward']
        backward = oneID[oneID['dir'] == 'backward']
        backward.sort_values(by='frame', inplace=True)
        forward['group'] = forward['frame'].diff().gt(1).cumsum()
        backward['group'] = backward['frame'].diff().gt(1).cumsum()
        for i in set(forward['group']):
            fn = forward[forward['group'] == i]
            bn = backward[backward['group'] == i]
            fn.drop('group', axis=1, inplace=True)
            bn.drop('group', axis=1, inplace=True)
            
            #choose one with more IDed tags
            if len(fn[fn['frontX'] != 'X']) > len(bn[bn['frontX'] != 'X']):
                combine = pd.concat([fn, combine], ignore_index=True)
                continue
            elif len(fn[fn['frontX'] != 'X']) < len(bn[bn['frontX'] != 'X']):
                combine = pd.concat([bn, combine], ignore_index=True)
                continue
            else: #choose one with smaller distance
                fx = fn.centroidX
                fy = fn.centroidY
                fdist = (fx.diff()**2).sum() + (fy.diff()**2).sum()
                bx = bn.centroidX
                by = bn.centroidY
                bdist = (bx.diff()**2).sum() + (by.diff()**2).sum()
                if bdist < fdist:
                    combine = pd.concat([bn, combine], ignore_index=True)
                    continue
                else: #choose forward
                    combine = pd.concat([fn, combine], ignore_index=True)
                    continue

    #interpolate
    out = pd.DataFrame()
    for i in set(combine['ID']):
        subset = combine[combine['ID'] == i]
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