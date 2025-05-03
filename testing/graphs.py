"""Compares TRAC-PATHS results to manually labelled tag locations. TRAC results will be compared to from frame, PATHS to from vid."""

import glob
import pandas as pd
import os
import json
import sys
from scipy.spatial.distance import cdist
import numpy as np

def JSONtoCSV():
    for dir in glob.glob('../data/from*'):
        for jsn in glob.glob(os.path.join(dir, '*.json')):
            readJSON = pd.DataFrame.from_dict(json.load(open(jsn))['shapes'])
            readJSON[['points']] = pd.DataFrame(readJSON.points.tolist(), index= readJSON.index)
            readJSON[['centroidX','centroidY']] = pd.DataFrame(readJSON.points.tolist(), index= readJSON.index)
            readJSON['ID'] = readJSON['label']
            readJSON['vidID'] = os.path.basename(jsn).split('_')[0]
            
            readJSON[['ID','centroidX','centroidY', 'vidID']].to_csv('Results/JSONs/'+os.path.basename(dir)+'_'+os.path.basename(jsn).split('_')[0]+'.csv', index = False)
        
    return 0

def dist(manualCSV, oneTest, distThres):
    manualpts = manualCSV[['centroidX','centroidY']].values
    testpts = oneTest[['centroidX','centroidY']].values

    distances = cdist(manualpts, testpts, metric='euclidean')
    distDF = pd.DataFrame(distances, index=manualCSV.ID, columns=oneTest.ID)

    testMatch = pd.DataFrame({'closestDist':distDF.min(axis=1), 'closestID':distDF.idxmin(axis=1), 'ID':distDF.index})

    testMatch['found'] = testMatch['closestDist']<distThres
    testMatch['same'] = testMatch['ID'] == testMatch['closestID']

    falseNeg = sum(np.logical_not(testMatch.same))
    truePos = sum(np.logical_and(testMatch.same, testMatch.found))
    mislabelled = sum(np.logical_and(np.logical_not(testMatch.same), testMatch.found))

    reverseMatch = pd.DataFrame({'closestDist':distDF.min(axis=0), 'closestID':distDF.idxmin(axis=0), 'ID':distDF.columns})
    falsePos = sum(reverseMatch['closestDist']>distThres)

    numPos = manualCSV.shape[0]

    return [truePos/numPos, falseNeg/numPos, falsePos, mislabelled/numPos]

def checkTRAC(testname, distThres):
    out = pd.DataFrame()
    for raw in glob.glob('Results'+'/*raw*.csv'):
        try:
            rawCSV = pd.read_csv(raw)
        except:
            continue
        
        print(raw)
        vidID = os.path.basename(raw).split('.')[0]
        testValue = os.path.basename(raw).split('_')[2]

        manualCSV = pd.read_csv('../fromVids/'+vidID+'.csv')

        checkResults = dist(manualCSV, rawCSV, distThres)
        oneRow =pd.DataFrame([[vidID, testname, testValue]+checkResults], columns = ['vidID', 'test', 'testValue', 'truePos', 'falseNeg', 'falsePos', 'mislabelled'])
        out = pd.concat([out, oneRow])
    
    out.to_csv(testname+'.csv', index=False)
    return out

def graph(out, testname):
    melted = pd.melt(out, id_vars=['vidID', 'test', 'testValue'], value_vars=['truePos', 'falseNeg', 'falsePos', 'mislabelled'],var_name='metric', value_name='count')
    melted.to_csv('Results/toGraph/'+testname + '_melted.csv')
    return 0

def CralltoCSV(folder):
    for crallJSON in glob.glob(os.path.join(folder, '*.json')):
        crallDF = pd.read_json(crallJSON)
        crallDF.to_csv(os.path.join(folder,os.path.basename(crallJSON).split('.')[0]+'_raw.csv'))
    return 0

def checkPATHS(testname, distThres):
    out = pd.DataFrame()
    for raw in glob.glob('Results/'+testname+'/*.csv'):
        if 'time' not in raw:
            try:
                rawCSV = pd.read_csv(raw)
            except:
                continue
            print(raw)
            testFrame = rawCSV[rawCSV['frame']==5040]
            
            vidID = os.path.basename(raw).split('_')[0]
            testValue = os.path.basename(raw).split('_')[2].replace('.csv', '')

            manualCSV = pd.read_csv('../fromVids/'+vidID+'.csv')

            checkResults = dist(manualCSV, testFrame, distThres)

            oneRow = pd.DataFrame([[vidID, testname, testValue]+checkResults], columns = ['vidID', 'test', 'testValue', 'truePos', 'falseNeg', 'falsePos', 'mislabelled'])
            out = pd.concat([out, oneRow])
        
    out.to_csv('Results/toGraph/'+testname+'.csv', index=False)
    return out

def checkManual(testname, distThres):
    out = pd.DataFrame()
    for raw in glob.glob('../fromFrames/'+vidID+'.csv'):
        vidID = raw.split('_')[1].replace('.csv', '')
        rawCSV = pd.read_csv(raw)
        oneTest = rawCSV[['ID','centroidX','centroidY']]
        testValue = 0

        manualCSV = pd.read_csv('../fromVids/'+vidID+'.csv')

        checkResults = dist(manualCSV, oneTest, distThres)
        oneRow = pd.DataFrame([[vidID, testname, testValue]+checkResults], columns = ['vidID', 'test', 'testValue', 'truePos', 'falseNeg', 'falsePos', 'mislabelled'])
        out = pd.concat([out, oneRow])
    
    return out

def main(argv):
    JSONtoCSV()

    distThres = 10
    
    print('testi')
    out = checkTRAC('testi', distThres)
    graph(out, 'testi')

    print('testsize')
    out = checkTRAC('testsize', distThres)
    graph(out, 'testsize')

    print('testwindow')
    out = checkPATHS('testwindow', distThres)
    graph(out, 'testwindow')

    print('testhz')
    out = checkPATHS('testhz', distThres)
    graph(out, 'testhz')

    print('crall')
    CralltoCSV('Results/CrallResults')
    out = checkTRAC('CrallResults', distThres)
    graph(out, 'crallResults')

    out = checkManual('manual labels (frame)', distThres)
    graph(out, 'manual')

if __name__ == "__main__": 
    """Makes sure the "main" function is called from command line"""  
    status = main(sys.argv)
    sys.exit(status)

"""
import glob
import pandas as pd
import os
import json
import sys
from scipy.spatial.distance import cdist
import numpy as np

combine = pd.DataFrame()
for raw in glob.glob('Results/testi/*raw*'):
    try:
        rawCSV = pd.read_csv(raw)
    except:
        continue

    combine = pd.concat([combine, rawCSV])
"""