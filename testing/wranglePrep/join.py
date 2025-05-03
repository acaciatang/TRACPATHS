import glob, os
import pandas as pd

if not os.path.exists('../Results/wranglePreped'):
    os.mkdir('../Results/wranglePreped')

for csv in glob.glob("../Results/wranglePrep/*.csv"):
    og = os.path.basename(csv).split('_')[0]

    if not os.path.exists(os.path.join('../Results/wranglePrep/',og)):
        print('creating folder: ' + og)
        os.mkdir(os.path.join('../Results/wranglePrep/', og))

    os.rename(csv, os.path.join('../Results/wranglePrep/', og, os.path.basename(csv)))
    
for dir in glob.glob("../Results/wranglePrep/*"):
    og = os.path.basename(dir)
    raw = pd.DataFrame()
    noID = pd.DataFrame()

    for csv in glob.glob(dir+'/*.csv'):
        chunk = int(csv.split('_')[-2])
        if 'raw' in csv:
            part = pd.read_csv(csv)
            part['frame'] = part['frame']+chunk*1000
            raw = pd.concat([raw, part])
        elif 'noID' in csv:
            part = pd.read_csv(csv)
            part['frame'] = part['frame']+chunk*1000
            noID = pd.concat([noID, part])

    raw.to_csv('../Results/wranglePreped/'+og+'_raw.csv',index=False)
    noID.to_csv('../Results/wranglePreped/'+og+'_noID.csv',index=False)