import pandas as pd
import glob, os

def renumber(folder):
    for file in glob.glob(os.path.join(folder, '*.csv')):
        if 'time' not in file:
            hz = int(file.split('_')[-1].replace('.csv', ''))
            OG = pd.read_csv(file)
            OG['frame'] = OG['frame']*hz
            OG.to_csv(file)

renumber('../Results/testhz')