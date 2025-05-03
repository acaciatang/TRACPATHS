import pandas as pd
import glob, os

def joinTime(folder):
    joinedTime = pd.DataFrame()
    for file in glob.glob(os.path.join(folder, '*time.csv')):
        timeCSV = pd.read_csv(file)
        joinedTime = pd.concat([joinedTime, timeCSV])

    joinedTime.to_csv(os.path.basename(folder)+'time.csv')

joinTime('/Volumes/MResData/MRes/methodspaper/reun_summer2024/Results/testwindow')