import glob, os

f = open("list.txt", "w")
for file in glob.glob('/Volumes/MResData/MRes/methodspaper/reun_summer2024/Results/wranglePreped/*raw.csv'):
    for i in range(1,11):
        f.write(os.path.basename(file).replace('_raw.csv', '') + ','+ str(i) + '\n')
f.close()