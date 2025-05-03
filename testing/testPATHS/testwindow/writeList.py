import glob, os

f = open("list.txt", "w")
for file in glob.glob('/Volumes/MResData/MRes/methodspaper/reun_summer2024/Results/wranglePreped/*raw.csv'):
    for i in [5, 50, 500, 5000, 50000]:
        f.write(os.path.basename(file).replace('_raw.csv', '') + ','+ str(i) + '\n')
f.close()