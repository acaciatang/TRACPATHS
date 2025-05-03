import glob, os

f = open('localRun.sh', 'w')

for i in glob.glob('/Users/acacia/Desktop/wranglePrep/wranglePreped/*'):
    i = i.split('_')[0]
    for j in range(10):
        j = str(j+1)
        f.write('python3 testhz.py ' + i + ' ' + j + '&\n')

f.close()