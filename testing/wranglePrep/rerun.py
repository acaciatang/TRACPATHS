import glob, os

f = open('localWrangle.sh', 'w')
for i in glob.glob('../OG/*'):
    base = os.path.basename(i).split('.')[0]
    if len(glob.glob(base+'*')) == 0:   
        f.write('python wranglePrep.py ../OG/'+ base + '.MP4&')
f.close()