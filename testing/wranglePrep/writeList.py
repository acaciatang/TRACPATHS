import glob

f = open("list.txt", "w")
for file in glob.glob('/Volumes/MResData/MRes/methodspaper/Data/video/chunks/*'):
        f.write(file.replace('/Volumes/MResData/MRes/methodspaper/Data/video/', '') + '\n')
f.close()