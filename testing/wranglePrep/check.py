import os
import sys

def main(argv):
    filename = os.path.join('/Volumes/MResData/MRes/methodspaper/Data/video/OG',argv[1])
    if os.path.exists(filename):
        if os.path.isfile(filename):
            print('Yes!')
        else:
            print(filename + ' is not a file')
    else:
        print(filename + ' does not exist')
    

if __name__ == "__main__": 
    """Makes sure the "main" function is called from command line"""  
    status = main(sys.argv)
    sys.exit(status)