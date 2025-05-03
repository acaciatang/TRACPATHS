from TRACPATHS.TRAC import *
import sys
import os

def main(argv):
    """ Main entry point of the program """
    #check that options given make sense
    filename = os.path.basename(argv[1])
    outname = os.path.basename(filename).split('.')[0]
    #run code
    sortrois(argv[1], outname, 'TagList.csv', red=True, write=False, i=4, minSize=500, maxSize=2500)
    return 0

if __name__ == "__main__": 
    """Makes sure the "main" function is called from command line"""  
    status = main(sys.argv)
    sys.exit(status)