# TRAC-PATHS

## Overview
TRAC-PATHS is a two-part algorithm comprising of Tag Recognition by Approximating Corners (TRAC) and Predictor of Approximate Track with Heuristic Search (PATHS).

TRAC carries out tag detection around corners to increase the chance of identifying tags. PATHS takes in a list of potential and identified tags and based on the principle that a tag should only move by a limited amount from frame to frame, reassigns identity to both potential and identified markers to decrease error rates and patch gaps in identification to give a more realistic track.

While TRAC is written specifically for 16-bit Beetag, PATHS can be used on data from tag-detection of any tags by other algorithms as long as data is re-formatted for processing.

<br>

## Requirements
TRAC-PATHS has been tested in MacOS and Linux environments, and has the following requirements:
- python3 (tested on python 3.8.20)
- python modules: opencv-python, numpy, pandas, matplotlib

To create the python environment, use requirements.txt like so:
```
conda create -n tracpaths python=3.8.20 pip
conda activate tracpaths
pip install -r requirements.txt
```

<br>

## Usage
This repository includes an example video in the *Example* directory, which will be used in the example code below.

Both TRAC and PATHS can be called directly in terminal.

TRAC takes the following arguments:
- -f, --filename': str, default path to Example/test.mp4
  - Path to video
- -t, --taglist: str, default path to /../Example/TagList.csv
  - Path to tag list
- -w, --write: bool, default True
  - Whether to write output video
- -o, --outname: str, default None
  - What to name outputs and where to put them
- -r, --red: bool, default True
  - Whether to only use the red channel in analysis
- -m, --minSize: int, default 500
  - Minimum size of white region to be considered
- -M, --maxSize: int, default 2500
  - Maximum size of white region to be considered
- -i, --i: int, default 5
  - How far around corners to search


For example, from this directory:
```
python3 TRACPATHS/TRAC.py -f Example/test.mp4 -t Example/TagList.csv -w True -o Results/test -r True -m 500 -M 2500 -i 5
```

The result should be three files in the *Results* directory, test_raw.csv, test_noID.csv, and test_raw.mp4. test_raw.csv contains data on identified tags, while test_noID.csv contains data on potential tags. test_raw.mp4 is a video with marked identified tags (in green, with the ID) and potential tags (in yellow).

To predict tracks from existing data on identified and potential tag locations, run PATHS.

PATHS takes the following arguments:
- -r, --rawname: str, default test_raw.csv
  - Path to video
- -n, --noidname: str, default test_noID.csv
  - Path to tag list
- -o, --outname: str, default None
  - What to name outputs and where to put them
- -j, --nojump: bool, default False
  - Whether to run algorithm to remove jumps in a path
- -d1, --dist1: int, default 30
  - Maximum distance to be considered part of a path
- -t1, --time1: int, default 30
  - Maximum frames passed to be considered part of a path
- -d2, --dist2: int, default 100
  - Maximum distance to extrapolate linearly
- -t2, --time2: int, default 100
  - Maximum frames to extrapolate linearly over

For example:
```
python3 TRACPATHS/PATHS.py -r Results/test_raw.csv -n Results/test_noID.csv -o Results/test -j False -d1 30 -t1 30 -d2 100 -t2 100
```

This should output test.csv into the Results directory, which will contain the predicted locations of tags through time.

TRACPATHS also includes modules for data visualization. They are drawCircles (which draws circles around the tags' predicted location), and drawTracks (which draws the predicted tracks through time).

dataVisualisation takes the following arguments:
- -f, --filename: str, default path to Example/test.mp4
  - Path to video
- -c, --csvname: str, default test.csv
  - Path to results from PATHS
- -o, --outname: str, default None
  - What to name outputs and where to put them

For example:
```
python3 TRACPATHS/dataVisualisation.py -f Example/test.mp4 -c Results/test.csv -o Results/test
```

This should give two videos: test_tracks.mp4 and test_circles.mp4. test_tracks.mp4 will show the paths taken by tags through time as predicted by TRAC-PATHS, and test_circles.mp4 will show circles around tags, labelled with their IDs.

<br>

## Maintainers
Acacia Tang -- [ttang53@wisc.edu](mailto:ttang53@wisc.edu)