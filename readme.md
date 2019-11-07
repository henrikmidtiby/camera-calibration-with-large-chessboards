## Install actions

```
python3 -m venv env
source env/bin/activate
pip install -r requirements
```

## Run the program

Enter the directory and issue the following commands:
```
source env/bin/activate
python calibration.py --input folder --output folder --fisheye
```

## Development goal

I would like this program to develop, such that it can be used for camera calibration.

The tasks to do so is the following.
1. Use the ChessBoardCornerDetector class to extract calibration points from a set of images. The list of images should be defined from the command line.
2. Use the calibration points to calibrate the camera using an approach similar to https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html
3. Output the determined camera parameters to the screen and a file.
