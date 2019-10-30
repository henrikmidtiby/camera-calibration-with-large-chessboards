import numpy as np
import cv2
import glob

from corner_detector import ChessBoardCornerDetector

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

# define all image types
types = ('*.JPG', '*.jpeg', '*.jpg', '*.png')
# make a list of all the image paths
files_grabbed = []
for files in types:
    files_grabbed.extend(glob.glob('input/' + files))
for fname in files_grabbed:
    print(fname)
    # read image
    img = cv2.imread(fname)
    # define detector
    cbcd = ChessBoardCornerDetector()
    # find all corners using the detector
    corners = cbcd.detect_chess_board_corners(fname)
    # count all the corners, necessary to define a np array with fixed size
    count = 0
    for key in corners.keys():
        count = count + len(corners[key])

    # make objects with a length the amount of corners
    objp = np.zeros(shape=(count, 3), dtype=np.float32)
    imgp = np.zeros(shape=(count, 2), dtype=np.float32)

    # fill up the vectors with the corners
    count2 = 0
    for key in corners.keys():
        for inner_key in corners[key].keys():
            objp[count2] = np.array([key, inner_key, 0])
            imgp[count2] = corners[key][inner_key]

            count2 = count2 + 1

    objpoints.append(objp)
    imgpoints.append(imgp)

# grab the first image so we can get the shape of the image (used to initialize the intrinsic camera matrix)
img = cv2.imread(files_grabbed[0])
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# calibrate the camera
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

for fname in files_grabbed:
    # read image
    img = cv2.imread(fname)
    # undistort images
    h,  w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

    # crop the image
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    cv2.imwrite('output/' + fname[6:],dst)

print("Calibration matrix: ")
print(mtx)
print("Distortion parameters (k1, k2, p1, p2, k3):")
print(dist)

file = open("output/camera_calibration.txt", "w")
file.write("Calibration matrix: \n")
for line in mtx:
    file.write(str(line) + '\n')
file.write("Distortion parameters (k1, k2, p1, p2, k3): \n")
file.write(str(dist))
file.close()
