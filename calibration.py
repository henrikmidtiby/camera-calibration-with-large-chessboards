import numpy as np
import cv2

from corner_detector import ChessBoardCornerDetector

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

img = cv2.imread('input/GOPR0011red.JPG')

cbcd = ChessBoardCornerDetector()
corners = cbcd.detect_chess_board_corners('input/GOPR0011red.JPG')

count = 0
for key in corners.keys():
    count = count + len(corners[key])

# make objects with a length the amount of corners
objp = np.zeros(shape=(count,3), dtype=np.float32)
imgp = np.zeros(shape=(count,2), dtype=np.float32)

# fill up the vectors with the corners
count2 = 0
for key in corners.keys():
    for inner_key in corners[key].keys():
        objp[count2] = np.array([key, inner_key, 0])
        imgp[count2] = corners[key][inner_key]

        count2 = count2 + 1

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


objpoints.append(objp)
imgpoints.append(imgp)

# calibrate the camera
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
print(ret)
print(mtx)
print(dist)
print(rvecs)
print(tvecs)

# undistort images
h,  w = img.shape[:2]
newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

# crop the image
x,y,w,h = roi
# dst = dst[y:y+h, x:x+w]
cv2.imwrite('output/calibresult.png',dst)
