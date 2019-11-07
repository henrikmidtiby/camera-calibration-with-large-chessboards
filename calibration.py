import numpy as np
import cv2
import argparse
from pathlib import Path
from corner_detector import ChessBoardCornerDetector

parser = argparse.ArgumentParser(description='Calibrate camera with multiple images, if no arguments are given, /input and /output in current folder are used')
parser.add_argument('-i', '--input', metavar='', type=lambda p: Path(p).absolute(), help='the input directory', default=Path(__file__).absolute().parent / "input")
parser.add_argument('-o', '--output', metavar='', type=lambda p: Path(p).absolute(), help='the output directory', default=Path(__file__).absolute().parent / "output")
parser.add_argument('-f', '--fisheye', dest='fisheye', action='store_true')
args = parser.parse_args()
min_percentage_coverage = 25
objpoints, imgpoints = [], []  # Every element is the list of one image
different_objp = np.zeros(shape=(593, 3), dtype=np.float32)
# different_imgp = []

def main():
    # make sure output directory exists, otherwise we make it
    args.output.mkdir(parents=False, exist_ok=True)
    # import path names of all images
    list_input = generate_list_of_images(args.input)
    if len(list_input) == 0:
        print("ERROR: No files found at the provided input location, program stopped")
        exit()

    # objpoints, imgpoints = [], []  # Every element is the list of one image
    # detect corners in every image
    for file_path in list_input:
        (objp, imgp, coverage) = detect_calibration_pattern_in_image(file_path, args.output)
        if coverage < min_percentage_coverage:
            print("ERROR: Less than " + str(min_percentage_coverage) +" percent of this image is covered with detected points, this image is excluded from the calibration")
        else:
            objpoints.append(objp)
            imgpoints.append(imgp)
    # calibrate camera
    matrix, distortion = calibrate_camera(str(list_input[0]), objpoints, imgpoints, args.fisheye)
    # undistort images
    path_to_undistorted_images = args.output / '5_undistorted_images'
    path_to_undistorted_images.mkdir(parents=False, exist_ok=True)
    undistort_images(list_input, path_to_undistorted_images, matrix, distortion, args.fisheye)
    # print output to screen and file
    print_output(matrix, distortion, args.fisheye)
    write_output(args.output, matrix, distortion, args.fisheye)


def generate_list_of_images(path_to_dir):
    """
    Returns a list of all the image paths in the provided directory
    """
    assert(path_to_dir.is_dir())
    file_paths_input = []
    for file in path_to_dir.iterdir():
        if file.suffix.lower() in ['.jpg', '.png', '.jpeg']:
            file_paths_input.append(file)

    return file_paths_input


def detect_calibration_pattern_in_image(file_path, output_folder):
    """
    Returns the coordinates of the detected corners in 3d object points (corresponds to the real world)
    and the corresponding coordinates in the image calibration plane
    """
    print(file_path.name)
    # define detector
    cbcd = ChessBoardCornerDetector(output_folder)
    # find all corners using the detector
    corners, coverage = cbcd.detect_chess_board_corners(file_path)
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

    return objp, imgp, coverage


def calibrate_camera(file_name_img, objpoints, imgpoints, fisheye = False):
    """
    Calibrate camera with the provided object points and image points
    The image is necessary to get the shape of the image to initialize the intrinsic camera matrix
    """
    img = cv2.imread(file_name_img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if fisheye:
        # the fisheye function is way more demanding on the format of the input...
        objpp = []
        imgpp = []
        for k in range(len(objpoints)):
            objpp.append(objpoints[k].reshape(1, -1, 3))
            imgpp.append(imgpoints[k].reshape(1, -1, 2))
        calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC+cv2.fisheye.CALIB_CHECK_COND+cv2.fisheye.CALIB_FIX_SKEW
        K = np.zeros((3, 3))
        D = np.zeros((4, 1))
        N_OK = len(objpoints)
        rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
        tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
        retval, mtx, dist, rvecs, tvecs	= cv2.fisheye.calibrate(objpp, imgpp, gray.shape[::-1], K, D, rvecs, tvecs, calibration_flags)

    else:
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    return mtx, dist


def print_output(mtx, dist, fisheye):
    """
    Write calibration matrix and distortion coefficients to screen
    """
    print("Calibration matrix: ")
    print(mtx)
    if fisheye:
        print("Distortion parameters (k1, k2, k3, k4):")
    else:
        print("Distortion parameters (k1, k2, p1, p2, k3):")
    print(dist)


def write_output(output, mtx, dist, fisheye):
    """
    Write calibration matrix and distortion coefficients to file
    """
    output_path = output / 'camera_calibration.txt'
    with output_path.open(mode="w") as f:
        f.write("Calibration matrix: \n")
        for line in mtx:
            f.write(str(line) + '\n')
        if fisheye:
            f.write("Distortion parameters (k1, k2, k3, k4):\n")
        else:
            f.write("Distortion parameters (k1, k2, p1, p2, k3):\n")
        f.write(str(dist))


def undistort_images(list_input, output, mtx, dist, fisheye):
    """
    Undistorts all images in the input folder and places them in the output folder
    """
    if fisheye:
        for fname in list_input:
            # read image
            img = cv2.imread(str(fname))
            # undistort images
            h,  w = img.shape[:2]
            map1, map2 = cv2.fisheye.initUndistortRectifyMap(mtx, dist, np.eye(3), mtx, (h, w), cv2.CV_16SC2)
            undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            cv2.imwrite(str(output / (fname.stem + '_undistorted' + fname.suffix)), undistorted_img)
    else:
        for fname in list_input:
            # read image
            img = cv2.imread(str(fname))
            # undistort images
            h,  w = img.shape[:2]
            newcamera_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

            dst = cv2.undistort(img, mtx, dist, None, newcamera_mtx)
            # crop the image
            x, y, w, h = roi
            dst = dst[y:y+h, x:x+w]
            cv2.imwrite(str(output / (fname.stem + '_undistorted' + fname.suffix)), dst)


main()
