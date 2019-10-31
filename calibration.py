import numpy as np
import cv2
import argparse
from pathlib import Path
from corner_detector import ChessBoardCornerDetector

parser = argparse.ArgumentParser(description='Calibrate camera with multiple images, if no arguments are given, /input and /output in current folder are used')
parser.add_argument('-i', '--input', metavar='', type=lambda p: Path(p).absolute(), help='the input directory', default=Path(__file__).absolute().parent / "input")
parser.add_argument('-o', '--output', metavar='', type=lambda p: Path(p).absolute(), help='the output directory', default=Path(__file__).absolute().parent / "output")
args = parser.parse_args()


def main():
    # make sure output directory exists, otherwise we make it
    args.output.mkdir(parents=False, exist_ok=True)
    # import path names of all images
    list_input = generate_list_of_images(args.input)
    if len(list_input) == 0:
        print("ERROR: No files found at the provided input location, program stopped")
        exit()

    objpoints, imgpoints = [], []  # Every element is the list of one image
    # detect corners in every image
    for file_path in list_input:
        (objp, imgp) = detect_calibration_pattern_in_image(file_path)
        objpoints.append(objp)
        imgpoints.append(imgp)
    # calibrate camera
    matrix, distortion = calibrate_camera(str(list_input[0]), objpoints, imgpoints)
    # undistort images
    undistort_images(list_input, args.output, matrix, distortion)
    # print output to screen and file
    print_output(matrix, distortion)
    write_output(args.output, matrix, distortion)


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


def detect_calibration_pattern_in_image(file_path):
    """
    Returns the coordinates of the detected corners in 3d object points (corresponds to the real world)
    and the corresponding coordinates in the image calibration plane
    """
    # define detector
    cbcd = ChessBoardCornerDetector()
    # find all corners using the detector
    corners = cbcd.detect_chess_board_corners(str(file_path))
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

    return objp, imgp


def calibrate_camera(file_name_img, objpoints, imgpoints):
    """
    Calibrate camera with the provided object points and image points
    The image is necessary to get the shape of the image to initialize the intrinsic camera matrix
    """
    img = cv2.imread(file_name_img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # calibrate the camera
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    return mtx, dist


def print_output(mtx, dist):
    """
    Write calibration matrix and distortion coefficients to screen
    """
    print("Calibration matrix: ")
    print(mtx)
    print("Distortion parameters (k1, k2, p1, p2, k3):")
    print(dist)


def write_output(output, mtx, dist):
    """
    Write calibration matrix and distortion coefficients to file
    """
    output_path = output / 'camera_calibration.txt'
    with output_path.open(mode="w") as f:
        f.write("Calibration matrix: \n")
        for line in mtx:
            f.write(str(line) + '\n')
        f.write("Distortion parameters (k1, k2, p1, p2, k3): \n")
        f.write(str(dist))


def undistort_images(list_input, output, mtx, dist):
    """
    Undistorts all images in the input folder and places them in the output folder
    """
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
        cv2.imwrite(str(output / fname.name), dst)


main()
