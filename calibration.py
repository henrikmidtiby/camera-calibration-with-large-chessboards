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
stats_before, stats_after, coverage_images = [], [], []
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
        (objp, imgp, coverage, statistics) = detect_calibration_pattern_in_image(file_path, args.output)
        stats_before.append(statistics)
        coverage_images.append(coverage)
        if coverage < min_percentage_coverage:
            print("ERROR: Less than " + str(min_percentage_coverage) +"% is covered with points, excluding from calibration")
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
    write_output(list_input, args.output, matrix, distortion, args.fisheye)


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
    cbcd = ChessBoardCornerDetector()
    # find all corners using the detector
    corners, coverage, statistics = cbcd.detect_chess_board_corners(file_path, output_folder)
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

    return objp, imgp, coverage, statistics


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


def write_output(list_input, output, mtx, dist, fisheye):
    """
    Write calibration matrix and distortion coefficients to file
    """
    output_path = output / 'camera_calibration.txt'
    with output_path.open(mode="w") as f:
        f.write("Calibration matrix: \n")
        for line in mtx:
            f.write(str(line) + '\n')
        if fisheye:
            f.write("\nDistortion parameters (k1, k2, k3, k4):\n")
        else:
            f.write("\nDistortion parameters (k1, k2, p1, p2, k3):\n")
        f.write(str(dist))
        f.write("\n\nSummary statistics:\n")
        hor_before, ver_before, hor_after, ver_after = 0.0,0.0,0.0,0.0
        good_images = 0
        f.write("Image".ljust(25) + "Coverage".ljust(12) + "Avg distor bef".ljust(17)+ "Avg distor aft\n" )
        for num, fname in enumerate(list_input):
            if coverage_images[num] >= min_percentage_coverage:
                avg_before = round(stats_before[num][0][1] + stats_before[num][1][1] /2 ,2)
                avg_after = round(stats_after[num][0][1] + stats_after[num][1][1] /2 ,2)
                hor_before += round(stats_before[num][0][1],2)
                ver_before += round(stats_before[num][1][1],2)
                hor_after += round(stats_after[num][0][1],2)
                ver_after += round(stats_after[num][1][1],2)
                f.write(str(fname.name).ljust(25) + (str(coverage_images[num]).rjust(3) + '%').ljust(12) + str(avg_before).ljust(17) + str(avg_after) + '\n')
                good_images += 1
            else:
                f.write(str(fname.name).ljust(25) + (str(coverage_images[num]).rjust(3) + '%').ljust(12) + " -----EXCLUDED-----\n" )
        avg_hor_before = round(hor_before / good_images,2)
        avg_ver_before = round(ver_before / good_images,2)
        avg_hor_after = round(hor_after / good_images,2)
        avg_ver_after = round(ver_after / good_images,2)
        f.write("\nAverage horizontal distortion before: " + str(avg_hor_before).ljust(6) + "pixels from ideal line")
        f.write("\nAverage vertical distortion before:   " + str(avg_ver_before).ljust(6) + "pixels from ideal line")
        f.write("\nAverage horizontal distortion after:  " + str(avg_hor_after).ljust(6) + "pixels from ideal line")
        f.write("\nAverage vertical distortion after:    " + str(avg_ver_after).ljust(6) + "pixels from ideal line")
        f.write("\nImages with a coverage lower than " + str(min_percentage_coverage) + "% are excluded from the calibration")

        f.write("\n\nExtended statistics:")
        for num, fname in enumerate(list_input):
            f.write("\n\t" + str(fname.name))
            if coverage_images[num] < min_percentage_coverage:
                f.write("\n\tPercentage of image covered with points: " + str(coverage_images[num]) + "%  -> EXCLUDED")
            else:
                f.write("\n\tPercentage of image covered with points: " + str(coverage_images[num]) + "%")
            f.write("\n\t\tBefore undistorting:")
            f.write("\n\t\t\tHorizontal points : " + str(stats_before[num][0][0]))
            f.write("\n\t\t\tAverage horizontal distortion: " + str(round(stats_before[num][0][1],2)))
            f.write("\n\t\t\tVertical points : " + str(stats_before[num][1][0]))
            f.write("\n\t\t\tAverage vertical distortion: " + str(round(stats_before[num][1][1],2)))
            f.write("\n\t\tAfter undistorting:")
            f.write("\n\t\t\tHorizontal points : " + str(stats_after[num][0][0]))
            f.write("\n\t\t\tAverage horizontal distortion: " + str(round(stats_after[num][0][1],2)))
            f.write("\n\t\t\tVertical points : " + str(stats_after[num][1][0]))
            f.write("\n\t\t\tAverage vertical distortion: " + str(round(stats_after[num][1][1],2)))



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
            cbcd = ChessBoardCornerDetector()
            # make stats
            statistics = cbcd.make_statistics(str(output / (fname.stem + '_undistorted' + fname.suffix)))
            stats_after.append(statistics)
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
            #dst = dst[y:y+h, x:x+w]
            cv2.imwrite(str(output / (fname.stem + '_undistorted' + fname.suffix)), dst)
            cbcd = ChessBoardCornerDetector()
            # make stats
            statistics = cbcd.make_statistics(str(output / (fname.stem + '_undistorted' + fname.suffix)))
            stats_after.append(statistics)

main()
