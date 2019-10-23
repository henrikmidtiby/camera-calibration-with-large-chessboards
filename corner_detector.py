import cv2
import numpy as np
import MarkerTracker
import math
import time
import collections


class CornerDetector():
    def __init__(self, path_to_image):
        t_start = time.time()

        # Load image
        self.img = cv2.imread(path_to_image)
        assert self.img is not None, "Failed to load image"

        # Calculate corner responses
        response = calculate_corner_responses(self.img)
        print("%8.2f, convolution" % (time.time() - t_start))
        cv2.imwrite('output/00_response.png', response)

        # Localized normalization of responses
        response_relative_to_neighbourhood = local_normalization(response, 511)
        print("%8.2f, relative response" % (time.time() - t_start))
        cv2.imwrite("output/25_response_relative_to_neighbourhood.png", response_relative_to_neighbourhood * 255)

        # Threshold responses
        relative_responses_thresholded = threshold_responses(response_relative_to_neighbourhood)
        cv2.imwrite("output/26_relative_response_thresholded.png", relative_responses_thresholded)

        # Locate centers of peaks
        centers = locate_centers_of_peaks(relative_responses_thresholded)

        # Select central center of mass
        selected_center = select_central_peak_location(centers)

        # Locate nearby centers
        neighbours = locate_nearby_centers(selected_center, centers)

        # Enumerate detected peaks
        calibration_points = enumerate_peaks(centers, self.img, neighbours, selected_center)
        print("%8.2f, grid mapping" % (time.time() - t_start))

        # Show detected calibration points
        canvas = show_detected_calibration_points(self.img, calibration_points)
        cv2.imwrite("output/30_local_maxima.png", canvas)


def calculate_corner_responses(img):
    locator = MarkerTracker.MarkerTracker(order=2,
                                          kernel_size=45,
                                          scale_factor=40)

    greyscale_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    response = locator.apply_convolution_with_complex_kernel(greyscale_image)
    return response


def local_normalization(response, neighbourhoodsize):
    _, max_val, _, _ = cv2.minMaxLoc(response)
    response_relative_to_neighbourhood = peaks_relative_to_neighbourhood(response, neighbourhoodsize, 0.05 * max_val)
    return response_relative_to_neighbourhood


def threshold_responses(response_relative_to_neighbourhood):
    _, relative_responses_thresholded = cv2.threshold(response_relative_to_neighbourhood, 0.5, 255, cv2.THRESH_BINARY)
    return relative_responses_thresholded


def locate_centers_of_peaks(relative_responses_thresholded):
    contours, t1 = cv2.findContours(np.uint8(relative_responses_thresholded), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    centers = list(map(get_center_of_mass, contours))
    return centers


def select_central_peak_location(centers):
    mean_position_of_centers = np.mean(centers, axis=0)
    central_centers_temp = np.array(list(filter(lambda c: abs(c[0] -
        mean_position_of_centers[0]) < 300, centers)))
    central_centers = np.array(list(filter(lambda c: (c[1] -
        mean_position_of_centers[1]) < 300, central_centers_temp)))
    return central_centers[0]


def locate_nearby_centers(selected_center, centers):
    maximum_distance_to_neighbours = 300
    neighbours = list(filter(lambda c: abs(selected_center[0] - c[0]) +
        abs(selected_center[1] - c[1]) < maximum_distance_to_neighbours, centers))
    return neighbours


def enumerate_peaks(centers, img, neighbours, selected_center):
    calibration_points = initialize_calibration_points(neighbours, selected_center)

    distance_threshold = 0.06
    points_to_examine_queue = list()

    for k in range(2):
        for x_index in list(calibration_points.keys()):
            for y_index in list(calibration_points[x_index].keys()):
                rule_one(calibration_points, centers, distance_threshold,
                        x_index, y_index, points_to_examine_queue)
                rule_two(calibration_points, centers, distance_threshold,
                        x_index, y_index, points_to_examine_queue)
                rule_three(calibration_points, centers, distance_threshold,
                        x_index, y_index, points_to_examine_queue)
                rule_four(calibration_points, centers, distance_threshold,
                        x_index, y_index, points_to_examine_queue)
                rule_five(calibration_points, centers, distance_threshold,
                        x_index, y_index, points_to_examine_queue)

    for x_index, y_index in points_to_examine_queue:
        rule_one(calibration_points, centers, distance_threshold,
                x_index, y_index, points_to_examine_queue)
        rule_two(calibration_points, centers, distance_threshold,
                x_index, y_index, points_to_examine_queue)
        rule_three(calibration_points, centers, distance_threshold,
                x_index, y_index, points_to_examine_queue)
        rule_four(calibration_points, centers, distance_threshold,
                x_index, y_index, points_to_examine_queue)
        rule_five(calibration_points, centers, distance_threshold,
                x_index, y_index, points_to_examine_queue)
        break

    return calibration_points


def show_detected_calibration_points(img, calibration_points):
    canvas = img.copy()
    for x_index, temp in calibration_points.items():
        for y_index, cal_point in temp.items():
            cv2.circle(canvas,
                       tuple(cal_point.astype(int)),
                       20,
                       (0, 255 * (y_index % 2), 255 * (x_index % 2)),
                       2)
    return canvas 


def initialize_calibration_points(neighbours, selected_center):
    closest_neighbour, _ = locate_nearest_neighbour(neighbours, selected_center)
    direction = selected_center - closest_neighbour
    rotation_matrix = np.array([[0, 1], [-1, 0]])
    hat_vector = np.matmul(direction, rotation_matrix)
    direction_b_neighbour, _ = locate_nearest_neighbour(neighbours,
                                                        selected_center + hat_vector,
                                                        minimum_distance_from_selected_center=-1)

    calibration_points = collections.defaultdict(dict)
    calibration_points[0][0] = selected_center
    calibration_points[1][0] = closest_neighbour
    calibration_points[0][1] = direction_b_neighbour

    return calibration_points


def rule_three(calibration_points, centers, distance_threshold, x_index,
        y_index, points_to_examine_queue):
    try:
        # Ensure that we don't overwrite already located
        # points.
        if y_index + 1 in calibration_points[x_index]:
            return
        position_one = calibration_points[x_index - 1][y_index]
        position_two = calibration_points[x_index - 1][y_index + 1]
        position_three = calibration_points[x_index][y_index]
        predicted_location = position_two + position_three - position_one
        location, distance = locate_nearest_neighbour(centers,
                                                      predicted_location,
                                                      minimum_distance_from_selected_center=-1)
        reference_distance = np.linalg.norm(position_three - position_one)
        if distance / reference_distance < distance_threshold:
            calibration_points[x_index][y_index + 1] = location
            print('Added point using rule 3')
            points_to_examine_queue.append((x_index, y_index + 1))
    except:
        pass


def rule_two(calibration_points, centers, distance_threshold, x_index, y_index,
        points_to_examine_queue):
    try:
        if y_index in calibration_points[x_index + 1]:
            return

        position_one = calibration_points[x_index - 1][y_index]
        position_two = calibration_points[x_index][y_index]
        predicted_location = 2 * position_two - position_one
        location, distance = locate_nearest_neighbour(centers,
                                                      predicted_location,
                                                      minimum_distance_from_selected_center=-1)
        reference_distance = np.linalg.norm(position_two - position_one)
        if distance / reference_distance < distance_threshold:
            calibration_points[x_index + 1][y_index] = location
            print('Added point using rule 2 (%d, %d) + (%d %d) = (%d %d)' %
                  (x_index - 1, y_index, x_index, y_index, x_index + 1, y_index))
            points_to_examine_queue.append((x_index + 1, y_index))
    except:
        pass


def rule_one(calibration_points, centers, distance_threshold, x_index, y_index,
        points_to_examine_queue):
    try:
        # Ensure that we don't overwrite already located
        # points.
        if y_index + 1 in calibration_points[x_index]:
            return
        position_one = calibration_points[x_index][y_index]
        position_two = calibration_points[x_index][y_index - 1]
        predicted_location = 2 * position_one - position_two
        location, distance = locate_nearest_neighbour(centers,
                                                      predicted_location,
                                                      minimum_distance_from_selected_center=-1)
        reference_distance = np.linalg.norm(position_two - position_one)
        if distance / reference_distance < distance_threshold:
            calibration_points[x_index][y_index + 1] = location
            print('Added point using rule 1')
            points_to_examine_queue.append((x_index, y_index + 1))
    except:
        pass


def rule_four(calibration_points, centers, distance_threshold, x_index,
        y_index, points_to_examine_queue):
    try:
        # Ensure that we don't overwrite already located
        # points.
        if y_index - 1 in calibration_points[x_index]:
            return
        position_one = calibration_points[x_index][y_index]
        position_two = calibration_points[x_index][y_index + 1]
        predicted_location = 2 * position_one - position_two
        location, distance = locate_nearest_neighbour(centers,
                                                      predicted_location,
                                                      minimum_distance_from_selected_center=-1)
        reference_distance = np.linalg.norm(position_two - position_one)
        if distance / reference_distance < distance_threshold:
            calibration_points[x_index][y_index - 1] = location
            print('Added point using rule 4')
            points_to_examine_queue.append((x_index, y_index - 1))
    except:
        pass


def rule_five(calibration_points, centers, distance_threshold, x_index,
        y_index, points_to_examine_queue):
    try:
        if y_index in calibration_points[x_index - 1]:
            return

        position_one = calibration_points[x_index + 1][y_index]
        position_two = calibration_points[x_index][y_index]
        predicted_location = 2 * position_two - position_one
        location, distance = locate_nearest_neighbour(centers,
                                                      predicted_location,
                                                      minimum_distance_from_selected_center=-1)
        reference_distance = np.linalg.norm(position_two - position_one)
        if distance / reference_distance < distance_threshold:
            calibration_points[x_index - 1][y_index] = location
            print('Added point using rule 5 (%d, %d) + (%d %d) = (%d %d)' %
                  (x_index + 1, y_index, x_index, y_index, x_index - 1, y_index))
            points_to_examine_queue.append((x_index - 1, y_index))
    except:
        pass


def locate_nearest_neighbour(neighbours,
                             selected_center,
                             minimum_distance_from_selected_center=0):
    min_distance = np.inf
    closest_neighbour = None
    for neighbour in neighbours:
        distance = distance_to_ref(selected_center)(neighbour)
        if distance < min_distance:
            if distance > minimum_distance_from_selected_center:
                min_distance = distance
                closest_neighbour = neighbour
    return closest_neighbour, min_distance


def distance_to_ref(ref_point):
    return lambda c: ((c[0] - ref_point[0]) ** 2 + (c[1] - ref_point[1]) ** 2) ** 0.5


def get_center_of_mass(contour):
    m = cv2.moments(contour)
    if m["m00"] > 0:
        cx = m["m10"] / m["m00"]
        cy = m["m01"] / m["m00"]
        result = np.array([cx, cy])
    else:
        result = np.array([contour[0][0][0], contour[0][0][1]])
    return result


def peaks_relative_to_neighbourhood(response, neighbourhoodsize, value_to_add):
    local_min_image = minimum_image_value_in_neighbourhood(response, neighbourhoodsize)
    local_max_image = maximum_image_value_in_neighbourhood(response, neighbourhoodsize)
    response_relative_to_neighbourhood = (response - local_min_image) / (
            value_to_add + local_max_image - local_min_image)
    return response_relative_to_neighbourhood


def minimum_image_value_in_neighbourhood(response, neighbourhood_size):
    """
    A fast method for determining the local minimum value in
    a neighbourhood for an entire image.
    """
    kernel_1 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    orig_size = response.shape
    for x in range(int(math.log(neighbourhood_size, 2))):
        eroded_response = cv2.morphologyEx(response, cv2.MORPH_ERODE, kernel_1)
        response = cv2.resize(eroded_response, None, fx=0.5, fy=0.5)
    local_min_image_temp = cv2.resize(response, (orig_size[1], orig_size[0]))
    return local_min_image_temp


def maximum_image_value_in_neighbourhood(response, neighbourhood_size):
    """
    A fast method for determining the local maximum value in
    a neighbourhood for an entire image.
    """
    kernel_1 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    orig_size = response.shape
    for x in range(int(math.log(neighbourhood_size, 2))):
        eroded_response = cv2.morphologyEx(response, cv2.MORPH_DILATE, kernel_1)
        response = cv2.resize(eroded_response, None, fx=0.5, fy=0.5)
    local_min_image_temp = cv2.resize(response, (orig_size[1], orig_size[0]))
    return local_min_image_temp


CornerDetector('input/GOPR0003red.JPG')
