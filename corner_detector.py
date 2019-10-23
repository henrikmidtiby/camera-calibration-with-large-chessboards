import cv2
import numpy as np
import MarkerLocator.MarkerTracker as MarkerTracker
import math
import time
import collections

CalibrationPoint = collections.namedtuple('CalibrationPoint',
                                          ['position', 'coordinate', 'direction_a', 'direction_b'])
Point = collections.namedtuple('Point',
                               ['x', 'y'])


def main():
    t_start = time.time()
    # img = cv2.imread('input/4122AC35FBF7C6F7B71089A50CDC1814.jpg')
    img = cv2.imread('input/GOPR0011.JPG')
    assert img is not None, "Failed to load image"
    locator = MarkerTracker.MarkerTracker(order=2,
                                          kernel_size=45,
                                          scale_factor=40)

    greyscale_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print("%8.2f, conversion to grayscale" % (time.time() - t_start))
    response = locator.apply_convolution_with_complex_kernel(greyscale_image)
    print("%8.2f, convolution" % (time.time() - t_start))
    response_normalised = cv2.normalize(response, None, 0, 255, cv2.NORM_MINMAX)

    cv2.imwrite('output/00_response.png', response)
    cv2.imwrite('output/05_response_normalized.png', response_normalised)

    _, max_val, _, _ = cv2.minMaxLoc(response)

    response_relative_to_neighbourhood = peaks_relative_to_neighbourhood(response, 511, 0.05 * max_val)
    print("%8.2f, relative response" % (time.time() - t_start))
    cv2.imwrite("output/25_response_relative_to_neighbourhood.png", response_relative_to_neighbourhood * 255)

    _, relative_responses_thresholded = cv2.threshold(response_relative_to_neighbourhood, 0.5, 255, cv2.THRESH_BINARY)
    cv2.imwrite("output/26_relative_response_thresholded.png", relative_responses_thresholded)

    print(relative_responses_thresholded.dtype)

    contours, t1 = cv2.findContours(np.uint8(relative_responses_thresholded), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    print("contours: ", len(contours))
    centers = list(map(get_center_of_mass, contours))
    print("centers:", len(centers))
    # Select a central center of mass
    # TODO: Make this adaptive, so it is not tied to the location 
    # of the chessboard and size of the image.

    print(img.shape)
    central_centers_temp = np.array(list(filter(lambda c: abs(c[0] - img.shape[0] / 2)< 300, centers)))
    print(central_centers_temp)
    print("central_centers_temp:", len(central_centers_temp))
    central_centers = np.array(list(filter(lambda c: (c[1] - img.shape[1] / 2) < 300, central_centers_temp)))
    print(central_centers)
    selected_center = central_centers[0]

    # Locate nearby centers
    neighbours = list(filter(lambda c: abs(selected_center[0] - c[0]) + abs(selected_center[1] - c[1]) < 300,
                        centers))

    # attempt_one(centers, img, neighbours, selected_center)
    attempt_two(centers, img, neighbours, selected_center)


def attempt_two(centers, img, neighbours, selected_center):
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

    distance_threshold = 0.06
    print(calibration_points)

    for k in range(40):
        for x_index in list(calibration_points.keys()):
            for y_index in list(calibration_points[x_index].keys()):
                rule_one(calibration_points, centers, distance_threshold, x_index, y_index)
                rule_two(calibration_points, centers, distance_threshold, x_index, y_index)
                rule_three(calibration_points, centers, distance_threshold, x_index, y_index)
                rule_four(calibration_points, centers, distance_threshold, x_index, y_index)
                rule_five(calibration_points, centers, distance_threshold, x_index, y_index)


    canvas = img.copy()
    for temp in calibration_points.values():
        for cal_point in temp.values():
            cv2.circle(canvas,
                       tuple(cal_point.astype(int)),
                       40,
                       (0, 0, 255),
                       2)
    cv2.imwrite("output/30_local_maxima.png", canvas)


def rule_three(calibration_points, centers, distance_threshold, x_index, y_index):
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
    except:
        pass


def rule_two(calibration_points, centers, distance_threshold, x_index, y_index):
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
    except:
        pass


def rule_one(calibration_points, centers, distance_threshold, x_index, y_index):
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
    except:
        pass


def rule_four(calibration_points, centers, distance_threshold, x_index, y_index):
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
    except:
        pass


def rule_five(calibration_points, centers, distance_threshold, x_index, y_index):
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
    except:
        pass



def attempt_one(centers, img, neighbours, selected_center):
    closest_neighbour, _ = locate_nearest_neighbour(neighbours, selected_center)
    direction = selected_center - closest_neighbour
    rotation_matrix = np.array([[0, 1], [-1, 0]])
    hat_vector = np.matmul(direction, rotation_matrix)
    direction_b_neighbour, _ = locate_nearest_neighbour(neighbours,
                                                        selected_center + hat_vector,
                                                        minimum_distance_from_selected_center=-1)
    center_point = CalibrationPoint(selected_center,
                                    [0, 0],
                                    direction,
                                    direction_b_neighbour - selected_center)
    print(center_point)
    calibration_points = []
    current_point = center_point
    for k in range(30):
        probe_location = current_point.position + current_point.direction_a
        neighbour_location, _ = locate_nearest_neighbour(centers,
                                                         probe_location,
                                                         -1)
        new_direction_a = neighbour_location - current_point.position
        cal_point = CalibrationPoint(neighbour_location,
                                     current_point.coordinate + np.array([1, 0]),
                                     new_direction_a,
                                     current_point.direction_b)
        current_point = cal_point
        calibration_points.append(cal_point)
    canvas = img.copy()
    for cal_point in calibration_points:
        print(cal_point)
        draw_calibration_point(canvas, cal_point)
    cv2.imwrite("output/30_local_maxima.png", canvas)


def draw_calibration_point(canvas, center_point):
    cv2.circle(img=canvas,
               center=tuple(center_point.position.astype(int)),
               radius=20,
               color=(255, 255, 0),
               thickness=1)
    cv2.circle(img=canvas,
               center=tuple((center_point.position + center_point.direction_a).astype(int)),
               radius=20,
               color=(255, 0, 255),
               thickness=1)
    cv2.circle(img=canvas,
               center=tuple((center_point.position + center_point.direction_b).astype(int)),
               radius=20,
               color=(255, 0, 255),
               thickness=1)


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


main()
