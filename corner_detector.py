import cv2
import numpy as np
import MarkerTracker
import math
import time
import collections
import sklearn.neighbors


class ChessBoardCornerDetector():
    def __init__(self):
        self.threshold_distance_for_central_location = 300
        self.maximum_distance_to_neighbours = 300
        self.distance_threshold = 0.06
        pass
        
    def detect_chess_board_corners(self, path_to_image):
        # t_start = time.time()

        # Load image
        self.img = cv2.imread(path_to_image)
        assert self.img is not None, "Failed to load image"

        # Calculate corner responses
        response = self.calculate_corner_responses(self.img)
        # print("%8.2f, convolution" % (time.time() - t_start))
        # cv2.imwrite('output/00_response.png', response)

        # Localized normalization of responses
        response_relative_to_neighbourhood = self.local_normalization(response, 511)
        # print("%8.2f, relative response" % (time.time() - t_start))
        # cv2.imwrite("output/25_response_relative_to_neighbourhood.png", response_relative_to_neighbourhood * 255)

        # Threshold responses
        relative_responses_thresholded = self.threshold_responses(response_relative_to_neighbourhood)
        # cv2.imwrite("output/26_relative_response_thresholded.png", relative_responses_thresholded)

        # Locate centers of peaks
        centers = self.locate_centers_of_peaks(relative_responses_thresholded)

        # Select central center of mass
        selected_center = self.select_central_peak_location(centers)

        # Locate nearby centers
        neighbours = self.locate_nearby_centers(selected_center, centers)

        # Enumerate detected peaks
        self.calibration_points = self.enumerate_peaks(centers, self.img, neighbours, selected_center)
        # print("%8.2f, grid mapping" % (time.time() - t_start))

        # Show detected calibration points
        # canvas = self.show_detected_calibration_points(self.img, self.calibration_points)
        # cv2.imwrite("output/30_local_maxima.png", canvas)

        return self.calibration_points


    def calculate_corner_responses(self, img):
        locator = MarkerTracker.MarkerTracker(order=2,
                                              kernel_size=45,
                                              scale_factor=40)

        greyscale_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        response = locator.apply_convolution_with_complex_kernel(greyscale_image)
        return response


    def local_normalization(self, response, neighbourhoodsize):
        _, max_val, _, _ = cv2.minMaxLoc(response)
        response_relative_to_neighbourhood = self.peaks_relative_to_neighbourhood(response, neighbourhoodsize, 0.05 * max_val)
        return response_relative_to_neighbourhood


    def threshold_responses(self, response_relative_to_neighbourhood):
        _, relative_responses_thresholded = cv2.threshold(response_relative_to_neighbourhood, 0.5, 255, cv2.THRESH_BINARY)
        return relative_responses_thresholded


    def locate_centers_of_peaks(self, relative_responses_thresholded):
        contours, t1 = cv2.findContours(np.uint8(relative_responses_thresholded), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        centers = list(map(self.get_center_of_mass, contours))
        return centers


    def select_central_peak_location(self, centers):
        mean_position_of_centers = np.mean(centers, axis=0)
        central_centers_temp = np.array(list(filter(lambda c: abs(c[0] -
            mean_position_of_centers[0]) <
            self.threshold_distance_for_central_location, centers)))
        central_centers = np.array(list(filter(lambda c: (c[1] -
            mean_position_of_centers[1]) <
            self.threshold_distance_for_central_location, central_centers_temp)))
        return central_centers[0]


    def locate_nearby_centers(self, selected_center, centers):
        neighbours = list(filter(lambda c: abs(selected_center[0] - c[0]) +
            abs(selected_center[1] - c[1]) < self.maximum_distance_to_neighbours, centers))
        return neighbours


    def enumerate_peaks(self, centers, img, neighbours, selected_center):
        self.centers = centers
        self.centers_kdtree = sklearn.neighbors.KDTree(np.array(self.centers))

        self.calibration_points = self.initialize_calibration_points(neighbours, selected_center)

        self.points_to_examine_queue = list()
        self.points_to_examine_queue.append((0, 0))
        self.points_to_examine_queue.append((1, 0))
        self.points_to_examine_queue.append((0, 1))

        for x_index, y_index in self.points_to_examine_queue:
            self.apply_all_rules_to_add_calibration_points(centers, x_index, y_index)

        return self.calibration_points


    def show_detected_calibration_points(self, img, calibration_points):
        canvas = img.copy()
        for x_index, temp in calibration_points.items():
            for y_index, cal_point in temp.items():
                cv2.circle(canvas,
                           tuple(cal_point.astype(int)),
                           20,
                           (0, 255 * (y_index % 2), 255 * (x_index % 2)),
                           2)
        return canvas 


    def initialize_calibration_points(self, neighbours, selected_center):
        closest_neighbour, _ = self.locate_nearest_neighbour(selected_center)
        direction = selected_center - closest_neighbour
        rotation_matrix = np.array([[0, 1], [-1, 0]])
        hat_vector = np.matmul(direction, rotation_matrix)
        direction_b_neighbour, _ = self.locate_nearest_neighbour(selected_center + hat_vector,
                                                            minimum_distance_from_selected_center=-1)

        calibration_points = collections.defaultdict(dict)
        calibration_points[0][0] = selected_center
        calibration_points[1][0] = closest_neighbour
        calibration_points[0][1] = direction_b_neighbour

        return calibration_points


    def apply_all_rules_to_add_calibration_points(self, centers, x_index, y_index):
        self.rule_one(centers, x_index, y_index)
        self.rule_two(centers, x_index, y_index)
        self.rule_three(centers, x_index, y_index)
        self.rule_four(centers, x_index, y_index)
        self.rule_five(centers, x_index, y_index)


    def rule_three(self, centers, x_index, y_index):
        try:
            # Ensure that we don't overwrite already located
            # points.
            if y_index + 1 in self.calibration_points[x_index]:
                return
            position_one = self.calibration_points[x_index - 1][y_index]
            position_two = self.calibration_points[x_index - 1][y_index + 1]
            position_three = self.calibration_points[x_index][y_index]
            predicted_location = position_two + position_three - position_one
            location, distance = self.locate_nearest_neighbour(predicted_location,
                                                          minimum_distance_from_selected_center=-1)
            reference_distance = np.linalg.norm(position_three - position_one)
            if distance / reference_distance < self.distance_threshold:
                self.calibration_points[x_index][y_index + 1] = location
                self.points_to_examine_queue.append((x_index, y_index + 1))
        except:
            pass


    def rule_two(self, centers, x_index, y_index):
        try:
            if y_index in self.calibration_points[x_index + 1]:
                return

            position_one = self.calibration_points[x_index - 1][y_index]
            position_two = self.calibration_points[x_index][y_index]
            predicted_location = 2 * position_two - position_one
            location, distance = self.locate_nearest_neighbour(predicted_location,
                                                          minimum_distance_from_selected_center=-1)
            reference_distance = np.linalg.norm(position_two - position_one)
            if distance / reference_distance < self.distance_threshold:
                self.calibration_points[x_index + 1][y_index] = location
                self.points_to_examine_queue.append((x_index + 1, y_index))
        except:
            pass


    def rule_one(self, centers, x_index, y_index):
        try:
            # Ensure that we don't overwrite already located
            # points.
            if y_index + 1 in self.calibration_points[x_index]:
                return
            position_one = self.calibration_points[x_index][y_index]
            position_two = self.calibration_points[x_index][y_index - 1]
            predicted_location = 2 * position_one - position_two
            location, distance = self.locate_nearest_neighbour(predicted_location,
                                                          minimum_distance_from_selected_center=-1)
            reference_distance = np.linalg.norm(position_two - position_one)
            if distance / reference_distance < self.distance_threshold:
                self.calibration_points[x_index][y_index + 1] = location
                self.points_to_examine_queue.append((x_index, y_index + 1))
        except:
            pass


    def rule_four(self, centers, x_index, y_index):
        try:
            # Ensure that we don't overwrite already located
            # points.
            if y_index - 1 in self.calibration_points[x_index]:
                return
            position_one = self.calibration_points[x_index][y_index]
            position_two = self.calibration_points[x_index][y_index + 1]
            predicted_location = 2 * position_one - position_two
            location, distance = self.locate_nearest_neighbour(predicted_location,
                                                          minimum_distance_from_selected_center=-1)
            reference_distance = np.linalg.norm(position_two - position_one)
            if distance / reference_distance < self.distance_threshold:
                self.calibration_points[x_index][y_index - 1] = location
                self.points_to_examine_queue.append((x_index, y_index - 1))
        except:
            pass


    def rule_five(self, centers, x_index, y_index):
        try:
            if y_index in self.calibration_points[x_index - 1]:
                return

            position_one = self.calibration_points[x_index + 1][y_index]
            position_two = self.calibration_points[x_index][y_index]
            predicted_location = 2 * position_two - position_one
            location, distance = self.locate_nearest_neighbour(predicted_location,
                                                          minimum_distance_from_selected_center=-1)
            reference_distance = np.linalg.norm(position_two - position_one)
            if distance / reference_distance < self.distance_threshold:
                self.calibration_points[x_index - 1][y_index] = location
                self.points_to_examine_queue.append((x_index - 1, y_index))
        except:
            pass


    def locate_nearest_neighbour(self,
                                 selected_center,
                                 minimum_distance_from_selected_center=0):
        reshaped_query_array = np.array(selected_center).reshape(1, -1)
        (distances, indices) = self.centers_kdtree.query(reshaped_query_array, 2)

        if distances[0][0] <= minimum_distance_from_selected_center:
            return self.centers[indices[0][1]], distances[0][1]
        else:
            return self.centers[indices[0][0]], distances[0][0]


    def distance_to_ref(self, ref_point):
        return lambda c: ((c[0] - ref_point[0]) ** 2 + (c[1] - ref_point[1]) ** 2) ** 0.5


    def get_center_of_mass(self, contour):
        m = cv2.moments(contour)
        if m["m00"] > 0:
            cx = m["m10"] / m["m00"]
            cy = m["m01"] / m["m00"]
            result = np.array([cx, cy])
        else:
            result = np.array([contour[0][0][0], contour[0][0][1]])
        return result


    def peaks_relative_to_neighbourhood(self, response, neighbourhoodsize, value_to_add):
        local_min_image = self.minimum_image_value_in_neighbourhood(response, neighbourhoodsize)
        local_max_image = self.maximum_image_value_in_neighbourhood(response, neighbourhoodsize)
        response_relative_to_neighbourhood = (response - local_min_image) / (
                value_to_add + local_max_image - local_min_image)
        return response_relative_to_neighbourhood


    def minimum_image_value_in_neighbourhood(self, response, neighbourhood_size):
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


    def maximum_image_value_in_neighbourhood(self, response, neighbourhood_size):
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


# cbcd = ChessBoardCornerDetector();
# cbcd.detect_chess_board_corners('input/GOPR0003red.JPG')
