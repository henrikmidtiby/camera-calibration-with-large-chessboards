from sklearn.neighbors import KDTree
import numpy as np
from icecream import ic
import collections

class PeakEnumerator():
    def __init__(self, centers):
        self.centers = centers
        self.central_peak_location = None
        self.distance_threshold = 0.13


    def select_central_peak_location(self):
        mean_position_of_centers = np.mean(
            self.centers, axis=0)
        
        central_center = np.array(
            sorted(list(self.centers), 
                   key=lambda c: np.sqrt((
                       c[0] - mean_position_of_centers[0]) ** 2 + 
                       (c[1] - mean_position_of_centers[1]) ** 2)))
        
        self.central_peak_location = central_center[0]
        return self.central_peak_location
    

    def enumerate_peaks(self):
        self.centers_kdtree = KDTree(np.array(self.centers))
        self.calibration_points = self.initialize_calibration_points(self.central_peak_location)
        self.build_examination_queue()
        self.analyse_elements_in_queue()
        return self.calibration_points


    def initialize_calibration_points(self, selected_center):
        closest_neighbour, _ = self.locate_nearest_neighbour(selected_center)
        direction = selected_center - closest_neighbour
        rotation_matrix = np.array([[0, 1], [-1, 0]])
        hat_vector = np.matmul(direction, rotation_matrix)
        direction_b_neighbour, _ = self.locate_nearest_neighbour(selected_center + hat_vector, minimum_distance_from_selected_center=-1)
        calibration_points = collections.defaultdict(dict)
        calibration_points[0][0] = selected_center
        calibration_points[1][0] = closest_neighbour
        calibration_points[0][1] = direction_b_neighbour

        return calibration_points


    def build_examination_queue(self):
        self.points_to_examine_queue = []
        for x_key, value in self.calibration_points.items():
            for y_key, _ in value.items():
                self.points_to_examine_queue.append((x_key, y_key))


    def analyse_elements_in_queue(self):
        for x_index, y_index in self.points_to_examine_queue:
            self.apply_all_rules_to_add_calibration_points(x_index, y_index)


    def apply_all_rules_to_add_calibration_points(self, x_index, y_index):
        self.rule_one(x_index, y_index) # Grow in +y direction
        self.rule_two(x_index, y_index) # Grow in +x direction
        self.rule_three(x_index, y_index) # Grow in +y direction, based on three points
        self.rule_four(x_index, y_index) # Grow in -y direction
        self.rule_five(x_index, y_index) # Grow in -x direction


    def rule_one(self, x_index, y_index):
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
        except KeyError:
            pass


    def rule_two(self, x_index, y_index):
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
        except KeyError:
            pass


    def rule_three(self, x_index, y_index):
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
        except KeyError:
            pass


    def rule_four(self, x_index, y_index):
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
            ic(reference_distance)
            if distance / reference_distance < self.distance_threshold:
                self.calibration_points[x_index][y_index - 1] = location
                self.points_to_examine_queue.append((x_index, y_index - 1))
        except KeyError:
            pass


    def rule_five(self, x_index, y_index):
        try:
            if y_index in self.calibration_points[x_index - 1]:
                return

            position_one = self.calibration_points[x_index + 1][y_index]
            position_two = self.calibration_points[x_index][y_index]
            predicted_location = 2 * position_two - position_one
            location, distance = self.locate_nearest_neighbour(predicted_location, minimum_distance_from_selected_center=-1)
            reference_distance = np.linalg.norm(position_two - position_one)
            if distance / reference_distance < self.distance_threshold:
                self.calibration_points[x_index - 1][y_index] = location
                self.points_to_examine_queue.append((x_index - 1, y_index))
        except KeyError:
            pass


    def locate_nearest_neighbour(self, selected_center, minimum_distance_from_selected_center=0):
        reshaped_query_array = np.array(selected_center).reshape(1, -1)
        (distances, indices) = self.centers_kdtree.query(reshaped_query_array, 2)
        if distances[0][0] <= minimum_distance_from_selected_center:
            return self.centers[indices[0][1]], distances[0][1]
        else:
            return self.centers[indices[0][0]], distances[0][0]

