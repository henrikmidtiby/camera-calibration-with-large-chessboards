import cv2
import numpy as np
import MarkerLocator.MarkerTracker as MarkerTracker
import math
import skimage
import time


def main():
    t_start = time.time()
    img = cv2.imread('input/4122AC35FBF7C6F7B71089A50CDC1814.jpg')
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

    im2, contours, hireachy = cv2.findContours(np.uint8(relative_responses_thresholded), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    centers = map(get_center_of_mass, contours)
    # Select a central center of mass

    central_centers = np.array(filter(lambda c: abs(c[0] - 3000) < 300 and abs(c[1] - 3000) < 300, centers))
    selected_center = central_centers[0]
    print(selected_center)

    # Locate nearby centers
    neighbours = filter(lambda c: abs(selected_center[0] - c[0]) + abs(selected_center[1] - c[1]) < 300,
                        centers)
    print(neighbours)

    print(map(distance_to_ref(selected_center), neighbours))

    # Using the skimage.feature.peak_local_max method.
    local_maxima = skimage.feature.peak_local_max(response,
                                                  min_distance=80,
                                                  threshold_rel=0.2)
    canvas = img.copy()
    for local_maximum in local_maxima:
        position = (local_maximum[1], local_maximum[0])
        canvas = cv2.circle(canvas, position, 20, (0, 0, 255), -1)
    cv2.imwrite("output/30_local_maxima.png", canvas)


def distance_to_ref(ref_point):
    return lambda c: ((c[0] - ref_point[0])**2 + (c[1] - ref_point[1])**2)**0.5


def get_center_of_mass(contour):
    m = cv2.moments(contour)
    cx = m["m10"] / m["m00"]
    cy = m["m01"] / m["m00"]
    return np.array([cx, cy])


def peaks_relative_to_neighbourhood(response, neighbourhoodsize, value_to_add):
    local_min_image = minimum_image_value_in_neightbourhood(response, neighbourhoodsize)
    local_max_image = maximum_image_value_in_neightbourhood(response, neighbourhoodsize)
    response_relative_to_neighbourhood = (response - local_min_image) / (
                value_to_add + local_max_image - local_min_image)
    return response_relative_to_neighbourhood


def minimum_image_value_in_neightbourhood(response, neighbourhood_size):
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


def maximum_image_value_in_neightbourhood(response, neighbourhood_size):
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
