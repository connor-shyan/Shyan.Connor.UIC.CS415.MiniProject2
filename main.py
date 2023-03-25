#
# CS 415 - Mini Project 2
# Connor Shyan
# UIC, Fall 2022
# Using Edge Detection and Hough Transform code tutorials as the base
#

import cv2
import numpy as np
from copy import copy, deepcopy
import matplotlib.pyplot as plt
import math


#
# Convolution Function
#
def convolution(im, kernel):
    kernel = np.flip(np.flip(kernel, axis=0), axis=1)
    im_height, im_width = im.shape
    kernel_size = kernel.shape[0]
    pad_size = int((kernel_size - 1) / 2)
    im_padded = np.zeros((im_height + pad_size * 2, im_width + pad_size * 2))
    im_padded[pad_size:-pad_size, pad_size:-pad_size] = im

    im_out = np.zeros_like(im)
    for x in range(im_width):
        for y in range(im_height):
            im_patch = im_padded[y:y + kernel_size, x:x + kernel_size]
            new_value = np.sum(kernel * im_patch)
            im_out[y, x] = new_value
    return im_out


#
# Function to get Gaussian Kernel
#
def get_gaussian_kernel(kernel_size, sigma):
    kernel_x = np.linspace(-(kernel_size // 2), kernel_size // 2, kernel_size)
    for i in range(kernel_size):
        kernel_x[i] = np.exp(-(kernel_x[i] / sigma) ** 2 / 2)
    kernel = np.outer(kernel_x.T, kernel_x.T)

    kernel *= 1.0 / kernel.sum()
    return kernel


#
# Function to compute gradient magnitude and direction
#
def compute_gradient(im):
    sobel_filter_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    sobel_filter_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    gradient_x = convolution(im, sobel_filter_x)
    gradient_y = convolution(im, sobel_filter_y)

    magnitude = np.sqrt(np.square(gradient_x) + np.square(gradient_y))
    magnitude *= 255.0 / magnitude.max()
    direction = np.arctan2(gradient_y, gradient_x)
    direction *= 180 / np.pi
    return magnitude, direction


#
# Non-Maximum Suppression Function
#
def nms(magnitude, direction):
    height, width = magnitude.shape
    res = np.zeros(magnitude.shape)
    direction[direction < 0] += 180  # (-180, 180) -> (0, 180)

    for y in range(1, height - 1):
        for x in range(1, width - 1):
            current_direction = direction[y, x]
            current_magnitude = magnitude[y, x]
            if (0 <= current_direction < 22.5) or (157.5 <= current_direction <= 180):
                p = magnitude[y, x - 1]
                r = magnitude[y, x + 1]

            elif 22.5 <= current_direction < 67.5:
                p = magnitude[y + 1, x + 1]
                r = magnitude[y - 1, x - 1]

            elif 67.5 <= current_direction < 112.5:
                p = magnitude[y - 1, x]
                r = magnitude[y + 1, x]

            else:
                p = magnitude[y - 1, x + 1]
                r = magnitude[y + 1, x - 1]

            if current_magnitude >= p and current_magnitude >= r:
                res[y, x] = current_magnitude

    return res


#
# Hysteresis Thresholding Function
#
def hysteresis_thresholding(low, high, magnitude):
    # Getting the shape of the image
    height, width = magnitude.shape
    res = np.zeros(magnitude.shape)

    # Loop through the pixels
    for y in range(0, height):
        for x in range(0, width):

            # Checking if gradient magnitude is equal to or higher than high threshold
            if magnitude[y, x] > high:
                # Marking pixel as strong edge
                res[y, x] = 255
            # Checking if gradient magnitude is lower than low threshold
            elif magnitude[y, x] < low:
                # Discarding pixel as noise
                res[y, x] = 0
            # Checking if gradient magnitude is in between
            else:
                # Marking pixel as weak edge
                res[y, x] = 127

    # Returning resulting pixel values
    return res


#
# Edge Linking Function
#
def edge_linking(hysteresis):
    # Getting the shape of the image
    height, width = hysteresis.shape
    res = deepcopy(hysteresis)

    # Loop through the pixels
    for y in range(1, height - 1):
        for x in range(1, width - 1):

            # Checking if pixel is weak edge
            if res[y, x] == 127:

                # Checking any of the 8 neighboring pixels are strong edges
                if 255 in [res[y - 1, x - 1], res[y - 1, x], res[y, x - 1], res[y - 1, x + 1],
                           res[y + 1, x - 1], res[y, x + 1], res[y + 1, x], res[y + 1, x + 1]]:
                    # Replace pixel with strong edge
                    res[y, x] = 255

                # None of the neighboring pixels are strong edges
                else:
                    # Discard pixel as noise
                    res[y, x] = 0

    # Return modified image
    return res


#
# Hough Transform function from code tutorial
#
def HoughTransform(edge_map):
    theta_values = np.deg2rad(np.arange(-90.0, 90.0))
    height, width = edge_map.shape
    diagonal_length = int(round(math.sqrt(width * width + height * height)))
    rho_values = np.linspace(-diagonal_length, diagonal_length, diagonal_length * 2 + 1)

    accumulator = np.zeros((len(rho_values), len(theta_values)), dtype=int)
    y_coordinates, x_coordinates = np.nonzero(edge_map)

    for edge_idx in range(len(x_coordinates)):
        x = x_coordinates[edge_idx]
        y = y_coordinates[edge_idx]
        for theta_idx in range(len(theta_values)):
            theta = theta_values[theta_idx]
            rho = int(round(x * np.cos(theta) + y * np.sin(theta)))
            accumulator[rho + diagonal_length, theta_idx] += 1
        # print("%d out of %d edges have voted" % (edge_idx+1, len(x_coordinates)))
        # cv2.imshow("Accumulator", (accumulator*255/accumulator.max()).astype(np.uint8))
        # cv2.waitKey(0)
    return accumulator, theta_values, rho_values


#
# Function to suppress non local maxima in the accumulator array from Hough Transform
#
def suppress_non_local_maxima(accumulator):
    # Getting the shape of the image
    height, width = accumulator.shape
    res = deepcopy(accumulator)

    # Loop through the pixels
    for y in range(2, height - 2):
        for x in range(2, width - 2):

            # Checking if current pixel is not local maximum in its 5x5 neighborhood
            neighborhood = [res[y - 1, x - 1], res[y - 1, x], res[y, x - 1], res[y - 1, x + 1],
                            res[y + 1, x - 1], res[y, x + 1], res[y + 1, x], res[y + 1, x + 1],
                            res[y - 2, x - 2], res[y - 2, x], res[y, x - 2], res[y - 2, x + 2],
                            res[y + 2, x - 2], res[y, x + 2], res[y + 2, x], res[y + 2, x + 2],
                            res[y - 2, x - 1], res[y - 2, x + 1], res[y + 2, x - 1], res[y + 2, x + 1],
                            res[y - 1, x - 2], res[y + 1, x - 2], res[y - 1, x + 2], res[y + 1, x + 2]]
            if not all(res[y, x] > i for i in neighborhood):
                # Suppress pixel
                res[y, x] = 0

    # Return modified accumulator array
    return res


# Reading lena.png and putting it through gaussian filtering and NMS
im_lena = cv2.imread("lena.png", 0)
im_lena = im_lena.astype(float)
gaussian_kernel = get_gaussian_kernel(9, 3)
im_smoothed = convolution(im_lena, gaussian_kernel)
gradient_magnitude, gradient_direction = compute_gradient(im_smoothed)
edge_nms = nms(gradient_magnitude, gradient_direction)

# Putting the image through hysteresis thresholding (first)
# Low Threshold = 12, High Threshold = 24
hysteresis1 = hysteresis_thresholding(12, 24, edge_nms)
im_out_ht1 = hysteresis1.astype(np.uint8)
cv2.imwrite('lena_hysteresis_1.png', im_out_ht1)
# cv2.imshow("After Hysteresis Thresholding 1", im_out_ht1)
# cv2.waitKey()
# cv2.destroyAllWindows()

# Putting the image through hysteresis thresholding (second)
# Low Threshold = 8, High Threshold = 32
hysteresis2 = hysteresis_thresholding(8, 32, edge_nms)
im_out_ht2 = hysteresis2.astype(np.uint8)
cv2.imwrite('lena_hysteresis_2.png', im_out_ht2)
# cv2.imshow("After Hysteresis Thresholding 2", im_out_ht2)
# cv2.waitKey()
# cv2.destroyAllWindows()

# Putting the image through hysteresis thresholding (third)
# Low Threshold = 25, High Threshold = 35
hysteresis3 = hysteresis_thresholding(25, 35, edge_nms)
im_out_ht3 = hysteresis3.astype(np.uint8)
cv2.imwrite('lena_hysteresis_3.png', im_out_ht3)
# cv2.imshow("After Hysteresis Thresholding 3", im_out_ht3)
# cv2.waitKey()
# cv2.destroyAllWindows()

# Putting the first image through edge linking
edge_link_1 = edge_linking(hysteresis1)
im_out_el1 = edge_link_1.astype(np.uint8)
cv2.imwrite('lena_edge_link_1.png', im_out_el1)
# cv2.imshow("After Edge Linking 1", im_out_el1)
# cv2.waitKey()
# cv2.destroyAllWindows()

# Putting the second image through edge linking
edge_link_2 = edge_linking(hysteresis2)
im_out_el2 = edge_link_2.astype(np.uint8)
cv2.imwrite('lena_edge_link_2.png', im_out_el2)
# cv2.imshow("After Edge Linking 2", im_out_el2)
# cv2.waitKey()
# cv2.destroyAllWindows()

# Putting the third image through edge linking
edge_link_3 = edge_linking(hysteresis3)
im_out_el3 = edge_link_3.astype(np.uint8)
cv2.imwrite('lena_edge_link_3.png', im_out_el3)
# cv2.imshow("After Edge Linking 3", im_out_el3)
# cv2.waitKey()
# cv2.destroyAllWindows()


# Putting paper.bmp through Hough Transform
im1 = cv2.imread('paper.bmp')
im_gray1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
edge_map1 = cv2.Canny(im_gray1, 70, 150)
accumulator1, theta_values1, rho_values1 = HoughTransform(edge_map1)

# Suppressing non local maxima in accumulator array of paper.bmp
snlm_accumulator1 = suppress_non_local_maxima(accumulator1)

# Using modified accumulator array for paper.bmp for line detection
lines1 = np.argwhere(snlm_accumulator1 > 30)
height1, width1 = im_gray1.shape
for line1 in lines1:
    rho = rho_values1[line1[0]]
    theta = theta_values1[line1[1]]
    slope = -np.cos(theta)/np.sin(theta)
    intercept = rho/np.sin(theta)
    x1, x2 = 0, width1
    y1 = int(slope*x1 + intercept)
    y2 = int(slope*x2 + intercept)
    cv2.line(im1, (x1, y1), (x2, y2), (0, 0, 255), 2)
cv2.imwrite('paper_edge_map.png', edge_map1)
im_out_hough1 = (snlm_accumulator1*255/snlm_accumulator1.max()).astype(np.uint8)
cv2.imwrite('paper_hough_transform.png', im_out_hough1)
cv2.imwrite('paper_line_detection.png', im1)
# cv2.imshow("Edge Map for paper.bmp", edge_map1)
# cv2.imshow("Hough Transform for paper.bmp", im_out_hough1)
# cv2.imshow("Line Detection for paper.bmp", im1)
# cv2.waitKey()
# cv2.destroyAllWindows()


# Putting shape.bmp through Hough Transform
im2 = cv2.imread('shape.bmp')
im_gray2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
edge_map2 = cv2.Canny(im_gray2, 70, 150)
accumulator2, theta_values2, rho_values2 = HoughTransform(edge_map2)

# Suppressing non local maxima in accumulator array of shape.bmp
snlm_accumulator2 = suppress_non_local_maxima(accumulator2)

# Using modified accumulator array for shape.bmp for line detection
lines2 = np.argwhere(snlm_accumulator2 > 30)
height2, width2 = im_gray2.shape
for line2 in lines2:
    rho = rho_values2[line2[0]]
    theta = theta_values2[line2[1]]
    slope = -np.cos(theta)/np.sin(theta)
    intercept = rho/np.sin(theta)
    x1, x2 = 0, width2
    y1 = int(slope*x1 + intercept)
    y2 = int(slope*x2 + intercept)
    cv2.line(im2, (x1, y1), (x2, y2), (0, 0, 255), 2)
cv2.imwrite('shape_edge_map.png', edge_map2)
im_out_hough2 = (snlm_accumulator2*255/snlm_accumulator2.max()).astype(np.uint8)
cv2.imwrite('shape_hough_transform.png', im_out_hough2)
cv2.imwrite('shape_line_detection.png', im2)
# cv2.imshow("Edge Map for shape.bmp", edge_map2)
# cv2.imshow("Hough Transform for shape.bmp", im_out_hough2)
# cv2.imshow("Line Detection for shape.bmp", im2)
# cv2.waitKey()
# cv2.destroyAllWindows()


# Applying cv2.Canny to lena.png
im_lena = cv2.imread('lena.png')
lena_gray = cv2.cvtColor(im_lena, cv2.COLOR_BGR2GRAY)
lena_cv2_canny = cv2.Canny(lena_gray, 70, 150)
cv2.imwrite('lena_cv2_canny.png', lena_cv2_canny)
# cv2.imshow("CV2 Canny on lena.png")
# cv2.waitKey()
# cv2.destroyAllWindows()


# Putting paper.bmp through Hough Transform using cv2.HoughLines for line detection
im3 = cv2.imread('paper.bmp')
im_gray3 = cv2.cvtColor(im3, cv2.COLOR_BGR2GRAY)
edge_map3 = cv2.Canny(im_gray3, 70, 150)
lines3 = cv2.HoughLines(edge_map3, 1, np.pi/180, 50)
height3, width3 = im_gray3.shape
for r_theta in lines3:
    rho, theta = np.array(r_theta[0], dtype=np.float64)
    slope = -np.cos(theta)/np.sin(theta)
    intercept = rho/np.sin(theta)
    x1, x2 = 0, width3
    y1 = int(slope*x1 + intercept)
    y2 = int(slope*x2 + intercept)
    cv2.line(im3, (x1, y1), (x2, y2), (0, 0, 255), 2)
cv2.imwrite('paper_cv2_hough_line_detection.png', im3)
# cv2.imshow("CV2 Hough Line Detection for paper.bmp", im3)
# cv2.waitKey()
# cv2.destroyAllWindows()

# Putting shape.bmp through Hough Transform using cv2.HoughLines for line detection
im4 = cv2.imread('shape.bmp')
im_gray4 = cv2.cvtColor(im4, cv2.COLOR_BGR2GRAY)
edge_map4 = cv2.Canny(im_gray4, 70, 150)
lines4 = cv2.HoughLines(edge_map4, 1, np.pi/180, 50)
height4, width4 = im_gray4.shape
for r_theta in lines4:
    rho, theta = np.array(r_theta[0], dtype=np.float64)
    slope = -np.cos(theta)/np.sin(theta)
    intercept = rho/np.sin(theta)
    x1, x2 = 0, width4
    y1 = int(slope*x1 + intercept)
    y2 = int(slope*x2 + intercept)
    cv2.line(im4, (x1, y1), (x2, y2), (0, 0, 255), 2)
cv2.imwrite('shape_cv2_hough_line_detection.png', im4)
# cv2.imshow("CV2 Hough Line Detection for shape.bmp", im4)
# cv2.waitKey()
# cv2.destroyAllWindows()
