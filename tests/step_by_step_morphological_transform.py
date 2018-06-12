from track_sphere.utils import grab_frame
import cv2 as cv
import os
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt

from track_sphere.extract_data_opencv import check_method_parameters, load_video_info, get_frame_data, get_method_objects, process_image, add_features_to_image

import os
import cv2 as cv

folder_in = '../example_data/'
filename_in = '20171207_magnet.avi'

# ======== Settings ========
# folder_in = '../raw_data/20180607_Sample6_bead_1/'
# filename_in = '20180611_Sample_6_Bead_7.avi'

frame_id = 9000

# method = 'fit_ellipse_with_bgs'
parameters = {
    'pre-processing': {'process_method': 'adaptive_thresh_gauss'},
    'extraction_parameters': {'method': 'fit_ellipse'}
}
# parameters['pre-processing']['roi'] = (20, 20, 160, 160)
# parameters['pre-processing']['roi'] = (20, 20, 200, 200)
# parameters['pre-processing']['roi'] = (10, 10, 230, 230)

file_in = os.path.join(folder_in, filename_in)

frame_in = grab_frame(file_in, verbose=True, frame_id=frame_id)

info = load_video_info(file_in)

parameters = check_method_parameters(parameters, info, verbose=True)

processing_parameters = parameters['pre-processing']
extraction_parameters = parameters['extraction_parameters']
# # get headers for the data
# data_header = get_data_header(extraction_parameters)
# create method dependent objects
method_objects = get_method_objects(parameters)

print('asa', parameters)
method_objects = get_method_objects(parameters)


print(
'processing_parameters',processing_parameters
)
frame_out, feature_list = process_image(frame_in, parameters=processing_parameters, method_objects=method_objects,
                                        verbose=True, return_features=True)

frame_out_pre = deepcopy(frame_out)
# convert to color
frame_out_pre = cv.cvtColor(frame_out_pre, cv.COLOR_GRAY2BGR)
add_features_to_image(frame_out_pre, feature_list, verbose=True)

print('===========>> extract')
data, fl = get_frame_data(frame_out, parameters=extraction_parameters, return_features=True, method_objects=None,
                          verbose=True)
feature_list += fl
frame_out = cv.cvtColor(frame_out, cv.COLOR_GRAY2BGR)
add_features_to_image(frame_out, feature_list, verbose=True)
print('pppppp', len(feature_list))
# print('pppppp', feature_list)



frame_out_morph = deepcopy(frame_out_pre)
# frame_out_morph = cv.bitwise_not(frame_out_morph)

k_size = 3
iterations = 3
kernel = np.ones((k_size,k_size),np.uint8)

kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (k_size,k_size))


erosion = cv.erode(frame_out_morph,kernel,iterations=iterations)

dilation = cv.dilate(frame_out_morph,kernel,iterations=iterations)

opening = cv.morphologyEx(frame_out_morph, cv.MORPH_OPEN, kernel,iterations=iterations)
closing = cv.morphologyEx(frame_out_morph, cv.MORPH_CLOSE, kernel,iterations=iterations)

grad = cv.morphologyEx(frame_out_morph, cv.MORPH_GRADIENT, kernel)

# fill = deepcopy(grad)
# # Floodfill from point (0, 0)
h, w = grad.shape[:2]
mask = np.zeros((h+2, w+2), np.uint8)
# cv.floodFill(fill, mask, (0,0), 255);

# cv.imshow("out", np.hstack([frame_in, frame_out_pre, frame_out]))

# cv.imwrite('/Users/rettentulla/PycharmProjects/track_sphere/processed_data/xx.jpg',
#            np.hstack([frame_in, frame_out_pre, frame_out]))
#
# cv.imwrite('/Users/rettentulla/PycharmProjects/track_sphere/processed_data/xx.jpg',
#            np.hstack([[erosion, dilation], [opening, closing]]))

init_pt = (0,120)

cv.floodFill(erosion, mask, init_pt, 255)

h, w = grad.shape[:2]
mask = np.zeros((h+2, w+2), np.uint8)
cv.floodFill(dilation, mask, init_pt, 255)

h, w = grad.shape[:2]
mask = np.zeros((h+2, w+2), np.uint8)
cv.floodFill(opening, mask, init_pt, 255)
cv.floodFill(closing, np.zeros((h+2, w+2), np.uint8), init_pt, 255)
cv.floodFill(grad, np.zeros((h+2, w+2), np.uint8), init_pt, 255)

# cv.imshow('out', np.hstack([frame_in, frame_out_pre, frame_out]))
# cv.imshow('out2', np.hstack([erosion, dilation, opening, closing, grad]))
# cv.imshow('out2', np.hstack([[erosion, dilation], [opening, closing]]))


combined = cv.bitwise_not(deepcopy(frame_out_pre))
k_size = 2
iterations = 1
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (k_size,k_size))
combined = cv.morphologyEx(combined, cv.MORPH_OPEN, kernel, iterations=iterations)

k_size = 17
iterations = 1
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (k_size,k_size))
combined = cv.morphologyEx(combined, cv.MORPH_CLOSE, kernel, iterations=iterations)

cv.floodFill(combined, np.zeros((h+2, w+2), np.uint8), init_pt, 255)

# cv.waitKey(0)
cv.imwrite('/Users/rettentulla/PycharmProjects/track_sphere/processed_data/xx.jpg',
           np.hstack([erosion, dilation, opening, closing, grad, combined]))

cv.imwrite('/Users/rettentulla/PycharmProjects/track_sphere/processed_data/xx.jpg',
           np.vstack(
               [np.hstack([erosion, dilation, opening]),
                np.hstack([ closing, grad, combined])]))

cv.imwrite('/Users/rettentulla/PycharmProjects/track_sphere/processed_data/xx2.jpg', np.hstack([frame_in, frame_out_pre, frame_out]))


cv.destroyAllWindows()
