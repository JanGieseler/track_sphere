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
folder_in = '../raw_data/20180607_Sample6_bead_1/'
filename_in = '20180611_Sample_6_Bead_7.avi'

frame_id = 9000

# method = 'fit_ellipse_with_bgs'
parameters = {
    'pre-processing': {'process_method': 'grabCut'},
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


cv.imshow("out", np.hstack([frame_in, frame_out_pre, frame_out]))

cv.imwrite('/Users/rettentulla/PycharmProjects/track_sphere/processed_data/xx.jpg',
           np.hstack([frame_in, frame_out_pre, frame_out]))


cv.waitKey(0)
