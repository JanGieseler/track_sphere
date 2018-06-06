## Example 0:
#  background subtraction using the method 'Bright px'
# Source file is 20171207_magnet.avi

import os
from glob import glob
from track_sphere.extract_data_opencv import *




################################################################################
#### for our test data
################################################################################

method = 'Bright px'
method = 'fit_blobs'
# method = 'fit_ellipse'

folder_in = '../example_data/'
filename_in = '20171207_magnet.avi'
folder_out = '../position_data/'

export_video = False
output_fps = 10
output_images = 1000

method_parameters = {}
if method == 'fit_blobs':
    method_parameters['winSize'] = (20, 20)
    method_parameters['initial_points'] = [[60, 109], [89, 66], [91, 108], [96, 142], [139, 113]]
    # method_parameters['initial_points'] = [[14, 104], [93,  57], [96, 107], [98, 140], [142,109]]
elif method == 'Bright px':
    pass
elif method == 'fit_ellipse':
    method_parameters['threshold'] = 'gaussian'
    method_parameters['blockSize'] = 35
    method_parameters['c'] = 11
    method_parameters['maxval'] = 255
    method_parameters['convex_hull'] = True

# ----- end settings --------


export_parameters = {
    'export_video': export_video,
    'output_fps': output_fps,
    'output_images': output_images
}

filename_out = filename_in.replace('.avi', '-{:s}.avi'.format(method))


file_in = os.path.join(folder_in, filename_in)
file_out = os.path.join(folder_out, filename_out)



if method == 'fit_blobs' and method_parameters['initial_points'] is None:
    method_parameters['initial_points'] = select_initial_points(file_in)


extract_position_data(file_in, file_out=file_out, min_frame=0, max_frame=None, verbose=False,
                      method=method, method_parameters=method_parameters, export_parameters=export_parameters)

################################################################################
#### for real data
################################################################################

# folder_in = '/Volumes/Elements/lev_data/20180523_Sample_6_bead_1/'
# folder_out = os.path.join('../data/', os.path.dirname(folder_in))
#
# print('source folder:'.format(folder_in))
# print('target folder:'.format(folder_out))
#
# print(glob(os.path.join(folder_in, '*.avi')))
#
# filename_in = '20171207_magnet.avi'
#
# filename_out = filename_in.replace('.avi', '-{:s}.avi'.format(method))
#
# file_in = os.path.join(folder_in, filename_in)
# file_out = os.path.join(folder_out, filename_out)
#
#
# # substract_background(file_in, file_out=file_out, max_frame=2000 ,output_images=200, verbose=False, method=method)
#
#
# export_parameters = {
#     'output_images': 200
# }
#
# extract_position_data(file_in, file_out=file_out, max_frame=1000, verbose=False,
#                       method=method,  export_parameters=export_parameters)