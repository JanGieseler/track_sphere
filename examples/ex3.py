## Example 3:
#  feature extraction using the method 'fit_ellipse'
# Source file is 20171207_magnet.avi

import os
from track_sphere.extract_data_opencv import *

method = 'fit_ellipse'


# ======== Settings ========

folder_in = '../example_data/'
# filename_in = '20180529_Sample6_bead_1_direct_thermal_01c_reencode.avi'
filename_in = '20171207_magnet.avi'

export_video = True
output_fps = 10
output_images=200

case = 'a' #
case = 'b' #
# case = 'video' #
case = 'c'

# ======== Settings a ========
if case == 'a':

    process_parameters = {'process_method': 'adaptive_thresh_gauss'}
    process_parameters['blockSize'] = 35
    process_parameters['c'] = 11
    process_parameters['maxval'] = 255
    process_parameters['convex_hull'] = False
    folder_out = '../example_out/ex3-a/'
# ======== Settings b========
elif case in ['b']:

    process_parameters = {'process_method': 'adaptive_thresh_mean'}
    # method_parameters['threshold'] = 'mean'
    # method_parameters['threshold'] = 100
    process_parameters['blockSize'] = 35
    process_parameters['c'] = 11
    process_parameters['maxval'] = 255
    process_parameters['convex_hull'] = True
    folder_out = '../example_out/ex3-b/'

elif case in ['c','video']:

    process_parameters = {'process_method': 'morph'}
    # process_parameters['blockSize'] = 35
    # process_parameters['c'] = 11
    # process_parameters['maxval'] = 255
    # process_parameters['convex_hull'] = False
    folder_out = '../example_out/ex3-c/'

# ======== run script ========
if case == 'video':
    export_video = True

    folder_out = '../example_out/ex3-video/'

filename_out = filename_in.replace('.avi', '-{:s}.avi'.format(method))

file_in = os.path.join(folder_in, filename_in)
file_out = os.path.join(folder_out, filename_out)

export_parameters = {
    'export_video': export_video,
    'output_fps': output_fps,
    'output_images': output_images
}

extraction_parameters = {'method': method}

parameters = {
    'pre-processing': process_parameters,
    'extraction_parameters': extraction_parameters,
    'export_parameters': export_parameters
}



extract_position_data(file_in, file_out=file_out, max_frame=1000, verbose=False,
                      parameters=parameters)



