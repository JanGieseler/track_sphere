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

# ======== Settings a ========
method_parameters = {}
method_parameters['threshold'] = 100
method_parameters['maxval'] = 255
method_parameters['num_features'] = 5
method_parameters['convex_hull'] = False
folder_out = '../example_out/ex3-a/'


# ======== Settings b========
method_parameters = {}
method_parameters['threshold'] = 'gaussian'
# method_parameters['threshold'] = 'mean'
# method_parameters['threshold'] = 100
method_parameters['blockSize'] = 35
method_parameters['c'] = 11
method_parameters['maxval'] = 255
method_parameters['convex_hull'] = True
folder_out = '../example_out/ex3-b/'

# ======== run script ========

if export_video:
    folder_out = '../example_out/ex3-video/'

filename_out = filename_in.replace('.avi', '-{:s}.avi'.format(method))

file_in = os.path.join(folder_in, filename_in)
file_out = os.path.join(folder_out, filename_out)

export_parameters = {
    'export_video': export_video,
    'output_fps': output_fps,
    'output_images': output_images
}


extract_position_data(file_in, file_out=file_out, max_frame=1000, verbose=False,
                      method=method, method_parameters=method_parameters, export_parameters=export_parameters)

