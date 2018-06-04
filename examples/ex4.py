## Example 4:
#  feature extraction using the method 'features_surf'
# Source file is 20171207_magnet.avi

import os
from track_sphere.extract_data_opencv import *

method = 'features_surf'


# ======== Settings ========

folder_in = '../example_data/'
filename_in = '20171207_magnet.avi'

export_video = False
output_fps = 10
output_images=200


# ======== Settings b========
method_parameters = {}
method_parameters['xfeatures'] = 100
method_parameters['HessianThreshold'] = 1000
method_parameters['num_features'] = 5
folder_out = '../example_out/ex4/'

# ======== run script ========

if export_video:
    folder_out = '../example_out/ex4-video/'

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

