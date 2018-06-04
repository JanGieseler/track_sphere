## Example 3:
#  background subtraction using the method 'fit_ellipse'
# Source file is 20171207_magnet.avi

import os
from track_sphere.extract_data_opencv import *

method = 'fit_ellipse'
method_parameters = {'detect_features': True} # set true or false of you also want to detect features


# ======== Settings ========

folder_in = './raw_data/'
# filename_in = '20180529_Sample6_bead_1_direct_thermal_01c_reencode.avi'
filename_in = '20180529_Sample6_bead_1_direct_thermal_01c_reencode.avi'
filename_in = '20180523_Sample6_bead_1_direct_thermal_03_reencode.avi'

export_video = True
output_fps = 10
output_images = 100

# ======== run script ========
if method_parameters['detect_features']:
    folder_out = './3/b/'
else:
    folder_out = './3/a/'

if export_video:
    folder_out = './3/video/'

filename_out = filename_in.replace('.avi', '-{:s}.avi'.format(method))

file_in = os.path.join(folder_in, filename_in)
file_out = os.path.join(folder_out, filename_out)

# set default values if not in dictionary

export_parameters = {
    'export_video': export_video,
    'output_fps': output_fps,
    'output_images': output_images
}

# extract_position_data(file_in, file_out=file_out, max_frame=2000, output_images=200, verbose=False,
#                       method=method, method_parameters=method_parameters,
#                       export_video=export_video, output_fps=output_fps)

extract_position_data(file_in, file_out=file_out, min_frame=0, max_frame=1000, verbose=False,
                      method=method, method_parameters=method_parameters, export_parameters=export_parameters)
