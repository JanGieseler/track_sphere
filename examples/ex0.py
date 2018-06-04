## Example 0:
#  background subtraction using the method 'Bright px'
# Source file is 20171207_magnet.avi

import os
from track_sphere.extract_data_opencv import *


method = 'Bright px'

folder_in = '../example_data/'
# filename_in = '20180529_Sample6_bead_1_direct_thermal_01c_reencode.avi'
filename_in = '20171207_magnet.avi'

folder_out = '../example_out/ex0/'
filename_out = filename_in.replace('.avi', '-{:s}.avi'.format(method))

file_in = os.path.join(folder_in, filename_in)
file_out = os.path.join(folder_out, filename_out)


# substract_background(file_in, file_out=file_out, max_frame=2000 ,output_images=200, verbose=False, method=method)


export_parameters = {
    'output_images': 200
}

extract_position_data(file_in, file_out=file_out, max_frame=1000, verbose=False,
                      method=method,  export_parameters=export_parameters)