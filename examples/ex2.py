## Example 12
#  background subtraction using the method 'grabCut'
# Source file is 20171207_magnet.avi

import os
from track_sphere.extract_data_opencv import *




rect = (20,20,160,160)
method = 'grabCut'
iterations = 5

method_parameters = {'roi':rect, 'iterations':iterations}

folder_in = '../example_data/'
# filename_in = '20180529_Sample6_bead_1_direct_thermal_01c_reencode.avi'
filename_in = '20171207_magnet.avi'

folder_out = '../example_out/ex2/'
filename_out = filename_in.replace('.avi', '-{:s}.avi'.format(method))

file_in = os.path.join(folder_in, filename_in)
file_out = os.path.join(folder_out, filename_out)


export_parameters = {
    'output_images': 200
}

# extract_position_data(file_in, file_out=file_out, max
extract_position_data(file_in, file_out=file_out, max_frame=1000, verbose=False,
                      method=method, export_parameters=export_parameters, method_parameters=method_parameters)