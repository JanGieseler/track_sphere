## Example 1:
#  background subtraction using the method 'BackgroundSubtractorMOG2'
# Source file is 20171207_magnet.avi

import os
from track_sphere.extract_data_opencv import *


process_method = 'BackgroundSubtractorMOG2'
method = 'fit_ellipse'
folder_in = '../example_data/'
# filename_in = '20180529_Sample6_bead_1_direct_thermal_01c_reencode.avi'
filename_in = '20171207_magnet.avi'

folder_out = '../example_out/ex1/'
filename_out = filename_in.replace('.avi', '-{:s}.avi'.format(method))

file_in = os.path.join(folder_in, filename_in)
file_out = os.path.join(folder_out, filename_out)

export_parameters = {
    'output_images': 200
}


extraction_parameters = {'method': method}
process_parameters = {'process_method': process_method}


parameters = {
    'pre-processing': process_parameters,
    'extraction_parameters': extraction_parameters,
    'export_parameters': export_parameters
}



extract_position_data(file_in, file_out=file_out, max_frame=1000, verbose=False,
                      parameters=parameters)