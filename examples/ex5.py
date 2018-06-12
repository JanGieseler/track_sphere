## Example 4:
#  feature extraction using the method 'features_surf'
# Source file is 20171207_magnet.avi

import os
from track_sphere.extract_data_opencv import *
from track_sphere.utils import select_initial_points
method = 'fit_blobs'
process_method = None

# ======== Settings ========

folder_in = '../example_data/'
filename_in = '20171207_magnet.avi'

export_video = True
output_fps = 10
output_images=200


# ======== Settings b========
extraction_parameters = {'method':method}
# method_parameters['maxval'] = 100
# method_parameters['convex_hull'] = 1000
extraction_parameters['initial_points'] = None # set none if you want to select points manually
extraction_parameters['initial_points'] = [[60, 109], [89, 66], [91, 108], [96, 142], [139, 113]]

folder_out = '../example_out/ex5/'

# ======== run script ========

if export_video:
    folder_out = '../example_out/ex5-video/'
else:
    folder_out = '../example_out/ex5/'

filename_out = filename_in.replace('.avi', '-{:s}.avi'.format(method))

file_in = os.path.join(folder_in, filename_in)
file_out = os.path.join(folder_out, filename_out)

export_parameters = {
    'export_video': export_video,
    'output_fps': output_fps,
    'output_images': output_images
}

if extraction_parameters['initial_points'] is None:
    extraction_parameters['initial_points'] = select_initial_points(file_in)


process_parameters = {'process_method': process_method}

parameters = {
    'pre-processing': process_parameters,
    'extraction_parameters': extraction_parameters,
    'export_parameters': export_parameters
}



extract_position_data(file_in, file_out=file_out, max_frame=1000, verbose=False,
                      parameters=parameters)

