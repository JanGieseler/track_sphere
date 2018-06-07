## Example 0:
#  background subtraction using the method 'Bright px'
# Source file is 20171207_magnet.avi

import os
from glob import glob
from track_sphere.extract_data_opencv import *


dataset = 'real'

method = 'Bright px'
method = 'fit_blobs'
# method = 'fit_ellipse'

export_video = False
output_fps = 10
output_images = 2

################################################################################
#### for our test data
################################################################################
if dataset == 'test':

    folder_in = '../example_data/'
    filename_in = '20171207_magnet.avi'
    folder_out = '../position_data/'

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
if dataset == 'real':
    folder_in = '../raw_data/'
    filename_in = '20180529_Sample6_bead_1_direct_thermal_01c_reencode.avi'
    # folder_in = '/Volumes/Elements/lev_data/20180523_Sample_6_bead_1/'
    folder_out = '../processed_data/position_data'

    method_parameters = {}
    if method == 'fit_blobs':
        method_parameters['winSize'] = (16, 16)
        method_parameters['initial_points'] = [[35, 60], [56, 60], [57, 82], [55, 35], [80, 60]]

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

    # 01c fails at frame 687030
    # 01c_reencode fails at frame 613536, 610000

    extract_position_data(file_in, file_out=file_out, min_frame=0, max_frame=50, verbose=False,
                          method=method, method_parameters=method_parameters, export_parameters=export_parameters)
