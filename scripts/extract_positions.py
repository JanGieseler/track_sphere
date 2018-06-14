## Example 0:
#  background subtraction using the method 'Bright px'
# Source file is 20171207_magnet.avi

import os
from glob import glob
from track_sphere.extract_data_opencv import *


dataset = 'real segemented'
dataset = 'real'
dataset = '20180607_Sample6_bead_1'
# dataset = 'test'

# dataset = 'test'

method = 'Bright px'
method = 'fit_blobs'
method = 'fit_ellipse'

export_video = False
output_fps = 2
# output_images = 1000
output_images = 10000

# max_frame = 610000
max_frame = 2000
# max_frame = None
# method_parameters = {'reencode':True}

process_method = 'morph'
################################################################################
## end settings ###
################################################################################



extraction_parameters = {'method': method}
process_parameters = {'process_method': process_method}
################################################################################
#### for our test data
################################################################################
if dataset == 'test':

    folder_in = '../example_data/'
    filename_in = '20171207_magnet.avi'
    folder_out = '../position_data/'

    if method == 'fit_blobs':
        extraction_parameters['winSize'] = (20, 20)
        extraction_parameters['initial_points'] = [[60, 109], [89, 66], [91, 108], [96, 142], [139, 113]]
        # extraction_parameters['initial_points'] = [[14, 104], [93,  57], [96, 107], [98, 140], [142,109]]
        extraction_parameters['initial_points'] = None
    elif method == 'Bright px':
        pass
    elif method == 'fit_ellipse':
        extraction_parameters['threshold'] = 'gaussian'
        extraction_parameters['blockSize'] = 35
        extraction_parameters['c'] = 11
        extraction_parameters['maxval'] = 255
        extraction_parameters['convex_hull'] = True

################################################################################
#### for real data: 20180607_Sample6_bead_1
################################################################################

if dataset == '20180607_Sample6_bead_1':
    folder_in = '../raw_data/20180607_Sample_6_bead_1/'
    # filename_in = '20180529_Sample6_bead_1_direct_thermal_01c.avi'
    filename_in = '20180607_Sample_6_Bead_1.avi'
    filename_in = '20180607_Sample_6_Bead_2.avi'
    # filename_in = '20180607_Sample_6_Bead_3.avi'
    filename_in = '20180608_Sample_6_Bead_4.avi'
    filename_in = '20180608_Sample_6_Bead_5.avi'
    # filename_in = '20180611_Sample_6_Bead_7.avi'
    # filename_in = '20180529_Sample6_bead_1_direct_thermal_01c-fixed.avi' # doesn't work!
    # folder_in = '/Volumes/Elements/lev_data/20180523_Sample_6_bead_1/'

    # folder_in = '../raw_data/20180529_Sample6_bead_1_direct_thermal_01c-segmented/'

    # processed_data
    folder_out = '../processed_data/position_data'
    # folder_out = '../'

    if method == 'fit_blobs':
        extraction_parameters['winSize'] = (16, 16)
        extraction_parameters['initial_points'] = [[35, 60], [56, 60], [57, 82], [55, 35], [80, 60]]
        extraction_parameters['initial_points'] = None
    elif method == 'Bright px':
        pass
    elif method == 'fit_ellipse':
        extraction_parameters['threshold'] = 'gaussian'
        extraction_parameters['blockSize'] = 51
        extraction_parameters['c'] = 11
        extraction_parameters['maxval'] = 255
        extraction_parameters['convex_hull'] = True

    # 01c fails at frame 687030
    # 01c_reencode fails at frame 613536, 610000







################################################################################
#### run the script
################################################################################



export_parameters = {
    'export_video': export_video,
    'output_fps': output_fps,
    'output_images': output_images
}

parameters = {
    'pre-processing': process_parameters,
    'extraction_parameters': extraction_parameters,
    'export_parameters': export_parameters
}


filename_out = filename_in.replace('.avi', '-{:s}.avi'.format(method))


file_in = os.path.join(folder_in, filename_in)
file_out = os.path.join(folder_out, filename_out)



if method == 'fit_blobs' and extraction_parameters['initial_points'] is None:
    extraction_parameters['initial_points'] = select_initial_points(file_in)


extract_position_data(file_in, file_out=file_out, min_frame=0, max_frame=max_frame, verbose=False,
                      parameters=parameters)
