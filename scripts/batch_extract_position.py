# This batch script extracts the position information from video files and writes them into a .csv file
# meta data is also saved into a .json file

import os
from glob import glob
from track_sphere.extract_data_opencv import *


# select one of the cases, case specific parameters are defined below
case = 'extract all 20180628_Sample_6_bead_1'
# case = 'create video relevitate'
# case = 'create video oscillation rotation'

################################################################################
#### define parameters for each case
################################################################################
if case == 'extract all 20180607_Sample_6_bead_1':
    ################################################################################
    ## method settings ###
    ################################################################################
    method = 'fit_ellipse'

    export_video = False
    output_fps = 2
    output_images = 1000
    output_images = 10000

    # max_frame = 25000
    max_frame = None
    min_frame = 0

    process_method = 'morph'

    # processed_data
    folder_out = '../processed_data/position_data'
    ################################################################################
    ## end settings ###
    ################################################################################
    extraction_parameters = {'method': method}
    process_parameters = {'process_method': process_method}
    ################################################################################
    #### preprocessing morph parameters
    ################################################################################
    extraction_parameters['blockSize'] = 51  # 35 default
    extraction_parameters['convex_hull'] = True

    extraction_parameters['k_size_close'] = 11  # 11 default
    extraction_parameters['k_size_noise'] = 3  # 3 default
    extraction_parameters['c'] = 11  # 11 default
    extraction_parameters['select_contour'] = 'all'  # 'longest' default (other option is 'all')
    #source folder
    folder_in = '../raw_data/20180607_Sample_6_bead_1/'




    video_files = sorted(glob(os.path.join(folder_in, '*.avi')))

    # video_files = sorted([f for f in video_files if int(f.split('.avi')[0].split('Bead_')[1].split('_')[0]) in list(range(118, 119))])
    video_files = sorted(
        [f for f in video_files if int(f.split('.avi')[0].split('Bead_')[1].split('_')[0]) in list(range(154, 180))])
elif case == 'extract all 20180628_Sample_6_bead_1':
    ################################################################################
    ## method settings ###
    ################################################################################
    method = 'fit_ellipse'

    export_video = False
    output_fps = 2
    output_images = 1000
    output_images = 10000
    # output_images = 1

    # max_frame = 25000
    max_frame = None
    min_frame = 0

    process_method = 'morph'

    # processed_data
    folder_out = '../processed_data/position_data'
    ################################################################################
    ## end settings ###
    ################################################################################
    extraction_parameters = {'method': method}
    process_parameters = {'process_method': process_method}
    ################################################################################
    #### preprocessing morph parameters
    ################################################################################
    extraction_parameters['blockSize'] = 51  # 35 default
    extraction_parameters['convex_hull'] = True

    extraction_parameters['k_size_close'] = 11  # 11 default
    extraction_parameters['k_size_noise'] = 5  # 3 default
    extraction_parameters['c'] = 11  # 11 default
    extraction_parameters['select_contour'] = 'all'  # 'longest' default (other option is 'all')
    extraction_parameters['select_contour'] = 'longest'  # 'longest' default (other option is 'all')
    #source folder
    folder_in = '../raw_data/20180628_Sample_6_Bead_1/'




    video_files = sorted(glob(os.path.join(folder_in, '*.avi')))

    f = video_files[0]
    print(f.split('.avi')[0].split('Bead_1_'))

    # video_files = sorted([f for f in video_files if int(f.split('.avi')[0].split('Bead_')[1].split('_')[0]) in list(range(118, 119))])
    video_files = sorted(
        [f for f in video_files if int(f.split('.avi')[0].split('Bead_1_')[1].split('_')[0]) in list(range(0, 180))])

    video_files = [video_files[1]]




elif case == 'create video relevitate':
    ################################################################################
    ## method settings ###
    ################################################################################
    method = 'fit_ellipse'

    export_video = True
    output_fps = 2
    output_images = 1

    # max_frame = 2000
    max_frame = 12000
    min_frame = 11880

    process_method = 'morph'

    # processed_data
    folder_out = '../processed_data/video'
    ################################################################################
    ## end settings ###
    ################################################################################
    extraction_parameters = {'method': method}
    process_parameters = {'process_method': process_method}
    ################################################################################
    #### for real data: 20180607_Sample6_bead_1
    ################################################################################
    extraction_parameters['threshold'] = 'mean'
    extraction_parameters['blockSize'] = 21
    extraction_parameters['c'] = 5
    extraction_parameters['maxval'] = 255
    extraction_parameters['convex_hull'] = True
    #source folder
    folder_in = '../raw_data/20180607_Sample_6_bead_1/'

    video_files = sorted(glob(os.path.join(folder_in, '20180613_Sample_6_Bead_25_levitate.avi')))
elif case == 'create video oscillation rotation':
    ################################################################################
    ## method settings ###
    ################################################################################
    method = 'fit_ellipse'

    export_video = True
    output_fps = 25
    output_images = 1

    min_frame = 0
    max_frame = 2000
    # max_frame = 12000
    # min_frame = 11880

    process_method = 'morph'

    # processed_data
    folder_out = '../processed_data/video'
    ################################################################################
    ## end settings ###
    ################################################################################
    extraction_parameters = {'method': method}
    process_parameters = {'process_method': process_method}
    ################################################################################
    #### for real data: 20180607_Sample6_bead_1
    ################################################################################
    extraction_parameters['threshold'] = 'mean'
    extraction_parameters['blockSize'] = 21
    extraction_parameters['c'] = 5
    extraction_parameters['maxval'] = 255
    extraction_parameters['convex_hull'] = True
    #source folder
    folder_in = '../raw_data/20180607_Sample_6_bead_1/'

    video_files = sorted(glob(os.path.join(folder_in, '*96.avi')))
    video_files = sorted(glob(os.path.join(folder_in, '*36.avi')))

################################################################################
#### run the script
################################################################################
for f in video_files:

    filename_in = os.path.basename(f)
    print(filename_in)

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

    extract_position_data(file_in, file_out=file_out, min_frame=min_frame, max_frame=max_frame, verbose=False,
                          parameters=parameters)
    # extract_position_data(file_in, file_out=file_out, min_frame=1440, max_frame=1460, verbose=False,
    #                       parameters=parameters)