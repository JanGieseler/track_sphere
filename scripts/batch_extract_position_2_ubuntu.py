# This batch script extracts the position information from video files and writes them into a .csv file
# meta data is also saved into a .json file
# this file is a copy of batch_extract_position and contains the setting for extracting videos on the ubuntu starting
# December 10th 2018

import os
from glob import glob
from track_sphere.extract_data_opencv import *


# select one of the cases, case specific parameters are defined below
case = 'extract all 20180628_Sample_6_bead_1'
case = 'extract top bright spot mc110 20181204_Sample_14_Bead_3'
case = 'extract top bright spot mc110 20191015_Sample_14_Bead_6'
case = 'extract top bright spot mc110 20190125_Sample_14_Bead_6'

process_parameters = {}

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
elif case == 'extract top bright spot mc110 20181204_Sample_14_Bead_3':
    ################################################################################
    ## method settings ###
    ################################################################################
    method = 'Bright px'

    export_video = False
    output_fps = 2
    # output_images = 1000
    output_images = 10000
    # output_images = 1

    # max_frame = 25000
    # max_frame = int(1e6)
    max_frame = None
    # max_frame = 10
    # min_frame = 1000
    min_frame = 0


    folder_out = '../processed_data/20181204_Sample_14_Bead_3/position_data'
    folder_out = '/run/user/1000/gvfs/smb-share:server=fs2k02.rc.fas.harvard.edu,share=lukin_lab/Lab/Lev/video_processed_data/20181204_Sample_14_Bead_3'
    ################################################################################
    ## end settings ###
    ################################################################################


    # source folder
    # folder_in = '../raw_data/20181204_Sample_14_Bead_3/'
    folder_in = '/run/user/1000/gvfs/smb-share:server=fs2k02.rc.fas.harvard.edu,share=lukin_lab/Lab/Lev/videos/20181204_Sample_14_Bead_3'


    ################################################################################
    ## end settings ###
    ################################################################################
    extraction_parameters = {'method': method}
    # option 2 select roi
    process_method = 'bilateral'
    process_parameters = {'process_method': process_method}
    process_parameters['filter_dimension'] = 9  # default 5
    process_parameters['sigmaColor'] = 120  # default 50
    process_parameters['sigmaSpace'] = 120  # default 50
    process_parameters['normalize'] = True  # default True

    # process_parameters['roi'] = (1, 22, 30, 30)  # default (60, 60, 30, 30)
    process_parameters['roi'] = (10, 10, 40, 40)  # default (60, 60, 30, 30)
    # process_parameters['roi'] = (76, 76, 40, 40)  # default (60, 60, 30, 30)

    # runs = list(range(3,5))
    run = 1
    runs = list(range(run, run + 1))

    # runs = [2,4,5,7]


    video_files = sorted(glob(os.path.join(folder_in, '*.avi')))
    # print(video_files)
    video_files = sorted(
        [f for f in video_files if int(f.split('.avi')[0].split('Bead_3_')[1].split('_')[0]) in runs])

    # print('video_files', video_files)

    # video_files = video_files[96:]
elif case == 'extract top bright spot mc110 20191015_Sample_14_Bead_6':
    ################################################################################
    ## method settings ###
    ################################################################################
    method = 'Bright px'

    export_video = False
    output_fps = 2
    # output_images = 1000
    output_images = 10000
    # output_images = 1

    # max_frame = 25000
    # max_frame = int(1e6)
    max_frame = None
    # max_frame = 10
    # min_frame = 1000
    min_frame = 0


    folder_out = '../processed_data/20191015_Sample_14_Bead_6/position_data'
    folder_out = '/run/user/1000/gvfs/smb-share:server=fs2k02.rc.fas.harvard.edu,share=lukin_lab/Lab/Lev/video_processed_data/20191015_Sample_14_Bead_6'
    ################################################################################
    ## end settings ###
    ################################################################################


    # source folder
    # folder_in = '../raw_data/20181204_Sample_14_Bead_3/'
    folder_in = '/run/user/1000/gvfs/smb-share:server=fs2k02.rc.fas.harvard.edu,share=lukin_lab/Lab/Lev/videos/20191015_Sample_14_Bead_6'


    ################################################################################
    ## end settings ###
    ################################################################################
    extraction_parameters = {'method': method}
    # option 2 select roi
    process_method = 'bilateral'
    process_parameters = {'process_method': process_method}
    process_parameters['filter_dimension'] = 9  # default 5
    process_parameters['sigmaColor'] = 120  # default 50
    process_parameters['sigmaSpace'] = 120  # default 50
    process_parameters['normalize'] = True  # default True

    # process_parameters['roi'] = (1, 22, 30, 30)  # default (60, 60, 30, 30)
    process_parameters['roi'] = (32, 32, 96, 96)  # default (60, 60, 30, 30)
    # process_parameters['roi'] = (76, 76, 40, 40)  # default (60, 60, 30, 30)

    # runs = list(range(3,5))
    run = 1
    runs = list(range(run, run + 1))

    # runs = [2,4,5,7]


    video_files = sorted(glob(os.path.join(folder_in, '*.avi')))
    print(video_files)
    video_files = sorted(
        [f for f in video_files if int(f.split('.avi')[0].split('Bead_6_run_')[1].split('_')[0]) in runs])

    print('video_files', video_files)

    # video_files = video_files[96:]
elif case == 'extract top bright spot mc110 20190125_Sample_14_Bead_6':
    ################################################################################
    ## method settings ###
    ################################################################################
    method = 'Bright px'

    export_video = False
    output_fps = 2
    output_images = 1000
    # output_images = 10000
    # output_images = 1

    # max_frame = 25000
    # max_frame = int(1e6)
    max_frame = None
    # max_frame = 10
    # min_frame = 1000
    min_frame = 0


    folder_out = '../processed_data/20190125_Sample_14_Bead_6/position_data'
    folder_out = '/run/user/1000/gvfs/smb-share:server=fs2k02.rc.fas.harvard.edu,share=lukin_lab/Lab/Lev/video_processed_data/20190125_Sample_14_Bead_6'
    ################################################################################
    ## end settings ###
    ################################################################################


    # source folder
    # folder_in = '../raw_data/20181204_Sample_14_Bead_3/'
    folder_in = '/run/user/1000/gvfs/smb-share:server=fs2k02.rc.fas.harvard.edu,share=lukin_lab/Lab/Lev/videos/20190125_Sample_14_Bead_6'


    ################################################################################
    ## end settings ###
    ################################################################################
    extraction_parameters = {'method': method}
    # option 2 select roi
    process_method = 'bilateral'
    process_parameters = {'process_method': process_method}
    process_parameters['filter_dimension'] = 9  # default 5
    process_parameters['sigmaColor'] = 120  # default 50
    process_parameters['sigmaSpace'] = 120  # default 50
    process_parameters['normalize'] = True  # default True

    # process_parameters['roi'] = (1, 22, 30, 30)  # default (60, 60, 30, 30)
    process_parameters['roi'] = (32, 32, 96, 96)  # default (60, 60, 30, 30)
    # process_parameters['roi'] = (76, 76, 40, 40)  # default (60, 60, 30, 30)

    # runs = list(range(3,5))
    run = 1
    runs = list(range(run, run + 1))

    # runs = [2,4,5,7]


    video_files = sorted(glob(os.path.join(folder_in, '*.avi')))
    print(video_files)
    video_files = sorted(
        [f for f in video_files if int(f.split('.avi')[0].split('Bead_6_run_')[1].split('_')[0]) in runs])

    print('video_files', video_files)

    # video_files = video_files[96:]

################################################################################
#### run the script
################################################################################
for f in video_files:

    filename_in = os.path.basename(f)
    print('file: ', filename_in)

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
