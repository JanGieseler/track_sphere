# This batch script extracts the position information from video files and writes them into a .csv file
# meta data is also saved into a .json file

import os
from glob import glob
from track_sphere.extract_data_opencv import *


# select one of the cases, case specific parameters are defined below
case = 'extract all 20180628_Sample_6_bead_1'
# case = 'extract all 20180628_Sample_6_bead_1 sideview'
# case = 'create video relevitate'
# case = 'create video oscillation rotation'
# case = 'extract top view full 20180710_M110_Sample_6_Bead_1'
case = 'extract sideview spot 20180710_M110_Sample_6_Bead_1'
# case = 'test'

# case = 'extract sideview spot 20180710_M110_Sample_6_Bead_1'
case = 'extract top view full 20180718_M110_Sample_6_Bead_1'
case = 'extract sideview spot 20180718_M110_Sample_6_Bead_1'
case = 'extract top view full 20180724_M110_Sample_6_Bead_1'
case = 'extract top bright spot 20180724_M110_Sample_6_Bead_1'
case = 'extract top bright spot ueye 20180724_M110_Sample_6_Bead_1'

case = 'extract top bright spot mc110 20180731_Sample_9_Bead_2'
# case = 'extract top full mc110 20180731_Sample_9_Bead_2'
case = 'extract top full mc110 20180801_Sample_9_Bead_2'
case = 'extract top bright spot mc110 20180801_Sample_9_Bead_2'
# case = 'extract top full mc110 20180802_Sample_9_Bead_2'
case = 'extract top full mc110 20180806_Sample_9_Bead_2'
# case = 'extract top bright spot mc110 20180806_Sample_9_Bead_2'
case = 'extract top bright spot mc110 20180810_Sample_10_Bead_B2'
case = 'extract top full mc110 20180821_Sample_10_Bead_F5'
case = 'extract top bright spot mc110 20180821_Sample_10_Bead_F5'
case = 'extract top full mc110 20180824_Sample_6_Bead_1'
# case = 'extract top bright spot mc110 20180824_Sample_6_Bead_1'
case = 'extract top full mc110 20180910_Sample_13_Bead_4'
case = 'extract top bright spot mc110 20181204_Sample_14_Bead_3'
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
    # max_frame = 1050
    # min_frame = 1000
    min_frame = 0

    process_method = 'morph'

    # processed_data
    folder_out = '../processed_data/position_data'
    ################################################################################
    ## end settings ###
    ################################################################################
    extraction_parameters = {'method': method}
    process_parameters = {'process_method': process_method}
    process_parameters['k_size_close'] = 11  # 11 default
    process_parameters['k_size_noise'] = 5  # 3 default
    process_parameters['c'] = 11  # 11 default
    process_parameters['select_contour'] = 'all'  # 'longest' default (other option is 'all')
    process_parameters['select_contour'] = 'longest'  # 'longest' default (other option is 'all')
    process_parameters['blockSize'] = 101  # 35 default

    # process_parameters['threshold_type'] = 'gauss'  # default is mean maxval
    process_parameters['maxval'] = 255  # default is 255

    ################################################################################
    #### preprocessing morph parameters
    ################################################################################

    extraction_parameters['convex_hull'] = True


    # extraction_parameters['select_contour'] = 'longest'  # 'longest' default (other option is 'all')
    #
    # extraction_parameters['blockSize'] = 21
    # extraction_parameters['select_contour'] = 'all'  # 'longest' default (other option is 'all')
    # extraction_parameters['c'] = 11  # 11 default
    # extraction_parameters['k_size_close'] = 5  # 11 default
    # extraction_parameters['k_size_noise'] = 3  # 3 default
    #source folder
    folder_in = '../raw_data/20180628_Sample_6_Bead_1/'




    video_files = sorted(glob(os.path.join(folder_in, '*.avi')))

    f = video_files[0]
    print(f.split('.avi')[0].split('Bead_1_'))

    # video_files = sorted([f for f in video_files if int(f.split('.avi')[0].split('Bead_')[1].split('_')[0]) in list(range(118, 119))])
    video_files = sorted(
        [f for f in video_files if int(f.split('.avi')[0].split('Bead_1_')[1].split('_')[0]) in list(range(0, 200))])

    # video_files = [video_files[18]]
    video_files = video_files[96:]
    # video_files = video_files[71:72]
elif case == 'extract all 20180628_Sample_6_bead_1 sideview':
    ################################################################################
    ## method settings ###
    ################################################################################
    method = 'Bright px'

    export_video = False
    output_fps = 2
    output_images = 1000
    output_images = 10000
    # output_images = 10

    # max_frame = 25000
    max_frame = None
    # max_frame = 1050
    # max_frame = 100
    min_frame = 0

    process_method = 'roi'

    process_parameters = {'process_method': process_method}
    process_parameters['roi'] = (20, 60, 30, 30)  # default (60, 60, 30, 30)

    # processed_data
    folder_out = '../processed_data/position_data'
    ################################################################################
    ## end settings ###
    ################################################################################
    #source folder
    folder_in = '../raw_data/20180628_Sample_6_Bead_1/'

    extraction_parameters = {'method': method}



    video_files = sorted(glob(os.path.join(folder_in, '*.avi')))

    f = video_files[0]
    print(f.split('.avi')[0].split('Bead_1_'))

    # video_files = sorted([f for f in video_files if int(f.split('.avi')[0].split('Bead_')[1].split('_')[0]) in list(range(118, 119))])
    video_files = sorted(
        [f for f in video_files if int(f.split('.avi')[0].split('Bead_1_')[1].split('_')[0]) in list(range(0, 200))])

    print(video_files[75:78])
    # files 77-80
    video_files = video_files[75:76]
    # video_files = video_files[71:72]
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
elif case == 'extract top view full 20180710_M110_Sample_6_Bead_1':
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
    # max_frame = 1050
    # min_frame = 1000
    min_frame = 0

    process_method = 'morph'

    # processed_data
    folder_out = '../processed_data/position_data'
    ################################################################################
    ## end settings ###
    ################################################################################
    extraction_parameters = {'method': method}
    process_parameters = {'process_method': process_method}
    process_parameters['k_size_close'] = 13  # 11 default
    process_parameters['k_size_noise'] = 3  # 3 default
    process_parameters['c'] = 11  # 11 default
    process_parameters['select_contour'] = 'all'  # 'longest' default (other option is 'all')
    process_parameters['select_contour'] = 'longest'  # 'longest' default (other option is 'all')
    process_parameters['blockSize'] = 91  # 35 default

    # process_parameters['threshold_type'] = 'gauss'  # default is mean maxval
    process_parameters['maxval'] = 255  # default is 255

    ################################################################################
    #### preprocessing morph parameters
    ################################################################################

    extraction_parameters['convex_hull'] = True

    #source folder
    folder_in = '../raw_data/20180710_M110_Sample_6_Bead_1/'

    video_files = sorted(glob(os.path.join(folder_in, '*.avi')))
    video_files = sorted(
        [f for f in video_files if int(f.split('.avi')[0].split('Bead_1_')[1].split('_')[0]) in list(range(0, 200))])

    # video_files = video_files[96:]
elif case == 'extract sideview spot 20180710_M110_Sample_6_Bead_1':
    ################################################################################
    ## method settings ###
    ################################################################################
    method = 'Bright px'

    export_video = False
    output_fps = 2


    # output_images = 1
    output_images = 10000
    # output_images = 1

    # max_frame = 25000
    # max_frame = 20
    # max_frame = 1050
    max_frame =None
    # max_frame = 100
    min_frame = 0

    #option 1 select roi
    # process_method = 'roi'
    # process_parameters = {'process_method': process_method}
    # process_parameters['roi'] = (0, 0, 64, 64)  # default (60, 60, 30, 30)

    # option 2 select roi
    process_method = 'bilateral'
    process_parameters = {'process_method': process_method}
    process_parameters['filter_dimension'] = 9  # default 5
    process_parameters['sigmaColor'] = 120  # default 50
    process_parameters['sigmaSpace'] = 120  # default 50
    process_parameters['normalize'] = True  # default True





    # processed_data
    folder_out = '../processed_data/position_data'
    ################################################################################
    ## end settings ###
    ################################################################################

    extraction_parameters = {'method': method}

    #source folder
    folder_in = '../raw_data/20180710_M110_Sample_6_Bead_1/'

    video_files = sorted(glob(os.path.join(folder_in, '*.avi')))
    video_files = sorted(
        [f for f in video_files if int(f.split('.avi')[0].split('Bead_1_')[1].split('_')[0]) in list(range(22, 200))])

    # video_files = video_files[96:]
elif case == 'extract top view full 20180718_M110_Sample_6_Bead_1':
    ################################################################################
    ## method settings ###
    ################################################################################
    method = 'fit_ellipse'

    export_video = False
    output_fps = 2
    output_images = 1000
    output_images = 1000
    # output_images = 1

    # max_frame = 25000
    max_frame = None
    # max_frame = 1050
    # min_frame = 1000
    min_frame = 0

    process_method = 'morph'

    # processed_data
    folder_out = '../processed_data/position_data'
    ################################################################################
    ## end settings ###
    ################################################################################
    extraction_parameters = {'method': method}
    process_parameters = {'process_method': process_method}
    process_parameters['k_size_close'] = 13  # 11 default
    process_parameters['k_size_noise'] = 3  # 3 default
    process_parameters['c'] = 11  # 11 default
    process_parameters['select_contour'] = 'all'  # 'longest' default (other option is 'all')
    process_parameters['select_contour'] = 'longest'  # 'longest' default (other option is 'all')
    process_parameters['blockSize'] = 91  # 35 default

    # process_parameters['threshold_type'] = 'gauss'  # default is mean maxval
    process_parameters['maxval'] = 255  # default is 255

    ################################################################################
    #### preprocessing morph parameters
    ################################################################################

    extraction_parameters['convex_hull'] = True

    #source folder
    folder_in = '../raw_data/20180718_M110_Sample_6_Bead_1/'
    runs = list(range(10, 100))
    run = 16
    runs = list(range(run, run+1))

    video_files = sorted(glob(os.path.join(folder_in, '*.avi')))
    video_files = sorted(
        [f for f in video_files if int(f.split('.avi')[0].split('Bead_1_')[1].split('_')[0]) in runs])

    print('video_files', video_files)

    # video_files = video_files[96:]
elif case == 'extract sideview spot 20180718_M110_Sample_6_Bead_1':
    ################################################################################
    ## method settings ###
    ################################################################################
    method = 'Bright px'

    export_video = False
    output_fps = 2


    # output_images = 1
    output_images = 10000
    # output_images = 1

    # max_frame = 25000
    # max_frame = 20
    # max_frame = 1050
    max_frame =None
    # max_frame = 100
    min_frame = 0

    #option 1 select roi
    # process_method = 'roi'
    # process_parameters = {'process_method': process_method}
    # process_parameters['roi'] = (0, 0, 64, 64)  # default (60, 60, 30, 30)

    # option 2 select roi
    process_method = 'bilateral'
    process_parameters = {'process_method': process_method}
    process_parameters['filter_dimension'] = 9  # default 5
    process_parameters['sigmaColor'] = 120  # default 50
    process_parameters['sigmaSpace'] = 120  # default 50
    process_parameters['normalize'] = True  # default True





    # processed_data
    folder_out = '../processed_data/position_data'
    ################################################################################
    ## end settings ###
    ################################################################################

    extraction_parameters = {'method': method}

    #source folder
    folder_in = '../raw_data/20180718_M110_Sample_6_Bead_1/'



    run= 34
    # runs = list(range(37, 36))
    runs = list(range(run, run+1))

    video_files = sorted(glob(os.path.join(folder_in, '*.avi')))
    video_files = sorted(
        [f for f in video_files if int(f.split('.avi')[0].split('Bead_1_')[1].split('_')[0]) in runs])

    # video_files = video_files[96:]
elif case == 'extract top view full 20180724_M110_Sample_6_Bead_1':
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
    # max_frame = 10
    # min_frame = 1000
    min_frame = 0

    process_method = 'morph'

    # processed_data
    folder_out = '../processed_data/position_data'
    ################################################################################
    ## end settings ###
    ################################################################################
    extraction_parameters = {'method': method}
    process_parameters = {'process_method': process_method}
    process_parameters['k_size_close'] = 13  # 11 default
    process_parameters['k_size_noise'] = 3  # 3 default
    process_parameters['c'] = 5  # 11 default
    process_parameters['select_contour'] = 'all'  # 'longest' default (other option is 'all')
    process_parameters['select_contour'] = 'longest'  # 'longest' default (other option is 'all')
    process_parameters['blockSize'] = 71  # 35 default

    # process_parameters['threshold_type'] = 'gauss'  # default is mean maxval
    # process_parameters['maxval'] = 255  # default is 255

    ################################################################################
    #### preprocessing morph parameters
    ################################################################################

    extraction_parameters['convex_hull'] = True

    #source folder
    folder_in = '../raw_data/20180724_Sample_6_Bead_1/'
    runs = list(range(7,8))
    run = 11
    runs = list(range(run, run+1))

    video_files = sorted(glob(os.path.join(folder_in, '*.avi')))
    video_files = sorted(
        [f for f in video_files if int(f.split('.avi')[0].split('Bead_1_')[1].split('_')[0]) in runs])

    print('video_files', video_files)

    # video_files = video_files[96:]
elif case == 'extract top bright spot 20180724_M110_Sample_6_Bead_1':
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
    max_frame = int(1e6)
    # max_frame = 10
    # min_frame = 1000
    min_frame = 0


    # processed_data
    folder_out = '../processed_data/position_data'
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



    #source folder
    folder_in = '../raw_data/20180724_Sample_6_Bead_1/'
    runs = list(range(8,9))
    run = 18
    runs = list(range(run, run+1))

    video_files = sorted(glob(os.path.join(folder_in, '*.avi')))
    video_files = sorted(
        [f for f in video_files if int(f.split('.avi')[0].split('Bead_1_')[1].split('_')[0]) in runs])

    print('video_files', video_files)

    # video_files = video_files[96:]
elif case == 'extract top bright spot ueye 20180724_M110_Sample_6_Bead_1':
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
    max_frame = int(1e6)
    max_frame = None
    # min_frame = 1000
    min_frame = 0


    # processed_data
    folder_out = '../processed_data/position_data'
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



    #source folder
    folder_in = '../raw_data/20180724_Sample_6_Bead_1/'
    runs = list(range(8,9))
    run = 29 #  29, 30
    runs = list(range(run, run+2))

    print(glob(os.path.join(folder_in, '*.avi')))
    video_files = sorted(glob(os.path.join(folder_in, '*Sample_6_Bead_1_*.avi')))


    video_files = sorted(
        [f for f in video_files if int(f.split('.avi')[0].split('_')[-1]) in runs])

    print('video_files', video_files)

    # video_files = video_files[96:]
elif case == 'extract top bright spot mc110 20180731_Sample_9_Bead_2':
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


    # processed_data
    folder_out = '../processed_data/20180731_Sample_9_Bead_2/position_data'
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



    #source folder
    folder_in = '../raw_data/20180731_Sample_9_Bead_2/'
    runs = list(range(17,19))
    # run = 6
    # runs = list(range(run, run+1))

    video_files = sorted(glob(os.path.join(folder_in, '*.avi')))
    video_files = sorted(
        [f for f in video_files if int(f.split('.avi')[0].split('Bead_2_')[1].split('_')[0]) in runs])

    print('video_files', video_files)

    # video_files = video_files[96:]
elif case == 'extract top full mc110 20180731_Sample_9_Bead_2':
    ################################################################################
    ## method settings ###
    ################################################################################
    ################################################################################
    method = 'fit_ellipse'

    export_video = False
    output_fps = 2
    output_images = 1000
    output_images = 10000
    # output_images = 1

    # max_frame = 25000
    max_frame = None
    # max_frame = 10
    # min_frame = 1000
    min_frame = 0

    process_method = 'morph'

    ################################################################################
    ## end settings ###
    ################################################################################
    extraction_parameters = {'method': method}
    process_parameters = {'process_method': process_method}
    process_parameters['k_size_close'] = 13  # 11 default
    process_parameters['k_size_noise'] = 5  # 3 default
    process_parameters['c'] = 5  # 11 default
    process_parameters['select_contour'] = 'all'  # 'longest' default (other option is 'all')
    process_parameters['select_contour'] = 'longest'  # 'longest' default (other option is 'all')
    process_parameters['blockSize'] = 51  # 35 default
    process_parameters['normalize'] = True  # default True


    # processed_data
    folder_out = '../processed_data/20180731_Sample_9_Bead_2/position_data'
    ################################################################################
    ## end settings ###
    ################################################################################


    # source folder
    folder_in = '../raw_data/20180731_Sample_9_Bead_2/'
    runs = list(range(8, 9))
    run = 4
    runs = list(range(run, run + 1))

    video_files = sorted(glob(os.path.join(folder_in, '*.avi')))
    video_files = sorted(
        [f for f in video_files if int(f.split('.avi')[0].split('Bead_2_')[1].split('_')[0]) in runs])

    print('video_files', video_files)

    # video_files = video_files[96:]
elif case == 'extract top full mc110 20180801_Sample_9_Bead_2':
    ################################################################################
    ## method settings ###
    ################################################################################
    ################################################################################
    method = 'fit_ellipse'

    export_video = False
    output_fps = 2
    output_images = 1000
    output_images = 10000
    # output_images = 1

    # max_frame = 25000
    max_frame = None
    # max_frame = 10
    # min_frame = 1000
    min_frame = 0

    process_method = 'morph'

    ################################################################################
    ## end settings ###
    ################################################################################
    extraction_parameters = {'method': method}
    process_parameters = {'process_method': process_method}
    process_parameters['k_size_close'] = 13  # 11 default
    process_parameters['k_size_noise'] = 7  # 3 default
    process_parameters['c'] = 5  # 11 default
    process_parameters['select_contour'] = 'all'  # 'longest' default (other option is 'all')
    process_parameters['select_contour'] = 'longest'  # 'longest' default (other option is 'all')
    process_parameters['blockSize'] = 61  # 35 default
    process_parameters['normalize'] = True  # default True


    # processed_data
    folder_out = '../processed_data/20180801_Sample_9_Bead_2/position_data'
    ################################################################################
    ## end settings ###
    ################################################################################


    # source folder
    folder_in = '../raw_data/20180801_Sample_9_Bead_2/'
    runs = list(range(8, 9))
    run = 1
    runs = list(range(run, run + 1))

    video_files = sorted(glob(os.path.join(folder_in, '*.avi')))
    video_files = sorted(
        [f for f in video_files if int(f.split('.avi')[0].split('Bead_2_')[1].split('_')[0]) in runs])

    print('video_files', video_files)

    # video_files = video_files[96:]
elif case == 'extract top bright spot mc110 20180801_Sample_9_Bead_2':
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


    # processed_data
    folder_out = '../processed_data/20180801_Sample_9_Bead_2/position_data'
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



    #source folder
    folder_in = '../raw_data/20180801_Sample_9_Bead_2/'
    runs = list(range(1,12))
    # run = 6
    # runs = list(range(run, run+1))

    video_files = sorted(glob(os.path.join(folder_in, '*.avi')))
    video_files = sorted(
        [f for f in video_files if int(f.split('.avi')[0].split('Bead_2_')[1].split('_')[0]) in runs])

    print('video_files', video_files)

    # video_files = video_files[96:]
elif case == 'extract top full mc110 20180802_Sample_9_Bead_2':
    ################################################################################
    ## method settings ###
    ################################################################################
    ################################################################################
    method = 'fit_ellipse'

    export_video = False
    output_fps = 2
    output_images = 1000
    output_images = 10000
    # output_images = 1

    # max_frame = 25000
    max_frame = None
    # max_frame = 10
    # min_frame = 1000
    min_frame = 0

    process_method = 'morph'

    ################################################################################
    ## end settings ###
    ################################################################################
    extraction_parameters = {'method': method}
    process_parameters = {'process_method': process_method}
    process_parameters['k_size_close'] = 13  # 11 default
    process_parameters['k_size_noise'] = 7  # 3 default
    process_parameters['c'] = 5  # 11 default
    process_parameters['select_contour'] = 'all'  # 'longest' default (other option is 'all')
    process_parameters['select_contour'] = 'longest'  # 'longest' default (other option is 'all')
    process_parameters['blockSize'] = 61  # 35 default
    process_parameters['normalize'] = True  # default True


    # processed_data
    folder_out = '../processed_data/20180802_Sample_9_Bead_2/position_data'
    ################################################################################
    ## end settings ###
    ################################################################################


    # source folder
    folder_in = '../raw_data/20180802_Sample_9_Bead_2/'
    runs = list(range(3,5))
    # run = 1
    # runs = list(range(run, run + 1))

    video_files = sorted(glob(os.path.join(folder_in, '*.avi')))
    video_files = sorted(
        [f for f in video_files if int(f.split('.avi')[0].split('Bead_2_')[1].split('_')[0]) in runs])

    print('video_files', video_files)

    # video_files = video_files[96:]
elif case == 'extract top full mc110 20180806_Sample_9_Bead_2':
    ################################################################################
    ## method settings ###
    ################################################################################
    ################################################################################
    method = 'fit_ellipse'

    export_video = False
    output_fps = 2
    output_images = 1000
    output_images = 10000
    # output_images = 1

    # max_frame = 25000
    max_frame = None
    # max_frame = 10
    # min_frame = 1000
    min_frame = 0

    process_method = 'morph'

    ################################################################################
    ## end settings ###
    ################################################################################
    extraction_parameters = {'method': method}
    process_parameters = {'process_method': process_method}
    process_parameters['k_size_close'] = 7  # 11 default 7 for 52, 53
    process_parameters['k_size_noise'] = 3  # 3 default
    process_parameters['c'] = 11  # 11 default,  11 for 52, 53
    # process_parameters['select_contour'] = 'all'  # 'longest' default (other option is 'all')
    process_parameters['select_contour'] = 'longest'  # 'longest' default (other option is 'all')
    process_parameters['blockSize'] = 91  # 35 default
    process_parameters['blockSize'] = 51  # 35 default,  51 for 52, 53
    # process_parameters['blockSize'] =   # 35 default
    process_parameters['normalize'] = True  # default True


    # processed_data
    folder_out = '../processed_data/20180806_Sample_9_Bead_2/position_data'
    ################################################################################
    ## end settings ###
    ################################################################################


    # source folder
    folder_in = '../raw_data/20180806_Sample_9_Bead_2/'
    runs = list(range(3,5))
    run = 58
    runs = list(range(run, run + 1))

    video_files = sorted(glob(os.path.join(folder_in, '*.avi')))
    video_files = sorted(
        [f for f in video_files if int(f.split('.avi')[0].split('Bead_2_')[1].split('_')[0]) in runs])

    print('video_files', video_files)

    # video_files = video_files[96:]
elif case == 'extract top bright spot mc110 20180806_Sample_9_Bead_2':
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


    # processed_data
    folder_out = '../processed_data/20180806_Sample_9_Bead_2/position_data'
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



    #source folder
    folder_in = '../raw_data/20180806_Sample_9_Bead_2/'
    runs = list(range(1,12))
    run = 46
    runs = list(range(run, run+2))

    video_files = sorted(glob(os.path.join(folder_in, '*.avi')))
    video_files = sorted(
        [f for f in video_files if int(f.split('.avi')[0].split('Bead_2_')[1].split('_')[0]) in runs])

    print('video_files', video_files)

    # video_files = video_files[96:]
elif case == 'extract top bright spot mc110 20180810_Sample_10_Bead_B2':
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


    # processed_data
    folder_out = '../processed_data/201800820_Sample_10_Bead_B2/position_data'
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

    process_parameters['roi'] = (22, 22, 20, 20)  # default (60, 60, 30, 30)



    #source folder
    folder_in = '../raw_data/201800820_Sample_10_Bead_B2/'
    runs = list(range(1,12))
    run = 1
    runs = list(range(run, run+1))

    video_files = sorted(glob(os.path.join(folder_in, '*.avi')))
    video_files = sorted(
        [f for f in video_files if int(f.split('.avi')[0].split('Bead_B2_')[1].split('_')[0]) in runs])

    print('video_files', video_files)

    # video_files = video_files[96:]
elif case == 'extract top full mc110 20180821_Sample_10_Bead_F5':
    ################################################################################
    ## method settings ###
    ################################################################################
    ################################################################################
    method = 'fit_ellipse'

    export_video = False
    output_fps = 2
    output_images = 1000
    output_images = 10000
    # output_images = 1

    # max_frame = 25000
    max_frame = None
    # max_frame = 10
    # min_frame = 1000
    min_frame = 0

    process_method = 'morph'

    ################################################################################
    ## end settings ###
    ################################################################################
    extraction_parameters = {'method': method}
    process_parameters = {'process_method': process_method}
    process_parameters['k_size_close'] = 11  # 11 default 7 for 52, 53
    process_parameters['k_size_noise'] = 5  # 3 default
    process_parameters['c'] = 11  # 11 default,  11 for 52, 53
    # process_parameters['select_contour'] = 'all'  # 'longest' default (other option is 'all')
    process_parameters['select_contour'] = 'longest'  # 'longest' default (other option is 'all')
    process_parameters['blockSize'] = 91  # 35 default
    process_parameters['blockSize'] = 51  # 35 default,  51 for 52, 53
    # process_parameters['blockSize'] =   # 35 default
    process_parameters['normalize'] = True  # default True


    # processed_data
    folder_out = '../processed_data/201800821_Sample_10_Bead_F5/position_data'
    ################################################################################
    ## end settings ###
    ################################################################################


    # source folder
    folder_in = '../raw_data/201800821_Sample_10_Bead_F5/'
    runs = list(range(3,5))
    run = 1
    runs = list(range(run, run + 1))

    video_files = sorted(glob(os.path.join(folder_in, '*.avi')))
    video_files = sorted(
        [f for f in video_files if int(f.split('.avi')[0].split('Bead_F5_')[1].split('_')[0]) in runs])

    print('video_files', video_files)

    # video_files = video_files[96:]
elif case == 'extract top bright spot mc110 20180821_Sample_10_Bead_F5':
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


    # processed_data
    folder_out = '../processed_data/20180821_Sample_10_Bead_F5/position_data'
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

    process_parameters['roi'] = (22, 22, 20, 20)  # default (60, 60, 30, 30)



    # source folder
    folder_in = '../raw_data/201800821_Sample_10_Bead_F5/'
    runs = list(range(3,5))
    run = 1
    runs = list(range(run, run + 1))

    video_files = sorted(glob(os.path.join(folder_in, '*.avi')))
    video_files = sorted(
        [f for f in video_files if int(f.split('.avi')[0].split('Bead_F5_')[1].split('_')[0]) in runs])

    print('video_files', video_files)

    # video_files = video_files[96:]
elif case == 'extract top full mc110 20180824_Sample_6_Bead_1':
    ################################################################################
    ## method settings ###
    ################################################################################
    ################################################################################
    method = 'fit_ellipse'

    export_video = False
    output_fps = 2
    output_images = 1000
    output_images = 10000
    # output_images = 1

    # max_frame = 25000
    max_frame = None
    # max_frame = 10
    # min_frame = 1000
    min_frame = 0

    process_method = 'morph'

    ################################################################################
    ## end settings ###
    ################################################################################
    extraction_parameters = {'method': method}
    process_parameters = {'process_method': process_method}
    # process_parameters['k_size_close'] = 11  # 11 default 7 for 52, 53
    process_parameters['k_size_close'] = 11  # 11 default 7 for 52, 53
    process_parameters['k_size_noise'] = 5  # 3 default
    process_parameters['c'] = 11  # 11 default,  11 for 52, 53
    # process_parameters['select_contour'] = 'all'  # 'longest' default (other option is 'all')
    process_parameters['select_contour'] = 'longest'  # 'longest' default (other option is 'all')
    process_parameters['blockSize'] = 121  # 35 default, 171 (71 for 48)
    # process_parameters['blockSize'] = 51  # 35 default,  51 for 52, 53
    # process_parameters['blockSize'] =   # 35 default
    process_parameters['normalize'] = True  # default True


    # processed_data
    folder_out = '../processed_data/20180824_Sample_6_Bead_1/position_data'
    ################################################################################
    ## end settings ###
    ################################################################################


    # source folder
    folder_in = '../raw_data/20180824_Sample_6_Bead_1/'
    runs = list(range(3,5))
    run = 61
    runs = list(range(run, run +1))

    video_files = sorted(glob(os.path.join(folder_in, '*.avi')))
    video_files = sorted(
        [f for f in video_files if int(f.split('.avi')[0].split('Bead_1_')[1].split('_')[0]) in runs])

    print('video_files', video_files)

    # video_files = video_files[96:]
elif case == 'extract top bright spot mc110 20180824_Sample_6_Bead_1':
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


    folder_out = '../processed_data/20180824_Sample_6_Bead_1/position_data'
    ################################################################################
    ## end settings ###
    ################################################################################


    # source folder
    folder_in = '../raw_data/20180824_Sample_6_Bead_1/'


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
    process_parameters['roi'] = (1, 11, 30, 40)  # default (60, 60, 30, 30)
    # process_parameters['roi'] = (76, 76, 40, 40)  # default (60, 60, 30, 30)

    # max_frame = 100000

    # source folder
    folder_in = '../raw_data/20180824_Sample_6_Bead_1/'
    runs = list(range(3,5))
    run = 52
    runs = list(range(run, run + 1))

    video_files = sorted(glob(os.path.join(folder_in, '*.avi')))
    video_files = sorted(
        [f for f in video_files if int(f.split('.avi')[0].split('Bead_1_')[1].split('_')[0]) in runs])

    print('video_files', video_files)

    # video_files = video_files[96:]
elif case == 'extract top full mc110 20180910_Sample_13_Bead_4':
    ################################################################################
    ## method settings ###
    ################################################################################
    ################################################################################
    method = 'fit_ellipse'

    export_video = False
    output_fps = 2
    output_images = 1000
    output_images = 10000
    # output_images = 1

    # max_frame = 25000
    max_frame = None
    # max_frame = 10
    # min_frame = 1000
    min_frame = 0

    process_method = 'morph'

    ################################################################################
    ## end settings ###
    ################################################################################
    extraction_parameters = {'method': method}
    process_parameters = {'process_method': process_method}
    # process_parameters['k_size_close'] = 11  # 11 default 7 for 52, 53
    process_parameters['k_size_close'] = 11  # 11 default 7 for 52, 53
    process_parameters['k_size_noise'] = 7  # 3 default
    process_parameters['c'] = 11  # 11 default,  11 for 52, 53
    # process_parameters['select_contour'] = 'all'  # 'longest' default (other option is 'all')
    process_parameters['select_contour'] = 'longest'  # 'longest' default (other option is 'all')
    process_parameters['blockSize'] = 151  # 35 default, (171 for 1)
    # process_parameters['blockSize'] = 51  # 35 default,  51 for 52, 53
    # process_parameters['blockSize'] =   # 35 default
    process_parameters['normalize'] = True  # default True


    # processed_data
    folder_out = '../processed_data/20180910_Sample_13_Bead_4/position_data'
    ################################################################################
    ## end settings ###
    ################################################################################


    # source folder
    folder_in = '../raw_data/20180910_Sample_13_Bead_4/'
    runs = list(range(3,5))
    run = 3
    runs = list(range(run, run +1))

    video_files = sorted(glob(os.path.join(folder_in, '*.avi')))
    video_files = sorted(
        [f for f in video_files if int(f.split('.avi')[0].split('Bead_4_')[1].split('_')[0]) in runs])

    print('video_files', video_files)

    # video_files = video_files[96:]
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
    ################################################################################
    ## end settings ###
    ################################################################################


    # source folder
    folder_in = '../raw_data/20181204_Sample_14_Bead_3/'


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


    video_files = sorted(glob(os.path.join(folder_in, '*.avi')))
    # print(video_files)
    video_files = sorted(
        [f for f in video_files if int(f.split('.avi')[0].split('Bead_3_')[1].split('_')[0]) in runs])

    print('video_files', video_files)

    # video_files = video_files[96:]


elif case == 'test':
    ################################################################################
    ## method settings ###
    ################################################################################
    method = 'moments_roi'
    method = 'fit_ellipse'
    method = 'Bright px'



    export_video = False
    output_fps = 2

    # output_images = 1
    output_images = 10000
    output_images = 1

    # max_frame = 25000
    # max_frame = 20
    # max_frame = 1050
    max_frame = 10
    # max_frame = 100
    min_frame = 0

    # option 1 select roi
    # process_method = 'roi'
    # process_parameters = {'process_method': process_method}
    # process_parameters['roi'] = (0, 0, 64, 64)  # default (60, 60, 30, 30)

    # option 2 select roi
    process_method = 'bilateral'
    process_parameters = {'process_method': process_method}
    process_parameters['filter_dimension'] = 15  # default 5
    process_parameters['sigmaColor'] = 200  # default 50
    # process_parameters['sigmaSpace'] = 120  # default 50
    process_parameters['normalize'] = True  # default True

    # processed_data
    folder_out = '../processed_data/position_data'
    ################################################################################
    ## end settings ###
    ################################################################################

    extraction_parameters = {'method': method}

    extraction_parameters['convex_hull'] = True

    # source folder
    folder_in = '../raw_data/20180710_M110_Sample_6_Bead_1/'

    video_files = sorted(glob(os.path.join(folder_in, '*.avi')))
    video_files = sorted(
        [f for f in video_files if int(f.split('.avi')[0].split('Bead_1_')[1].split('_')[0]) in list(range(3, 200))])

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
