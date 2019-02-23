# This batch script extracts the position information from video files and writes them into a .csv file
# meta data is also saved into a .json file

# ====== Feb 23rd 2019 ===========================================================================
# ===== copied batch_extract_position.py to run on ubuntu computer to recover old analysis =======
# deleted the cases that are not needed
# ================================================================================================


import os
from glob import glob
from track_sphere.extract_data_opencv import *

# raw_data_path = '/run/user/1000/gvfs/smb-share:server=fs2k02.rc.fas.harvard.edu,share=lukin_lab/Lab/Lev/videos/'
raw_data_path = '/run/user/1000/gvfs/smb-share:server=fs2k02.rc.fas.harvard.edu,share=lukin_lab/Lab/Lev/videos/'
folder_out_base_path = '/run/user/1000/gvfs/smb-share:server=fs2k02.rc.fas.harvard.edu,share=lukin_lab/Lab/Lev/video_processed_data/'
# select one of the cases, case specific parameters are defined below
case = 'extract all 20180628_Sample_6_bead_1 - Lev 7 Q xy'
case = 'extract all 20180628_Sample_6_bead_1 sideview - Lev 7 Q z'
case = 'extract sideview spot 20180718_M110_Sample_6_Bead_1 - Lev 8 Q xyz'
case = 'extract top bright spot 20180724_M110_Sample_6_Bead_1 - Lev 9 Q '
# case = 'extract top bright spot ueye 20180724_M110_Sample_6_Bead_1 - Lev 9 Q '
process_parameters = {}

################################################################################
#### define parameters for each case
################################################################################
if case == 'extract all 20180628_Sample_6_bead_1 - Lev 7 Q xy':
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
    # max_frame = 100
    # min_frame = 1000
    min_frame = 0

    process_method = 'morph'

    # processed_data
    # folder_out = '../processed_data/position_data'
    folder_out = folder_out_base_path + '20180628_Sample_6_bead_1'

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
    process_parameters['blockSize'] = 35  # 101, (35 default)

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
    # folder_in = '../raw_data/20180628_Sample_6_Bead_1/'
    # raw_data_path = '/run/user/1000/gvfs/smb-share:server=fs2k02.rc.fas.harvard.edu,share=lukin_lab/Lab/Lev/videos/20180628_Sample_6_Bead_1/'

    folder_in = raw_data_path+'20180628_Sample_6_Bead_1/'

    video_files = sorted(glob(folder_in+'*.avi'))

    f = video_files[0]
    print(f.split('.avi')[0].split('Bead_1_'))

    runs = [73, 75, 92, 93, 95] # x
    runs += [76, 91, 96, 100] # y




    video_files = sorted(glob(os.path.join(folder_in, '*.avi')))
    # print(video_files)
    video_files = sorted(
        [f for f in video_files if int(f.split('.avi')[0].split('Bead_1_')[1].split('_')[0]) in runs])
elif case == 'extract all 20180628_Sample_6_bead_1 sideview - Lev 7 Q z':
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
    # max_frame = 100
    # max_frame = 100
    min_frame = 0

    process_method = 'roi'

    process_parameters = {'process_method': process_method}
    process_parameters['roi'] = (35, 35, 50, 50)  # default (60, 60, 30, 30)

    # processed_data
    # folder_out = '../processed_data/position_data'
    folder_out = folder_out_base_path + '20180628_Sample_6_bead_1'

    ################################################################################
    ## end settings ###
    ################################################################################
    #source folder
    # folder_in = '../raw_data/20180628_Sample_6_Bead_1/'
    folder_in = raw_data_path + '20180628_Sample_6_Bead_1/'

    extraction_parameters = {'method': method}



    video_files = sorted(glob(os.path.join(folder_in, '*.avi')))

    f = video_files[0]
    print(f.split('.avi')[0].split('Bead_1_'))

    runs = [13, 18, 19, 20, 21] # z


    video_files = sorted(glob(os.path.join(folder_in, '*.avi')))
    # print(video_files)
    video_files = sorted(
        [f for f in video_files if int(f.split('.avi')[0].split('Bead_1_')[1].split('_')[0]) in runs])
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
elif case == 'extract sideview spot 20180718_M110_Sample_6_Bead_1 - Lev 8 Q xyz':
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
    # folder_out = '../processed_data/position_data'
    folder_out = folder_out_base_path + '20180718_M110_Sample_6_Bead_1'
    ################################################################################
    ## end settings ###
    ################################################################################

    extraction_parameters = {'method': method}

    #source folder
    # folder_in = '../raw_data/20180718_M110_Sample_6_Bead_1/'
    folder_in = raw_data_path+'20180718_M110_Sample_6_Bead_1/'

    video_files = sorted(glob(folder_in+'*.avi'))

    f = video_files[0]
    print(f.split('.avi')[0].split('Bead_1_'))

    runs = [23, 29] # x
    runs += [24, 27] # y
    runs += [21, 35]  # z




    video_files = sorted(glob(os.path.join(folder_in, '*.avi')))
    # print(video_files)
    video_files = sorted(
        [f for f in video_files if int(f.split('.avi')[0].split('Bead_1_')[1].split('_')[0]) in runs])
    # video_files = video_files[96:]
elif case == 'extract top bright spot 20180724_M110_Sample_6_Bead_1 - Lev 9 Q ':
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
    # folder_out = '../processed_data/position_data'

    folder_out = folder_out_base_path + '20180724_Sample_6_Bead_1'
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
    # folder_in = '../raw_data/20180724_Sample_6_Bead_1/'
    folder_in = raw_data_path + '20180724_M110_Sample_6_Bead_1/'

    video_files = sorted(glob(folder_in+'*.avi'))

    f = video_files[0]
    print(f.split('.avi')[0].split('_')[-1])

    runs = [16, 20]  # x
    runs += [17]  # y
    runs += [28, 30]  # z




    video_files = sorted(glob(os.path.join(folder_in, '*.avi')))
    # for v in video_files:
    #     print(v)
    # # print(video_files)
    # video_files = sorted(
    #     [f for f in video_files if int(f.split('.avi')[0].split('Bead_1_')[1].split('_')[0]) in runs])

    video_files = sorted(
        [f for f in video_files if int(f.split('.avi')[0].split('_')[-1]) in runs])

    # video_files = video_files[96:]
elif case == 'extract top bright spot ueye 20180724_M110_Sample_6_Bead_1 - Lev 9 Q ':
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
    # max_frame = 100
    # min_frame = 1000
    min_frame = 0


    # processed_data
    folder_out = '../processed_data/position_data'
    folder_out = folder_out_base_path + '20180724_Sample_6_Bead_1'
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
    folder_in = raw_data_path + '20180724_M110_Sample_6_Bead_1/'

    video_files = sorted(glob(folder_in+'*.avi'))
    runs = list(range(8,9))
    run = 18 #  29, 30
    runs = list(range(run, run+1))



    video_files = sorted(
        [f for f in video_files if int(f.split('.avi')[0].split('_')[-1]) in runs])

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
