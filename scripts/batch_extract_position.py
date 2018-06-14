import os
from glob import glob
from track_sphere.extract_data_opencv import *


method = 'fit_ellipse'

export_video = False
output_fps = 2
# output_images = 1000
output_images = 10000

max_frame = 2000
max_frame = None


process_method = 'morph'
################################################################################
## end settings ###
################################################################################



extraction_parameters = {'method': method}
process_parameters = {'process_method': process_method}


################################################################################
#### for real data: 20180607_Sample6_bead_1
################################################################################

folder_in = '../raw_data/20180607_Sample_6_bead_1/'
# filename_in = '20180529_Sample6_bead_1_direct_thermal_01c.avi'
# filename_in = '20180607_Sample_6_Bead_1.avi'
# filename_in = '20180607_Sample_6_Bead_2.avi'
# # filename_in = '20180607_Sample_6_Bead_3.avi'
# filename_in = '20180608_Sample_6_Bead_4.avi'
# filename_in = '20180608_Sample_6_Bead_5.avi'
# filename_in = '20180611_Sample_6_Bead_7.avi'
# filename_in = '20180529_Sample6_bead_1_direct_thermal_01c-fixed.avi' # doesn't work!
# folder_in = '/Volumes/Elements/lev_data/20180523_Sample_6_bead_1/'

extraction_parameters['threshold'] = 'gaussian'
extraction_parameters['blockSize'] = 51
extraction_parameters['c'] = 11
extraction_parameters['maxval'] = 255
extraction_parameters['convex_hull'] = True

# processed_data
folder_out = '../processed_data/position_data'

video_files = sorted(glob(os.path.join(folder_in, '*.avi')))



################################################################################
#### run the script
################################################################################
# f=video_files[3]
# print(f)
# data = json.loads(f)
# print(data)
for f in video_files[3:]:

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


    extract_position_data(file_in, file_out=file_out, min_frame=0, max_frame=None, verbose=False,
                          parameters=parameters)
