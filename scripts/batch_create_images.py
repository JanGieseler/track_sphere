
# probably obsolete

import os
from track_sphere.plot_data import plot_rotations_vs_time, plot_ellipse_spectra_zoom
from track_sphere.read_write import load_time_trace
from track_sphere.utils import get_position_file_names
from glob import glob

import matplotlib.pyplot as plt

source_folder_positions = '../processed_data/20180607_Sample_6_bead_1/position_data/'
method = 'fit_ellipse'


image_type = 'rotation vs time'


################################################################################
#### image_type specific settings
################################################################################
if image_type == 'ellipse_spectra_zoom':
    position_file_names = get_position_file_names(source_folder_positions, method=method)
    image_folder = '../images/'
    annotation_dict = {'2r': [13.16877399924, 0.6],
                       'm': [544.131481993, 0.6],
                       'r': [6.58438699962, 0.85],
                       'x': [171.52045677, 0.85],
                       'y': [177.690573472, 0.6],
                       'z': [402.859033756, 0.6]}

    freq_window = 0.5
    method = 'fit_ellipse'
    plot_type = 'lin'


elif image_type == 'rotation vs time':
    position_file_names = get_position_file_names(source_folder_positions, method=method)
    # select the subset of interest
    position_file_names = position_file_names[32:]
    image_folder = '../images/20180607_Sample_6_bead_1/rotations_vs_time/'


################################################################################
#### run the script
################################################################################
if image_type == 'ellipse_spectra_zoom':
    # NOT FINISHED / OLD STUFF
    for filename in position_file_names:
        print(filename)
        data, info = load_time_trace(filename, source_folder_positions=source_folder_positions, verbose=False)
        fig = plot_ellipse_spectra_zoom(data, info, annotation_dict=annotation_dict, freq_window=freq_window, n_avrg=None, plot_type = plot_type, normalize=True)
        fig.suptitle('mode spectra ' + info['filename'], fontsize=20)
        print(os.path.join(image_folder, info['filename'].replace('.avi', '-' + method + '-modes-' + plot_type + '.jpg')))
        # fig.savefig(os.path.join(image_folder, info['filename'].replace('.avi', '-' + method + '-modes-' + plot_type + '.jpg')))
        fig.savefig('test.jpg')

elif image_type == 'rotation vs time':
    for filename in position_file_names:
        data, info = load_time_trace(filename, source_folder_positions=source_folder_positions, verbose=False)
        time_step = 1./info['info']['FrameRate']
        fig, axes = plot_rotations_vs_time(data, time_step, n_avrg=1, n_avrg_unwrapped=1)
        output_image_filename = os.path.join(image_folder, filename.replace('.dat', '-rotations.png'))
        fig.savefig(output_image_filename)