# This batch script extracts the rotation angle ellipse position data files and addes the result
# to the .json file that comes with the position data fil

import os
from glob import glob
import matplotlib.pyplot as plt
from track_sphere.read_write import load_time_trace, update_info
from track_sphere.utils import get_rotation_frequency, get_position_file_names, get_mode_frequency_fft, get_rotation_frequency_fit_slope

source_folder_positions = '../processed_data/20180607_Sample_6_bead_1/position_data/'
image_folder = '../images/20180607_Sample_6_bead_1/modes/'
method = 'fit_ellipse'




experiment = 'long term run 2 x'
experiment = 'long term run 2 y'
experiment = 'long term run 2 xyz'

print_only_names = True
print_only_names = False




################################################################################
#### experiment specific settings
################################################################################
if experiment == 'long term run 2 x':

    interval_width = 5
    interval_width_zoom = 0.9
    fo = 60
    mode = 'x'
    analysis_method = 1
    # get all the files and sort them by the run number
    position_file_names = get_position_file_names(source_folder_positions, method=method)
    # select the subset of interest
    position_file_names = position_file_names[29:]  # all
elif experiment == 'long term run 2 y':

    interval_width = 5
    interval_width_zoom = 0.9
    fo = 70
    mode = 'y'
    analysis_method = 1
    # get all the files and sort them by the run number
    position_file_names = get_position_file_names(source_folder_positions, method=method)
    # select the subset of interest
    position_file_names = position_file_names[29:]  # all

elif experiment == 'long term run 2 xyz':

    interval_width = [1, 1, 5]
    interval_width_zoom = 0.5
    fo = [60, 61.1, 70]

    analysis_method = 2
    # get all the files and sort them by the run number
    position_file_names = get_position_file_names(source_folder_positions, method=method)
    # select the subset of interest
    position_file_names = position_file_names[29:]  # all

################################################################################
#### run the script
################################################################################
if print_only_names:
    analysis_method = 0
axes = None

for i, filename in enumerate(position_file_names):

    print(filename)


    # ================================================
    # ==== method 1 = get the frequency from fft for single mode
    # ================================================
    if analysis_method == 1:
        data, info = load_time_trace(filename, source_folder_positions=source_folder_positions, verbose=False)

        # retrieve frequencies and figure
        fig, axes, freqs = get_mode_frequency_fft(data, mode, info, return_figure=True,
                                              interval_width=interval_width,
                                              interval_width_zoom=interval_width_zoom,
                                              fo=fo)

        # save figure
        image_filename = os.path.join(image_folder, filename.replace('.dat', '-' + mode + '-fft.png'))
        fig.savefig(image_filename)
        plt.close(fig)

        # update info (json) file
        for key, value in freqs.items():
            print(key, value)
            update_info(filename, key, value, folder_positions=source_folder_positions, dataset='ellipse', verbose=True)
    # ================================================
    # ==== method 2 = get the frequency from fft for all modes
    # ================================================
    elif analysis_method == 2:
        data, info = load_time_trace(filename, source_folder_positions=source_folder_positions, verbose=False)
        for i, mode in enumerate(['x', 'y', 'z']):
            # retrieve frequencies and figure
            fig, axes, freqs = get_mode_frequency_fft(data, mode, info, return_figure=True,
                                                  interval_width=interval_width[i],
                                                  interval_width_zoom=interval_width_zoom,
                                                  fo=fo[i])

            # save figure
            image_filename = os.path.join(image_folder, filename.replace('.dat', '-' + mode + '-fft.png'))
            fig.savefig(image_filename)
            plt.close(fig)

            # update info (json) file
            for key, value in freqs.items():
                print(key, value)
                update_info(filename, key, value, folder_positions=source_folder_positions, dataset='ellipse', verbose=True)


