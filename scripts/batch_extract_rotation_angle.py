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



experiment = 'long term run 2a'
experiment = 'long term run 2 fit slope'
experiment = 'long term run r-unwrap'
# experiment = 'long term run r-mode'
experiment = 'long term run 5 fit slope'
experiment = 'long term run 5 r-mode'

print_only_names = True
print_only_names = False




################################################################################
#### experiment specific settings
################################################################################
if experiment == 'long term run 2a':

    interval_width = 200
    interval_width_zoom = 0.9
    fo = 101

    analysis_method = 1
    mode = 'r'
    # get all the files and sort them by the run number
    position_file_names = get_position_file_names(source_folder_positions, method=method)
    # select the subset of interest
    position_file_names = position_file_names[29:]  # all

    # position_file_names = position_file_names[29:32]
    # position_file_names = position_file_names[32:35]
elif experiment == 'long term run r-mode':

    interval_width = None
    interval_width_zoom = 0.9
    fo = None

    analysis_method = 2

    mode = 'r'
    # get all the files and sort them by the run number
    position_file_names = get_position_file_names(source_folder_positions, method=method)
    # select the subset of interest
    position_file_names = position_file_names[29:]  # all
elif experiment == 'long term run r-unwrap':

    interval_width = None
    interval_width_zoom = 0.9
    fo = None

    analysis_method = 2
    mode = 'r-unwrap'
    # get all the files and sort them by the run number
    position_file_names = get_position_file_names(source_folder_positions, method=method)
    # select the subset of interest
    position_file_names = position_file_names[29:]  # all
elif experiment == 'long term run 2 fit slope':

    analysis_method = 3
    # get all the files and sort them by the run number
    position_file_names = get_position_file_names(source_folder_positions, method=method)
    # select the subset of interest
    # position_file_names = position_file_names[29:]  # all
    # position_file_names = position_file_names[48:]
    position_file_names = position_file_names[29:]
elif experiment == 'long term run 5 fit slope':

    analysis_method = 3
    # get all the files and sort them by the run number
    position_file_names = get_position_file_names(source_folder_positions, method=method, runs=list(range(140, 180)))
elif experiment == 'long term run 5 r-mode':

    interval_width = None
    interval_width_zoom = 0.9
    fo = None
    analysis_method = 2
    mode = 'r'
    # get all the files and sort them by the run number
    position_file_names = get_position_file_names(source_folder_positions, method=method, runs=list(range(118, 180)))
elif experiment == 'run':
    run = 19
    position_file_names = get_position_file_names(source_folder_positions, method=method)
    position_file_names = position_file_names[run-1:run]


################################################################################
#### run the script
################################################################################
if print_only_names:
    analysis_method = 0
axes = None

for i, filename in enumerate(position_file_names):

    print(filename)

    # ================================================
    # ==== method 1 = get the frequency from the phase
    # ================================================
    if analysis_method == 1:
        data, info = load_time_trace(filename, source_folder_positions=source_folder_positions, verbose=False)
        fig, axes, freqs = get_rotation_frequency(data, info, return_figure=True, exclude_percent=0.2, nmax=500)

        # save figure
        image_filename = os.path.join(image_folder, filename.replace('.dat', '-r-phase.png'))
        fig.savefig(image_filename)
        plt.close(fig)

        update_info(filename, 'rotation_freq', freqs, folder_positions=source_folder_positions, dataset='ellipse', verbose=True)
    # ================================================
    # ==== method 2 = get the frequency from fft
    # ================================================
    elif analysis_method == 2:

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
    # ==== method 3 = get the frequency from a fit to the slope of the unwrapped phase
    # ================================================
    if analysis_method == 3:
        data, info = load_time_trace(filename, source_folder_positions=source_folder_positions, verbose=False)
        fig, axes, freqs = get_rotation_frequency_fit_slope(data, info, return_figure=True, nmax=500)

        fig.suptitle(filename)
        # save figure
        image_filename = os.path.join(image_folder, filename.replace('.dat', '-r-slope.png'))
        fig.savefig(image_filename)
        plt.close(fig)

        update_info(filename, 'rotation_freq_slope_fit', freqs, folder_positions=source_folder_positions, dataset='ellipse', verbose=True)