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
experiment = 'long term run 5 xyzm'
experiment = 'long term run 6 xyzm'
experiment = 'long term run 7 xy'
experiment = 'long term run 7b xyz'
experiment = 'long term run 7c xyr alias'
# experiment = 'run 7d xyrz'
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

elif experiment == 'long term run 5 xyzm':

    modes = 'xyzm'
    interval_width = [1, 1, 5, 40]
    interval_width_zoom = [0.5, 0.5, 2, 10]
    fo = [61, 58.5, 70, 550]

    position_file_names = get_position_file_names(source_folder_positions, method=method, runs=list(range(118, 180)))

elif experiment == 'long term run 6 xyzm':

    source_folder_positions = '../processed_data/20180628_Sample_6_Bead_1/position_data/'
    modes = 'xyzm'
    interval_width = [1, 1, 5, 40]
    interval_width_zoom = [0.5, 0.5, 2, 10]
    fo = [40, 38, 70, 540]

    position_file_names = get_position_file_names(source_folder_positions, method=method, runs=list(range(0, 4)))

elif experiment == 'long term run 7 xy':

    source_folder_positions = '../processed_data/20180628_Sample_6_Bead_1/position_data/'
    modes = 'xy'
    interval_width = [10, 10]
    interval_width_zoom = [0.5, 0.5]
    fo = [572, 608]

    position_file_names = get_position_file_names(source_folder_positions, method=method, runs=list(range(5, 100)))
elif experiment == 'long term run 7b xyz':

    source_folder_positions = '../processed_data/20180628_Sample_6_Bead_1/position_data/'
    modes = 'xy'
    interval_width = [100, 50]
    interval_width_zoom = [1, 1]
    fo = [500, 570]

    position_file_names = get_position_file_names(source_folder_positions, method=method, runs=list(range(22, 38)))

elif experiment == 'long term run 7c xyr alias':

    source_folder_positions = '../processed_data/20180628_Sample_6_Bead_1/position_data/'
    image_folder = '../images/20180628_Sample_6_bead_1/modes/'
    modes = 'xyr'
    interval_width = [20, 20, 58]
    interval_width_zoom = [1, 1, 1]
    fo = [834, 810, 30]

    position_file_names = get_position_file_names(source_folder_positions, method=method, runs=list(range(91, 102)))

    modes = 'r'
    interval_width = [58]
    interval_width_zoom = [1]
    fo = [30]

    position_file_names = get_position_file_names(source_folder_positions, method=method, runs=list(range(91, 102)))

elif experiment == 'run 7d xyrz':

    source_folder_positions = '../processed_data/20180628_Sample_6_Bead_1/position_data/'
    image_folder = '../images/20180628_Sample_6_bead_1/modes/'
    modes = 'xyrz'
    interval_width = [150, 150, 30, 40]
    interval_width_zoom = [1, 1, 1, 1]
    fo = [800, 800, 30, 60]

    position_file_names = get_position_file_names(source_folder_positions, method=method, runs=list(range(86, 91)))


################################################################################
#### run the script
################################################################################
if print_only_names:
    analysis_method = 0
axes = None

assert len(modes) == len(interval_width)
assert len(modes) == len(fo)
assert len(modes) == len(interval_width_zoom)

for i, filename in enumerate(position_file_names):

    print(filename)


    # # ================================================
    # # ==== get the frequency from fft for single mode
    # # ================================================
    data, info = load_time_trace(filename, source_folder_positions=source_folder_positions, verbose=False)
    for i, mode in enumerate(modes):
        # retrieve frequencies and figure
        fig, axes, freqs = get_mode_frequency_fft(data, mode, info, return_figure=True,
                                              interval_width=interval_width[i],
                                              interval_width_zoom=interval_width_zoom[i],
                                              fo=fo[i])

        # save figure
        image_filename = os.path.join(image_folder, filename.replace('.dat', '-' + mode + '-fft.png'))
        fig.savefig(image_filename)
        plt.close(fig)

        # update info (json) file
        for key, value in freqs.items():
            print(key, value)
            update_info(filename, key, value, folder_positions=source_folder_positions, dataset='ellipse', verbose=True)


