# This batch script extracts the rotation angle ellipse position data files and addes the result
# to the .json file that comes with the position data fil

import os
from glob import glob
import matplotlib.pyplot as plt
from track_sphere.read_write import load_time_trace, update_info
from track_sphere.utils import get_rotation_frequency, get_position_file_names, get_mode_frequency

source_folder_positions = '../processed_data/20180607_Sample_6_bead_1/position_data/'
image_folder = '../images/20180607_Sample_6_bead_1/modes/'
method = 'fit_ellipse'


interval_width = 200
interval_width_zoom = 0.9
fo = 101




################################################################################
#### run the script
################################################################################
# get all the files and sort them by the run number

position_file_names = get_position_file_names(source_folder_positions, method=method)
# position_file_names = sorted([os.path.basename(f) for f in glob(source_folder_positions + '*-'+method+'.dat')])
# position_file_names = sorted(position_file_names, key=lambda f: int(f.split('-')[0].split('Bead_')[1].split('_')[0]))

position_file_names = position_file_names[62:]
position_file_names

axes = None

# run = 19
# for i, filename in enumerate(position_file_names[run-1:run]):

for i, filename in enumerate(position_file_names):

    print(filename)

    # ================================================
    # ==== method 1 = get the frequency from the phase
    # ================================================
    # data, info = load_time_trace(filename, source_folder_positions=source_folder_positions, verbose=False)
    # ret = get_rotation_frequency(data, info, n_avrg=20)
    #
    # update_info(filename, 'rotation_freq', {k:v for k, v in zip(['mean', 'std', 'time', 'n_avrg'], ret)}
    # , folder_positions=source_folder_positions, dataset='ellipse', verbose=True)

    # ================================================
    # ==== method 2 = get the frequency from fft
    # ================================================

    mode = 'r'
    data, info = load_time_trace(filename, source_folder_positions=source_folder_positions, verbose=False)
    # retrieve frequencies and figure
    fig, axes, freqs = get_mode_frequency(data, mode, info, return_figure=True,
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
