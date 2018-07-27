
from track_sphere.utils import find_peaks_in_psd, get_position_file_names, power_spectral_density
from track_sphere.plot_data import plot_frequencies_zoom
from track_sphere.read_write import load_time_trace
import numpy as np
import os

experiment = 'levitation 8 bright px'
# experiment = 'levitation 8 fit ellipse'
experiment = 'levitation 7 bright px steady state'

if experiment == 'levitation 8 bright px':
    source_folder_positions = '../processed_data/20180718_M110_Sample_6_Bead_1/position_data/'
    method = 'Bright px'

    fmin = 0
    fmax = None
    nbin = 100  ## for peak finding
    nbin_zoom = 1  ## for peak plotting (zoom)
    max_number_of_peaks = 10
    height_threshold_factor = 3
    distance = 5

    df_zoom = 10

    run = 12  # runs 4,5,6,7,8, 13
    runs = list(range(run, run+1))
    runs = [11,12]
    position_file_names = get_position_file_names(source_folder_positions, method=method, runs=runs)
elif experiment == 'levitation 8 fit ellipse':
    source_folder_positions = '../processed_data/20180718_M110_Sample_6_Bead_1/position_data/'
    method = 'fit_ellipse'

    fmin = 0
    fmax = None
    nbin = 5  ## for peak finding
    nbin_zoom = 1  ## for peak plotting (zoom)
    max_number_of_peaks = 10
    height_threshold_factor = 3
    distance = 5

    df_zoom = 10

    run = 12  # runs 4,5,6,7,8, 13
    runs = list(range(run, run + 1))

    runs = [1, 2, 10]
    position_file_names = get_position_file_names(source_folder_positions, method=method, runs=runs)
elif experiment == 'levitation 7 bright px steady state':
    source_folder_positions = '../processed_data/20180710_M110_Sample_6_Bead_1/position_data/'
    method = 'Bright px'

    fmin = 0
    fmax = None
    nbin = 10  ## for peak finding
    nbin_zoom = 1  ## for peak plotting (zoom)
    max_number_of_peaks = 10
    height_threshold_factor = 3
    distance = 5

    df_zoom = 10

    run = 12  # runs 4,5,6,7,8, 13
    runs = list(range(run, run+1))
    runs = [17]+list(range(28,32))
    position_file_names = get_position_file_names(source_folder_positions, method=method, runs=runs)
modes = 'xy'
psd_data = {m:None for m in modes}
for filename in position_file_names:
    print(filename)
    data, info = load_time_trace(filename, source_folder_positions=source_folder_positions, verbose=False)
    dt = 1./info['info']['FrameRate']
    for mode in modes:
        x = data[method.lower().split('_')[-1] + ' ' + mode]
        x -= np.mean(x)
        f, p = power_spectral_density(x, time_step=dt)
        psd_data[mode] = p
    psd_data['f'] = f


    df_peaks = find_peaks_in_psd(psd_data, fmin=fmin, fmax=fmax, nbin=nbin, max_number_of_peaks=max_number_of_peaks,
                      height_threshold_factor=height_threshold_factor, distance=distance)


    folder = os.path.join(os.path.dirname(source_folder_positions[:-1]), 'peak_data')
    if not os.path.exists(folder):
        os.makedirs(folder)

    image_folder = os.path.join(folder, filename.split('-')[0]+'-peaks')
    if not os.path.exists(image_folder):
        os.makedirs(image_folder)


    # save images
    plot_frequencies_zoom(psd_data, df_peaks, image_folder, nbin_zoom, df_zoom)
    # save peak data
    df_peaks.to_csv(os.path.join(folder, filename.split('-')[0]+'-peaks.csv'), index=False)
