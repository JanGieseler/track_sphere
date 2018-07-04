## Example 0:
#  background subtraction using the method 'Bright px'
# Source file is 20171207_magnet.avi

import os
from glob import glob
import numpy as np
import pandas as pd
from track_sphere.read_write import load_info, load_info_to_dataframe, load_time_trace
from track_sphere.utils import get_position_file_names, power_spectral_density


source_folder_positions = '../processed_data/20180607_Sample_6_bead_1/position_data/'
target_folder_spectra = '../processed_data/20180607_Sample_6_bead_1/psd_data/'



# image_folder = '../images/20180607_Sample_6_bead_1/modes/'
method = 'fit_ellipse'

dataset = 'relaxation_run2'
# dataset = 'relaxation_run3'
# dataset = 'relaxation_run4'
dataset = 'relaxation_run5a'
dataset = 'relaxation_run5b'
dataset = 'relaxation_run5c'
# dataset = 'relaxation_run5d'
dataset = 'relaxation_run5g'
dataset = 'relaxation_run6'
dataset = 'relaxation_run7a'
dataset = 'relaxation_run7b'
################################################################################
## end settings ###
################################################################################

################################################################################
#### for real data: 20180607_Sample6_bead_1
################################################################################

if dataset == 'relaxation_run5a':
    position_file_names = get_position_file_names(source_folder_positions, method=method, runs=list(range(118, 123)))
    modes = 'xyr'
elif dataset == 'relaxation_run5b':
    position_file_names = get_position_file_names(source_folder_positions, method=method, runs=list(range(123, 128)))
    modes = 'xyr'
elif dataset == 'relaxation_run5c':
    position_file_names = get_position_file_names(source_folder_positions, method=method, runs=list(range(128, 132)))
    modes = 'xyr'
elif dataset == 'relaxation_run5d':
    position_file_names = get_position_file_names(source_folder_positions, method=method, runs=list(range(134, 141)))
    modes = 'xyr'
elif dataset == 'relaxation_run5g':
    position_file_names = get_position_file_names(source_folder_positions, method=method, runs=list(range(155, 158)))
    modes = 'xyr'
elif dataset == 'relaxation_run4':
    position_file_names = get_position_file_names(source_folder_positions, method=method)
    position_file_names = position_file_names[29:]  # all
    modes = 'xy'
elif dataset == 'relaxation_run3':
    position_file_names = get_position_file_names(source_folder_positions, method=method)
    position_file_names = position_file_names[18:22]  # all
    modes = 'xy'
elif dataset == 'relaxation_run2':
    position_file_names = get_position_file_names(source_folder_positions, method=method)
    position_file_names = position_file_names[7:16]  # all
    modes = 'xy'
elif dataset == 'relaxation_run6':
    source_folder_positions = '../processed_data/20180628_Sample_6_Bead_1/position_data/'
    target_folder_spectra = '../processed_data/20180628_Sample_6_Bead_1/psd_data/'
    position_file_names = get_position_file_names(source_folder_positions, method=method, runs=list(range(0, 4)))
    modes = 'xyr'
elif dataset == 'relaxation_run7a':
    source_folder_positions = '../processed_data/20180628_Sample_6_Bead_1/position_data/'
    target_folder_spectra = '../processed_data/20180628_Sample_6_Bead_1/psd_data/'
    position_file_names = get_position_file_names(source_folder_positions, method=method, runs=list(range(5, 20)))
    modes = 'xyr'
elif dataset == 'relaxation_run7b':
    source_folder_positions = '../processed_data/20180628_Sample_6_Bead_1/position_data/'
    target_folder_spectra = '../processed_data/20180628_Sample_6_Bead_1/psd_data/'
    position_file_names = get_position_file_names(source_folder_positions, method=method, runs=list(range(22, 38)))
    modes = 'xyzr'


################################################################################
#### run the script
################################################################################
info_in = load_info(position_file_names[0], folder_positions=source_folder_positions)
df_modes = load_info_to_dataframe(position_file_names, source_folder_positions, verbose=False)
nmax = int(df_modes.describe()['FrameCount']['min'])
fps = df_modes.describe()['FrameRate']['mean']
dt = 1./fps

psd_data = { m:[] for m in modes}
# psd_data = {'x': [], 'y': []}

for i, filename in enumerate(position_file_names):
    print(filename)
#
    # info = load_info(filename, folder_positions=source_folder_positions, verbose=False, return_filname=False)
    data, info = load_time_trace(filename, source_folder_positions=source_folder_positions, verbose=False)

    for mode in modes:
        if mode == 'r':
            x = data['ellipse angle'][0:nmax]
        elif mode == 'z':
            x = data['ellipse a'][0:nmax] * data['ellipse b'][0:nmax]
        else:
            x = data['ellipse ' + mode][0:nmax]
        x -= np.mean(x)
        f, p = power_spectral_density(x, time_step=dt)
        psd_data[mode].append(p)


if not os.path.exists(target_folder_spectra):
    os.mkdir(target_folder_spectra)


out_dict = {'f': f}
for mode in modes:
    out_dict['pm ('+mode+')'] = np.mean(psd_data[mode], axis=0)
    out_dict['ps ('+mode+')'] = np.std(psd_data[mode], axis=0)

    # export mode full data
    df = pd.DataFrame.from_dict({k:v for k, v in zip(position_file_names, psd_data[mode])})
    out_filename = dataset + '-' + mode + '.dat'
    df.to_csv(os.path.join(target_folder_spectra, out_filename))

# export average
df = pd.DataFrame.from_dict(out_dict)
out_filename = dataset + '.dat'
df.to_csv(os.path.join(target_folder_spectra, out_filename))






